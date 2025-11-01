把 marinenav_env 搬到 Isaac Lab 上做 GPU 向量化，可以把“1 个 Python 环境 × 多机器人”的串行计算，升级为“上千个并行环境 × 多机器人”的纯 Torch/CUDA 张量计算，通常训练吞吐能提升一个数量级以上（取决于你并行的 env 数和观测/动力学复杂度）。

下面给你一条务实的改造路径：从“最少侵入式”的代码重构开始，到真正落地成 Isaac Lab 的 RLTask 与 VecEnv 管理。核心思想是三步：
- 用 Torch 张量重写你环境中的所有数值计算（包括流场、动力学、感知、奖励与终止），并把数据结构升维，支持 [E, R, …] 的批量；
- 用 Isaac Lab 的 RLTask 封装该批量环境，交给 SimulationApp/VecEnv 做 GPU 并行调度；
- 在不改你的策略/训练器接口的前提下，加一个轻量 wrapper，把 Isaac Lab 的 batch 观测打包回你现有的 (self_state, objects, mask) 结构，方便无痛复用 AC-IQN / IQN / SAC 等算法。

一、要点总览（你需要改什么）
- 数据组织从“列表 + for 循环”变为 Torch 张量广播：
  - 位置/姿态/速度：pos [E, R, 2]，theta [E, R]，vel_r/vel [E, R, 3]
  - 推力/舵角：left/right_thrust [E, R]
  - 漩涡核 cores：cores_xy [E, C, 2]，clockwise [E, C]，Gamma [E, C]
  - 障碍 obs：obs_xy [E, O, 2]，obs_r [E, O]
- 彻底去掉 scipy.spatial.KDTree 与 Python 循环，改用 torch.cdist / topk 筛最近对象；所有数学函数用 torch.*
- 感知与可见性：基于相对位置 + 角度扇区 + 距离阈值的向量化筛选；对 objects 结果做 pad 到 max_obj_num，并输出 mask（你现有 Agent 已支持 mask）
- 漩涡流速 get_velocity：对所有 cores 做广播叠加，必要时只取每个点最近的 K 个核心（topk）以降算力，替代 KDTree 与“外侧遮蔽”的启发式
- 动力学 compute_motion：把 Fossen 3-DOF/阻尼矩阵/推进力矩全部用 Torch 计算（支持 [E,R] 批量）
- 奖励/终止：距离差奖励、COLREGs 罚项、碰撞/到达/超时 done，全向量化
- Isaac Lab 集成：定义 MarineNavTask(RLTask)，提供 pre_physics_step、post_physics_step、reset_idx、get_observations、calculate_metrics、is_done。把 headless 配置成 GPU VecEnv
- 训练循环：从单环境改为多环境批量收集。你可以：
  - 方案A（推荐）：轻改 Trainer，让它把 [E,R] 批量展平成 [E×R] 个并行“智能体样本”，统一入 ReplayBuffer

二、关键改造：数学计算向量化（Torch/CUDA）
以漩涡叠加流速为例（替代 get_velocity 与 KDTree）。注意 piecewise 速度与顺/逆时针切向方向。

```python
# isaac_task/field.py
import torch

def compute_tangent_vel(pos_er, cores_xy_ec, clockwise_eC, Gamma_eC, core_r, device):
    # pos_er: [E,R,2]
    # cores_xy_ec: [E,C,2]
    # clockwise_eC: [E,C] bool / 0-1
    # Gamma_eC: [E,C]
    # 返回 v_currents: [E,R,2]，叠加所有核心后流速

    E, R, _ = pos_er.shape
    C = cores_xy_ec.shape[1]
    # 扩展维度做广播 [E,R,2] vs [E,C,2] -> [E,R,C,2]
    p = pos_er.unsqueeze(2)          # [E,R,1,2]
    c = cores_xy_ec.unsqueeze(1)     # [E,1,C,2]
    v_rad = p - c                    # [E,R,C,2]
    dist = torch.linalg.norm(v_rad, dim=-1).clamp_min(1e-6)  # [E,R,C]
    n = v_rad / dist.unsqueeze(-1)   # 单位径向

    # 旋转90度得到切向方向（顺逆时针）
    rot_cw = torch.tensor([[0., -1.],[1., 0.]], device=device)
    rot_ccw= torch.tensor([[0.,  1.],[-1.,0.]], device=device)
    # clockwise: True 用 rot_cw，否则 rot_ccw
    rot = torch.where(clockwise_eC.unsqueeze(0).unsqueeze(0).unsqueeze(-1).bool(),
                      rot_cw, rot_ccw)   # [E,R,C,2,2] via broadcast
    # 为简化，构造与 n 同形状的旋转：先把 rot 拓到 [E,1,C,2,2] 再 broadcast 到 [E,R,C,2,2]
    rot = rot.unsqueeze(0).unsqueeze(0).expand(E,R,C,2,2)
    t_hat = torch.matmul(rot, n.unsqueeze(-1)).squeeze(-1)  # [E,R,C,2]

    # 分段速度：d<=r 线性，d>r 1/d 衰减
    # speed = Gamma/(2*pi*r^2)*d (inside) else Gamma/(2*pi*d)
    two_pi = 2.0*torch.pi
    inside = dist <= core_r
    speed_inside  = Gamma_eC.unsqueeze(1) / (two_pi*core_r*core_r) * dist
    speed_outside = Gamma_eC.unsqueeze(1) / (two_pi*dist)
    speed = torch.where(inside, speed_inside, speed_outside)  # [E,R,C]

    v = t_hat * speed.unsqueeze(-1)  # [E,R,C,2]

    # 可选：只取每个点最近的K个核心减少算力
    # K = min(4, C)
    # d_top, idx_top = torch.topk(dist, k=K, largest=False)
    # 构建mask仅保留topk的核，略

    v_sum = v.sum(dim=2)  # [E,R,2]
    return v_sum
```

把 3-DOF 船模也向量化（这里给核心骨架，照你 robot.py 的矩阵式做广播）：

```python
# isaac_task/dynamics.py
import torch

def step_dynamics(state, action_lr, params, dt, N):
    # state: 字典/命名元组，包含 pos [E,R,2], theta [E,R], vel_r [E,R,3], vel [E,R,3],
    #        left/right_thrust/pos [E,R], 以及矩阵常量 m/Izz/导数/阻尼等，均为 [E,R] 或标量
    # action_lr: 连续动作 [-1,1]×2 映射到 推力变化/秒，和你原逻辑一致
    # params: 包含物理系数/几何参数
    # 这里省略细节，仅提示：所有运算保持张量形状一致，避免 for 循环。
    pass
```

感知/观测打包（含扇区、距离和 top-k 最近对象，输出定长列表 + mask）：

```python
# isaac_task/observation.py
import torch

def build_observation(pos_er, theta_er, vel_er, vel_r_er,
                      goal_er, left_thrust_er, right_thrust_er,
                      obs_xy_eo, obs_r_eo, robots_mask_er,  # robots_mask: 排除自身/失活
                      detect_range, detect_angle, max_obj_num):
    # 1) 自身特征（与你现有的7维一致）
    #   - 目标在自车坐标系：R(-theta)* (goal - pos)
    #   - 速度在自车坐标系：R(-theta)* vel[:2]
    cos, sin = torch.cos(theta_er), torch.sin(theta_er)
    R_T = torch.stack([torch.stack([ cos, sin], dim=-1),
                       torch.stack([-sin, cos], dim=-1)], dim=-2)  # [E,R,2,2]

    goal_vec = goal_er - pos_er                    # [E,R,2]
    goal_r = torch.matmul(R_T, goal_vec.unsqueeze(-1)).squeeze(-1)  # [E,R,2]

    v_abs = vel_er[..., :2]                        # [E,R,2]
    v_r = torch.matmul(R_T, v_abs.unsqueeze(-1)).squeeze(-1)        # [E,R,2]

    self_state = torch.stack([
        goal_r[...,0], goal_r[...,1],
        v_r[...,0],   v_r[...,1],
        vel_er[...,2], left_thrust_er, right_thrust_er
    ], dim=-1)  # [E,R,7]

    # 2) 其他机器人与静态障碍，统一当作“对象”：pos/vel/r
    #   2.1 机器人
    # 构造 pairwise 自车-他车（排除自身/失活），计算相对位置，转到自车坐标
    # 简化伪码：用 cdist 和广播，筛选距离<detect_range，角度在扇区内
    # 2.2 障碍
    # 2.3 合并后取最近的前 k 个，pad 到 max_obj_num，输出 [E,R,max_obj_num,5] 及 mask [E,R,max_obj_num]
    # 注意保留与 CPU 版本一致的噪声逻辑（可选：先去掉噪声，确认一致性后再加）

    objects, mask = ...  # [E,R,K,5], [E,R,K]  (K=max_obj_num)
    return self_state, objects, mask
```

三、把它塞进 Isaac Lab 的 RLTask
新建一个任务（不依赖物理网格，训练时甚至可以不生成几何体，单纯做张量计算即可；需要可视化时再实例化 USD/prim）：

```python
# isaac_task/marine_nav_task.py
import torch
from omni.isaac.lab.envs import RLTask  # Isaac Lab 基类（名称以你本机版本为准）
from .field import compute_tangent_vel
from .observation import build_observation
from .dynamics import step_dynamics

class MarineNavTask(RLTask):
    def __init__(self, cfg, device="cuda:0", **kwargs):
        super().__init__(cfg, **kwargs)
        self.device = torch.device(device)
        # 从 cfg 读取 E,R,C,O、半径阈值、奖励权重、时间上限等；并创建所有状态张量，放到 device
        self.num_envs = cfg.env.num_envs
        self.R = cfg.env.num_robots
        self.C = cfg.env.num_cores
        self.O = cfg.env.num_obstacles
        self.max_obj_num = cfg.env.max_obj_num

        # 预分配张量缓冲（位置、速度、推力、目标、核心/障碍参数等）
        # self.pos = torch.zeros(self.num_envs, self.R, 2, device=self.device)
        # ...
        self.done = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def set_up_scene(self, scene):
        # 训练可 headless，不必真的 spawn 几何体
        # 若需可视化，创建简单 prim 代表机器人/障碍/涡核
        pass

    def pre_physics_step(self, actions):
        # actions: [E,R,2] 连续动作（-1,1）
        # 映射动作 -> 推力变化，推进 N 子步，向前积分
        # 先计算海流：v_current = compute_tangent_vel(...)
        # 再调用 step_dynamics(...) 更新 self.vel_r/self.vel/theta/pos/推力
        pass

    def get_observations(self):
        self_obs, obj_obs, mask = build_observation(..., max_obj_num=self.max_obj_num)
        # Isaac Lab 标准接口通常返回扁平 torch.Tensor，但你也可保留三元组用于自定义
        return {
            "policy": {
                "self": self_obs,            # [E,R,7]
                "objects": obj_obs,          # [E,R,K,5]
                "objects_mask": mask,        # [E,R,K]
            }
        }

    def calculate_metrics(self):
        # 基于距离差、COLREGs 罚项、时间步惩罚等，输出 [E,R] 的 reward
        # 例：r = -0.1 + (dist_before - dist_after) + colregs_penalty + goal_bonus + collision_penalty
        self.rew = ...

    def is_done(self):
        # 条件：任一碰撞、到达、或长度>=上限
        # 任务层面 done 通常是 [E]；你也可以管理 per-robot 的“失活”mask，保持与原始设计一致
        self.done = ...
        return self.done

    def reset_idx(self, env_ids):
        # 对指定 env 重新采样 start/goal、cores、obstacles，并清零机器人状态
        # 你可以在 CPU 生成后 torch.as_tensor().to(device)，或直接用 torch 在 GPU 上拒绝采样
        pass
```

简单的 YAML（示意）：

```yaml
# cfg/marinenav_task.yaml
env:
  num_envs: 1024
  num_robots: 6
  num_cores: 8
  num_obstacles: 8
  map_width: 55.0
  map_height: 55.0
  core_r: 0.5
  detect_range: 20.0
  detect_angle: 6.28318530718  # 2*pi
  max_obj_num: 5
  max_episode_len: 1000
  reward:
    timestep_penalty: -0.1
    colregs_penalty_scale: 0.1
    collision_penalty: -5.0
    goal_bonus: 10.0
```

运行方式（示意，按你本机 Isaac Lab 的入口脚本为准）：
- 训练：headless, GPU 上 1024 env
- 如使用你自己的 Agent/Trainer，则用下节的 wrapper；如用 Isaac Lab 自带的 RL 框架（RSL-RL / RL Games），则写对应的 policy 接口。

四、复用你现有的 Agent/Trainer（最少侵入）
你的 Agent 期望的观测是三元组：(self_state_batch, object_batch, object_mask)，训练器逐机器人收集转移。我们做一个 wrapper，把 Isaac Lab 任务的批量观测展平成 [E×R]，并按你现有格式返回。

```python
# isaac_task/wrapper.py
import torch
import numpy as np

class IsaacMarineNavVecWrapper:
    def __init__(self, task, device="cuda:0"):
        self.task = task
        self.device = torch.device(device)
        self.E = task.num_envs
        self.R = task.R
        self.max_obj = task.max_obj_num

    def reset(self):
        self.task.reset()
        return self._pack_obs(self.task.get_observations())

    def step(self, actions, is_continuous_action=True):
        # actions: 你原来是 list[robots]；这里要求 [E*R,2] 或 list -> 我们 reshape 回 [E,R,2]
        if isinstance(actions, list):
            actions = np.array([a if a is not None else [0.0,0.0] for a in actions],
                               dtype=np.float32).reshape(self.E, self.R, -1)
        act = torch.as_tensor(actions, device=self.device)
        self.task.pre_physics_step(act)
        obs = self.task.get_observations()
        rews = self.task.rew      # [E,R]
        done = self.task.is_done()# [E]
        infos = {}                # 可填每个机器人状态

        states = self._pack_obs(obs)
        rewards = rews.reshape(-1).detach().cpu().numpy().tolist()
        # 将 env-level done 扩展到 robot-level，如果你有 per-robot 失活 mask，可按 mask 写入 dones
        dones = np.repeat(done.detach().cpu().numpy(), self.R).tolist()
        infos = [{"state":"normal"}]*(self.E*self.R)
        return states, rewards, dones, infos

    def _pack_obs(self, obs_dict):
        self_obs = obs_dict["policy"]["self"]         # [E,R,7]
        obj_obs  = obs_dict["policy"]["objects"]      # [E,R,K,5]
        mask     = obs_dict["policy"]["objects_mask"] # [E,R,K]
        E,R,K,_  = obj_obs.shape
        self_b   = self_obs.reshape(E*R, -1).detach().cpu().numpy()
        obj_b    = obj_obs.reshape(E*R, K, -1).detach().cpu().numpy()
        mask_b   = mask.reshape(E*R, K).detach().cpu().numpy()
        return (self_b, obj_b, mask_b)
```

然后在你 train_RL_agents.py 里，把 MarineNavEnv3(train) / MarineNavEnv3(eval) 换成上面的 wrapper 实例即可。Trainer 的循环几乎不用动（它本来就逐机器人处理），只是现在一次 step 会返回 E×R 个“机器人样本”。

五、对比你现有代码需要改/移除的点
- 移除 scipy.spatial.KDTree；改为 torch.cdist + topk
- 去掉所有 Python for 机器人/核心/障碍 的循环；统一用广播
- robot.py 的类不再逐实例创建；机器人参数放进批量张量
- perception_output/compute_COLREGs_* 改成张量版；COLREGs 逻辑可以先做一个“只对最近动态对象”的近似（与原先“找到第一个触发规则对象”一致），用 topk(1) 实现
- marinenav_env.py 的 reset_with_eval_config/episode_data/保存轨迹等，如果在训练阶段会成为性能瓶颈，建议只在评估模式下开启；训练期仅收奖励和 done

六、常见坑与建议
- 只在 GPU 上做计算：避免在每步把大张量搬回 CPU。ReplayBuffer 采样时再搬到 GPU；或把 Buffer 也改为 GPU（收益取决于规模）
- 先“功能等价”，再扩并行：建议先 E=64 做对齐验证（无噪声、相同随机种子），确保单步行为与旧环境一致，再拉到 E=512/1024+
- 使用 topk 限制核心/对象数，避免 C、O 很大时的 O(E×R×C/O) 成本爆炸
- 随机数：环境与策略各自独立 seed；GPU 上的随机可能和 CPU 有微差异，评估时注意
- 性能调参：把 max_obj_num 设成刚好够用就好；角度筛选先做，距离筛选再做，尽量早期裁剪；广播时注意中间张量大小

七、最小可运行骨架（组合）
给一个更完整的“只依赖 Torch 的 Isaac 任务最小骨架”，你可以按它填入细节，跑起来后再逐步替换 Trainer 里的环境实例。

```python
# run_isaac_marine_nav.py（示意，不含 GUI）
from isaac_task.marine_nav_task import MarineNavTask
from isaac_task.wrapper import IsaacMarineNavVecWrapper
from policy.agent import Agent
from policy.trainer import Trainer
import torch

def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    cfg = load_yaml("cfg/marinenav_task.yaml")  # 伪代码
    task = MarineNavTask(cfg, device=device)
    env = IsaacMarineNavVecWrapper(task, device=device)  # 训练环境（批量）
    eval_task = MarineNavTask(cfg, device=device)        # 评估也可用同一任务，但 E 较小
    eval_env = IsaacMarineNavVecWrapper(eval_task, device=device)

    agent = Agent(device=device, seed=0, agent_type="AC-IQN")  # 你的算法不变
    trainer = Trainer(train_env=env,
                      eval_env=eval_env,
                      eval_schedule=make_eval_schedule(),  # 可用你原来的
                      rl_agent=agent)
    # 注意：Trainer 内部无需知道 Isaac Lab，wrapper 给了同样的接口
    trainer.learn(total_timesteps=1_000_000,
                  eval_freq=50_000,
                  eval_log_path="./logs")

if __name__ == "__main__":
    main()
```