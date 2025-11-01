import torch
import numpy as np
from typing import List, Tuple, Optional, Dict, Any

class _RobotProxy:
    """最小代理，提供Trainer需要的属性/方法。"""
    def __init__(self, env, e_idx, r_idx):
        self._env = env
        self._e = e_idx
        self._r = r_idx

    @property
    def deactivated(self):
        return bool(self._env.task.deactivated[self._e, self._r].item())
    
    @deactivated.setter
    def deactivated(self, value):
        self._env.task.deactivated[self._e, self._r] = torch.as_tensor(
            value, dtype=torch.bool, device=self._env.task.deactivated.device
        )

    @property
    def collision(self):
        return bool(self._env.task.collision[self._e, self._r].item())
    
    @collision.setter
    def collision(self, value):
        self._env.task.collision[self._e, self._r] = torch.as_tensor(
            value, dtype=torch.bool, device=self._env.task.collision.device
        )

    @property
    def reach_goal(self):
        return bool(self._env.task.reach_goal[self._e, self._r].item())
    
    @reach_goal.setter
    def reach_goal(self, value):
        self._env.task.reach_goal[self._e, self._r] = torch.as_tensor(
            value, dtype=torch.bool, device=self._env.task.reach_goal.device
        )

    @property
    def dt(self):
        return float(self._env.task.dt)

    @property
    def N(self):
        return int(self._env.task.N)

    def compute_step_energy_cost(self):
        # 近似使用|thrust|之和 * dt * N（与原逻辑保持一致量纲）
        lt = float(self._env.task.left_thrust[self._e, self._r].abs().item())
        rt = float(self._env.task.right_thrust[self._e, self._r].abs().item())
        return (lt + rt) * self.dt * self.N


class IsaacMarineNavVecWrapper:
    """
    Isaac Lab环境包装器，兼容原有训练代码接口
    将批量环境展平为原有格式
    """
    
    def __init__(self, task, device="cuda:0"):
        self.task = task
        self.device = torch.device(device)
        self.E = task.E
        self.R = task.R
        self.max_obj = task.K
        # 预建robots代理列表（展平成E*R）
        self.robots = [_RobotProxy(self, e, r) for e in range(self.E) for r in range(self.R)]
        # 确保任务已初始化
        if hasattr(task, 'reset'):
            task.reset()
    
    def reset(self) -> Tuple[List, List, List]:
        """重置环境，返回原有格式的观测"""
        # 重置Isaac Lab任务
        all_env_ids = torch.arange(self.E, device=self.device)
        self.task.reset_idx(all_env_ids)
        obs_list = self._get_observations()
        # 生成collisions/reach_goals（逐机器人）
        collisions = [bool(self.task.collision[e, r].item()) for e in range(self.E) for r in range(self.R)]
        reach_goals = [bool(self.task.reach_goal[e, r].item()) for e in range(self.E) for r in range(self.R)]
        return obs_list, collisions, reach_goals
    
    def step(self, actions: List, is_continuous_action: bool = True) -> Tuple[List, List, List, List]:
        """
        执行动作，返回原有格式的结果
        
        参数:
        - actions: list，长度为机器人总数 [total_robots]
        - is_continuous_action: 是否连续动作
        
        返回:
        - observations: [(self_obs, objects_obs), ...]
        - rewards: [reward1, reward2, ...]
        - dones: [done1, done2, ...]  
        - infos: [{"state": "normal"}, ...]
        """
        # 将actions从列表转换为张量
        if isinstance(actions, list):
            # 假设actions是每个机器人的动作，展平为[E*R, 2]
            if len(actions) == self.E * self.R:
                actions_array = np.array(actions, dtype=np.float32).reshape(self.E, self.R, -1)
            else:
                # 假设actions是每个环境的动作列表
                actions_array = np.array(actions, dtype=np.float32)
        else:
            actions_array = actions
        act_tensor = torch.as_tensor(actions_array, device=self.device)
        if act_tensor.dim() == 2 and act_tensor.shape == (self.E * self.R, 2):
            act_tensor = act_tensor.reshape(self.E, self.R, 2)
        elif act_tensor.dim() == 1 and act_tensor.numel() == self.E * self.R * 2:
            act_tensor = act_tensor.reshape(self.E, self.R, 2)

        # 执行动作
        self.task.pre_physics_step(act_tensor)
        
        # 观测/奖励/终止
        observations = self._get_observations()
        rewards_tensor = self.task.calculate_metrics()
        rewards = rewards_tensor.reshape(-1).detach().cpu().numpy().tolist()
        
        # 更新失活标志（必须在计算奖励后调用）
        self.task.is_done()
        
        # done 逐机器人：碰撞/到达 或 超时
        dones = []
        timeout_env = self.task.episode_step >= self.task.max_episode_len
        for e in range(self.E):
            env_timeout = bool(timeout_env[e].item()) if isinstance(timeout_env, torch.Tensor) else bool(timeout_env)
            for r in range(self.R):
                dr = bool(self.task.collision[e, r].item()) or bool(self.task.reach_goal[e, r].item()) or env_timeout
                dones.append(dr)
        
        infos = self._get_infos()
        return observations, rewards, dones, infos
    
    def _get_observations(self) -> List[Tuple]:
        """获取观测并转换为原有格式"""
        # 获取Isaac Lab观测
        obs_dict = self.task.get_observations()
        self_obs = obs_dict["policy"]["self"]
        obj_obs = obs_dict["policy"]["objects"]
        obj_mask = obs_dict["policy"]["objects_mask"]
        E, R, K, _ = obj_obs.shape
        self_obs_flat = self_obs.reshape(E * R, -1).detach().cpu().numpy()
        obj_obs_flat = obj_obs.reshape(E * R, K, -1).detach().cpu().numpy()
        obj_mask_flat = obj_mask.reshape(E * R, K).detach().cpu().numpy().astype(bool)
        # Trainer 的ReplayBuffer会自己补mask，这里保持原格式返回 (self, objects)
        observations = []
        for i in range(E * R):
            valid_objects = obj_obs_flat[i][obj_mask_flat[i]]
            observations.append((self_obs_flat[i], valid_objects))
        return observations
    
    def _get_infos(self) -> List[Dict]:
        """获取信息"""
        # 生成信息字典
        infos = []
        
        for e in range(self.E):
            for r in range(self.R):
                if self.task.collision[e, r]:
                    state = "collision"
                elif self.task.reach_goal[e, r]:
                    state = "reach goal"
                elif self.task.episode_step >= self.task.max_episode_len:
                    state = "too long episode"
                else:
                    state = "normal"
                infos.append({"state": state})
        
        return infos

    # 供Trainer调用的辅助接口
    def check_all_deactivated(self) -> bool:
        return bool(torch.all(self.task.collision | self.task.reach_goal, dim=-1).all().item())

    def check_any_collision(self) -> bool:
        return bool(self.task.collision.any().item())

    def check_all_reach_goal(self) -> bool:
        return bool(self.task.reach_goal.all().item())

    # 评估回放接口占位（可后续完善完全对齐）
    def reset_with_eval_config(self, config: Dict):
        # 简化：当前先使用随机重置，未来按config恢复（cores/obstacles/robots）。
        return self.reset()

    def get_action_space_dimension(self) -> int:
        """获取动作空间维度"""
        return 2  # 连续动作：左推力和右推力变化
    
    def get_num_robots(self) -> int:
        """获取机器人总数"""
        return self.E * self.R
    
    def get_env_count(self) -> int:
        """获取环境数量"""
        return self.E
    
    def set_curriculum_schedule(self, schedule: Dict):
        """设置课程学习计划"""
        # 这里可以实现课程学习逻辑
        # 暂时跳过
        pass
    
    def is_eval_env(self, is_eval: bool):
        """设置评估模式"""
        # Isaac Lab模式下，这个可能不需要
        pass
    
    def save_episode(self, filename: str):
        """保存 episode 数据"""
        # Isaac Lab模式下，这个功能可能需要特殊实现
        # 或者可以跳过，因为主要关注训练性能
        pass
    
    def episode_data(self) -> Dict:
        """获取 episode 数据"""
        # 返回当前episode的数据，用于可视化或分析
        data = {
            "env_count": self.E,
            "robot_count": self.R,
            "episode_step": self.task.episode_step.tolist(),
            "positions": self.task.pos.tolist(),
            "goals": self.task.goals.tolist(),
            "collisions": self.task.collision.tolist(),
            "reach_goals": self.task.reach_goal.tolist()
        }
        return data


class MarineNavEnvWrapper:
    """
    简化的环境包装器，用于测试和开发
    """
    
    def __init__(self, num_envs=64, num_robots=6, device="cuda:0"):
        from .marine_nav_task import MarineNavTask, MarineNavTaskCfg
        
        # 创建任务配置
        cfg = MarineNavTaskCfg()
        cfg.num_envs = num_envs
        cfg.num_robots = num_robots
        
        # 创建任务
        self.task = MarineNavTask(cfg, device=device)
        self.wrapper = IsaacMarineNavVecWrapper(self.task, device=device)
    
    def reset(self):
        return self.wrapper.reset()
    
    def step(self, actions, is_continuous_action=True):
        return self.wrapper.step(actions, is_continuous_action)
    
    def get_action_space_dimension(self):
        return self.wrapper.get_action_space_dimension()
    
    def get_num_robots(self):
        return self.wrapper.get_num_robots()
