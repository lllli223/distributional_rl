import torch
import numpy as np

try:
    # Isaac Lab 2.3.0+ structure (optional for compatibility with docs)
    from omni.isaac.lab.utils import configclass
except ImportError:
    # Fallback: create a simple decorator for config classes
    def configclass(cls):  # type: ignore
        """Simple decorator for configuration classes"""
        return cls

from .field import compute_tangent_vel
from .observation import build_observation, add_perception_noise, compute_colregs_penalty
from .dynamics import step_dynamics


@configclass
class MarineNavTaskCfg:
    """Marine Navigation Task Configuration compatible with Isaac Lab 2.3.0"""
    num_envs: int = 1024
    num_robots: int = 6
    num_cores: int = 8
    num_obstacles: int = 8
    max_obj_num: int = 5
    
    # Environment parameters
    map_width: float = 55.0
    map_height: float = 55.0
    core_r: float = 0.5
    detect_range: float = 20.0
    detect_angle: float = 2 * np.pi  # 360 degrees
    max_episode_len: int = 1000
    
    # Reward parameters
    timestep_penalty: float = -0.1
    colregs_penalty_scale: float = 0.1
    collision_penalty: float = -5.0
    goal_bonus: float = 10.0
    goal_distance_threshold: float = 2.0
    
    # Physics parameters
    dt: float = 0.05
    N: int = 10  # steps per action


class MarineNavTask:
    """Marine Navigation Task compatible with Isaac Lab 2.3.0
    
    This is a custom GPU-accelerated task that uses Isaac Lab 2.3.0 conventions
    but implements custom physics without requiring USD scenes or heavy simulation.
    Compatible with gymnasium API and Isaac Lab's ManagerBasedRLEnv patterns.
    """
    def __init__(self, cfg: MarineNavTaskCfg, device="cuda:0", **kwargs):
        self.cfg = cfg
        self.device = torch.device(device)
        
        # Initialize configuration
        self.E = cfg.num_envs
        self.R = cfg.num_robots
        self.C = cfg.num_cores
        self.O = cfg.num_obstacles
        self.K = cfg.max_obj_num
        
        # Environment parameters
        self.width = cfg.map_width
        self.height = cfg.map_height
        self.core_r = cfg.core_r
        self.detect_range = cfg.detect_range
        self.detect_angle = cfg.detect_angle
        self.max_episode_len = cfg.max_episode_len
        
        # Reward parameters
        self.timestep_penalty = cfg.timestep_penalty
        self.colregs_penalty_scale = cfg.colregs_penalty_scale
        self.collision_penalty = cfg.collision_penalty
        self.goal_bonus = cfg.goal_bonus
        self.goal_threshold = cfg.goal_distance_threshold
        
        # Physics parameters
        self.dt = cfg.dt
        self.N = cfg.N
        
        # Physics constants (from robot.py)
        self.mass = 400.0  # kg
        self.Izz = 450.0   # kg*m^2
        self.length = 5.0  # m
        self.width_vessel = 2.5  # m
        self.r_collision = self.detect_r = 0.5 * np.sqrt(self.length**2 + self.width_vessel**2)
        
        # Initialize state tensors
        self._allocate_state_tensors()
        
        # Episode tracking
        self.episode_lengths = torch.zeros(self.E, device=self.device, dtype=torch.int32)

    def _allocate_state_tensors(self):
        """分配状态张量"""
        # Robot states [E, R, ...]
        self.pos = torch.zeros(self.E, self.R, 2, device=self.device)
        self.theta = torch.zeros(self.E, self.R, device=self.device)
        self.vel_r = torch.zeros(self.E, self.R, 3, device=self.device)  # relative velocity
        self.vel = torch.zeros(self.E, self.R, 3, device=self.device)    # absolute velocity
        self.left_thrust = torch.zeros(self.E, self.R, device=self.device)
        self.right_thrust = torch.zeros(self.E, self.R, device=self.device)
        
        # Environment states
        self.cores_xy = torch.zeros(self.E, self.C, 2, device=self.device)
        self.cores_clockwise = torch.zeros(self.E, self.C, device=self.device, dtype=torch.bool)
        self.cores_Gamma = torch.zeros(self.E, self.C, device=self.device)
        
        self.obstacles_xy = torch.zeros(self.E, self.O, 2, device=self.device)
        self.obstacles_r = torch.zeros(self.E, self.O, device=self.device)
        
        # Goals and starts
        self.goals = torch.zeros(self.E, self.R, 2, device=self.device)
        self.starts = torch.zeros(self.E, self.R, 2, device=self.device)
        
        # Status tracking
        self.collision = torch.zeros(self.E, self.R, device=self.device, dtype=torch.bool)
        self.reach_goal = torch.zeros(self.E, self.R, device=self.device, dtype=torch.bool)
        self.deactivated = torch.zeros(self.E, self.R, device=self.device, dtype=torch.bool)
        
        # Episode and step counters
        self.episode_step = torch.zeros(self.E, device=self.device, dtype=torch.int32)

    def set_up_scene(self, scene=None):
        """兼容Isaac Lab接口的占位实现（无USD场景要求）。"""
        # 训练模式下可以不需要可视化的几何体
        return scene

    def reset(self):
        """重置所有环境"""
        env_ids = torch.arange(self.E, device=self.device)
        self.reset_idx(env_ids)

    def reset_idx(self, env_ids):
        """重置指定环境"""
        if len(env_ids) == 0:
            return
        
        # 重新采样环境和机器人状态
        self._sample_environment(env_ids)
        self._reset_robots(env_ids)
        
        # 重置episode计数器
        self.episode_step[env_ids] = 0

    def _sample_environment(self, env_ids):
        """批量采样环境状态（漩涡核心和障碍物）"""
        n = len(env_ids)
        if n == 0:
            return
        
        # 批量采样漩涡核心 [n, C, 2]
        cores_xy = torch.rand(n, self.C, 2, device=self.device) * torch.tensor(
            [self.width, self.height], device=self.device)
        clockwise = torch.rand(n, self.C, device=self.device) > 0.5
        v_edge = torch.rand(n, self.C, device=self.device) * 3 + 3  # [3, 6]
        Gamma = 2 * np.pi * self.core_r * v_edge  # [n, C]
        
        # 批量赋值
        self.cores_xy[env_ids] = cores_xy
        self.cores_clockwise[env_ids] = clockwise
        self.cores_Gamma[env_ids] = Gamma
        
        # 批量采样障碍物 [n, O, 2]
        obstacles_xy = torch.rand(n, self.O, 2, device=self.device) * torch.tensor(
            [self.width-10, self.height-10], device=self.device) + 5
        obstacles_r = torch.rand(n, self.O, device=self.device) + 1  # [1, 2]
        
        # 批量赋值
        self.obstacles_xy[env_ids] = obstacles_xy
        self.obstacles_r[env_ids] = obstacles_r

    def _reset_robots(self, env_ids):
        """批量重置机器人状态"""
        n = len(env_ids)
        if n == 0:
            return
        
        # 批量生成起点和终点（距离足够远）
        for env_idx, env_id in enumerate(env_ids):
            for r in range(self.R):
                start, goal = self._generate_start_goal(env_id)
                self.starts[env_id, r] = start
                self.goals[env_id, r] = goal
                self.pos[env_id, r] = start
        
        # 批量设置机器人初始状态
        self.theta[env_ids] = torch.rand(n, self.R, device=self.device) * 2 * np.pi
        self.vel_r[env_ids] = 0.0
        self.left_thrust[env_ids] = 0.0
        self.right_thrust[env_ids] = 0.0
        
        # 批量计算初始流速
        initial_velocity = self._compute_current_velocity(self.pos[env_ids])  # [n, R, 2]
        # 扩展到3D速度（添加角速度维度）
        self.vel[env_ids, :, :2] = initial_velocity
        self.vel[env_ids, :, 2] = 0.0
        
        # 重置状态标志
        self.collision[env_ids] = False
        self.reach_goal[env_ids] = False
        self.deactivated[env_ids] = False

    def _generate_start_goal(self, env_id):
        """生成起点和终点，确保距离足够"""
        for _ in range(50):  # 最多尝试50次
            start = torch.rand(2, device=self.device) * torch.tensor([self.width-4, self.height-4], device=self.device) + 2
            goal = torch.rand(2, device=self.device) * torch.tensor([self.width-4, self.height-4], device=self.device) + 2
            
            if torch.norm(goal - start) >= 30.0:
                return start, goal
        
        # 如果找不到合适的，直接返回
        start = torch.tensor([5.0, 5.0], device=self.device)
        goal = torch.tensor([self.width-5, self.height-5], device=self.device)
        return start, goal

    def pre_physics_step(self, actions):
        """执行动作并推进环境一个控制步（包含N个积分子步）。"""
        # actions: [E, R, 2] 连续动作 [-1,1]
        actions = actions.to(self.device)

        # 准备物理参数
        physics_params = {
            'mass': torch.full((self.E, self.R), self.mass, device=self.device),
            'Izz': torch.full((self.E, self.R), self.Izz, device=self.device),
            'length': torch.full((self.E, self.R), self.length, device=self.device),
            'width': torch.full((self.E, self.R), self.width_vessel, device=self.device)
        }
        
        state = {
            'pos': self.pos,
            'theta': self.theta,
            'vel_r': self.vel_r,
            'vel': self.vel,
            'left_thrust': self.left_thrust,
            'right_thrust': self.right_thrust
        }

        dt = self.dt
        for _ in range(self.N):
            # 1) 计算当前位置的流速（current）
            current_velocity = self._compute_current_velocity(state['pos'])  # [E,R,2]
            # 2) 绝对速度 = 相对速度 + 流速
            abs_vel = state['vel_r'].clone()
            abs_vel[..., :2] = abs_vel[..., :2] + current_velocity
            state['vel'] = abs_vel
            
            # 3) 先用绝对速度推进位姿（与原环境一致）
            self.pos = state['pos'] = state['pos'] + abs_vel[..., :2] * dt
            self.theta = state['theta'] = torch.remainder(state['theta'] + abs_vel[..., 2] * dt, 2 * np.pi)

            # 4) 再更新相对速度与推力（船体动力学）
            state = step_dynamics(state, actions, physics_params, dt, self.device)
            self.vel_r = state['vel_r']
            self.left_thrust = state['left_thrust']
            self.right_thrust = state['right_thrust']
            # 绝对速度将在下一子步重算

        # 更新绝对速度（控制步末）
        current_velocity = self._compute_current_velocity(self.pos)
        self.vel = self.vel_r.clone()
        self.vel[..., :2] = self.vel[..., :2] + current_velocity
        
        # 步计数
        self.episode_step += 1

    def _compute_current_velocity(self, pos_er):
        """计算当前位置的流速"""
        return compute_tangent_vel(
            pos_er, 
            self.cores_xy, 
            self.cores_clockwise, 
            self.cores_Gamma, 
            self.core_r, 
            self.device
        )

    def get_observations(self):
        """获取观测"""
        # 构建观测
        self_obs, obj_obs, obj_mask, collision = build_observation(
            self.pos, self.theta, self.vel, self.vel_r,
            self.goals, self.left_thrust, self.right_thrust,
            self.obstacles_xy, self.obstacles_r,
            self.pos, torch.full((self.E, self.R), self.r_collision, device=self.device),
            self.detect_range, self.detect_angle, self.K, self.device
        )
        
        # 添加感知噪声（可选）
        # self_obs, obj_obs = add_perception_noise(self_obs, obj_obs, obj_mask, self.device)
        
        # 更新碰撞状态
        self.collision = collision
        
        return {
            "policy": {
                "self": self_obs,
                "objects": obj_obs,
                "objects_mask": obj_mask,
            }
        }

    def calculate_metrics(self):
        """计算奖励（修正COLREGs惩罚方向：违规应扣分）。"""
        reward = torch.full((self.E, self.R), self.timestep_penalty, device=self.device)
        distances_to_goal = torch.norm(self.goals - self.pos, dim=-1)
        if hasattr(self, '_last_distances'):
            reward += (self._last_distances - distances_to_goal)
        self._last_distances = distances_to_goal
        
        colregs_count = compute_colregs_penalty(
            self.pos, self.vel, 
            torch.full((self.E, self.R), self.r_collision, device=self.device),
            self.theta, self.detect_angle, self.device
        )
        # 扣分：scale为正，乘以负号
        reward -= colregs_count * self.colregs_penalty_scale
        
        reward[self.collision] += self.collision_penalty
        goal_reached = distances_to_goal <= self.goal_threshold
        reward[goal_reached] += self.goal_bonus
        self.reach_goal = goal_reached
        
        return reward

    def is_done(self):
        """环境结束仅在超时或所有机器人都失活（碰撞或到达）时为True。"""
        # 更新失活标志
        self.deactivated = self.deactivated | self.collision | self.reach_goal
        timeout = self.episode_step >= self.max_episode_len
        all_robots_done = torch.all(self.deactivated, dim=-1)
        env_done = timeout | all_robots_done
        return env_done

    def get_termination_conditions(self):
        """获取终止条件"""
        return {
            "time_out": self.episode_step >= self.max_episode_len,
            "collision": self.collision,
            "goal_reached": self.reach_goal
        }
