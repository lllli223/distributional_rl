import math
import torch
import numpy as np
from typing import Optional, Tuple, List

# ============================================================
# GPU-optimized velocity field computation
# ============================================================

def compute_speed_torch(Gamma: torch.Tensor, d: torch.Tensor, r: float) -> torch.Tensor:
    """速度标量计算（与CPU版本一致）"""
    two_pi = 2.0 * math.pi
    r_sq = r * r
    d_safe = torch.clamp(d, min=1e-12)
    inner = (Gamma * d_safe) / (two_pi * r_sq)
    outer = Gamma / (two_pi * d_safe)
    return torch.where(d_safe <= r + 1e-12, inner, outer)


def _get_velocity_vectorized(
    pos: torch.Tensor,           # [B,R,2]
    cores_pos: torch.Tensor,     # [B,C,2]
    cores_gamma: torch.Tensor,   # [B,C]
    cores_sign: torch.Tensor,    # [B,C]
    r: float
) -> torch.Tensor:
    """
    GPU向量化流场计算（与CPU版本数值一致，包含遮蔽机制）
    返回 [B,R,3]
    """
    B, R = pos.shape[0], pos.shape[1]
    C = 0 if cores_pos is None else cores_pos.shape[1]
    
    if C == 0:
        return torch.zeros((B, R, 3), device=pos.device, dtype=pos.dtype)
    
    # [B,R,C,2]: 每个robot到每个core的相对位置
    rel = cores_pos.unsqueeze(1) - pos.unsqueeze(2)  # [B,C,2] -> [B,1,C,2], [B,R,2] -> [B,R,1,2]
    
    # [B,R,C]: 距离
    d = torch.linalg.norm(rel, dim=-1)  # [B,R,C]
    
    # 按距离排序（保持与CPU版本一致）
    d_sorted, sort_idx = torch.sort(d, dim=-1)  # [B,R,C]
    
    # 重排rel, gamma, sign
    batch_idx = torch.arange(B, device=pos.device).view(B, 1, 1, 1).expand(B, R, C, 2)
    robot_idx = torch.arange(R, device=pos.device).view(1, R, 1, 1).expand(B, R, C, 2)
    core_idx = sort_idx.unsqueeze(-1).expand(B, R, C, 2)
    
    rel_sorted = rel.gather(2, core_idx)  # [B,R,C,2]
    gamma_sorted = cores_gamma.unsqueeze(1).expand(B, R, -1).gather(2, sort_idx)  # [B,R,C]
    sign_sorted = cores_sign.unsqueeze(1).expand(B, R, -1).gather(2, sort_idx)  # [B,R,C]
    
    # 遮蔽机制：与CPU版本完全一致
    # CPU的continue在内层循环，不会跳出外层循环，所以实际上没有遮蔽！
    # 所有cores都会贡献速度
    
    # 归一化径向向量
    d_safe = torch.clamp(d_sorted, min=1e-12).unsqueeze(-1)  # [B,R,C,1]
    r_hat = rel_sorted / d_safe  # [B,R,C,2]
    
    # 切向向量（根据旋转方向）
    t_x = -sign_sorted * r_hat[..., 1]  # [B,R,C]
    t_y = sign_sorted * r_hat[..., 0]   # [B,R,C]
    
    # 速度标量
    speed = compute_speed_torch(gamma_sorted, d_sorted, r)  # [B,R,C]
    
    # 合成速度（所有core贡献求和）
    vx = (t_x * speed).sum(dim=-1)  # [B,R]
    vy = (t_y * speed).sum(dim=-1)  # [B,R]
    
    return torch.stack([vx, vy, torch.zeros_like(vx)], dim=-1)  # [B,R,3]


class TorchMarineNavEnv:
    """
    GPU 批量版 MarineNav 环境（子任务 2/5：对齐动力学积分）

    核心改动：
      - 环境参数与 MarineNavEnv3 对齐
      - reset: schedule 课程 + 约束生成（start/goal、cores、obstacles）
      - 流场计算：一致版 piecewise + 遮蔽规则（_get_velocity_consistent）
      - 动力学：附加质量 A、线性阻尼 D、非线性阻尼 D_n、Coriolis C_RB + C_A、N 子步积分
      - 动作接口：连续/离散动作解析、推力变化率积分
      - 返回接口：reset/step 仍返回 GPU torch.Tensor
    """
    def __init__(
        self,
        B: int = 1,
        R: int = 6,
        C: int = 8,
        O: int = 8,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
        seed: int = 0,
        schedule: Optional[dict] = None,
        is_eval_env: bool = False
    ):
        # 设备与 dtype
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.dtype = dtype

        # 批量与数量
        self.B = int(B)
        self.R = int(R)
        self.C = int(C)
        self.O = int(O)

        # MarineNavEnv3 参数对齐
        self.width = 55
        self.height = 55
        self.r = 0.5  # 统一命名: vortex core 半径
        self.v_rel_max = 1.0
        self.p = 0.8
        self.v_range = [3.0, 3.0]   # edge speed range
        self.obs_range = [1.0, 1.0]
        self.clear_r = 10.0
        self.angular_speed_max = np.pi / 2
        self.angular_speed_penalty = -1.0
        self.steering_reward_angle_max = np.pi / 3
        self.steering_reward_speed_min = 1.0
        self.steering_reward = 0.5
        self.timestep_penalty = -0.1
        self.COLREGs_penalty = -0.1
        self.collision_penalty = -5.0
        self.goal_reward = 10.0
        self.min_start_goal_dis = 30.0

        # 兼容旧命名
        self.r_core = self.r

        # 课程 schedule 与计数
        self.schedule = schedule
        self.is_eval_env = is_eval_env
        self.episode_timesteps = 0
        self.total_timesteps = 0
        # 使用torch.Generator实现GPU原生随机数生成（替代CPU numpy.RandomState）
        self._rg = torch.Generator(device=self.device)
        if seed is not None:
            self._rg.manual_seed(seed)

        # 机器人动力学参数（与 Robot 类默认值一致）
        self.dt = 0.05
        self.N = 10  # 子步数
        self.goal_dis = 2.0
        self.max_obj_num = 5
        
        # 几何
        self.length = 2.0
        self.width_m = 0.8  # 机器人宽度（避免与 env.width 混淆）
        
        # 推力器参数
        self.min_thrust = -1e4
        self.max_thrust = 1e4
        self.left_pos = 0.0   # 推力器角度（弧度）
        self.right_pos = 0.0
        
        # 动力学常量（从 robot.py 复制）
        self.m = 50.0
        self.Izz = 10.0
        self.xDotU = 25.0
        self.yDotV = 50.0
        self.yDotR = 2.0
        self.nDotR = 10.0
        self.nDotV = 2.0
        self.xU = 5.0
        self.xUU = 25.0
        self.yV = 10.0
        self.yVV = 50.0
        self.yR = 2.0
        self.yRV = 5.0
        self.yVR = 5.0
        self.yRR = 10.0
        self.nR = 2.0
        self.nRR = 10.0
        self.nV = 2.0
        self.nVV = 5.0
        self.nRV = 2.0
        self.nVR = 2.0
        
        # 动作空间（离散动作选项）
        self.left_thrust_change = np.array([-1000, -500, 0, 500, 1000], dtype=np.float32)
        self.right_thrust_change = np.array([-1000, -500, 0, 500, 1000], dtype=np.float32)
        # 生成所有动作组合
        self.actions = []
        for l in self.left_thrust_change:
            for r in self.right_thrust_change:
                self.actions.append((float(l), float(r)))
        self.actions_tensor = torch.tensor(self.actions, device=self.device, dtype=self.dtype)
        
        # 预计算动力学矩阵（批量 GPU）
        self._build_dynamics_matrices()

        # 状态与环境张量（初始化为空，reset 时重建）
        self.pos = None             # [B,R,2]
        self.theta = None           # [B,R]
        self.velocity_r = None      # [B,R,3]
        self.velocity = None        # [B,R,3]
        self.left_thrust = None     # [B,R]
        self.right_thrust = None    # [B,R]
        self.goal = None            # [B,R,2]
        self.cores_pos = None       # [B,C,2]
        self.cores_gamma = None     # [B,C]
        self.cores_sign = None      # [B,C] (+1.0 clockwise, -1.0 else)
        self.obs_pos = None         # [B,O,2]
        self.obs_r = None           # [B,O]
        
        # Robot state tracking
        self.reach_goal_flags = None    # [B,R] bool
        self.collision_flags = None     # [B,R] bool
        self.deactivated_flags = None   # [B,R] bool

    def _build_dynamics_matrices(self):
        """
        预计算动力学矩阵（与 robot.py 一致）
        A_const = M + MA (附加质量 + 刚体质量)
        D = 线性阻尼矩阵
        在 step 中构造 D_n = 非线性阻尼（速度依赖）
        C_RB = 刚体 Coriolis
        C_A = 附加质量 Coriolis
        """
        dev, dtp = self.device, self.dtype
        R = self.R
        
        # 附加质量矩阵 MA
        MA = torch.zeros((R, 3, 3), device=dev, dtype=dtp)
        MA[:, 0, 0] = self.xDotU
        MA[:, 1, 1] = self.yDotV
        MA[:, 1, 2] = self.yDotR
        MA[:, 2, 1] = self.nDotV
        MA[:, 2, 2] = self.nDotR
        
        # 刚体质量矩阵 M
        M = torch.zeros((R, 3, 3), device=dev, dtype=dtp)
        M[:, 0, 0] = self.m
        M[:, 1, 1] = self.m
        M[:, 2, 2] = self.Izz
        
        # 总惯性矩阵 A_const = M + MA
        self.A_const = M + MA  # [R,3,3]
        
        # 线性阻尼矩阵 D
        D = torch.zeros((R, 3, 3), device=dev, dtype=dtp)
        D[:, 0, 0] = -self.xU
        D[:, 1, 1] = -self.yV
        D[:, 1, 2] = -self.yR
        D[:, 2, 1] = -self.nV
        D[:, 2, 2] = -self.nR
        self.D = D  # [R,3,3]
        
        # Cholesky 分解 A_const（用于快速求解 Ax=b）
        self.L_chol = torch.linalg.cholesky(self.A_const)  # [R,3,3]
        
    def _safe(self, t: torch.Tensor, pos: float = 1e6, neg: float = -1e6) -> torch.Tensor:
        return torch.nan_to_num(t, nan=0.0, posinf=pos, neginf=neg)
        
    def _cholesky_solve_batch(self, L: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        批量 Cholesky 求解 A x = b，其中 L = cholesky(A)
        L: [B,R,3,3], b: [B,R,3]
        返回 x: [B,R,3]
        """
        # 先解 L y = b
        y = torch.linalg.solve_triangular(L, b.unsqueeze(-1), upper=False).squeeze(-1)
        # 再解 L^T x = y
        x = torch.linalg.solve_triangular(L.transpose(-2, -1), y.unsqueeze(-1), upper=True).squeeze(-1)
        return x

    # =========================
    # reset + 生成约束（一致化）
    # =========================
    def reset(self):
        # schedule 支持
        if self.schedule is not None:
            steps = self.schedule["timesteps"]
            diffs = np.array(steps) - self.total_timesteps
            idx = int(np.sum(diffs <= 0) - 1)
            idx = max(0, min(idx, len(steps) - 1))
            self.R = int(self.schedule["num_robots"][idx])
            self.C = int(self.schedule["num_cores"][idx])
            self.O = int(self.schedule["num_obstacles"][idx])
            self.min_start_goal_dis = float(self.schedule["min_start_goal_dis"][idx])

            print("\n======== training schedule ========")
            print("num of robots: ", self.R)
            print("num of cores: ", self.C)
            print("num of obstacles: ", self.O)
            print("min start goal dis: ", self.min_start_goal_dis)
            print("======== training schedule ========\n")

        self._build_dynamics_matrices()

        # 重新分配状态与环境张量
        B, R, C, O = self.B, self.R, self.C, self.O
        dev, dtp = self.device, self.dtype

        self.pos = torch.zeros((B, R, 2), device=dev, dtype=dtp)
        self.theta = torch.zeros((B, R), device=dev, dtype=dtp)
        self.velocity_r = torch.zeros((B, R, 3), device=dev, dtype=dtp)
        self.velocity = torch.zeros((B, R, 3), device=dev, dtype=dtp)
        self.left_thrust = torch.zeros((B, R), device=dev, dtype=dtp)
        self.right_thrust = torch.zeros((B, R), device=dev, dtype=dtp)
        self.goal = torch.zeros((B, R, 2), device=dev, dtype=dtp)

        self.cores_pos = torch.zeros((B, C, 2), device=dev, dtype=dtp) if C > 0 else None
        self.cores_gamma = torch.zeros((B, C), device=dev, dtype=dtp) if C > 0 else None
        self.cores_sign = torch.zeros((B, C), device=dev, dtype=dtp) if C > 0 else None

        self.obs_pos = torch.zeros((B, O, 2), device=dev, dtype=dtp) if O > 0 else None
        self.obs_r = torch.zeros((B, O), device=dev, dtype=dtp) if O > 0 else None
        
        # Robot state flags
        self.reach_goal_flags = torch.zeros((B, R), device=dev, dtype=torch.bool)
        self.collision_flags = torch.zeros((B, R), device=dev, dtype=torch.bool)
        self.deactivated_flags = torch.zeros((B, R), device=dev, dtype=torch.bool)

        # 生成：先机器人，再核心，再障碍
        for b in range(B):
            starts = []
            goals = []

            # GPU批量生成所有机器人的start/goal
            starts_tensor = self._rand_uniform_2d_gpu(2.0, self.width - 2.0, R)
            goals_tensor = self._rand_uniform_2d_gpu(2.0, self.height - 2.0, R)

            # 直接赋值GPU张量（无需tensor转换）
            self.pos[b, :, :] = starts_tensor
            self.goal[b, :, :] = goals_tensor
            # 初始朝向与自身体速度
            self.theta[b, :] = torch.empty(R, device=dev, dtype=dtp).uniform_(0.0, 2 * np.pi, generator=self._rg)
            self.velocity_r[b, :, :] = 0.0
            self.velocity[b, :, :] = 0.0
            self.left_thrust[b, :] = 0.0
            self.right_thrust[b, :] = 0.0

            # 将核心生成的循环改为GPU批量：
            if C > 0:
                # GPU批量生成核心参数
                cores_pos = self._rand_uniform_2d_gpu(0.0, self.width, C)
                cores_pos[:, 0] = torch.empty(C, device=self.device, dtype=self.dtype).uniform_(0.0, self.width, generator=self._rg)
                cores_pos[:, 1] = torch.empty(C, device=self.device, dtype=self.dtype).uniform_(0.0, self.height, generator=self._rg)
                
                # 随机方向和速度
                cores_sign = torch.empty(C, device=self.device, dtype=self.dtype).uniform_(0, 1, generator=self._rg)  # [0,1]
                cores_sign = (cores_sign < 0.5).float() * 2 - 1  # 转为{-1, +1}
                cores_gamma = torch.empty(C, device=self.device, dtype=self.dtype).uniform_(self.v_range[0], self.v_range[1], generator=self._rg)
                cores_gamma = 2.0 * math.pi * self.r * cores_gamma
                
                self.cores_pos[b, :C, :] = cores_pos
                self.cores_gamma[b, :C] = cores_gamma
                self.cores_sign[b, :C] = cores_sign

            # 将障碍物生成的循环改为GPU批量：
            if O > 0:
                # GPU批量生成障碍物参数
                obs_pos = self._rand_uniform_2d_gpu(5.0, max(self.width - 5.0, 5.1), O)
                obs_pos[:, 0] = torch.empty(O, device=self.device, dtype=self.dtype).uniform_(5.0, self.width - 5.0, generator=self._rg)
                obs_pos[:, 1] = torch.empty(O, device=self.device, dtype=self.dtype).uniform_(5.0, self.height - 5.0, generator=self._rg)
                obs_r = torch.empty(O, device=self.device, dtype=self.dtype).uniform_(self.obs_range[0], self.obs_range[1], generator=self._rg)
                
                self.obs_pos[b, :O, :] = obs_pos
                self.obs_r[b, :O] = obs_r
        # 使用GPU向量化流场设定初始 current 并更新世界速度
        v_curr = self.get_velocity_field(self.pos)  # [B,R,3]
        self.velocity = self.velocity_r + v_curr

        self.episode_timesteps = 0
        # 返回GPU张量观测
        obs, _, _ = self._perception()
        return obs, None, None

    # ================
    # Step（完整动力学积分）
    # ================
    def _compute_COLREGs_penalty_gpu(self):
        """
        GPU批量化COLREGs罚项计算
        返回 [B,R] 的罚项张量
        """
        B, R = self.B, self.R
        dev, dtp = self.device, self.dtype
        
        if R <= 1:
            return torch.zeros((B, R), device=dev, dtype=dtp)
        
        penalties = torch.zeros((B, R), device=dev, dtype=dtp)
        
        # COLREGs区域参数（从robot.py复制）
        left_crossing_zone_x_dim = torch.tensor([0.0, 30.0], device=dev, dtype=dtp)
        left_crossing_zone_y_dim_front = torch.tensor([-30.0, 0.0], device=dev, dtype=dtp)
        head_on_zone_x_dim = 30.0
        head_on_zone_y_dim = 30.0
        
        cos_t = torch.cos(self.theta)  # [B,R]
        sin_t = torch.sin(self.theta)
        
        # 对每个机器人，检查与其他活跃机器人的COLREGs关系
        for b in range(B):
            for r in range(R):
                if self.deactivated_flags[b, r]:
                    continue
                
                ego_vel = self.velocity[b, r, :2]  # [2]
                ego_vel_norm = torch.linalg.norm(ego_vel)
                
                if ego_vel_norm < 0.5:
                    continue
                
                # 转到机体坐标系
                ego_vel_robot_x = ego_vel[0] * cos_t[b, r] + ego_vel[1] * sin_t[b, r]
                ego_vel_robot_y = -ego_vel[0] * sin_t[b, r] + ego_vel[1] * cos_t[b, r]
                ego_v_angle = torch.atan2(ego_vel_robot_y, ego_vel_robot_x)
                
                for other_r in range(R):
                    if other_r == r or self.deactivated_flags[b, other_r]:
                        continue
                    
                    # 其他机器人的速度
                    other_vel = self.velocity[b, other_r, :2]
                    other_vel_norm = torch.linalg.norm(other_vel)
                    
                    if other_vel_norm < 0.5:
                        continue
                    
                    # 其他机器人在ego机体坐标系下的位置和速度
                    rel_pos = self.pos[b, other_r] - self.pos[b, r]  # [2]
                    rel_pos_robot_x = rel_pos[0] * cos_t[b, r] + rel_pos[1] * sin_t[b, r]
                    rel_pos_robot_y = -rel_pos[0] * sin_t[b, r] + rel_pos[1] * cos_t[b, r]
                    
                    # 将ego投影到other的坐标系
                    other_cos = torch.cos(self.theta[b, other_r])
                    other_sin = torch.sin(self.theta[b, other_r])
                    
                    ego_rel_pos = -rel_pos  # other看ego的相对位置
                    ego_p_proj_x = ego_rel_pos[0] * other_cos + ego_rel_pos[1] * other_sin
                    ego_p_proj_y = -ego_rel_pos[0] * other_sin + ego_rel_pos[1] * other_cos
                    
                    ego_v_proj_x = ego_vel[0] * other_cos + ego_vel[1] * other_sin
                    ego_v_proj_y = -ego_vel[0] * other_sin + ego_vel[1] * other_cos
                    
                    # 检查left crossing zone
                    x_in_range = (ego_p_proj_x >= left_crossing_zone_x_dim[0]) & (ego_p_proj_x <= left_crossing_zone_x_dim[1])
                    y_in_range = (ego_p_proj_y >= left_crossing_zone_y_dim_front[0]) & (ego_p_proj_y <= 0.0)
                    
                    x_diff = ego_p_proj_x - left_crossing_zone_x_dim[1]
                    y_diff = ego_p_proj_y - left_crossing_zone_y_dim_front[1]
                    grad = left_crossing_zone_y_dim_front[1] / left_crossing_zone_x_dim[1]
                    in_triangle = y_diff > grad * x_diff
                    
                    pos_in_left_crossing = x_in_range & y_in_range & (~in_triangle)
                    
                    ego_v_proj_angle = torch.atan2(ego_v_proj_y, ego_v_proj_x)
                    angle_in_left_crossing = (ego_v_proj_angle >= torch.pi / 4) & (ego_v_proj_angle <= 3 * torch.pi / 4)
                    
                    in_left_crossing_zone = pos_in_left_crossing & angle_in_left_crossing
                    
                    # 检查head on zone
                    x_in_head_on = (ego_p_proj_x >= 0.0) & (ego_p_proj_x <= head_on_zone_x_dim)
                    y_in_head_on = (ego_p_proj_y >= -0.5 * head_on_zone_y_dim) & (ego_p_proj_y <= 0.5 * head_on_zone_y_dim)
                    pos_in_head_on = x_in_head_on & y_in_head_on
                    
                    angle_in_head_on = torch.abs(ego_v_proj_angle) > 3 * torch.pi / 4
                    in_head_on_zone = pos_in_head_on & angle_in_head_on
                    
                    # 如果在COLREGs区域，计算转向角
                    if in_left_crossing_zone or in_head_on_zone:
                        obj_p_angle = torch.atan2(rel_pos_robot_y, rel_pos_robot_x)
                        base_1 = self.r + 1.0
                        dist = torch.linalg.norm(rel_pos)
                        
                        if dist > base_1:
                            add_angle_1 = torch.asin(base_1 / dist)
                            tangent_len = torch.sqrt(dist ** 2 - base_1 ** 2)
                            add_angle_2 = torch.atan2(torch.tensor(self.r, device=dev, dtype=dtp), tangent_len)
                            
                            desired_dir = obj_p_angle + add_angle_1 + add_angle_2
                            # wrap to [-pi, pi]
                            desired_dir = torch.atan2(torch.sin(desired_dir), torch.cos(desired_dir))
                            
                            phi = desired_dir - ego_v_angle
                            phi = torch.atan2(torch.sin(phi), torch.cos(phi))
                            
                            # 只有需要右转时才应用COLREGs罚项
                            if phi > 0:
                                penalties[b, r] += self.COLREGs_penalty * phi
        
        return penalties

    def step(self, actions=None, is_continuous_action=True):
        """
        GPU优化的动力学积分（与CPU版本数值一致）
        """
        B, R = self.B, self.R
        dev, dtp = self.device, self.dtype

        if actions is None:
            actions = [None] * (B * R)
        
        goal_rel_before = self._safe(self.goal - self.pos)
        dis_before = torch.linalg.norm(goal_rel_before, dim=-1)

        # GPU批量解析动作
        l_change = torch.zeros((B, R), device=dev, dtype=dtp)
        r_change = torch.zeros((B, R), device=dev, dtype=dtp)
        
        if isinstance(actions, torch.Tensor):
            if actions.dim() == 3:
                if is_continuous_action:
                    l_change = actions[:, :, 0] * 1000.0
                    r_change = actions[:, :, 1] * 1000.0
                else:
                    idx_tensor = actions[:, :, 0].to(torch.long)
                    idx_tensor = torch.clamp(idx_tensor, 0, self.actions_tensor.shape[0] - 1)
                    vals = self.actions_tensor[idx_tensor]
                    l_change = vals[:, :, 0]
                    r_change = vals[:, :, 1]
            else:  # [B*R, 2] or [B*R]
                actions_2d = actions.reshape(B, R, -1)
                if is_continuous_action:
                    l_change = actions_2d[:, :, 0] * 1000.0
                    r_change = actions_2d[:, :, 1] * 1000.0
                else:
                    idx_tensor = actions_2d[:, :, 0].to(torch.long)
                    idx_tensor = torch.clamp(idx_tensor, 0, self.actions_tensor.shape[0] - 1)
                    vals = self.actions_tensor[idx_tensor]
                    l_change = vals[:, :, 0]
                    r_change = vals[:, :, 1]
        else:
            # List输入（保持兼容性）
            for br_idx, act in enumerate(actions):
                if act is None:
                    continue
                b, r = br_idx // R, br_idx % R
                if is_continuous_action:
                    if isinstance(act, torch.Tensor):
                        l_change[b, r] = act[0] * 1000.0
                        r_change[b, r] = act[1] * 1000.0
                    else:
                        l_change[b, r] = float(act[0]) * 1000.0
                        r_change[b, r] = float(act[1]) * 1000.0
                else:
                    idx = int(act) if not isinstance(act, torch.Tensor) else int(act.item())
                    idx = max(0, min(idx, len(self.actions) - 1))
                    l_change[b, r], r_change[b, r] = self.actions[idx]

        # N 子步积分
        dt_sub = self.dt / self.N
        two_pi = 2.0 * math.pi
        
        for k in range(self.N):
            # 子步 0：更新推力（按总动作变化 × dt / N）
            if k == 0:
                self.left_thrust = torch.clamp(
                    self.left_thrust + l_change * self.dt * self.N,
                    min=self.min_thrust,
                    max=self.max_thrust
                )
                self.right_thrust = torch.clamp(
                    self.right_thrust + r_change * self.dt * self.N,
                    min=self.min_thrust,
                    max=self.max_thrust
                )
            
            v_curr = self.get_velocity_field(self.pos)
            
            # 世界速度 = 自身体速度（相对流） + 流场
            self.velocity = self.velocity_r + v_curr
            
            self.pos = self.pos + self.velocity[:, :, :2] * dt_sub
            self.theta = (self.theta + self.velocity[:, :, 2] * dt_sub) % two_pi
            
            # 旋转矩阵 R_wr（世界→机体）
            cos_t = torch.cos(self.theta)  # [B,R]
            sin_t = torch.sin(self.theta)
            # R_wr = [[cos,-sin],[sin,cos]]
            # R_rw = R_wr^T = [[cos,sin],[-sin,cos]]
            
            # velocity_r 在机体坐标系下的表示（用于计算阻尼）
            u_r = self.velocity_r[:, :, 0] * cos_t + self.velocity_r[:, :, 1] * sin_t
            v_r = -self.velocity_r[:, :, 0] * sin_t + self.velocity_r[:, :, 1] * cos_t
            
            # velocity 在机体坐标系下（用于 Coriolis C_RB）
            u = self.velocity[:, :, 0] * cos_t + self.velocity[:, :, 1] * sin_t
            v = -self.velocity[:, :, 0] * sin_t + self.velocity[:, :, 1] * cos_t
            r_ang = self.velocity[:, :, 2]
            
            # Coriolis 刚体 C_RB（与 robot.py 一致）
            C_RB = torch.zeros((B, R, 3, 3), device=dev, dtype=dtp)
            C_RB[:, :, 0, 1] = -self.m * r_ang
            C_RB[:, :, 1, 0] = self.m * r_ang
            
            # Coriolis 附加质量 C_A
            C_A = torch.zeros((B, R, 3, 3), device=dev, dtype=dtp)
            C_A[:, :, 0, 2] = self.yDotV * v_r + self.yDotR * r_ang
            C_A[:, :, 1, 2] = -self.xDotU * u_r
            C_A[:, :, 2, 0] = -self.yDotV * v_r - self.yDotR * r_ang
            C_A[:, :, 2, 1] = self.xDotU * u_r
            
            # 非线性阻尼 D_n（速度依赖）
            abs_u_r = torch.abs(u_r)
            abs_v_r = torch.abs(v_r)
            abs_r = torch.abs(r_ang)
            D_n = torch.zeros((B, R, 3, 3), device=dev, dtype=dtp)
            D_n[:, :, 0, 0] = -self.xUU * abs_u_r
            D_n[:, :, 1, 1] = -(self.yVV * abs_v_r + self.yRV * abs_r)
            D_n[:, :, 1, 2] = -(self.yVR * abs_v_r + self.yRR * abs_r)
            D_n[:, :, 2, 1] = -(self.nVV * abs_v_r + self.nRV * abs_r)
            D_n[:, :, 2, 2] = -(self.nVR * abs_v_r + self.nRR * abs_r)
            
            # 总阻尼矩阵 N = C_A + D + D_n
            N_mat = C_A + self.D.unsqueeze(0).expand(B, -1, -1, -1) + D_n
            
            # 推力产生的广义力（机体坐标系）
            F_x_left = self.left_thrust * torch.cos(torch.tensor(self.left_pos, device=dev, dtype=dtp))
            F_y_left = self.left_thrust * torch.sin(torch.tensor(self.left_pos, device=dev, dtype=dtp))
            F_x_right = self.right_thrust * torch.cos(torch.tensor(self.right_pos, device=dev, dtype=dtp))
            F_y_right = self.right_thrust * torch.sin(torch.tensor(self.right_pos, device=dev, dtype=dtp))
            
            M_x_left = F_x_left * (self.width_m * 0.5)
            M_y_left = -F_y_left * (self.length * 0.5)
            M_x_right = -F_x_right * (self.width_m * 0.5)
            M_y_right = -F_y_right * (self.length * 0.5)
            
            F_x = F_x_left + F_x_right
            F_y = F_y_left + F_y_right
            M_n = M_x_left + M_y_left + M_x_right + M_y_right
            tau_p = torch.stack([F_x, F_y, M_n], dim=-1)  # [B,R,3]
            
            # 速度向量（机体坐标系）
            V_rf = torch.stack([u, v, r_ang], dim=-1)       # [B,R,3]
            V_r_rf = torch.stack([u_r, v_r, r_ang], dim=-1)  # [B,R,3]
            
            # 右端项 b = -C_RB·V - N·V_r + τ
            b = -torch.einsum('brij,brj->bri', C_RB, V_rf) \
                - torch.einsum('brij,brj->bri', N_mat, V_r_rf) \
                + tau_p
            
            # 求解 A·acc = b（使用 Cholesky 分解）
            L_batch = self.L_chol.unsqueeze(0).expand(B, -1, -1, -1)  # [B,R,3,3]
            acc = self._cholesky_solve_batch(L_batch, b)  # [B,R,3]
            
            # 积分速度
            V_r_rf_new = V_r_rf + acc * dt_sub
            
            # 转回世界坐标系（velocity_r 在世界坐标，但物理方程在机体坐标求解）
            u_r_new = V_r_rf_new[:, :, 0]
            v_r_new = V_r_rf_new[:, :, 1]
            self.velocity_r[:, :, 0] = u_r_new * cos_t - v_r_new * sin_t
            self.velocity_r[:, :, 1] = u_r_new * sin_t + v_r_new * cos_t
            self.velocity_r[:, :, 2] = V_r_rf_new[:, :, 2]
        
        self.pos = self._safe(self.pos)
        self.theta = self._safe(self.theta, pos=0.0, neg=0.0)
        self.velocity_r = self._safe(self.velocity_r)
        self.velocity = self._safe(self.velocity)
        goal_rel_after = self._safe(self.goal - self.pos)
        dis_after = torch.linalg.norm(goal_rel_after, dim=-1)
        rewards = torch.zeros((B, R), device=dev, dtype=dtp)
        
        # Only compute rewards for active robots
        active_mask = ~self.deactivated_flags
        rewards[active_mask] += self.timestep_penalty
        rewards[active_mask] += (dis_before - dis_after)[active_mask]
        
        # 计算COLREGs罚项
        colregs_penalties = self._compute_COLREGs_penalty_gpu()
        rewards += colregs_penalties
        
        rewards = self._safe(rewards, pos=0.0, neg=0.0)
        
        # 调用 perception 并检测碰撞和到达
        observations, collisions_detected, reach_goals_detected = self._perception()
        
        # 终止条件与状态信息（按机器人构建列表）
        dones = []
        infos = []
        rewards_list = []
        
        for b in range(B):
            for r in range(R):
                if self.deactivated_flags[b, r]:
                    dones.append(True)
                    if self.collision_flags[b, r]:
                        infos.append({"state": "deactivated after collision"})
                    elif self.reach_goal_flags[b, r]:
                        infos.append({"state": "deactivated after reaching goal"})
                    else:
                        infos.append({"state": "deactivated"})
                    rewards_list.append(float(rewards[b, r]))
                    continue
                
                # 更新碰撞和到达标志
                if collisions_detected[b * R + r]:
                    self.collision_flags[b, r] = True
                if reach_goals_detected[b * R + r]:
                    self.reach_goal_flags[b, r] = True
                
                # 检查终止条件
                if self.episode_timesteps >= 1000:
                    dones.append(True)
                    infos.append({"state": "too long episode"})
                elif self.collision_flags[b, r]:
                    rewards[b, r] += self.collision_penalty
                    dones.append(True)
                    infos.append({"state": "collision"})
                    self.deactivated_flags[b, r] = True
                elif self.reach_goal_flags[b, r]:
                    rewards[b, r] += self.goal_reward
                    dones.append(True)
                    infos.append({"state": "reach goal"})
                    self.deactivated_flags[b, r] = True
                else:
                    dones.append(False)
                    infos.append({"state": "normal"})
                
                rewards_list.append(float(rewards[b, r]))
        
        self.episode_timesteps += 1
        self.total_timesteps += 1
        
        # 返回观测、奖励、终止、信息（与 CPU 版本接口一致）
        return observations, rewards_list, dones, infos

    # =========================
    # 观测（GPU批量化版本）
    # =========================
    def _perception(self):
        """
        GPU批量化感知（完全GPU计算，最小化CPU-GPU传输）
        返回：
          - observations: 列表（长度 B*R），每项为 (self_obs, obj_obs)
          - collisions: 列表（长度 B*R），每项为 bool
          - reach_goals: 列表（长度 B*R），每项为 bool
        """
        B, R = self.B, self.R
        dev, dtp = self.device, self.dtype
        
        # 批量旋转矩阵 [B,R,2,2]
        theta_safe = self._safe(self.theta, pos=0.0, neg=0.0)
        cos_t = torch.cos(theta_safe)
        sin_t = torch.sin(theta_safe)
        
        # 批量目标相对位置 [B,R,2]
        goal_rel_world = self._safe(self.goal - self.pos)
        goal_rel_robot_x = goal_rel_world[:,:,0] * cos_t + goal_rel_world[:,:,1] * sin_t
        goal_rel_robot_y = -goal_rel_world[:,:,0] * sin_t + goal_rel_world[:,:,1] * cos_t
        
        # 批量速度转换 [B,R,2]
        vel_r = self._safe(self.velocity_r)
        vel_r_robot_x = vel_r[:,:,0] * cos_t + vel_r[:,:,1] * sin_t
        vel_r_robot_y = -vel_r[:,:,0] * sin_t + vel_r[:,:,1] * cos_t
        
        # 批量到达检测 [B,R]
        dis_to_goal = torch.linalg.norm(goal_rel_world, dim=-1)
        reach_goals_tensor = dis_to_goal <= self.goal_dis
        
        # 批量碰撞检测 [B,R]
        collisions_tensor = torch.zeros((B, R), dtype=torch.bool, device=dev)
        if self.obs_pos is not None and self.obs_pos.shape[1] > 0:
            obs_rel = self._safe(self.obs_pos).unsqueeze(1) - self._safe(self.pos).unsqueeze(2)
            obs_dis = torch.linalg.norm(obs_rel, dim=-1)  # [B,R,O]
            collisions_tensor = (obs_dis <= (self.obs_r.unsqueeze(1) + self.r)).any(dim=-1)
        
        # 机器人间碰撞检测（向量化）
        if R > 1:
            pos_diff = self._safe(self.pos).unsqueeze(2) - self._safe(self.pos).unsqueeze(1)
            robot_dis = torch.linalg.norm(pos_diff, dim=-1)  # [B,R,R]
            robot_dis = robot_dis + torch.eye(R, device=dev) * 1e6  # 避免自碰撞
            active_mask = ~self.deactivated_flags  # [B,R]
            active_pairs = active_mask.unsqueeze(2) & active_mask.unsqueeze(1)  # [B,R,R]
            robot_collisions = ((robot_dis <= 2 * self.r) & active_pairs).any(dim=-1)  # [B,R]
            collisions_tensor = collisions_tensor | robot_collisions
        
        # 批量对象观测（k-nearest障碍物）
        obj_obs_batch = None
        if self.obs_pos is not None and self.obs_pos.shape[1] > 0:
            O = self.obs_pos.shape[1]
            K = min(self.max_obj_num, O)
            
            # [B,R,O,2]
            obs_rel_world = self._safe(self.obs_pos).unsqueeze(1) - self._safe(self.pos).unsqueeze(2)
            obs_dis_all = torch.linalg.norm(obs_rel_world, dim=-1)  # [B,R,O]
            
            # k-nearest indices [B,R,K]
            _, nearest_idx = torch.topk(obs_dis_all, K, dim=-1, largest=False)
            
            # 收集k-nearest障碍物 [B,R,K,2]
            batch_idx = torch.arange(B, device=dev).view(B,1,1).expand(B,R,K)
            robot_idx = torch.arange(R, device=dev).view(1,R,1).expand(B,R,K)
            obs_rel_nearest = obs_rel_world[batch_idx, robot_idx, nearest_idx]  # [B,R,K,2]
            obs_r_nearest = self.obs_r.unsqueeze(1).expand(B,R,O)[batch_idx, robot_idx, nearest_idx]  # [B,R,K]
            
            # 转到机体坐标系 [B,R,K,2]
            obs_rel_robot_x = obs_rel_nearest[:,:,:,0] * cos_t.unsqueeze(-1) + obs_rel_nearest[:,:,:,1] * sin_t.unsqueeze(-1)
            obs_rel_robot_y = -obs_rel_nearest[:,:,:,0] * sin_t.unsqueeze(-1) + obs_rel_nearest[:,:,:,1] * cos_t.unsqueeze(-1)
            
            # [B,R,K,5]: [x, y, vx, vy, radius]
            obj_obs_batch = torch.stack([
                obs_rel_robot_x,
                obs_rel_robot_y,
                torch.zeros_like(obs_rel_robot_x),
                torch.zeros_like(obs_rel_robot_x),
                obs_r_nearest
            ], dim=-1)
        
        # 转换为列表格式（单次CPU-GPU传输）
        observations = []
        collisions = []
        reach_goals = []
        
        # 一次性转移到CPU
        goal_rel_robot_x_cpu = goal_rel_robot_x.cpu().numpy()
        goal_rel_robot_y_cpu = goal_rel_robot_y.cpu().numpy()
        vel_r_robot_x_cpu = vel_r_robot_x.cpu().numpy()
        vel_r_robot_y_cpu = vel_r_robot_y.cpu().numpy()
        velocity_r_z_cpu = self.velocity_r[:,:,2].cpu().numpy()
        left_thrust_cpu = self.left_thrust.cpu().numpy()
        right_thrust_cpu = self.right_thrust.cpu().numpy()
        collisions_cpu = collisions_tensor.cpu().numpy()
        reach_goals_cpu = reach_goals_tensor.cpu().numpy()
        deactivated_cpu = self.deactivated_flags.cpu().numpy()
        collision_flags_cpu = self.collision_flags.cpu().numpy()
        reach_goal_flags_cpu = self.reach_goal_flags.cpu().numpy()
        
        if obj_obs_batch is not None:
            obj_obs_batch_cpu = obj_obs_batch.cpu().numpy()
        
        for b in range(B):
            for r in range(R):
                if deactivated_cpu[b, r]:
                    observations.append((None, None))
                    collisions.append(bool(collision_flags_cpu[b, r]))
                    reach_goals.append(bool(reach_goal_flags_cpu[b, r]))
                    continue
                
                # 自身观测
                self_obs = [
                    float(goal_rel_robot_x_cpu[b, r]),
                    float(goal_rel_robot_y_cpu[b, r]),
                    float(vel_r_robot_x_cpu[b, r]),
                    float(vel_r_robot_y_cpu[b, r]),
                    float(velocity_r_z_cpu[b, r]),
                    float(left_thrust_cpu[b, r]),
                    float(right_thrust_cpu[b, r])
                ]
                
                # 对象观测
                obj_obs_list = []
                if obj_obs_batch is not None:
                    for k in range(obj_obs_batch_cpu.shape[2]):
                        obj_obs_list.append(obj_obs_batch_cpu[b, r, k].tolist())
                
                observations.append((self_obs, obj_obs_list))
                collisions.append(bool(collisions_cpu[b, r]))
                reach_goals.append(bool(reach_goals_cpu[b, r]))
        
        return observations, collisions, reach_goals

    def get_tensor_observation(self):
        """
        返回张量版观测（设备端）：
        - self_obs: [B,R,7] -> [goal_rel_x, goal_rel_y, vel_r_x, vel_r_y, vel_r_z, left_thrust, right_thrust]
        - obj_obs: [B,R,max_obj_num,5] -> [x, y, vx, vy, radius]（不足部分零填充）
        - obj_mask: [B,R,max_obj_num] -> 1 表示真实观测，0 表示填充
        - collisions: [B,R] bool
        - reach_goals: [B,R] bool
        """
        B, R = self.B, self.R
        dev, dtp = self.device, self.dtype

        cos_t = torch.cos(self.theta)
        sin_t = torch.sin(self.theta)

        goal_rel_world = self.goal - self.pos  # [B,R,2]
        goal_rel_robot_x = goal_rel_world[:, :, 0] * cos_t + goal_rel_world[:, :, 1] * sin_t
        goal_rel_robot_y = -goal_rel_world[:, :, 0] * sin_t + goal_rel_world[:, :, 1] * cos_t

        vel_r_robot_x = self.velocity_r[:, :, 0] * cos_t + self.velocity_r[:, :, 1] * sin_t
        vel_r_robot_y = -self.velocity_r[:, :, 0] * sin_t + self.velocity_r[:, :, 1] * cos_t

        dis_to_goal = torch.linalg.norm(goal_rel_world, dim=-1)
        reach_goals_tensor = dis_to_goal <= self.goal_dis

        collisions_tensor = torch.zeros((B, R), dtype=torch.bool, device=dev)
        if self.obs_pos is not None and self.obs_pos.shape[1] > 0:
            obs_rel = self.obs_pos.unsqueeze(1) - self.pos.unsqueeze(2)  # [B,R,O,2]
            obs_dis = torch.linalg.norm(obs_rel, dim=-1)
            collisions_tensor = (obs_dis <= (self.obs_r.unsqueeze(1) + self.r)).any(dim=-1)

        if R > 1:
            pos_diff = self.pos.unsqueeze(2) - self.pos.unsqueeze(1)
            robot_dis = torch.linalg.norm(pos_diff, dim=-1)
            robot_dis = robot_dis + torch.eye(R, device=dev) * 1e6
            active_mask = ~self.deactivated_flags
            active_pairs = active_mask.unsqueeze(2) & active_mask.unsqueeze(1)
            robot_collisions = ((robot_dis <= 2 * self.r) & active_pairs).any(dim=-1)
            collisions_tensor = collisions_tensor | robot_collisions

        # 对象观测张量，包含静态障碍物和其他机器人
        obj_obs = torch.zeros((B, R, self.max_obj_num, 5), device=dev, dtype=dtp)
        obj_mask = torch.zeros((B, R, self.max_obj_num), device=dev, dtype=dtp)
        
        # 收集所有对象（静态障碍物 + 其他活跃机器人）
        all_obj_pos = []
        all_obj_vel = []
        all_obj_r = []
        
        # 静态障碍物
        if self.obs_pos is not None and self.obs_pos.shape[1] > 0:
            all_obj_pos.append(self.obs_pos)  # [B,O,2]
            all_obj_vel.append(torch.zeros((B, self.obs_pos.shape[1], 2), device=dev, dtype=dtp))
            all_obj_r.append(self.obs_r)  # [B,O]
        
        # 其他活跃机器人
        if R > 1:
            # 将所有机器人的位置、速度、半径作为动态对象
            all_obj_pos.append(self.pos)  # [B,R,2]
            all_obj_vel.append(self.velocity[:, :, :2])  # [B,R,2]
            all_obj_r.append(torch.full((B, R), self.r, device=dev, dtype=dtp))  # [B,R]
        
        # 合并所有对象
        if len(all_obj_pos) > 0:
            all_obj_pos_cat = torch.cat(all_obj_pos, dim=1)  # [B,N_total,2]
            all_obj_vel_cat = torch.cat(all_obj_vel, dim=1)  # [B,N_total,2]
            all_obj_r_cat = torch.cat(all_obj_r, dim=1)  # [B,N_total]
            
            N_total = all_obj_pos_cat.shape[1]
            K = min(self.max_obj_num, N_total)
            
            # 对每个机器人，计算到所有对象的距离并选择最近的K个
            obj_rel_world = all_obj_pos_cat.unsqueeze(1) - self.pos.unsqueeze(2)  # [B,R,N_total,2]
            obj_dis_all = torch.linalg.norm(obj_rel_world, dim=-1)  # [B,R,N_total]
            
            # 如果有其他机器人，需要排除自己
            if R > 1 and self.obs_pos is not None:
                # 静态障碍物数量
                O = self.obs_pos.shape[1]
                # 为每个机器人，将自己的距离设为无穷大
                for r in range(R):
                    if O + r < N_total:
                        obj_dis_all[:, r, O + r] = 1e6
            
            _, nearest_idx = torch.topk(obj_dis_all, K, dim=-1, largest=False)
            batch_idx = torch.arange(B, device=dev).view(B, 1, 1).expand(B, R, K)
            robot_idx = torch.arange(R, device=dev).view(1, R, 1).expand(B, R, K)
            
            obj_rel_nearest = obj_rel_world[batch_idx, robot_idx, nearest_idx]  # [B,R,K,2]
            obj_vel_nearest = all_obj_vel_cat.unsqueeze(1).expand(B, R, N_total, 2)[batch_idx, robot_idx, nearest_idx]  # [B,R,K,2]
            obj_r_nearest = all_obj_r_cat.unsqueeze(1).expand(B, R, N_total)[batch_idx, robot_idx, nearest_idx]  # [B,R,K]
            
            # 转到机体坐标系
            obj_rel_robot_x = obj_rel_nearest[:, :, :, 0] * cos_t.unsqueeze(-1) + obj_rel_nearest[:, :, :, 1] * sin_t.unsqueeze(-1)
            obj_rel_robot_y = -obj_rel_nearest[:, :, :, 0] * sin_t.unsqueeze(-1) + obj_rel_nearest[:, :, :, 1] * cos_t.unsqueeze(-1)
            obj_vel_robot_x = obj_vel_nearest[:, :, :, 0] * cos_t.unsqueeze(-1) + obj_vel_nearest[:, :, :, 1] * sin_t.unsqueeze(-1)
            obj_vel_robot_y = -obj_vel_nearest[:, :, :, 0] * sin_t.unsqueeze(-1) + obj_vel_nearest[:, :, :, 1] * cos_t.unsqueeze(-1)
            
            obj_obs[:, :, :K, :] = torch.stack([
                obj_rel_robot_x,
                obj_rel_robot_y,
                obj_vel_robot_x,
                obj_vel_robot_y,
                obj_r_nearest
            ], dim=-1)
            obj_mask[:, :, :K] = 1.0

        self_obs = torch.stack([
            goal_rel_robot_x,
            goal_rel_robot_y,
            vel_r_robot_x,
            vel_r_robot_y,
            self.velocity_r[:, :, 2],
            self.left_thrust,
            self.right_thrust
        ], dim=-1)

        return self_obs, obj_obs, obj_mask, collisions_tensor, reach_goals_tensor

    # =========================
    # 公共：GPU向量化流场接口
    # =========================
    def get_velocity_field(self, pos: torch.Tensor) -> torch.Tensor:
        """
        GPU向量化流场查询。返回 [B,R,3]
        """
        if self.cores_pos is None or self.cores_pos.shape[1] == 0:
            return torch.zeros((self.B, self.R, 3), device=self.device, dtype=self.dtype)
        pos_safe = self._safe(pos)
        return _get_velocity_vectorized(pos_safe, self.cores_pos, self.cores_gamma, self.cores_sign, self.r)

    # =========================
    # 工具：随机/约束检查
    # =========================
    def _rand_uniform_2d_gpu(self, low: float, high: float, num_samples: int = 1) -> torch.Tensor:
        """GPU张量版2D均匀分布随机数生成，返回 [num_samples, 2]"""
        return torch.empty((num_samples, 2), device=self.device, dtype=self.dtype).uniform_(low, high, generator=self._rg)

    def _rand_uniform_2d(self, low: float, high: float) -> Tuple[float, float]:
        x = self._rd.uniform(low, high)
        y = self._rd.uniform(low, high)
        return (x, y)

    def _check_start_goal(
        self,
        start: Tuple[float, float],
        goal: Tuple[float, float],
        starts: list,
        goals: list
    ) -> bool:
        # 最小起终距离
        if np.linalg.norm(np.array(goal) - np.array(start)) < self.min_start_goal_dis:
            return False
        # 与已选机器人 start/goal 清空区
        for s in starts:
            if np.linalg.norm(np.array(s) - np.array(start)) <= self.clear_r:
                return False
        for g in goals:
            if np.linalg.norm(np.array(g) - np.array(goal)) <= self.clear_r:
                return False
        return True

    def _check_core_single(
        self,
        core: Tuple[float, float, bool, float],  # (x, y, clockwise, Gamma)
        cores: list,
        starts: list,
        goals: list
    ) -> bool:
        x, y, clockwise, Gamma = core

        # 边界
        if x - self.r < 0.0 or x + self.r > self.width:
            return False
        if y - self.r < 0.0 or y + self.r > self.height:
            return False

        # 与所有机器人 start/goal 的清空区
        cpos = np.array([x, y], dtype=float)
        for s in starts:
            if np.linalg.norm(cpos - np.array(s)) < (self.r + self.clear_r):
                return False
        for g in goals:
            if np.linalg.norm(cpos - np.array(g)) < (self.r + self.clear_r):
                return False

        # 与已选核心的相互约束（同向/反向）
        for c in cores:
            xi, yi, clockwise_i, Gamma_i = c
            dx = xi - x
            dy = yi - y
            dis = math.hypot(dx, dy)

            # 安全：避免 dis - 2r <= 0
            if dis <= 2.0 * self.r:
                return False

            if clockwise_i == clockwise:
                boundary_i = Gamma_i / (2.0 * math.pi * self.v_rel_max)
                boundary_j = Gamma   / (2.0 * math.pi * self.v_rel_max)
                if dis < (boundary_i + boundary_j):
                    return False
            else:
                Gamma_l = max(Gamma_i, Gamma)
                Gamma_s = min(Gamma_i, Gamma)
                v_1 = Gamma_l / (2.0 * math.pi * max(dis - 2.0 * self.r, 1e-6))
                v_2 = Gamma_s / (2.0 * math.pi * self.r)
                if v_1 > self.p * v_2:
                    return False

        return True

    def _check_obstacle_single(
        self,
        obs: Tuple[float, float, float],  # (x, y, r)
        obs_list: list,
        cores: list,
        starts: list,
        goals: list
    ) -> bool:
        x, y, r_obs = obs

        # 边界
        if x - r_obs < 0.0 or x + r_obs > self.width:
            return False
        if y - r_obs < 0.0 or y + r_obs > self.height:
            return False

        # 与所有机器人 start/goal 清空区
        opos = np.array([x, y], dtype=float)
        for s in starts:
            if np.linalg.norm(opos - np.array(s)) < (r_obs + self.clear_r):
                return False
        for g in goals:
            if np.linalg.norm(opos - np.array(g)) < (r_obs + self.clear_r):
                return False

        # 与所有核心距离 > r_core + r_obs
        for c in cores:
            cx, cy = c[0], c[1]
            if math.hypot(cx - x, cy - y) <= (self.r + r_obs):
                return False

        # 障碍之间不相交
        for o in obs_list:
            ox, oy, orad = o
            if math.hypot(ox - x, oy - y) <= (orad + r_obs):
                return False

        return True

    def _compute_rewards_tensor(self, collisions, reach_goals):
        """
        批量GPU奖励计算
        返回: [B,R] GPU张量
        """
        B, R = self.B, self.R
        rewards = torch.full((B, R), self.timestep_penalty, device=self.device, dtype=self.dtype)
        
        rewards[collisions] += self.collision_penalty
        rewards[reach_goals] += self.goal_reward
        
        angular_speed = torch.abs(self.velocity_r[:, :, 2])
        exceed_mask = angular_speed > self.angular_speed_max
        rewards[exceed_mask] += self.angular_speed_penalty
        
        colregs_penalties = self._compute_COLREGs_penalty_gpu()
        rewards += colregs_penalties
        
        return rewards

    def step_tensor(self, actions_t, is_continuous_action=False):
        """
        GPU张量版step（零CPU-GPU传输）
        
        输入：
          - actions_t: [B,R,2] 或 [B,R,1] GPU张量
          - is_continuous_action: bool
        
        返回：
          - self_obs: [B,R,7] GPU张量
          - obj_obs: [B,R,K,5] GPU张量
          - obj_mask: [B,R,K] GPU张量
          - rewards: [B,R] GPU张量
          - dones: [B,R] GPU张量（bool）
          - collisions: [B,R] GPU张量（bool）
          - reach_goals: [B,R] GPU张量（bool）
        """
        B, R = self.B, self.R
        dev, dtp = self.device, self.dtype
        
        l_change = torch.zeros((B, R), device=dev, dtype=dtp)
        r_change = torch.zeros((B, R), device=dev, dtype=dtp)
        
        if actions_t.dim() == 3:
            if is_continuous_action:
                l_change = actions_t[:, :, 0] * 1000.0
                r_change = actions_t[:, :, 1] * 1000.0
            else:
                idx_tensor = actions_t[:, :, 0].to(torch.long)
                idx_tensor = torch.clamp(idx_tensor, 0, self.actions_tensor.shape[0] - 1)
                vals = self.actions_tensor[idx_tensor]
                l_change = vals[:, :, 0]
                r_change = vals[:, :, 1]
        
        dt_sub = self.dt / self.N
        two_pi = 2.0 * math.pi
        
        for k in range(self.N):
            if k == 0:
                self.left_thrust = torch.clamp(
                    self.left_thrust + l_change * self.dt * self.N,
                    min=self.min_thrust,
                    max=self.max_thrust
                )
                self.right_thrust = torch.clamp(
                    self.right_thrust + r_change * self.dt * self.N,
                    min=self.min_thrust,
                    max=self.max_thrust
                )
            
            v_curr = self.get_velocity_field(self.pos)
            self.velocity = self.velocity_r + v_curr
            self.pos = self.pos + self.velocity[:, :, :2] * dt_sub
            self.theta = (self.theta + self.velocity[:, :, 2] * dt_sub) % two_pi
            
            cos_t = torch.cos(self.theta)
            sin_t = torch.sin(self.theta)
            
            u_r = self.velocity_r[:, :, 0] * cos_t + self.velocity_r[:, :, 1] * sin_t
            v_r = -self.velocity_r[:, :, 0] * sin_t + self.velocity_r[:, :, 1] * cos_t
            u = self.velocity[:, :, 0] * cos_t + self.velocity[:, :, 1] * sin_t
            v = -self.velocity[:, :, 0] * sin_t + self.velocity[:, :, 1] * cos_t
            r_ang = self.velocity[:, :, 2]
            
            C_RB = torch.zeros((B, R, 3, 3), device=dev, dtype=dtp)
            C_RB[:, :, 0, 1] = -self.m * r_ang
            C_RB[:, :, 1, 0] = self.m * r_ang
            
            C_A = torch.zeros((B, R, 3, 3), device=dev, dtype=dtp)
            C_A[:, :, 0, 2] = self.yDotV * v_r + self.yDotR * r_ang
            C_A[:, :, 1, 2] = -self.xDotU * u_r
            C_A[:, :, 2, 0] = -self.yDotV * v_r - self.yDotR * r_ang
            C_A[:, :, 2, 1] = self.xDotU * u_r
            
            abs_u_r = torch.abs(u_r)
            abs_v_r = torch.abs(v_r)
            abs_r = torch.abs(r_ang)
            D_n = torch.zeros((B, R, 3, 3), device=dev, dtype=dtp)
            D_n[:, :, 0, 0] = -self.xUU * abs_u_r
            D_n[:, :, 1, 1] = -(self.yVV * abs_v_r + self.yRV * abs_r)
            D_n[:, :, 1, 2] = -(self.yVR * abs_v_r + self.yRR * abs_r)
            D_n[:, :, 2, 1] = -(self.nVV * abs_v_r + self.nRV * abs_r)
            D_n[:, :, 2, 2] = -(self.nVR * abs_v_r + self.nRR * abs_r)
            
            N_mat = C_A + self.D.unsqueeze(0).expand(B, -1, -1, -1) + D_n
            
            F_x_left = self.left_thrust * torch.cos(torch.tensor(self.left_pos, device=dev, dtype=dtp))
            F_y_left = self.left_thrust * torch.sin(torch.tensor(self.left_pos, device=dev, dtype=dtp))
            F_x_right = self.right_thrust * torch.cos(torch.tensor(self.right_pos, device=dev, dtype=dtp))
            F_y_right = self.right_thrust * torch.sin(torch.tensor(self.right_pos, device=dev, dtype=dtp))
            
            M_x_left = F_x_left * (self.width_m * 0.5)
            M_y_left = -F_y_left * (self.length * 0.5)
            M_x_right = -F_x_right * (self.width_m * 0.5)
            M_y_right = -F_y_right * (self.length * 0.5)
            
            F_x = F_x_left + F_x_right
            F_y = F_y_left + F_y_right
            M_n = M_x_left + M_y_left + M_x_right + M_y_right
            tau_p = torch.stack([F_x, F_y, M_n], dim=-1)
            
            V_rf = torch.stack([u, v, r_ang], dim=-1)
            V_r_rf = torch.stack([u_r, v_r, r_ang], dim=-1)
            
            b = -torch.einsum('brij,brj->bri', C_RB, V_rf) \
                - torch.einsum('brij,brj->bri', N_mat, V_r_rf) \
                + tau_p
            
            L_batch = self.L_chol.unsqueeze(0).expand(B, -1, -1, -1)
            acc = self._cholesky_solve_batch(L_batch, b)
            
            V_r_rf_new = V_r_rf + acc * dt_sub
            
            u_r_new = V_r_rf_new[:, :, 0]
            v_r_new = V_r_rf_new[:, :, 1]
            self.velocity_r[:, :, 0] = u_r_new * cos_t - v_r_new * sin_t
            self.velocity_r[:, :, 1] = u_r_new * sin_t + v_r_new * cos_t
            self.velocity_r[:, :, 2] = V_r_rf_new[:, :, 2]
        
        self.pos = self._safe(self.pos)
        self.theta = self._safe(self.theta, pos=0.0, neg=0.0)
        self.velocity_r = self._safe(self.velocity_r)
        self.velocity = self._safe(self.velocity)
        
        self_obs, obj_obs, obj_mask, collisions, reach_goals = self.get_tensor_observation()
        
        rewards = self._compute_rewards_tensor(collisions, reach_goals)
        
        dones = collisions | reach_goals
        
        self.deactivated_flags = self.deactivated_flags | dones
        self.collision_flags = self.collision_flags | collisions
        self.reach_goal_flags = self.reach_goal_flags | reach_goals
        
        self.episode_timesteps += 1
        self.total_timesteps += 1
        
        return self_obs, obj_obs, obj_mask, rewards, dones, collisions, reach_goals

    def get_action_space_dimension(self):
        """返回动作空间维度（连续动作为 2）"""
        return 2
