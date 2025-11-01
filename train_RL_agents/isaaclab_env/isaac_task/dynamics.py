import torch
import numpy as np

def step_dynamics(state, action_lr, params, dt, N, device):
    """
    向量化船舶动力学计算（不更新位置与姿态）
    - 输入/输出位置与姿态由调用方管理；此函数仅更新相对速度与推力。
    """
    E, R, _ = state['pos'].shape
    
    # 提取状态
    pos = state['pos']
    theta = state['theta']
    vel_r = state['vel_r']
    vel = state['vel']
    left_thrust = state['left_thrust']
    right_thrust = state['right_thrust']
    
    # 物理参数
    m = params['mass']
    Izz = params['Izz']
    length = params['length']
    width = params['width']
    
    # 动作映射 [-1,1] -> 推力变化/秒
    thrust_change_scale = 1000.0
    delta_left = action_lr[..., 0].clamp(-1.0, 1.0) * thrust_change_scale * dt * N
    delta_right = action_lr[..., 1].clamp(-1.0, 1.0) * thrust_change_scale * dt * N
    
    # 更新推进器推力（与原环境一致的范围）
    new_left_thrust = torch.clamp(left_thrust + delta_left, -500.0, 1000.0)
    new_right_thrust = torch.clamp(right_thrust + delta_right, -500.0, 1000.0)
    
    # 计算旋转矩阵（世界->机器人）
    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)
    R_wr = torch.stack([
        torch.stack([cos_theta, -sin_theta], dim=-1),
        torch.stack([sin_theta,  cos_theta], dim=-1)
    ], dim=-2)
    R_rw = torch.transpose(R_wr, -2, -1)
    
    # 机器人坐标下速度
    vel_r_2d = vel_r[..., :2]
    vel_2d = vel[..., :2]
    vel_r_robot = torch.matmul(R_rw, vel_r_2d.unsqueeze(-1)).squeeze(-1)
    vel_robot = torch.matmul(R_rw, vel_2d.unsqueeze(-1)).squeeze(-1)
    u_r, v_r = vel_r_robot[..., 0], vel_r_robot[..., 1]
    u, v = vel_robot[..., 0], vel_robot[..., 1]
    r = vel[..., 2]
    
    # 水动力系数（与原robot.py一致）
    xDotU = 20.0
    yDotV = 0.0
    yDotR = 0.0
    nDotR = -980.0
    nDotV = 0.0
    xU = -100.0
    xUU = -150.0
    yV = -100.0
    yVV = -150.0
    yR = 0.0
    yRV = 0.0
    yVR = 0.0
    yRR = 0.0
    nR = -980.0
    nRR = -950.0
    nV = 0.0
    nVV = 0.0
    nRV = 0.0
    nVR = 0.0
    
    # 科里奥利/附加质量/阻尼
    C_RB = torch.zeros(E, R, 3, 3, device=device)
    C_RB[..., 0, 1] = -m * r
    C_RB[..., 1, 0] =  m * r
    
    C_A = torch.zeros(E, R, 3, 3, device=device)
    C_A[..., 0, 2] = yDotV * v_r + yDotR * r
    C_A[..., 1, 2] = -xDotU * u_r
    C_A[..., 2, 0] = -yDotV * v_r - yDotR * r
    C_A[..., 2, 1] =  xDotU * u_r
    
    D_n = torch.zeros(E, R, 3, 3, device=device)
    D_n[..., 0, 0] = -(xUU * torch.abs(u_r))
    D_n[..., 1, 1] = -(yVV * torch.abs(v_r) + yRV * torch.abs(r))
    D_n[..., 1, 2] = -(yVR * torch.abs(v_r) + yRR * torch.abs(r))
    D_n[..., 2, 1] = -(nVV * torch.abs(v_r) + nRV * torch.abs(r))
    D_n[..., 2, 2] = -(nVR * torch.abs(v_r) + nRR * torch.abs(r))
    
    D_linear = torch.zeros(E, R, 3, 3, device=device)
    D_linear[..., 0, 0] = -xU
    D_linear[..., 1, 1] = -yV
    D_linear[..., 2, 2] = -nR
    D_total = D_linear + D_n + C_A
    
    M_RB = torch.zeros(E, R, 3, 3, device=device)
    M_RB[..., 0, 0] = m
    M_RB[..., 1, 1] = m
    M_RB[..., 2, 2] = Izz
    
    M_A = torch.zeros(E, R, 3, 3, device=device)
    M_A[..., 0, 0] = -xDotU
    M_A[..., 1, 1] = -yDotV
    M_A[..., 1, 2] = -yDotR
    M_A[..., 2, 1] = -nDotV
    M_A[..., 2, 2] = -nDotR
    M_total = M_RB + M_A
    
    # 推进力
    left_pos = torch.zeros(E, R, device=device)
    right_pos = torch.zeros(E, R, device=device)
    F_x_left = new_left_thrust * torch.cos(left_pos)
    F_y_left = new_left_thrust * torch.sin(left_pos)
    F_x_right = new_right_thrust * torch.cos(right_pos)
    F_y_right = new_right_thrust * torch.sin(right_pos)
    M_x_left = F_x_left * width / 2
    M_y_left = -F_y_left * length / 2
    M_x_right = -F_x_right * width / 2
    M_y_right = -F_y_right * length / 2
    F_x = F_x_left + F_x_right
    F_y = F_y_left + F_y_right
    M_n = M_x_left + M_y_left + M_x_right + M_y_right
    tau_p = torch.stack([F_x, F_y, M_n], dim=-1)
    
    V = torch.stack([u, v, r], dim=-1)
    V_r = torch.stack([u_r, v_r, r], dim=-1)
    left_side = tau_p - torch.matmul(C_RB, V.unsqueeze(-1)).squeeze(-1) - torch.matmul(D_total, V_r.unsqueeze(-1)).squeeze(-1)
    try:
        acc = torch.linalg.solve(M_total, left_side.unsqueeze(-1)).squeeze(-1)
    except RuntimeError:
        acc = torch.matmul(torch.linalg.pinv(M_total), left_side.unsqueeze(-1)).squeeze(-1)
    
    # 更新相对速度（机器人坐标）
    new_V_r = V_r + acc * dt
    
    # 转回世界坐标
    new_vel_r_2d_world = torch.matmul(R_wr, new_V_r[..., :2].unsqueeze(-1)).squeeze(-1)
    new_vel_r_world = torch.cat([new_vel_r_2d_world, new_V_r[..., 2:3]], dim=-1)
    
    # 返回更新后的状态（pos/theta不在此更新）
    new_state = {
        'pos': pos,
        'theta': theta,
        'vel_r': new_vel_r_world,
        'vel': vel,  # 由调用方在每个子步外部重算（= vel_r + current）
        'left_thrust': new_left_thrust,
        'right_thrust': new_right_thrust
    }
    
    return new_state

def compute_motion_matrices(mass, Izz, device):
    E, R = mass.shape
    M_RB = torch.zeros(E, R, 3, 3, device=device)
    M_RB[..., 0, 0] = mass
    M_RB[..., 1, 1] = mass
    M_RB[..., 2, 2] = Izz
    return M_RB
