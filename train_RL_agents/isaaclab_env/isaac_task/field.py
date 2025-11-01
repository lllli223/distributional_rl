import torch

def compute_tangent_vel(pos_er, cores_xy_ec, clockwise_eC, Gamma_eC, core_r, device):
    """
    向量化计算漩涡流速场，替代原有的KDTree方法
    
    参数:
    - pos_er: [E,R,2] 位置张量
    - cores_xy_ec: [E,C,2] 漩涡核心位置张量  
    - clockwise_eC: [E,C] 漩涡方向，True为顺时针
    - Gamma_eC: [E,C] 漩涡强度
    - core_r: 漩涡核心半径
    - device: 计算设备
    
    返回:
    - v_currents: [E,R,2] 叠加后流速场
    """
    
    E, R, _ = pos_er.shape
    C = cores_xy_ec.shape[1]
    
    # 扩展维度做广播 [E,R,2] vs [E,C,2] -> [E,R,C,2]
    p = pos_er.unsqueeze(2)          # [E,R,1,2]
    c = cores_xy_ec.unsqueeze(1)     # [E,1,C,2]
    v_rad = p - c                    # [E,R,C,2]
    dist = torch.linalg.norm(v_rad, dim=-1).clamp_min(1e-6)  # [E,R,C]
    n = v_rad / dist.unsqueeze(-1)   # 单位径向 [E,R,C,2]

    # 构造旋转矩阵 - 先在CPU上构建，再移到device
    rot_cw = torch.tensor([[0., -1.],[1., 0.]], device=device)
    rot_ccw = torch.tensor([[0.,  1.],[-1.,0.]], device=device)
    
    # 根据漩涡方向选择旋转矩阵 [E,R,C,2,2]
    # 将 clockwise_eC [E,C] 扩展到 [E,R,C,1,1] 用于选择旋转矩阵
    clockwise_expanded = clockwise_eC.unsqueeze(1).unsqueeze(-1).unsqueeze(-1).expand(E, R, C, 1, 1)
    # 构造旋转矩阵 [E,R,C,2,2]
    rot_cw_expanded = rot_cw.unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(E, R, C, 2, 2)
    rot_ccw_expanded = rot_ccw.unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(E, R, C, 2, 2)
    rot = torch.where(clockwise_expanded.bool(), rot_cw_expanded, rot_ccw_expanded)
    
    # 计算切向单位向量
    t_hat = torch.matmul(rot, n.unsqueeze(-1)).squeeze(-1)  # [E,R,C,2]

    # 计算速度大小 - 分段函数
    two_pi = 2.0 * torch.pi
    inside = dist <= core_r
    
    speed_inside = Gamma_eC.unsqueeze(1) / (two_pi * core_r * core_r) * dist  # [E,R,C]
    speed_outside = Gamma_eC.unsqueeze(1) / (two_pi * dist)  # [E,R,C]
    speed = torch.where(inside, speed_inside, speed_outside)  # [E,R,C]

    # 计算速度向量 [E,R,C,2]
    v = t_hat * speed.unsqueeze(-1)

    # 可选：只取每个点最近的K个核心减少算力
    K = min(4, C)
    if K < C:
        d_top, idx_top = torch.topk(dist, k=K, largest=False)
        mask = torch.zeros_like(dist, dtype=torch.bool)
        mask.scatter_(2, idx_top, True)
        v = v * mask.unsqueeze(-1)

    # 叠加所有核心的速度
    v_sum = v.sum(dim=2)  # [E,R,2]
    return v_sum

def compute_velocity_batch(pos_batch, cores_batch, clockwise_batch, Gamma_batch, core_r, device):
    """
    批量计算多个位置的流速，适用于不同环境配置
    pos_batch: [B,2] 批处理位置
    cores_batch: [C,2] 所有漩涡核心位置
    clockwise_batch: [C] 漩涡方向
    Gamma_batch: [C] 漩涡强度
    """
    B = pos_batch.shape[0]
    C = cores_batch.shape[0]
    
    # 为批处理扩展维度
    pos_expanded = pos_batch.unsqueeze(0)  # [1,B,2]
    cores_expanded = cores_batch.unsqueeze(1)  # [C,1,2]
    
    # 计算径向向量和距离
    v_rad = pos_expanded - cores_expanded  # [C,B,2]
    dist = torch.linalg.norm(v_rad, dim=-1).clamp_min(1e-6)  # [C,B]
    n = v_rad / dist.unsqueeze(-1)  # [C,B,2]
    
    # 旋转矩阵
    rot_cw = torch.tensor([[0., -1.],[1., 0.]], device=device)
    rot_ccw = torch.tensor([[0.,  1.],[-1.,0.]], device=device)
    
    rot = torch.where(clockwise_batch.unsqueeze(1).unsqueeze(-1).bool(),
                      rot_cw, rot_ccw)  # [C,1,2,2]
    rot = rot.expand(C, B, 2, 2)  # [C,B,2,2]
    
    # 切向单位向量
    t_hat = torch.matmul(rot, n.unsqueeze(-1)).squeeze(-1)  # [C,B,2]
    
    # 速度大小计算
    two_pi = 2.0 * torch.pi
    inside = dist <= core_r
    speed_inside = Gamma_batch.unsqueeze(1) / (two_pi * core_r * core_r) * dist
    speed_outside = Gamma_batch.unsqueeze(1) / (two_pi * dist)
    speed = torch.where(inside, speed_inside, speed_outside)
    
    # 速度向量
    v = t_hat * speed.unsqueeze(-1)
    
    # 叠加所有漩涡的速度
    v_total = v.sum(dim=0)  # [B,2]
    
    return v_total
