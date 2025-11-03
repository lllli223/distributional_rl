import torch
import math


def compute_vortex_velocity(pos, cores_xy, clockwise, Gamma, core_r, device):
    """
    Compute tangential velocity from vortex cores (vectorized).
    
    Args:
        pos: [E, R, 2] - position of robots
        cores_xy: [E, C, 2] - positions of vortex cores
        clockwise: [E, C] - clockwise direction (bool/0-1)
        Gamma: [E, C] - circulation strength
        core_r: scalar - core radius
        device: torch device
    
    Returns:
        v_sum: [E, R, 2] - summed velocity from all cores
    """
    E, R, _ = pos.shape
    C = cores_xy.shape[1]
    
    if C == 0:
        return torch.zeros(E, R, 2, device=device)
    
    p = pos.unsqueeze(2)
    c = cores_xy.unsqueeze(1)
    v_rad = p - c
    dist = torch.linalg.norm(v_rad, dim=-1).clamp_min(1e-6)
    n = v_rad / dist.unsqueeze(-1)
    
    rot_cw = torch.tensor([[0., -1.], [1., 0.]], device=device)
    rot_ccw = torch.tensor([[0., 1.], [-1., 0.]], device=device)
    
    clockwise_expanded = clockwise.view(E, 1, C, 1, 1).bool()
    rot = torch.where(clockwise_expanded, 
                      rot_cw.view(1, 1, 1, 2, 2), 
                      rot_ccw.view(1, 1, 1, 2, 2))
    rot = rot.expand(E, R, C, 2, 2)
    
    t_hat = torch.matmul(rot, n.unsqueeze(-1)).squeeze(-1)
    
    two_pi = 2.0 * math.pi
    inside = dist <= core_r
    speed_inside = Gamma.unsqueeze(1) / (two_pi * core_r * core_r) * dist
    speed_outside = Gamma.unsqueeze(1) / (two_pi * dist)
    speed = torch.where(inside, speed_inside, speed_outside)
    
    v = t_hat * speed.unsqueeze(-1)
    v_sum = v.sum(dim=2)
    
    return v_sum
