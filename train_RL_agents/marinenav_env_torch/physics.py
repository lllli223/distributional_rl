"""
Torch-based physics and flowfield utilities for MarineNav environments.

This module provides GPU-friendly vectorized kernels for:
- Computing the 2D flowfield induced by multiple vortex cores at many query points
- Angle wrapping utility in torch
- Batched geometry utilities for multi-agent projection, distance, and detection

The functions are written to operate fully on torch tensors and support broadcasting
across batch and multi-agent dimensions.
"""
from typing import Optional

try:
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore


def wrap_to_pi(angle: "torch.Tensor") -> "torch.Tensor":
    """Wrap angles to [-pi, pi) elementwise.

    Args:
        angle: Tensor of any shape.
    Returns:
        Tensor of same shape with values wrapped to [-pi, pi).
    """
    if torch is None:
        raise ImportError("physics.wrap_to_pi requires torch to be installed.")
    pi = torch.pi if hasattr(torch, "pi") else torch.tensor(3.141592653589793, device=angle.device, dtype=angle.dtype)
    twopi = 2.0 * pi
    # Use fmod for numerical stability and GPU friendliness
    wrapped = (angle + pi) % twopi - pi
    return wrapped



def flow_velocity(
    xy: "torch.Tensor",
    core_centers: "torch.Tensor",
    clockwise: "torch.Tensor",
    gamma: "torch.Tensor",
    r: float,
    topk: Optional[int] = None,
) -> "torch.Tensor":
    """Compute 2D flow velocities at query points induced by vortex cores.

    Vectorized, GPU-friendly implementation matching the piecewise speed law:
        v(d) = Gamma / (2*pi*r*r) * d,    if d <= r
             = Gamma / (2*pi*d),          else

    Tangential direction is set by clockwise flag per core.

    Args:
        xy: Tensor of shape [B, A, 2], query positions.
        core_centers: Tensor of shape [B, C, 2] or [C, 2], vortex core centers.
        clockwise: Bool/byte/int Tensor of shape [B, C] or [C], True for clockwise rotation.
        gamma: Tensor of shape [B, C] or [C], circulation strengths per core.
        r: Core radius (float).
        topk: If provided and <= C, only the k nearest cores are used per query.

    Returns:
        Tensor of shape [B, A, 2] with the resulting 2D velocities.
    """
    if torch is None:
        raise ImportError("physics.flow_velocity requires torch to be installed.")

    if core_centers.dim() == 2:
        # [C, 2] -> [1, C, 2]
        core_centers = core_centers.unsqueeze(0)
    if clockwise.dim() == 1:
        clockwise = clockwise.unsqueeze(0)
    if gamma.dim() == 1:
        gamma = gamma.unsqueeze(0)

    B = xy.shape[0]
    # Broadcast core tensors over batch if needed
    if core_centers.shape[0] == 1 and B > 1:
        core_centers = core_centers.expand(B, -1, -1)
    if clockwise.shape[0] == 1 and B > 1:
        clockwise = clockwise.expand(B, -1)
    if gamma.shape[0] == 1 and B > 1:
        gamma = gamma.expand(B, -1)

    # Compute deltas and distances: [B, A, C, 2] and [B, A, C]
    delta = core_centers[:, None, :, :] - xy[:, :, None, :]
    d = torch.linalg.norm(delta, dim=-1)
    d_safe = torch.clamp_min(d, 1e-8)

    # Unit radial vectors: [B, A, C, 2]
    radial = delta / d_safe[..., None]
    radial_x = radial[..., 0]
    radial_y = radial[..., 1]

    # Tangential directions based on clockwise flag
    cw = clockwise.to(dtype=torch.bool)[:, None, :]  # [B, 1, C]
    tan_x = torch.where(cw, -radial_y, radial_y)  # [B, A, C]
    tan_y = torch.where(cw,  radial_x, -radial_x)  # [B, A, C]

    # Piecewise speed law
    gamma_bc = gamma[:, None, :]  # [B, 1, C]
    pi = torch.pi if hasattr(torch, "pi") else torch.tensor(3.141592653589793, device=xy.device, dtype=xy.dtype)
    speed_inner = gamma_bc / (2.0 * pi * (r * r)) * d  # [B, A, C]
    speed_outer = gamma_bc / (2.0 * pi * d_safe)
    speed = torch.where(d <= r, speed_inner, speed_outer)  # [B, A, C]

    if topk is not None:
        C = d.shape[-1]
        k = min(int(topk), int(C))
        if k < C:
            vals, inds = torch.topk(d, k=k, dim=-1, largest=False)  # [B, A, k]
            tan_x = torch.gather(tan_x, dim=-1, index=inds)
            tan_y = torch.gather(tan_y, dim=-1, index=inds)
            speed = torch.gather(speed, dim=-1, index=inds)

    vx = torch.sum(tan_x * speed, dim=-1)
    vy = torch.sum(tan_y * speed, dim=-1)
    v = torch.stack((vx, vy), dim=-1)  # [B, A, 2]
    return v


def project_to_robot_frame_batch(
    x: "torch.Tensor",
    robot_xytheta: "torch.Tensor",
    is_vector: bool = True,
) -> "torch.Tensor":
    """Project points or vectors from world to each robot's frame in batch.

    Args:
        x: Tensor [..., 2], typically [B, A, 2] or [B, A, N, 2].
        robot_xytheta: Tensor [B, A, 3] with (x, y, theta) per robot in world frame.
        is_vector: If True, treats x as vectors; if False, as points (applies translation).

    Returns:
        Tensor with same leading shape as x, last dim 2, in robot frames.
    """
    if torch is None:
        raise ImportError("physics.project_to_robot_frame_batch requires torch to be installed.")
    # Ensure shapes
    if robot_xytheta.size(-1) != 3:
        raise ValueError("robot_xytheta must have last dimension 3: (x, y, theta)")
    # Compute R_rw (world->robot) per robot
    theta = robot_xytheta[..., 2]
    c = torch.cos(theta)
    s = torch.sin(theta)
    # R_rw = [[c, s], [-s, c]]
    Rxx = c
    Rxy = s
    Ryx = -s
    Ryy = c

    # Align shapes for broadcasting
    # Expand robot pose to x's shape excluding the last coord dim
    # robot_xy: [B, A, 2] -> broadcast to x[..., :2]
    robot_xy = robot_xytheta[..., :2]
    # Compute rotation of x
    x0 = x[..., 0]
    x1 = x[..., 1]
    xr0 = Rxx * x0 + Rxy * x1
    xr1 = Ryx * x0 + Ryy * x1

    if not is_vector:
        # Apply translation t_rw = -R_rw * t_wr
        tx = robot_xy[..., 0]
        ty = robot_xy[..., 1]
        t0 = -(Rxx * tx + Rxy * ty)
        t1 = -(Ryx * tx + Ryy * ty)
        xr0 = xr0 + t0
        xr1 = xr1 + t1

    return torch.stack((xr0, xr1), dim=-1)


def compute_distance_batch(
    ego_xy: "torch.Tensor",
    obj_xy: "torch.Tensor",
    ego_r: "torch.Tensor",
    obj_r: "torch.Tensor",
    in_robot_frame: bool = False,
) -> "torch.Tensor":
    """Compute clearance distance between ego robots and objects in batch.

    d = ||p_obj - p_ego||_2 - r_obj - r_ego (or ||p_obj|| if in_robot_frame)

    Args:
        ego_xy: [B, A, 2]
        obj_xy: [B, A, N, 2] if not in_robot_frame else [B, A, N, 2] (already in robot frame)
        ego_r:  [B, A] or broadcastable
        obj_r:  [B, A, N] or broadcastable
        in_robot_frame: if True, ego position assumed at origin in robot frames.
    Returns:
        distances: [B, A, N]
    """
    if torch is None:
        raise ImportError("physics.compute_distance_batch requires torch to be installed.")

    if in_robot_frame:
        d = torch.linalg.norm(obj_xy, dim=-1)
    else:
        # Broadcast ego position to object shape
        d = torch.linalg.norm(obj_xy - ego_xy.unsqueeze(-2), dim=-1)
    # Subtract radii (broadcasting)
    d = d - obj_r - ego_r.unsqueeze(-1)
    return d


def check_detection_batch(
    robot_xytheta: "torch.Tensor",
    obj_xy: "torch.Tensor",
    obj_r: "torch.Tensor",
    perception_range: float,
    perception_angle: float,
) -> "torch.Tensor":
    """Detection mask for objects in batch following Robot.check_detection logic.

    An object is detected if:
      - Its distance in robot frame <= perception_range + obj_r
      - Its angle in robot frame within [-0.5*perception_angle, 0.5*perception_angle]

    Args:
        robot_xytheta: [B, A, 3]
        obj_xy: [B, A, N, 2] (world-frame)
        obj_r:  [B, A, N]
        perception_range: float
        perception_angle: float
    Returns:
        mask: bool tensor [B, A, N]
    """
    if torch is None:
        raise ImportError("physics.check_detection_batch requires torch to be installed.")

    # Project objects into robot frames
    obj_r_xy = project_to_robot_frame_batch(obj_xy, robot_xytheta, is_vector=False)
    dist = torch.linalg.norm(obj_r_xy, dim=-1)
    # Range check
    in_range = dist <= (perception_range + obj_r)
    # Angle check
    ang = torch.atan2(obj_r_xy[..., 1], obj_r_xy[..., 0])
    half = 0.5 * perception_angle
    in_angle = (ang >= -half) & (ang <= half)

    return in_range & in_angle
