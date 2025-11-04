"""
Torch-based physics and flowfield utilities for MarineNav environments.

This module provides GPU-friendly vectorized kernels for:
- Computing the 2D flowfield induced by multiple vortex cores at many query points
- Angle wrapping utility in torch

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
