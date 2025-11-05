import argparse
import os
from typing import Optional

import torch
from torch import nn
from torch.optim import Adam

from tensordict import TensorDict
from torchrl.collectors.collectors import SyncDataCollector
from torchrl.data.replay_buffers import TensorDictReplayBuffer
from torchrl.data.replay_buffers.storages import LazyTensorStorage

from marinenav_env_torch import MarineNavTorchRLEnv


class MLPPolicy(nn.Module):
    def __init__(self, obs_dim: int = 32, action_dim: int = 2, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh(),  # output in [-1, 1]
        )

    @torch.no_grad()
    def act(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)

    def forward(self, tensordict: TensorDict) -> TensorDict:
        obs = tensordict.get("observation")
        action = self.net(obs)
        tensordict.set("action", action)
        return tensordict


def make_env(num_envs: int, device: str, seed: int, schedule: Optional[dict] = None) -> MarineNavTorchRLEnv:
    env = MarineNavTorchRLEnv(
        seed=seed,
        schedule=schedule,
        is_eval_env=False,
        device=device,
        dtype=torch.float32,
        num_envs=num_envs,
    )
    return env


def train(
    device: str = "cuda",
    seed: int = 0,
    num_envs: int = 8,
    frames_per_batch: int = 4096,
    total_frames: int = 200_000,
    replay_size: int = 200_000,
    grad_updates_per_batch: int = 1,
    lr: float = 3e-4,
    log_dir: Optional[str] = None,
):
    torch.manual_seed(seed)

    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        print(f"Requested CUDA device but CUDA is not available. Falling back to CPU.")
        device = torch.device("cpu")

    # Build env on device
    env = make_env(num_envs=num_envs, device=device, seed=seed)

    # Simple on-device policy
    obs_dim = env.specs["observation_dim"]
    action_dim = env.specs["action_dim"]
    policy = MLPPolicy(obs_dim=obs_dim, action_dim=action_dim).to(device)
    optim = Adam(policy.parameters(), lr=lr)

    # On-device replay buffer
    storage = LazyTensorStorage(replay_size, device=device)
    replay_buffer = TensorDictReplayBuffer(storage=storage)

    # Collector on device
    collector = SyncDataCollector(
        env,
        policy,
        frames_per_batch=frames_per_batch,
        total_frames=total_frames,
        device=device,
    )

    # Minimal actor-only training loop (behavioural cloning on immediate reward proxy)
    # This is a placeholder to demonstrate on-device collection + replay.
    # Replace with your preferred RL algorithm.

    it = 0
    for batch in collector:
        it += 1
        # Ensure batch stays on device (collector already places it on `device`)
        batch = batch.to(device, non_blocking=True)

        # Store transitions in on-device replay buffer
        # Flatten time and env/agent dims into a single dimension
        try:
            flat = batch.reshape(-1)
        except Exception:
            # fallback for older tensordict versions
            flat = batch.view(-1)
        replay_buffer.extend(flat)

        # Sample and do a dummy update that nudges actions towards zero,
        # just to exercise the GPU compute pipeline without CPU transfers.
        for _ in range(grad_updates_per_batch):
            if len(replay_buffer) < 1024:
                break
            td = replay_buffer.sample(1024).to(device)
            obs = td.get("observation")
            # policy forward (train mode)
            pred_action = policy.net(obs)
            # simple L2 loss to keep actions bounded and exercise training
            loss = (pred_action.pow(2)).mean()
            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()

        if log_dir and it % 10 == 0:
            os.makedirs(log_dir, exist_ok=True)
            with open(os.path.join(log_dir, "progress.txt"), "a+") as f:
                f.write(f"iter={it}, buffer_size={len(replay_buffer)}\n")

        if collector._frames >= total_frames:
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--frames-per-batch", type=int, default=4096)
    parser.add_argument("--total-frames", type=int, default=200_000)
    parser.add_argument("--replay-size", type=int, default=200_000)
    parser.add_argument("--grad-updates-per-batch", type=int, default=1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--log-dir", type=str, default=None)
    args = parser.parse_args()

    train(
        device=args.device,
        seed=args.seed,
        num_envs=args.num_envs,
        frames_per_batch=args.frames_per_batch,
        total_frames=args.total_frames,
        replay_size=args.replay_size,
        grad_updates_per_batch=args.grad_updates_per_batch,
        lr=args.lr,
        log_dir=args.log_dir,
    )
