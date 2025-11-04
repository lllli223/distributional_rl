import numpy as np
from typing import Optional, Tuple, List, Any

# Optional imports: torch / tensordict / torchrl
try:
    import torch
    from tensordict import TensorDict
    from torchrl.envs import EnvBase
except Exception:  # pragma: no cover - allow importing this module without torchrl installed
    torch = None  # type: ignore
    TensorDict = None  # type: ignore
    class EnvBase:  # type: ignore
        pass

# Import the original numpy-based environment
from marinenav_env.envs.marinenav_env import MarineNavEnv3


def _pad_and_stack_observations(
    observations: List[Tuple[Optional[List[float]], Optional[List[List[float]]]]],
    max_obj_num: int = 5,
    dtype: Any = float,
) -> np.ndarray:
    """
    Convert the variable-length observation from MarineNavEnv3 to a fixed-shape array.

    Each robot observation is a tuple: (self_state, object_list)
      - self_state: list of length 7
      - object_list: list of up to max_obj_num objects, each [px, py, vx, vy, r]

    This function pads the object_list to max_obj_num with zeros and flattens it,
    returning an array of shape [num_robots, 7 + 5 * max_obj_num] = [num_robots, 32].
    """
    num_robots = len(observations)
    obs_dim = 7 + 5 * max_obj_num

    out = np.zeros((num_robots, obs_dim), dtype=dtype)
    for i, (self_state, obj_list) in enumerate(observations):
        if self_state is None or obj_list is None:
            # deactivated robot: keep zeros
            continue
        # self_state: length 7
        self_state_arr = np.asarray(self_state, dtype=dtype)
        if self_state_arr.shape[0] != 7:
            # be conservative if config changes; clip or pad
            tmp = np.zeros(7, dtype=dtype)
            n = min(7, self_state_arr.shape[0])
            tmp[:n] = self_state_arr[:n]
            self_state_arr = tmp

        # object list: list of lists, each of length 5
        # take up to max_obj_num (already sorted in the env), then pad
        flat_objs: List[float] = []
        for j, obj in enumerate(obj_list[:max_obj_num]):
            arr = np.asarray(obj, dtype=dtype)
            if arr.shape[0] != 5:
                tmp = np.zeros(5, dtype=dtype)
                n = min(5, arr.shape[0])
                tmp[:n] = arr[:n]
                arr = tmp
            flat_objs.extend(arr.tolist())
        # pad remaining objects
        if len(obj_list) < max_obj_num:
            pad_elems = 5 * (max_obj_num - len(obj_list))
            flat_objs.extend([0.0] * pad_elems)

        obj_arr = np.asarray(flat_objs, dtype=dtype)

        out[i, :7] = self_state_arr
        out[i, 7:] = obj_arr
    return out


def _pad_and_stack_batch_observations(
    batch_observations: List[List[Tuple[Optional[List[float]], Optional[List[List[float]]]]]],
    max_obj_num: int,
    dtype: Any = float,
) -> np.ndarray:
    """
    Convert a list over envs of observations into a single fixed-size array.

    Args:
        batch_observations: list of length B, each item is the observations list for an env
                            (length A_i, typically all A_i equal), where each element is
                            (self_state, object_list) as in _pad_and_stack_observations.
        max_obj_num: maximum number of objects per robot to pad to.
    Returns:
        np.ndarray of shape [B, A, 7 + 5 * max_obj_num].
    """
    B = len(batch_observations)
    assert B > 0
    # Compute A per env, ensure consistent across envs
    As = [len(obs_list) for obs_list in batch_observations]
    A0 = As[0]
    for a in As:
        if a != A0:
            raise ValueError("All envs must have the same number of robots (agents) to stack into a batch.")
    A = A0
    obs_dim = 7 + 5 * max_obj_num
    out = np.zeros((B, A, obs_dim), dtype=dtype)
    for b in range(B):
        out[b] = _pad_and_stack_observations(batch_observations[b], max_obj_num=max_obj_num, dtype=dtype)
    return out


class MarineNavTorchRLEnv(EnvBase):
    """
    TorchRL wrapper for MarineNavEnv3 supporting a multi-env, multi-agent batch API.

    - Keeps the original numpy physics and logic per env (CPU for now)
    - Converts variable-length observations into fixed-size tensors [B, A, 32]
    - Exposes a TensorDict API compatible with TorchRL collectors

    Shapes
      - action (step input):  [B, A, 2], values in [-1, 1]
      - observation (reset/step next): [B, A, 32]
      - reward (step next): [B, A]
      - done (step next): [B, A]
    """

    def __init__(
        self,
        seed: int = 0,
        schedule: Optional[dict] = None,
        is_eval_env: bool = False,
        device: Optional[str] = None,
        dtype: Optional["torch.dtype"] = None,
        num_envs: int = 1,
    ) -> None:
        if torch is None or TensorDict is None:
            raise ImportError(
                "MarineNavTorchRLEnv requires torch, tensordict, and torchrl to be installed."
            )
        if device is None:
            device = "cpu"
        if dtype is None:
            dtype = torch.float32

        super().__init__(device=device)
        self.device = torch.device(device)
        self.dtype = dtype

        if num_envs <= 0:
            raise ValueError("num_envs must be >= 1")
        self.num_envs = int(num_envs)

        # Create underlying numpy envs (one per batch env)
        # Use different seeds per env to decorrelate rollouts
        self._envs: List[MarineNavEnv3] = []
        for i in range(self.num_envs):
            env_seed = int(seed + i) if seed is not None else None
            self._envs.append(MarineNavEnv3(seed=env_seed, schedule=schedule, is_eval_env=is_eval_env))

        # Reset once to establish num_robots and initial observation shape
        batch_obs: List[List[Tuple[Optional[List[float]], Optional[List[List[float]]]]]] = []
        for e in self._envs:
            observations, collisions, reach_goals = e.reset()
            batch_obs.append(observations)
        # robots per env must match
        num_robots_list = [len(o) for o in batch_obs]
        if not all(n == num_robots_list[0] for n in num_robots_list):
            raise ValueError("All envs must have the same number of robots (agents).")
        self.num_robots = num_robots_list[0]

        # Determine max_obj_num from first robot (assumed consistent across robots and envs)
        self.max_obj_num = (
            self._envs[0].robots[0].perception.max_obj_num if len(self._envs[0].robots) > 0 else 5
        )

        # Convert to fixed-shape observation [B, A, 32]
        obs_np = _pad_and_stack_batch_observations(batch_obs, max_obj_num=self.max_obj_num, dtype=np.float32)
        self._last_obs = torch.as_tensor(obs_np, device=self.device, dtype=self.dtype)

    # TorchRL EnvBase API
    def _reset(self, tensordict: Optional[TensorDict] = None) -> TensorDict:
        batch_obs: List[List[Tuple[Optional[List[float]], Optional[List[List[float]]]]]] = []
        for e in self._envs:
            observations, collisions, reach_goals = e.reset()
            batch_obs.append(observations)
        # If curriculum changes num_robots, adapt wrapper (ensure all envs match)
        num_robots_list = [len(o) for o in batch_obs]
        if not all(n == num_robots_list[0] for n in num_robots_list):
            raise ValueError("All envs must have the same number of robots (agents) after reset.")
        new_num_robots = num_robots_list[0]
        if new_num_robots != self.num_robots:
            self.num_robots = new_num_robots
            self.max_obj_num = (
                self._envs[0].robots[0].perception.max_obj_num if len(self._envs[0].robots) > 0 else self.max_obj_num
            )
        obs_np = _pad_and_stack_batch_observations(batch_obs, max_obj_num=self.max_obj_num, dtype=np.float32)
        obs = torch.as_tensor(obs_np, device=self.device, dtype=self.dtype)
        self._last_obs = obs
        td = TensorDict(
            {
                "observation": obs,
            },
            batch_size=[self.num_envs, self.num_robots],
            device=self.device,
        )
        return td

    def _step(self, tensordict: TensorDict) -> TensorDict:
        assert "action" in tensordict.keys(), "action key is required in tensordict"
        action = tensordict.get("action")
        # Expect shape [B, A, 2] or [A, 2] when B == 1
        if action.device != self.device:
            action = action.to(self.device)
        if action.dtype != torch.float32:
            action = action.to(torch.float32)

        if action.dim() == 2:
            # [A, 2] -> [1, A, 2]
            action = action.unsqueeze(0)
        elif action.dim() != 3:
            raise ValueError(f"action must have shape [B, A, 2] or [A, 2]; got {tuple(action.shape)}")

        B, A, D = action.shape
        if B != self.num_envs or A != self.num_robots or D != 2:
            raise ValueError(
                f"action shape {tuple(action.shape)} does not match expected [B={self.num_envs}, A={self.num_robots}, 2]"
            )

        action_np = action.detach().cpu().numpy()

        batch_obs: List[List[Tuple[Optional[List[float]], Optional[List[List[float]]]]]] = []
        batch_rewards: List[List[float]] = []
        batch_dones: List[List[bool]] = []

        for i, env in enumerate(self._envs):
            # Convert to list of (l, r) pairs for this env
            actions_list: List[List[float]] = action_np[i].tolist()
            observations, rewards, dones, infos = env.step(actions_list, is_continuous_action=True)
            batch_obs.append(observations)
            batch_rewards.append(rewards)
            batch_dones.append(dones)

        # Convert observation to fixed shape tensor [B, A, 32]
        obs_np = _pad_and_stack_batch_observations(batch_obs, max_obj_num=self.max_obj_num, dtype=np.float32)
        obs = torch.as_tensor(obs_np, device=self.device, dtype=self.dtype)
        rew = torch.as_tensor(np.asarray(batch_rewards, dtype=np.float32), device=self.device, dtype=self.dtype)
        done = torch.as_tensor(np.asarray(batch_dones, dtype=np.bool_), device=self.device)

        # Keep latest observation
        self._last_obs = obs

        next_td = TensorDict(
            {
                "observation": self._last_obs,
                "reward": rew,
                "done": done,
            },
            batch_size=[self.num_envs, self.num_robots],
            device=self.device,
        )
        # TorchRL convention: return a TensorDict with a "next" entry
        out = TensorDict(
            {
                "next": next_td,
            },
            batch_size=[self.num_envs, self.num_robots],
            device=self.device,
        )
        return out

    # Convenience utilities
    @property
    def specs(self) -> dict:
        """Return basic specs information as a python dict (optional).
        This keeps the wrapper lightweight and version-agnostic across TorchRL releases.
        """
        return {
            "num_envs": self.num_envs,
            "num_robots": self.num_robots,
            "observation_dim": 7 + 5 * self.max_obj_num,
            "action_dim": 2,
        }

    def set_seed(self, seed: Optional[int] = None) -> None:
        if seed is None:
            return
        # Recreate the underlying env random state by re-instantiation semantics
        schedule = self._envs[0].schedule if self._envs else None
        is_eval_env = self._envs[0].is_eval_env if self._envs else False

        # Re-instantiate envs with new seeds
        self._envs = []
        for i in range(self.num_envs):
            env_seed = int(seed + i)
            self._envs.append(MarineNavEnv3(seed=env_seed, schedule=schedule, is_eval_env=is_eval_env))

        # Reset and refresh cached shapes
        batch_obs: List[List[Tuple[Optional[List[float]], Optional[List[List[float]]]]]] = []
        for e in self._envs:
            observations, collisions, reach_goals = e.reset()
            batch_obs.append(observations)
        num_robots_list = [len(o) for o in batch_obs]
        if not all(n == num_robots_list[0] for n in num_robots_list):
            raise ValueError("All envs must have the same number of robots (agents).")
        self.num_robots = num_robots_list[0]
        self.max_obj_num = (
            self._envs[0].robots[0].perception.max_obj_num if len(self._envs[0].robots) > 0 else self.max_obj_num
        )
        obs_np = _pad_and_stack_batch_observations(batch_obs, max_obj_num=self.max_obj_num, dtype=np.float32)
        self._last_obs = torch.as_tensor(obs_np, device=self.device, dtype=self.dtype)
