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


class MarineNavTorchRLEnv(EnvBase):
    """
    TorchRL wrapper for MarineNavEnv3 that:
      - keeps the original numpy physics and logic
      - converts variable-length observations into fixed-size tensors [num_robots, 32]
      - exposes a TensorDict API compatible with TorchRL collectors (CPU-only for now)

    Keys
      - input (step):  action: Tensor [num_robots, 2], values in [-1, 1]
      - output (reset): observation: Tensor [num_robots, 32]
      - output (step):  next: { observation [num_robots, 32], reward [num_robots], done [num_robots] }
    """

    def __init__(
        self,
        seed: int = 0,
        schedule: Optional[dict] = None,
        is_eval_env: bool = False,
        device: Optional[str] = None,
        dtype: Optional["torch.dtype"] = None,
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

        # Underlying numpy env
        self._env = MarineNavEnv3(seed=seed, schedule=schedule, is_eval_env=is_eval_env)

        # Reset once to establish num_robots and initial observation shape
        observations, collisions, reach_goals = self._env.reset()
        self.num_robots = len(observations)
        # Determine max_obj_num from first robot (assumed consistent across robots)
        self.max_obj_num = (
            self._env.robots[0].perception.max_obj_num if len(self._env.robots) > 0 else 5
        )
        # Convert to fixed-shape observation
        obs_np = _pad_and_stack_observations(observations, max_obj_num=self.max_obj_num, dtype=np.float32)
        self._last_obs = torch.as_tensor(obs_np, device=self.device, dtype=self.dtype)

    # TorchRL EnvBase API
    def _reset(self, tensordict: Optional[TensorDict] = None) -> TensorDict:
        observations, collisions, reach_goals = self._env.reset()
        # If curriculum changes num_robots, adapt wrapper
        new_num_robots = len(observations)
        if new_num_robots != self.num_robots:
            self.num_robots = new_num_robots
            self.max_obj_num = (
                self._env.robots[0].perception.max_obj_num if len(self._env.robots) > 0 else self.max_obj_num
            )
        obs_np = _pad_and_stack_observations(observations, max_obj_num=self.max_obj_num, dtype=np.float32)
        obs = torch.as_tensor(obs_np, device=self.device, dtype=self.dtype)
        self._last_obs = obs
        td = TensorDict(
            {
                "observation": obs,
            },
            batch_size=[self.num_robots],
            device=self.device,
        )
        return td

    def _step(self, tensordict: TensorDict) -> TensorDict:
        assert "action" in tensordict.keys(), "action key is required in tensordict"
        action = tensordict.get("action")
        # Expect shape [num_robots, 2]
        if action.device != self.device:
            action = action.to(self.device)
        action_np = action.detach().to(torch.float32).cpu().numpy()
        # Convert to list of (l, r) pairs
        actions_list: List[List[float]] = action_np.tolist()

        observations, rewards, dones, infos = self._env.step(actions_list, is_continuous_action=True)

        # Convert observation to fixed shape tensor
        obs_np = _pad_and_stack_observations(observations, max_obj_num=self.max_obj_num, dtype=np.float32)
        obs = torch.as_tensor(obs_np, device=self.device, dtype=self.dtype)
        rew = torch.as_tensor(np.asarray(rewards, dtype=np.float32), device=self.device, dtype=self.dtype)
        done = torch.as_tensor(np.asarray(dones, dtype=np.bool_), device=self.device)

        # Keep latest observation
        self._last_obs = obs

        next_td = TensorDict(
            {
                "observation": self._last_obs,
                "reward": rew,
                "done": done,
            },
            batch_size=[self.num_robots],
            device=self.device,
        )
        # TorchRL convention: return a TensorDict with a "next" entry
        out = TensorDict(
            {
                "next": next_td,
            },
            batch_size=[self.num_robots],
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
            "num_robots": self.num_robots,
            "observation_dim": 7 + 5 * self.max_obj_num,
            "action_dim": 2,
        }

    def set_seed(self, seed: Optional[int] = None) -> None:
        if seed is None:
            return
        # Recreate the underlying env random state by re-instantiation semantics
        schedule = self._env.schedule
        is_eval_env = self._env.is_eval_env
        # Preserve current configuration where possible
        self._env = MarineNavEnv3(seed=seed, schedule=schedule, is_eval_env=is_eval_env)
        observations, collisions, reach_goals = self._env.reset()
        self.num_robots = len(observations)
        self.max_obj_num = (
            self._env.robots[0].perception.max_obj_num if len(self._env.robots) > 0 else self.max_obj_num
        )
        obs_np = _pad_and_stack_observations(observations, max_obj_num=self.max_obj_num, dtype=np.float32)
        self._last_obs = torch.as_tensor(obs_np, device=self.device, dtype=self.dtype)
