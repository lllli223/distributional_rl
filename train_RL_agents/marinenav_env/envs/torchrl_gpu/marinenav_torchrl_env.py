import torch
import numpy as np
from torchrl.envs import EnvBase
from tensordict import TensorDict
from torchrl.data import Composite, Unbounded, DiscreteTensorSpec, Bounded
from marinenav_env.envs.marinenav_env import MarineNavEnv3


class MarineNavTorchRLEnv(EnvBase):
    """
    TorchRL-compatible wrapper for MarineNavEnv3 with explicit E×R batching.
    - EnvBase(batch_size=[E]) where E is the number of parallel environments.
    - Each environment contains R robots. Observations/actions are shaped (E, R, ...).
    """

    def __init__(self, seed=0, schedule=None, is_eval_env=False, device="cuda", max_obj_num=5, env_batch_size=1):
        super().__init__(device=device, batch_size=[env_batch_size])

        self.seed_value = seed
        self.schedule = schedule
        self.is_eval_env = is_eval_env
        self.max_obj_num = max_obj_num
        self.E = int(env_batch_size)

        # Underlying CPU envs (one per E). Keep consistent schedule across all.
        self._envs = [
            MarineNavEnv3(seed=seed + e, schedule=schedule, is_eval_env=is_eval_env)
            for e in range(self.E)
        ]

        # Assume same num_robots across envs (schedule applies equally)
        self.num_robots = self._envs[0].num_robots
        self.self_dim = 7
        self.obj_dim = 5

        self._make_spec()

        # Pre-allocated buffers (on device) for formatting observations quickly
        self._alloc_buffers()

    # ---- Specs ----
    def _make_spec(self):
        # Specs are per env; EnvBase(batch_size=[E]) adds the E leading batch dim
        self.observation_spec = Composite(
            self_state=Unbounded(
                shape=(self.num_robots, self.self_dim),
                device=self.device,
                dtype=torch.float32,
            ),
            objects_state=Unbounded(
                shape=(self.num_robots, self.max_obj_num, self.obj_dim),
                device=self.device,
                dtype=torch.float32,
            ),
            objects_mask=Bounded(
                low=0,
                high=1,
                shape=(self.num_robots, self.max_obj_num),
                device=self.device,
                dtype=torch.float32,
            ),
            shape=(self.E,),
        )

        # Action is either continuous [R, 2] or discrete index [R, 1]. We expose continuous bounds.
        self.action_spec = Bounded(
            low=-1.0,
            high=1.0,
            shape=(self.num_robots, 2),
            device=self.device,
            dtype=torch.float32,
        )

        self.reward_spec = Unbounded(
            shape=(self.num_robots, 1),
            device=self.device,
            dtype=torch.float32,
        )

        self.done_spec = DiscreteTensorSpec(
            n=2,
            shape=(self.num_robots, 1),
            device=self.device,
            dtype=torch.bool,
        )

    def _alloc_buffers(self):
        E, R = self.E, self.num_robots
        self._buf_self = torch.zeros((E, R, self.self_dim), dtype=torch.float32, device=self.device)
        self._buf_obj = torch.zeros((E, R, self.max_obj_num, self.obj_dim), dtype=torch.float32, device=self.device)
        self._buf_mask = torch.zeros((E, R, self.max_obj_num), dtype=torch.float32, device=self.device)
        self._buf_rew = torch.zeros((E, R, 1), dtype=torch.float32, device=self.device)
        self._buf_done = torch.zeros((E, R, 1), dtype=torch.bool, device=self.device)

    def _resize_buffers(self, new_num_robots: int):
        if new_num_robots != self.num_robots:
            self.num_robots = int(new_num_robots)
            self._make_spec()
            self._alloc_buffers()

    # ---- EnvBase API ----
    def reset(self, tensordict=None, **kwargs):
        return self._reset(tensordict)

    def step(self, tensordict):
        return self._step(tensordict)

    def _reset(self, tensordict=None):
        # Reset all envs for simplicity (partial reset not required by current trainer)
        all_states = []
        for e in range(self.E):
            states, _, _ = self._envs[e].reset()
            all_states.append(states)
        # In case schedule changed num_robots
        self._resize_buffers(self._envs[0].num_robots)
        return self._format_observation_batched(all_states)

    def _step(self, tensordict):
        actions = tensordict["action"]  # shape (E, R, A) or (E, R, 1)
        if actions.device != torch.device("cpu"):
            actions_cpu = actions.detach().to("cpu")
        else:
            actions_cpu = actions
        E, R = self.E, self.num_robots

        # Prepare output buffers
        self._buf_rew.zero_()
        self._buf_done.zero_()

        next_states_all = []
        for e in range(E):
            act_e = actions_cpu[e].numpy()
            # Determine discrete vs continuous by last dim
            is_discrete = act_e.shape[-1] == 1
            if is_discrete:
                action_list = [int(act_e[i, 0]) for i in range(min(len(self._envs[e].robots), act_e.shape[0]))]
                is_continuous_action = False
            else:
                action_list = [act_e[i] for i in range(min(len(self._envs[e].robots), act_e.shape[0]))]
                is_continuous_action = True

            next_states, rewards, dones, infos = self._envs[e].step(action_list, is_continuous_action)
            next_states_all.append(next_states)

            # Pad rewards/dones to R
            rewards_array = np.array(rewards, dtype=np.float32).reshape(-1)
            dones_array = np.array(dones, dtype=bool).reshape(-1)
            limit = min(R, rewards_array.shape[0])
            if limit > 0:
                self._buf_rew[e, :limit, 0] = torch.as_tensor(rewards_array[:limit], device=self.device)
                self._buf_done[e, :limit, 0] = torch.as_tensor(dones_array[:limit], device=self.device)

        obs_td = self._format_observation_batched(next_states_all)
        obs_td["reward"] = self._buf_rew.clone()
        obs_td["done"] = self._buf_done.clone()
        return obs_td

    # ---- Helpers ----
    def _format_observation_batched(self, batched_states):
        # batched_states: list of len E; each item is list of robot states
        E, R = self.E, self.num_robots
        self._buf_self.zero_()
        self._buf_obj.zero_()
        self._buf_mask.zero_()

        for e in range(E):
            states = batched_states[e]
            actual_states = len(states)
            actual_robots = len(self._envs[e].robots)
            limit = min(R, actual_states)
            for i in range(limit):
                state = states[i]
                if state is None:
                    continue
                # Skip deactivated robots to keep masks/rewards consistent
                if i < actual_robots and self._envs[e].robots[i].deactivated:
                    continue
                self_state, obj_list = state
                self._buf_self[e, i] = torch.as_tensor(np.asarray(self_state, dtype=np.float32), device=self.device)
                for j, obj in enumerate(obj_list[: self.max_obj_num]):
                    self._buf_obj[e, i, j] = torch.as_tensor(np.asarray(obj, dtype=np.float32), device=self.device)
                    self._buf_mask[e, i, j] = 1.0

        return TensorDict(
            {
                "self_state": self._buf_self.clone(),
                "objects_state": self._buf_obj.clone(),
                "objects_mask": self._buf_mask.clone(),
            },
            batch_size=[self.E],
            device=self.device,
        )

    # ---- Seed / Eval compatibility ----
    def _set_seed(self, seed: int):
        self.seed_value = seed
        for e, env in enumerate(self._envs):
            env.seed = seed + e
            env.rd = np.random.RandomState(seed + e)

    def reset_with_eval_config(self, eval_config):
        """For evaluation compatibility with original trainer (E must be 1)."""
        if self.E != 1:
            raise RuntimeError("reset_with_eval_config only supported with env_batch_size=1 (E=1)")
        state, _, _ = self._envs[0].reset_with_eval_config(eval_config)
        # Ensure buffers sized correctly (schedule may change R)
        self._resize_buffers(self._envs[0].num_robots)
        return self._format_observation_batched([state])

    def check_all_deactivated(self):
        if self.E != 1:
            # For training we do not rely on CPU robot shim; return False to avoid early resets
            return False
        return self._envs[0].check_all_deactivated()

    def check_all_reach_goal(self):
        if self.E != 1:
            return False
        return self._envs[0].check_all_reach_goal()

    def check_any_collision(self):
        if self.E != 1:
            return False
        return self._envs[0].check_any_collision()

    @property
    def robots(self):
        # Avoid creating training-time CPU shim; only expose in eval mode with E=1
        if self.E != 1:
            return []
        return self._envs[0].robots
