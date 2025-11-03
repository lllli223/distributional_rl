import torch
import numpy as np
from torchrl.envs import EnvBase
from tensordict import TensorDict
from torchrl.data import CompositeSpec, UnboundedContinuousTensorSpec, DiscreteTensorSpec, BoundedTensorSpec
from marinenav_env.envs.marinenav_env import MarineNavEnv3


class MarineNavTorchRLEnv(EnvBase):
    """
    TorchRL-compatible wrapper for MarineNavEnv3.
    Converts multi-robot observations into batched TensorDict format for GPU acceleration.
    """
    
    def __init__(self, seed=0, schedule=None, is_eval_env=False, device="cuda", max_obj_num=5):
        super().__init__(device=device, batch_size=[])
        
        self.seed_value = seed
        self.schedule = schedule
        self.is_eval_env = is_eval_env
        self.max_obj_num = max_obj_num
        
        self._env = MarineNavEnv3(seed=seed, schedule=schedule, is_eval_env=is_eval_env)
        
        self.num_robots = self._env.num_robots
        self.self_dim = 7
        self.obj_dim = 5
        
        self._make_spec()
    
    def _make_spec(self):
        self.observation_spec = CompositeSpec(
            self_state=UnboundedContinuousTensorSpec(
                shape=(self.num_robots, self.self_dim),
                device=self.device,
                dtype=torch.float32
            ),
            objects_state=UnboundedContinuousTensorSpec(
                shape=(self.num_robots, self.max_obj_num, self.obj_dim),
                device=self.device,
                dtype=torch.float32
            ),
            objects_mask=BoundedTensorSpec(
                low=0,
                high=1,
                shape=(self.num_robots, self.max_obj_num),
                device=self.device,
                dtype=torch.float32
            ),
            shape=()
        )
        
        self.action_spec = BoundedTensorSpec(
            low=-1.0,
            high=1.0,
            shape=(self.num_robots, 2),
            device=self.device,
            dtype=torch.float32
        )
        
        self.reward_spec = UnboundedContinuousTensorSpec(
            shape=(self.num_robots, 1),
            device=self.device,
            dtype=torch.float32
        )
        
        self.done_spec = DiscreteTensorSpec(
            n=2,
            shape=(self.num_robots, 1),
            device=self.device,
            dtype=torch.bool
        )
    
    def reset(self, tensordict=None, **kwargs):
        return self._reset(tensordict)
    
    def step(self, tensordict):
        return self._step(tensordict)
    
    def _reset(self, tensordict=None):
        states, _, _ = self._env.reset()
        return self._format_observation(states)
    
    def _step(self, tensordict):
        actions = tensordict["action"].cpu().numpy()
        
        action_list = []
        for i in range(self.num_robots):
            if i < len(self._env.robots) and not self._env.robots[i].deactivated:
                action_list.append(actions[i])
            else:
                action_list.append(None)
        
        is_continuous_action = True
        next_states, rewards, dones, infos = self._env.step(action_list, is_continuous_action)
        
        obs_td = self._format_observation(next_states)
        
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=self.device).view(self.num_robots, 1)
        dones_tensor = torch.tensor(dones, dtype=torch.bool, device=self.device).view(self.num_robots, 1)
        
        obs_td["reward"] = rewards_tensor
        obs_td["done"] = dones_tensor
        
        return obs_td
    
    def _format_observation(self, states):
        self_states = []
        objects_states = []
        objects_masks = []
        
        for i, state in enumerate(states):
            if state is None or (i < len(self._env.robots) and self._env.robots[i].deactivated):
                self_states.append(np.zeros(self.self_dim))
                objects_states.append(np.zeros((self.max_obj_num, self.obj_dim)))
                objects_masks.append(np.zeros(self.max_obj_num))
            else:
                self_state, obj_list = state
                self_states.append(np.array(self_state))
                
                obj_arr = np.zeros((self.max_obj_num, self.obj_dim))
                mask = np.zeros(self.max_obj_num)
                
                for j, obj in enumerate(obj_list[:self.max_obj_num]):
                    obj_arr[j] = np.array(obj)
                    mask[j] = 1.0
                
                objects_states.append(obj_arr)
                objects_masks.append(mask)
        
        self_state_tensor = torch.tensor(np.array(self_states), dtype=torch.float32, device=self.device)
        objects_state_tensor = torch.tensor(np.array(objects_states), dtype=torch.float32, device=self.device)
        objects_mask_tensor = torch.tensor(np.array(objects_masks), dtype=torch.float32, device=self.device)
        
        return TensorDict({
            "self_state": self_state_tensor,
            "objects_state": objects_state_tensor,
            "objects_mask": objects_mask_tensor,
        }, batch_size=[])
    
    def _set_seed(self, seed: int):
        self.seed_value = seed
        self._env.seed = seed
        self._env.rd = np.random.RandomState(seed)
    
    def reset_with_eval_config(self, eval_config):
        """For evaluation compatibility with original trainer"""
        state, _, _ = self._env.reset_with_eval_config(eval_config)
        return self._format_observation(state)
    
    def check_all_deactivated(self):
        return self._env.check_all_deactivated()
    
    def check_all_reach_goal(self):
        return self._env.check_all_reach_goal()
    
    def check_any_collision(self):
        return self._env.check_any_collision()
    
    @property
    def robots(self):
        return self._env.robots
