import random
from collections import deque
import torch
import copy
from tensordict import TensorDict
from torchrl.data import ReplayBuffer as TorchRLReplayBuffer, RandomSampler

try:
    from torchrl.data import TensorStorage
except ImportError:  # pragma: no cover - older versions may not expose TensorStorage
    TensorStorage = None
try:
    from torchrl.data import LazyTensorStorage
except ImportError:  # pragma: no cover - fallback for very old versions
    LazyTensorStorage = None


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""
    def __init__(self, buffer_size, batch_size, obj_len, max_object_num, seed=249):
        """
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.obj_len = obj_len
        self.max_obj_num = max_object_num
        self.seed = random.seed(seed)
    
    def add(self,item):
        """Add a new experience to memory."""
        self.memory.append(item)
    
    def sample(self, batch_size=None):
        """Randomly sample a batch of experiences from memory."""
        if batch_size is None:
            batch_size = self.batch_size
        samples = random.sample(self.memory, k=batch_size)
        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []
        for sample in samples:
            state, action, reward, next_state, done = sample 
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)

        states = self.state_batch(states)
        next_states = self.state_batch(next_states)

        return states, actions, rewards, next_states, dones

    def size(self):
        """Return the current size of internal memory."""
        return len(self.memory)

    def state_batch(self,states):
        self_state_batch = []
        object_states_batch = []

        for state in states:
            self_state_batch.append(state[0])
            object_states_batch.append(state[1])

        # padded object observations and masks (1 and 0 indicate whether the element is observation or not)
        max_curr_obj_num = max(len(sublist) for sublist in object_states_batch)
        if max_curr_obj_num == 0:
            padded_object_states_batch = []
            padded_object_states_batch_mask = []
        else:
            padded_object_states_batch = [sublist + [[0.]*self.obj_len] * (self.max_obj_num - len(sublist)) for sublist in object_states_batch]
            padded_object_states_batch_mask = [[1.]*len(sublist) + [0.]*(self.max_obj_num-len(sublist)) for sublist in object_states_batch]
            
        
        return (self_state_batch,padded_object_states_batch,padded_object_states_batch_mask)


class TensorDictReplayBuffer:
    """
    GPU-resident replay buffer built on top of torchrl.data.ReplayBuffer with TensorStorage.
    Stores and returns TensorDict batches living on CUDA without any CPU transfers.

    Schema for each transition (batch dimension = B):
      - self_state: float32 [B, 7]
      - objects_state: float32 [B, max_obj_num, 5]
      - objects_mask: float32 [B, max_obj_num]
      - action: float32 [B, act_dim] (continuous) or int64 [B, 1] (discrete)
      - reward: float32 [B, 1]
      - done: bool [B, 1]
      - next/self_state, next/objects_state, next/objects_mask
    """

    def __init__(self, capacity, device="cuda"):
        self.device = torch.device(device)

        if LazyTensorStorage is None:
            raise RuntimeError("LazyTensorStorage is required but not available in the current TorchRL version.")

        storage = LazyTensorStorage(max_size=capacity, device=self.device)
        self._rb = TorchRLReplayBuffer(storage=storage, sampler=RandomSampler())

    def add(self, td_or_batch_td: TensorDict):
        """
        Add a batch of transitions in a single call. The input must be a TensorDict with batch size [B].
        """
        if not isinstance(td_or_batch_td, TensorDict):
            raise TypeError("Expected a TensorDict with a batch of transitions.")
        # Ensure it lives on the correct device
        td_or_batch_td = td_or_batch_td.to(self.device)
        # Extend supports batched TensorDict to add B items at once
        self._rb.extend(td_or_batch_td)

    def sample(self, batch_size):
        """Uniformly sample a batch and return a tuple compatible with agents' train_* methods."""
        td = self._rb.sample(batch_size)
        # Ensure device is correct
        td = td.to(self.device)

        # Build tuples to keep backward compatibility with Agent state_to_tensor()
        states = (
            td.get("self_state"),
            td.get("objects_state"),
            td.get("objects_mask"),
        )
        next_states = (
            td.get(("next", "self_state")),
            td.get(("next", "objects_state")),
            td.get(("next", "objects_mask")),
        )
        actions = td.get("action")
        rewards = td.get("reward")
        dones = td.get("done")
        return states, actions, rewards, next_states, dones

    def size(self):
        try:
            return len(self._rb._storage)
        except Exception:
            # Fallback to ReplayBuffer length which should proxy to storage
            return len(self._rb)
