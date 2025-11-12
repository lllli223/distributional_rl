import random
from collections import deque
import torch
import copy

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""
    def __init__(self, buffer_size, batch_size, obj_len, max_object_num, seed=249, device=None):
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
        self.device = device
    
    def add(self,item):
        """Add a new experience to memory."""
        self.memory.append(item)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        samples = random.sample(self.memory, k=self.batch_size)
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
        if self.device is None:
            if max_curr_obj_num == 0:
                padded_object_states_batch = []
                padded_object_states_batch_mask = []
            else:
                padded_object_states_batch = [sublist + [[0.]*self.obj_len] * (self.max_obj_num - len(sublist)) for sublist in object_states_batch]
                padded_object_states_batch_mask = [[1.]*len(sublist) + [0.]*(self.max_obj_num-len(sublist)) for sublist in object_states_batch]
            return (self_state_batch,padded_object_states_batch,padded_object_states_batch_mask)
        else:
            self_state_tensor = torch.tensor(self_state_batch, dtype=torch.float32, device=self.device)
            if max_curr_obj_num == 0:
                return (self_state_tensor, None, None)
            padded_states = [sublist + [[0.]*self.obj_len] * (self.max_obj_num - len(sublist)) for sublist in object_states_batch]
            padded_masks = [[1.]*len(sublist) + [0.]*(self.max_obj_num-len(sublist)) for sublist in object_states_batch]
            object_tensor = torch.tensor(padded_states, dtype=torch.float32, device=self.device)
            mask_tensor = torch.tensor(padded_masks, dtype=torch.float32, device=self.device)
            return (self_state_tensor, object_tensor, mask_tensor)


# 在 policy/replay_buffer.py 中添加
class TensorReplayBuffer:
    """GPU张量版ReplayBuffer（零CPU-GPU传输）"""

    def __init__(self, buffer_size, batch_size, device, action_dim=1):
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.device = device
        self.action_dim = action_dim  # 支持多维动作
        self.position = 0
        self.size = 0

        # 预分配GPU内存（关键优化）
        self.self_obs_buffer = torch.zeros((buffer_size, 7), device=device, dtype=torch.float32)
        self.obj_obs_buffer = torch.zeros((buffer_size, 5, 5), device=device, dtype=torch.float32)
        self.obj_mask_buffer = torch.zeros((buffer_size, 5), device=device, dtype=torch.float32)
        
        # 根据动作维度设置action_buffer
        if action_dim == 1:
            self.action_buffer = torch.zeros((buffer_size,), device=device, dtype=torch.long)  # 离散动作
            self.is_continuous = False
        else:
            self.action_buffer = torch.zeros((buffer_size, action_dim), device=device, dtype=torch.float32)  # 连续动作
            self.is_continuous = True
        self.reward_buffer = torch.zeros((buffer_size,), device=device, dtype=torch.float32)
        self.next_self_obs_buffer = torch.zeros((buffer_size, 7), device=device, dtype=torch.float32)
        self.next_obj_obs_buffer = torch.zeros((buffer_size, 5, 5), device=device, dtype=torch.float32)
        self.next_obj_mask_buffer = torch.zeros((buffer_size, 5), device=device, dtype=torch.float32)
        self.done_buffer = torch.zeros((buffer_size,), device=device, dtype=torch.bool)

    def add_batch(self, self_obs, obj_obs, obj_mask, actions, rewards,
                  next_self_obs, next_obj_obs, next_obj_mask, dones):
        """
        批量添加经验（GPU张量输入，零拷贝）

        输入：
          - self_obs: [B,R,7] GPU张量
          - obj_obs: [B,R,K,5] GPU张量
          - obj_mask: [B,R,K] GPU张量
          - actions: [B,R] GPU张量（离散）或 [B,R,2]（连续）
          - rewards: [B,R] GPU张量
          - next_*: 同上
          - dones: [B,R] GPU张量（bool）
        """
        B, R = self_obs.shape[0], self_obs.shape[1]

        # 展平为 [B*R, ...]
        self_obs_flat = self_obs.reshape(-1, 7)
        # 安全处理obj_obs维度
        if obj_obs is None or obj_obs.numel() == 0:
            # 没有对象观测，创建空的占位符
            obj_obs_flat = torch.zeros((B * R, 0, 5), device=self_obs.device, dtype=self_obs.dtype)
        elif len(obj_obs.shape) == 3:
            # [B,R,5] -> [B,R,1,5]
            obj_obs_flat = obj_obs.unsqueeze(2).reshape(-1, 1, 5)
        elif len(obj_obs.shape) == 4:
            # [B,R,K,5] -> [B*R,K,5]
            obj_obs_flat = obj_obs.reshape(-1, obj_obs.shape[2], obj_obs.shape[3])
        else:
            raise ValueError(f"obj_obs unexpected shape: {obj_obs.shape}")
        # 安全处理obj_mask维度
        if len(obj_mask.shape) == 2:
            # [B,R] -> [B,R,1]
            obj_mask_flat = obj_mask.unsqueeze(2).reshape(-1, 1)
        elif len(obj_mask.shape) == 3:
            # [B,R,K] -> [B*R,K]
            obj_mask_flat = obj_mask.reshape(-1, obj_mask.shape[2])
        else:
            raise ValueError(f"obj_mask unexpected shape: {obj_mask.shape}")
        actions_flat = actions.reshape(-1)  # 展平为1维张量
        rewards_flat = rewards.reshape(-1)
        next_self_obs_flat = next_self_obs.reshape(-1, 7)
        # 安全处理next_obj_obs维度
        if next_obj_obs is None or next_obj_obs.numel() == 0:
            next_obj_obs_flat = torch.zeros((B * R, 0, 5), device=self_obs.device, dtype=self_obs.dtype)
        elif len(next_obj_obs.shape) == 3:
            next_obj_obs_flat = next_obj_obs.unsqueeze(2).reshape(-1, 1, 5)
        elif len(next_obj_obs.shape) == 4:
            next_obj_obs_flat = next_obj_obs.reshape(-1, next_obj_obs.shape[2], next_obj_obs.shape[3])
        else:
            raise ValueError(f"next_obj_obs unexpected shape: {next_obj_obs.shape}")
        # 安全处理next_obj_mask维度
        if len(next_obj_mask.shape) == 2:
            next_obj_mask_flat = next_obj_mask.unsqueeze(2).reshape(-1, 1)
        elif len(next_obj_mask.shape) == 3:
            next_obj_mask_flat = next_obj_mask.reshape(-1, next_obj_mask.shape[2])
        else:
            raise ValueError(f"next_obj_mask unexpected shape: {next_obj_mask.shape}")
        dones_flat = dones.reshape(-1)

        num_samples = B * R

        for i in range(num_samples):
            idx = self.position

            # 直接GPU内存拷贝（无CPU传输）
            self.self_obs_buffer[idx] = self_obs_flat[i]
            self.obj_obs_buffer[idx] = obj_obs_flat[i]
            self.obj_mask_buffer[idx] = obj_mask_flat[i]
            self.action_buffer[idx] = actions_flat[i]
            self.reward_buffer[idx] = rewards_flat[i]
            self.next_self_obs_buffer[idx] = next_self_obs_flat[i]
            self.next_obj_obs_buffer[idx] = next_obj_obs_flat[i]
            self.next_obj_mask_buffer[idx] = next_obj_mask_flat[i]
            self.done_buffer[idx] = dones_flat[i]

            self.position = (self.position + 1) % self.buffer_size
            self.size = min(self.size + 1, self.buffer_size)

    def sample(self):
        """
        采样batch（全GPU操作）

        返回：GPU张量元组，可直接用于训练
        """
        indices = torch.randint(0, self.size, (self.batch_size,), device=self.device)

        # 直接索引GPU内存（零拷贝）
        states = (
            self.self_obs_buffer[indices],
            self.obj_obs_buffer[indices],
            self.obj_mask_buffer[indices]
        )
        actions = self.action_buffer[indices]
        rewards = self.reward_buffer[indices]
        next_states = (
            self.next_self_obs_buffer[indices],
            self.next_obj_obs_buffer[indices],
            self.next_obj_mask_buffer[indices]
        )
        dones = self.done_buffer[indices]

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return self.size
