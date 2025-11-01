import random
from collections import deque
import torch
import copy

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
        if max_curr_obj_num == 0:
            padded_object_states_batch = []
            padded_object_states_batch_mask = []
        else:
            padded_object_states_batch = [sublist + [[0.]*self.obj_len] * (self.max_obj_num - len(sublist)) for sublist in object_states_batch]
            padded_object_states_batch_mask = [[1.]*len(sublist) + [0.]*(self.max_obj_num-len(sublist)) for sublist in object_states_batch]
            
        
        return (self_state_batch,padded_object_states_batch,padded_object_states_batch_mask)


    

