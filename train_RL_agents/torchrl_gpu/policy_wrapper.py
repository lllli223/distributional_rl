import torch
import torch.nn as nn


class PolicyWrapper(nn.Module):
    """
    Wraps existing Agent for TorchRL compatibility.
    Converts TensorDict observations to the format expected by existing agents.
    Optimized for batched GPU inference without CPU transfers.
    """

    def __init__(self, agent, max_obj_num=5, epsilon=0.0):
        super().__init__()
        self.agent = agent
        self.max_obj_num = max_obj_num
        self.epsilon = epsilon
        self.training_mode = True
        self.is_continuous = self.agent.agent_type in ("AC-IQN", "DDPG", "SAC")
        if self.is_continuous:
            self.action_dim = len(self.agent.value_ranges_of_action)
            value_ranges = torch.tensor(self.agent.value_ranges_of_action, dtype=torch.float32)
            self.register_buffer("action_low", value_ranges[:, 0])
            self.register_buffer("action_high", value_ranges[:, 1])
        else:
            self.action_dim = 1
            self.register_buffer("action_low", torch.zeros(0, dtype=torch.float32))
            self.register_buffer("action_high", torch.zeros(0, dtype=torch.float32))

    def forward(self, tensordict):
        self_state = tensordict["self_state"]
        objects_state = tensordict["objects_state"]
        objects_mask = tensordict["objects_mask"]

        batch_size = self_state.shape[0]
        device = self_state.device

        with torch.no_grad():
            if self.is_continuous:
                actions = self._forward_continuous_batched(
                    self_state, objects_state, objects_mask, batch_size, device
                )
            else:
                actions = self._forward_discrete_batched(
                    self_state, objects_state, objects_mask, batch_size, device
                )

        return tensordict.set("action", actions)

    def _forward_continuous_batched(self, self_state, objects_state, objects_mask, batch_size, device):
        state_tuple = (self_state, objects_state, objects_mask)
        
        if self.agent.agent_type == "AC-IQN":
            if self.training_mode:
                self.agent.policy_local.actor.train()
            else:
                self.agent.policy_local.actor.eval()
            actions = self.agent.policy_local.actor(state_tuple)
        elif self.agent.agent_type == "DDPG":
            if self.training_mode:
                self.agent.policy_local.actor.train()
            else:
                self.agent.policy_local.actor.eval()
            actions = self.agent.policy_local.actor(state_tuple)
        elif self.agent.agent_type == "SAC":
            if self.training_mode:
                self.agent.policy_local.actor.train()
            else:
                self.agent.policy_local.actor.eval()
            actions, _ = self.agent.policy_local.actor(state_tuple)
        else:
            raise RuntimeError(f"Agent type {self.agent.agent_type} not implemented!")

        actions = actions.to(device=device, dtype=torch.float32)

        if self.training_mode and self.epsilon > 0:
            explore_mask = torch.rand(batch_size, device=device) < self.epsilon
            if explore_mask.any():
                low = self.action_low.to(device).unsqueeze(0)
                high = self.action_high.to(device).unsqueeze(0)
                random_actions = low + (high - low) * torch.rand(batch_size, self.action_dim, device=device)
                actions = torch.where(explore_mask.unsqueeze(1), random_actions, actions)

        return actions

    def _forward_discrete_batched(self, self_state, objects_state, objects_mask, batch_size, device):
        state_tuple = (self_state, objects_state, objects_mask)
        
        if self.agent.agent_type == "IQN":
            if self.training_mode:
                self.agent.policy_local.train()
            else:
                self.agent.policy_local.eval()
            quantiles, _ = self.agent.policy_local(state_tuple, self.agent.policy_local.K, cvar=1.0)
            action_values = quantiles.mean(dim=1)
        elif self.agent.agent_type == "DQN":
            if self.training_mode:
                self.agent.policy_local.train()
            else:
                self.agent.policy_local.eval()
            action_values = self.agent.policy_local(state_tuple)
        elif self.agent.agent_type == "Rainbow":
            if self.training_mode:
                self.agent.policy_local.train()
            else:
                self.agent.policy_local.eval()
            action_value_probs = self.agent.policy_local(state_tuple)
            action_values = (action_value_probs * self.agent.support.unsqueeze(0).unsqueeze(0)).sum(2)
        else:
            raise RuntimeError(f"Agent type {self.agent.agent_type} not implemented!")

        if self.training_mode and self.epsilon > 0:
            explore_mask = torch.rand(batch_size, device=device) < self.epsilon
            greedy_actions = action_values.argmax(dim=1)
            random_actions = torch.randint(0, self.agent.action_size, (batch_size,), device=device)
            action_indices = torch.where(explore_mask, random_actions, greedy_actions)
        else:
            action_indices = action_values.argmax(dim=1)

        actions = action_indices.unsqueeze(1).float()
        
        return actions

    def set_epsilon(self, epsilon):
        self.epsilon = epsilon

    def set_training_mode(self, mode):
        self.training_mode = mode
