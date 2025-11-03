import numpy as np
import torch
import torch.nn as nn


class PolicyWrapper(nn.Module):
    """
    Wraps existing Agent for TorchRL compatibility.
    Converts TensorDict observations to the format expected by existing agents.
    """

    def __init__(self, agent, max_obj_num=5, epsilon=0.0):
        super().__init__()
        self.agent = agent
        self.max_obj_num = max_obj_num
        self.epsilon = epsilon
        self.training_mode = True
        self.is_continuous = self.agent.agent_type in ("AC-IQN", "DDPG", "SAC")
        self.action_dim = 2 if self.is_continuous else 1

    def forward(self, tensordict):
        self_state = tensordict["self_state"]
        objects_state = tensordict["objects_state"]
        objects_mask = tensordict["objects_mask"]

        batch_size = self_state.shape[0]
        device = self_state.device

        actions = []
        for i in range(batch_size):
            self_obs = self_state[i].cpu().numpy()
            obj_obs = objects_state[i].cpu().numpy()
            mask = objects_mask[i].cpu().numpy()

            obj_list = [obj_obs[j].tolist() for j in range(self.max_obj_num) if mask[j] > 0.5]

            state = (self_obs.tolist(), obj_list)

            if self.training_mode:
                if self.agent.agent_type == "AC-IQN":
                    action = self.agent.act_ac_iqn(state, self.epsilon, use_eval=False)
                elif self.agent.agent_type == "IQN":
                    action, _, _ = self.agent.act_iqn(state, self.epsilon, use_eval=False)
                elif self.agent.agent_type == "DDPG":
                    action = self.agent.act_ddpg(state, self.epsilon, use_eval=False)
                elif self.agent.agent_type == "DQN":
                    action = self.agent.act_dqn(state, self.epsilon, use_eval=False)
                elif self.agent.agent_type == "SAC":
                    action = self.agent.act_sac(state, self.epsilon, use_eval=False)
                elif self.agent.agent_type == "Rainbow":
                    action = self.agent.act_rainbow(state, self.epsilon, use_eval=False)
                else:
                    raise RuntimeError(f"Agent type {self.agent.agent_type} not implemented!")
            else:
                if self.agent.agent_type == "AC-IQN":
                    action = self.agent.act_ac_iqn(state)
                elif self.agent.agent_type == "IQN":
                    action, _, _ = self.agent.act_iqn(state)
                elif self.agent.agent_type == "DDPG":
                    action = self.agent.act_ddpg(state)
                elif self.agent.agent_type == "DQN":
                    action = self.agent.act_dqn(state)
                elif self.agent.agent_type == "SAC":
                    action = self.agent.act_sac(state)
                elif self.agent.agent_type == "Rainbow":
                    action = self.agent.act_rainbow(state)
                else:
                    raise RuntimeError(f"Agent type {self.agent.agent_type} not implemented!")

            if self.is_continuous:
                action_vec = np.asarray(action, dtype=np.float32).reshape(-1)
            else:
                action_vec = np.asarray([action], dtype=np.float32)

            padded_action = np.zeros(self.action_dim, dtype=np.float32)
            padded_action[: min(len(action_vec), self.action_dim)] = action_vec[: self.action_dim]
            actions.append(padded_action)

        action_tensor = torch.tensor(actions, dtype=torch.float32, device=device)

        return tensordict.set("action", action_tensor)

    def set_epsilon(self, epsilon):
        self.epsilon = epsilon

    def set_training_mode(self, mode):
        self.training_mode = mode
