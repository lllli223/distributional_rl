import torch
import torch.optim as optim
from policy.AC_IQN_model import AC_IQN_Policy
from policy.IQN_model import IQN_Policy
from policy.DDPG_model import DDPG_Policy
from policy.DQN_model import DQN_Policy
from policy.Rainbow_model import Rainbow_Policy
from policy.SAC_model import SAC_Policy
from policy.replay_buffer import ReplayBuffer
from policy.replay_memory_rainbow import ReplayMemory
from marinenav_env.envs.utils.robot import Robot
import numpy as np
import random
import time
from torch.nn import functional as F 
import copy

class Agent():
    def __init__(self, 
                 self_dimension=7,
                 object_dimension=5,
                 max_object_num=5,
                 self_feature_dimension=56,
                 object_feature_dimension=40,
                 concat_feature_dimension=256,
                 hidden_dimension=128,
                #  value_ranges_of_action=[[-500.0,1000.0],[-500.0,1000.0]], # for AC-IQN
                 value_ranges_of_action=[[-1.0,1.0],[-1.0,1.0]], # for AC-IQN
                 action_size=25, # for IQN
                 multi_steps=3,
                 BATCH_SIZE=64, 
                 BUFFER_SIZE=1_000_000,
                 LR=1e-4, 
                 TAU=1.0, 
                 GAMMA=0.99,  
                 device="cpu", 
                 seed=0,
                 training=True,
                 agent_type=None
                 ):
        
        self.device = device
        self.LR = LR
        self.TAU = TAU
        self.GAMMA = GAMMA
        self.BUFFER_SIZE = BUFFER_SIZE
        self.BATCH_SIZE = BATCH_SIZE
        self.training = training
        self.value_ranges_of_action = copy.deepcopy(value_ranges_of_action)
        self.action_size = action_size
        self.agent_type = agent_type

        if training:
            if agent_type == "AC-IQN":
                self.policy_local = AC_IQN_Policy(self_dimension,
                                                  object_dimension,
                                                  max_object_num,
                                                  self_feature_dimension,
                                                  object_feature_dimension,
                                                  concat_feature_dimension,
                                                  hidden_dimension,
                                                  value_ranges_of_action,
                                                  device,
                                                  seed)
                self.policy_target = AC_IQN_Policy(self_dimension,
                                                   object_dimension,
                                                   max_object_num,
                                                   self_feature_dimension,
                                                   object_feature_dimension,
                                                   concat_feature_dimension,
                                                   hidden_dimension,
                                                   value_ranges_of_action,
                                                   device,
                                                   seed)
                self.actor_optimizer = optim.Adam(self.policy_local.actor.parameters(), lr=self.LR)
                self.critic_optimizer = optim.Adam(self.policy_local.critic.parameters(), lr=self.LR)
            elif agent_type == "IQN":
                self.policy_local = IQN_Policy(self_dimension,
                                               object_dimension,
                                               max_object_num,
                                               self_feature_dimension,
                                               object_feature_dimension,
                                               concat_feature_dimension,
                                               hidden_dimension,
                                               action_size,
                                               device,
                                               seed).to(device)
                self.policy_target = IQN_Policy(self_dimension,
                                                object_dimension,
                                                max_object_num,
                                                self_feature_dimension,
                                                object_feature_dimension,
                                                concat_feature_dimension,
                                                hidden_dimension,
                                                action_size,
                                                device,
                                                seed).to(device)
                self.optimizer = optim.Adam(self.policy_local.parameters(), lr=self.LR)
            elif agent_type == "DDPG":
                self.policy_local = DDPG_Policy(self_dimension,
                                                object_dimension,
                                                max_object_num,
                                                self_feature_dimension,
                                                object_feature_dimension,
                                                concat_feature_dimension,
                                                hidden_dimension,
                                                value_ranges_of_action,
                                                device,
                                                seed)
                self.policy_target = DDPG_Policy(self_dimension,
                                                object_dimension,
                                                max_object_num,
                                                self_feature_dimension,
                                                object_feature_dimension,
                                                concat_feature_dimension,
                                                hidden_dimension,
                                                value_ranges_of_action,
                                                device,
                                                seed)
                self.actor_optimizer = optim.Adam(self.policy_local.actor.parameters(), lr=self.LR)
                self.critic_optimizer = optim.Adam(self.policy_local.critic.parameters(), lr=self.LR)
            elif agent_type == "DQN":
                self.policy_local = DQN_Policy(self_dimension,
                                               object_dimension,
                                               max_object_num,
                                               self_feature_dimension,
                                               object_feature_dimension,
                                               concat_feature_dimension,
                                               hidden_dimension,
                                               action_size,
                                               device,
                                               seed).to(device)
                self.policy_target = DQN_Policy(self_dimension,
                                                object_dimension,
                                                max_object_num,
                                                self_feature_dimension,
                                                object_feature_dimension,
                                                concat_feature_dimension,
                                                hidden_dimension,
                                                action_size,
                                                device,
                                                seed).to(device)
                self.optimizer = optim.Adam(self.policy_local.parameters(), lr=self.LR)
            elif agent_type == "SAC":
                self.alpha = 0.078
                self.policy_local = SAC_Policy(self_dimension,
                                               object_dimension,
                                               max_object_num,
                                               self_feature_dimension,
                                               object_feature_dimension,
                                               concat_feature_dimension,
                                               hidden_dimension,
                                               value_ranges_of_action,
                                               device,
                                               seed)
                self.policy_target = SAC_Policy(self_dimension,
                                                object_dimension,
                                                max_object_num,
                                                self_feature_dimension,
                                                object_feature_dimension,
                                                concat_feature_dimension,
                                                hidden_dimension,
                                                value_ranges_of_action,
                                                device,
                                                seed)
                self.actor_optimizer = optim.Adam(self.policy_local.actor.parameters(), lr=self.LR)
                self.critic_1_optimizer = optim.Adam(self.policy_local.critic_1.parameters(), lr=self.LR)
                self.critic_2_optimizer = optim.Adam(self.policy_local.critic_2.parameters(), lr=self.LR)
            elif agent_type == "Rainbow":
                self.atoms = 51
                self.Vmin = -1.0
                self.Vmax = 1.0
                self.support = torch.linspace(self.Vmin, self.Vmax, self.atoms).to(device)
                self.delta_z = (self.Vmax - self.Vmin) / (self.atoms - 1)
                self.n = multi_steps
                self.policy_local = Rainbow_Policy(self_dimension,
                                                   object_dimension,
                                                   max_object_num,
                                                   self_feature_dimension,
                                                   object_feature_dimension,
                                                   concat_feature_dimension,
                                                   hidden_dimension,
                                                   action_size,
                                                   self.atoms,
                                                   device,
                                                   seed).to(device)
                self.policy_target = Rainbow_Policy(self_dimension,
                                                    object_dimension,
                                                    max_object_num,
                                                    self_feature_dimension,
                                                    object_feature_dimension,
                                                    concat_feature_dimension,
                                                    hidden_dimension,
                                                    action_size,
                                                    self.atoms,
                                                    device,
                                                    seed).to(device)
                self.optimizer = optim.Adam(self.policy_local.parameters(), lr=self.LR)
            else:
                raise RuntimeError("Agent type not implemented!")

            if agent_type == "Rainbow":
                self.memory = ReplayMemory(device,BUFFER_SIZE)
            else:
                self.memory = ReplayBuffer(BUFFER_SIZE,BATCH_SIZE,object_dimension,max_object_num)
    
    def act_ac_iqn(self, state, eps=0.0, cvar=1.0, use_eval=True):
        # epsilon-greedy action selection
        if random.random() > eps:
            state = self.state_to_tensor(self.memory.state_batch([state]))
            if use_eval:
                self.policy_local.actor.eval()
            else:
                self.policy_local.actor.train()
            with torch.no_grad():
                action = self.policy_local.actor(state)
                # action,_ = self.policy_local.actor(state, self.policy_local.actor.K, cvar)
                action = action.cpu().data.numpy()[0].tolist()
            self.policy_local.actor.train()
        else:
            action = []
            for min_value, max_value in self.value_ranges_of_action:
                action.append(np.random.uniform(low=min_value,high=max_value))

        return action

    def act_iqn(self, state, eps=0.0, cvar=1.0, use_eval=True):
        """Returns action index and quantiles 
        Params
        ======
            frame: to adjust epsilon
            state (array_like): current state
        """
        state = self.state_to_tensor(self.memory.state_batch([state]))
        if use_eval:
            self.policy_local.eval()
        else:
            self.policy_local.train()
        with torch.no_grad():
            quantiles, taus = self.policy_local(state, self.policy_local.K, cvar)
            action_values = quantiles.mean(dim=1)
        self.policy_local.train()

        # epsilon-greedy action selection
        if random.random() > eps:
            action = np.argmax(action_values.cpu().data.numpy())
        else:
            action = random.choice(np.arange(self.action_size))
        
        return action, quantiles.cpu().data.numpy(), taus.cpu().data.numpy()
    
    def act_ddpg(self, state, eps=0.0, use_eval=True):
        # epsilon-greedy action selection
        if random.random() > eps:
            state = self.state_to_tensor(self.memory.state_batch([state]))
            if use_eval:
                self.policy_local.actor.eval()
            else:
                self.policy_local.actor.train()
            with torch.no_grad():
                action = self.policy_local.actor(state)
                action = action.cpu().data.numpy()[0].tolist()
            self.policy_local.actor.train()
        else:
            action = []
            for min_value, max_value in self.value_ranges_of_action:
                action.append(np.random.uniform(low=min_value,high=max_value))

        return action
    
    def act_dqn(self, state, eps=0.0, use_eval=True):
        state = self.state_to_tensor(self.memory.state_batch([state]))
        if use_eval:
            self.policy_local.eval()
        else:
            self.policy_local.train()
        with torch.no_grad():
            action_values = self.policy_local(state)
        self.policy_local.train()

        # epsilon-greedy action selection
        if random.random() > eps:
            action = np.argmax(action_values.cpu().data.numpy())
        else:
            action = random.choice(np.arange(self.action_size))

        return action
    
    def act_sac(self, state, eps=0.0, use_eval=True):
        # epsilon-greedy action selection
        if random.random() > eps:
            state = self.state_to_tensor(self.memory.state_batch([state]))
            if use_eval:
                self.policy_local.actor.eval()
            else:
                self.policy_local.actor.train()
            with torch.no_grad():
                action,_ = self.policy_local.actor(state)
                action = action.cpu().data.numpy()[0].tolist()
            self.policy_local.actor.train()
        else:
            action = []
            for min_value, max_value in self.value_ranges_of_action:
                action.append(np.random.uniform(low=min_value,high=max_value))

        return action
    
    def act_rainbow(self, state, eps=0.0, use_eval=True):
        state = self.state_to_tensor(self.memory.state_batch([state]))
        if use_eval:
            self.policy_local.eval()
        else:
            self.policy_local.train()
        with torch.no_grad():
            action_value_probs = self.policy_local(state)
        self.policy_local.train()

        # epsilon-greedy action selection
        if random.random() > eps:
            action = (action_value_probs * self.support).sum(2).argmax(1).item()
        else:
            action = random.choice(np.arange(self.action_size))

        return action

    # def act_adaptive_ac_iqn(self, state, eps=0.0):
    #     cvar = self.adjust_cvar(state)
    #     action = self.act_ac_iqn(state,eps,cvar)
    #     return action

    # def act_adaptive_iqn(self, state, eps=0.0):
    #     """adaptively tune the CVaR value, compute action index and quantiles
    #     Params
    #     ======
    #         frame: to adjust epsilon
    #         state (array_like): current state
    #     """
    #     cvar = self.adjust_cvar(state)
    #     action, quantiles, taus = self.act_iqn(state, eps, cvar)
    #     return action, quantiles, taus, cvar

    # def adjust_cvar(self,state):
    #     # scale CVaR value according to the closest distance to obstacles
        
    #     assert len(state) == 2, "The number of elements in state must be 2!"

    #     _,objects = state 
        
    #     closest_dist = 20.0
    #     if len(objects) > 0:
    #         closest_object = objects[0]
    #         closest_dist = np.linalg.norm(closest_object[:2]) - closest_object[4] - 2.8
        
    #     cvar = 1.0
    #     if closest_dist < 10.0:
    #         cvar = closest_dist / 10.0

    #     return cvar
        
    def state_to_tensor(self,states):
        self_state_batch,object_batch,object_batch_mask = states
        
        self_state_batch = torch.tensor(self_state_batch).float().to(self.device)
        empty = (len(object_batch) == 0)
        object_batch = None if empty else torch.tensor(object_batch).float().to(self.device)
        object_batch_mask = None if empty else torch.tensor(object_batch_mask).float().to(self.device)

        return (self_state_batch,object_batch,object_batch_mask)

    def train(self):
        if self.agent_type == "AC-IQN":
            return self.train_AC_IQN()
        elif self.agent_type == "IQN":
            return self.train_IQN()
        elif self.agent_type == "DDPG":
            return self.train_DDPG()
        elif self.agent_type == "DQN":
            return self.train_DQN()
        elif self.agent_type == "SAC":
            return self.train_SAC()
        elif self.agent_type == "Rainbow":
            return self.train_Rainbow()
        else:
            raise RuntimeError("Agent type not implemented!")

    def train_AC_IQN(self):
        states, actions, rewards, next_states, dones = self.memory.sample()
        states = self.state_to_tensor(states)
        actions = torch.tensor(actions).float().to(self.device)
        rewards = torch.tensor(rewards).unsqueeze(-1).float().to(self.device)
        next_states = self.state_to_tensor(next_states)
        dones = torch.tensor(dones).unsqueeze(-1).float().to(self.device)

        # update critic network
        self.critic_optimizer.zero_grad()
        next_actions = self.policy_target.actor(next_states)
        # next_actions,_ = self.policy_target.actor(next_states)
        next_actions = next_actions.detach()
        Q_targets_next,_ = self.policy_target.critic(next_states,next_actions)
        Q_targets_next = Q_targets_next.detach().unsqueeze(1)
        Q_targets = rewards.unsqueeze(-1) + (self.GAMMA * Q_targets_next * (1. - dones.unsqueeze(-1)))

        Q_expected,taus = self.policy_local.critic(states,actions)
        Q_expected = Q_expected.unsqueeze(-1)

        td_error = Q_targets - Q_expected
        assert td_error.shape == (self.BATCH_SIZE, 8, 8), "wrong td error shape"
        huber_l = calculate_huber_loss(td_error, 1.0)
        quantil_l = abs(taus -(td_error.detach() < 0).float()) * huber_l / 1.0 # Quantile Huber loss
        
        critic_loss = quantil_l.sum(dim=1).mean(dim=1) # keepdim=True if per weights get multiple
        critic_loss = critic_loss.mean()

        critic_loss.backward() # minimize the critic loss
        torch.nn.utils.clip_grad_norm_(self.policy_local.critic.parameters(), 0.5)
        self.critic_optimizer.step()

        # update actor network
        self.actor_optimizer.zero_grad()
        actor_outputs = self.policy_local.actor(states)
        # actor_outputs,_ = self.policy_local.actor(states)
        actor_loss,_ = self.policy_local.critic(states,actor_outputs)
        actor_loss = -actor_loss.mean()
        
        actor_loss.backward() # minimize the actor loss
        torch.nn.utils.clip_grad_norm_(self.policy_local.actor.parameters(), 0.5)
        self.actor_optimizer.step()

        # print("\ncritic loss: ",critic_loss.detach().cpu().numpy())
        # print("actor loss: ",actor_loss.detach().cpu().numpy(),"\n")

        return critic_loss.detach().cpu().numpy(), actor_loss.detach().cpu().numpy()
    
    def train_IQN(self):
        """Update value parameters using given batch of experience tuples
        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """

        states, actions, rewards, next_states, dones = self.memory.sample()
        states = self.state_to_tensor(states)
        actions = torch.tensor(actions).unsqueeze(-1).to(self.device)
        rewards = torch.tensor(rewards).unsqueeze(-1).float().to(self.device)
        next_states = self.state_to_tensor(next_states)
        dones = torch.tensor(dones).unsqueeze(-1).float().to(self.device)

        self.optimizer.zero_grad()
        # Get max predicted Q values (for next states) from target model
        Q_targets_next,_ = self.policy_target(next_states)
        Q_targets_next = Q_targets_next.detach().max(2)[0].unsqueeze(1) # (batch_size, 1, N)
        
        # Compute Q targets for current states 
        Q_targets = rewards.unsqueeze(-1) + (self.GAMMA * Q_targets_next * (1. - dones.unsqueeze(-1)))
        # Get expected Q values from local model
        Q_expected,taus = self.policy_local(states)
        Q_expected = Q_expected.gather(2, actions.unsqueeze(-1).expand(self.BATCH_SIZE, 8, 1))

        # Quantile Huber loss
        td_error = Q_targets - Q_expected
        assert td_error.shape == (self.BATCH_SIZE, 8, 8), "wrong td error shape"
        huber_l = calculate_huber_loss(td_error, 1.0)
        quantil_l = abs(taus -(td_error.detach() < 0).float()) * huber_l / 1.0
        
        loss = quantil_l.sum(dim=1).mean(dim=1) # keepdim=True if per weights get multiple
        loss = loss.mean()

        # minimize the loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_local.parameters(), 0.5)
        self.optimizer.step()

        # print("\nIQN loss: ",loss.detach().cpu().numpy(),"\n")

        return loss.detach().cpu().numpy()
    
    def train_DDPG(self):
        states, actions, rewards, next_states, dones = self.memory.sample()
        states = self.state_to_tensor(states)
        actions = torch.tensor(actions).float().to(self.device)
        rewards = torch.tensor(rewards).unsqueeze(-1).float().to(self.device)
        next_states = self.state_to_tensor(next_states)
        dones = torch.tensor(dones).unsqueeze(-1).float().to(self.device)

        # update critic network
        self.critic_optimizer.zero_grad()
        next_actions = self.policy_target.actor(next_states)
        next_actions = next_actions.detach()
        Q_targets_next = self.policy_target.critic(next_states,next_actions)
        Q_targets_next = Q_targets_next.detach()
        Q_targets = rewards + self.GAMMA * Q_targets_next * (1. - dones)

        Q_expected = self.policy_local.critic(states,actions)

        # compute critic loss
        critic_loss = F.smooth_l1_loss(Q_expected,Q_targets)

        critic_loss.backward() # minimize the critic loss
        torch.nn.utils.clip_grad_norm_(self.policy_local.critic.parameters(), 0.5)
        self.critic_optimizer.step()

        # update actor network
        self.actor_optimizer.zero_grad()
        actor_outputs = self.policy_local.actor(states)
        actor_loss = self.policy_local.critic(states,actor_outputs)
        actor_loss = -actor_loss.mean()
        
        actor_loss.backward() # minimize the actor loss
        torch.nn.utils.clip_grad_norm_(self.policy_local.actor.parameters(), 0.5)
        self.actor_optimizer.step()

        # print("\ncritic loss: ",critic_loss.detach().cpu().numpy())
        # print("actor loss: ",actor_loss.detach().cpu().numpy(),"\n")

        return critic_loss.detach().cpu().numpy(), actor_loss.detach().cpu().numpy()
    
    def train_DQN(self):
        states, actions, rewards, next_states, dones = self.memory.sample()
        states = self.state_to_tensor(states)
        actions = torch.tensor(actions).unsqueeze(-1).to(self.device)
        rewards = torch.tensor(rewards).unsqueeze(-1).float().to(self.device)
        next_states = self.state_to_tensor(next_states)
        dones = torch.tensor(dones).unsqueeze(-1).float().to(self.device)

        self.optimizer.zero_grad()

        # compute target values
        Q_targets_next = self.policy_target(next_states)
        Q_targets_next,_ = Q_targets_next.max(dim=1,keepdim=True)
        Q_targets = rewards + (1-dones) * self.GAMMA * Q_targets_next

        # compute expected values
        Q_expected = self.policy_local(states)
        Q_expected = Q_expected.gather(1,actions)

        # compute loss
        loss = F.smooth_l1_loss(Q_expected,Q_targets)

        # minimize the loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_local.parameters(), 0.5)
        self.optimizer.step()

        return loss.detach().cpu().numpy()
    
    def train_SAC(self):
        ### Based on the function in line 226 of (https://github.com/thu-ml/tianshou/blob/master/tianshou/policy/modelfree/sac.py)
        states, actions, rewards, next_states, dones = self.memory.sample()
        states = self.state_to_tensor(states)
        actions = torch.tensor(actions).float().to(self.device)
        rewards = torch.tensor(rewards).unsqueeze(-1).float().to(self.device)
        next_states = self.state_to_tensor(next_states)
        dones = torch.tensor(dones).unsqueeze(-1).float().to(self.device)

        with torch.no_grad():
            next_actions, next_log_probs = self.policy_target.actor(next_states)
            target_values_1 = self.policy_target.critic_1(next_states,next_actions)
            target_values_2 = self.policy_target.critic_2(next_states,next_actions)
            target_values = torch.min(target_values_1,target_values_2) - self.alpha * next_log_probs
            target_values = rewards + self.GAMMA * target_values * (1. - dones)

        # compute critic losses
        values_1 = self.policy_local.critic_1(states,actions)
        values_1_loss = F.mse_loss(values_1,target_values)

        values_2 = self.policy_local.critic_2(states,actions)
        values_2_loss = F.mse_loss(values_2,target_values)

        # update critic networks
        self.critic_1_optimizer.zero_grad()
        values_1_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_local.critic_1.parameters(),0.5)
        self.critic_1_optimizer.step()

        self.critic_2_optimizer.zero_grad()
        values_2_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_local.critic_2.parameters(),0.5)
        self.critic_2_optimizer.step()

        # compute actor loss
        new_actions, log_probs = self.policy_local.actor(states)
        values_1_new = self.policy_local.critic_1(states,new_actions)
        values_2_new = self.policy_local.critic_2(states,new_actions)
        values_new = torch.min(values_1_new,values_2_new)

        actor_loss = (self.alpha * log_probs - values_new).mean()

        # update actor network
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_local.actor.parameters(),0.5)
        self.actor_optimizer.step()

        return values_1_loss.detach().cpu().numpy(), values_2_loss.detach().cpu().numpy(), actor_loss.detach().cpu().numpy()
    
    def train_Rainbow(self):
        ### Based on the function in line 61 of (https://github.com/Kaixhin/Rainbow/blob/master/agent.py)
        
        # Sample transitions
        idxs, states, actions, returns, next_states, nonterminals, weights = self.memory.sample(self.BATCH_SIZE)

        # Calculate current state probabilities (online network noise already sampled)
        log_ps = self.policy_local(states, log=True)  # Log probabilities log p(s_t, ·; θonline)
        log_ps_a = log_ps[range(self.BATCH_SIZE), actions]  # log p(s_t, a_t; θonline)

        with torch.no_grad():
            # Calculate nth next state probabilities
            pns = self.policy_local(next_states)  # Probabilities p(s_t+n, ·; θonline)
            dns = self.support.expand_as(pns) * pns  # Distribution d_t+n = (z, p(s_t+n, ·; θonline))
            argmax_indices_ns = dns.sum(2).argmax(1)  # Perform argmax action selection using online network: argmax_a[(z, p(s_t+n, a; θonline))]
            self.policy_target.reset_noise()  # Sample new target net noise
            pns = self.policy_target(next_states)  # Probabilities p(s_t+n, ·; θtarget)
            pns_a = pns[range(self.BATCH_SIZE), argmax_indices_ns]  # Double-Q probabilities p(s_t+n, argmax_a[(z, p(s_t+n, a; θonline))]; θtarget)

            # Compute Tz (Bellman operator T applied to z)
            Tz = returns.unsqueeze(1) + nonterminals * (self.GAMMA ** self.n) * self.support.unsqueeze(0)  # Tz = R^n + (γ^n)z (accounting for terminal states)
            Tz = Tz.clamp(min=self.Vmin, max=self.Vmax)  # Clamp between supported values
            # Compute L2 projection of Tz onto fixed support z
            b = (Tz - self.Vmin) / self.delta_z  # b = (Tz - Vmin) / Δz
            l, u = b.floor().to(torch.int64), b.ceil().to(torch.int64)
            # Fix disappearing probability mass when l = b = u (b is int)
            l[(u > 0) * (l == u)] -= 1
            u[(l < (self.atoms - 1)) * (l == u)] += 1

            # Distribute probability of Tz
            # m = states.new_zeros(self.BATCH_SIZE, self.atoms)
            m = torch.zeros(self.BATCH_SIZE, self.atoms, dtype=torch.float32, device=self.device)
            offset = torch.linspace(0, ((self.BATCH_SIZE - 1) * self.atoms), self.BATCH_SIZE).unsqueeze(1).expand(self.BATCH_SIZE, self.atoms).to(actions)
            m.view(-1).index_add_(0, (l + offset).view(-1), (pns_a * (u.float() - b)).view(-1))  # m_l = m_l + p(s_t+n, a*)(u - b)
            m.view(-1).index_add_(0, (u + offset).view(-1), (pns_a * (b - l.float())).view(-1))  # m_u = m_u + p(s_t+n, a*)(b - l)

        loss = -torch.sum(m * log_ps_a, 1)  # Cross-entropy loss (minimises DKL(m||p(s_t, a_t)))
        self.policy_local.zero_grad()
        (weights * loss).mean().backward()  # Backpropagate importance-weighted minibatch loss
        torch.nn.utils.clip_grad_norm_(self.policy_local.parameters(), 0.5)
        self.optimizer.step()

        self.memory.update_priorities(idxs, loss.detach().cpu().numpy())  # Update priorities of sampled transitions

        return loss.detach().cpu().numpy()

    def soft_update(self):
        """Soft update model parameters.
        θ_target = τ * θ_local + (1 - τ) * θ_target
        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter
        """
        if self.agent_type == "AC-IQN":
            for target_param, local_param in zip(self.policy_target.actor.parameters(), self.policy_local.actor.parameters()):
                target_param.data.copy_(self.TAU * local_param.data + (1.0 - self.TAU) * target_param.data)
            for target_param, local_param in zip(self.policy_target.critic.parameters(), self.policy_local.critic.parameters()):
                target_param.data.copy_(self.TAU * local_param.data + (1.0 - self.TAU) * target_param.data)
        elif self.agent_type == "IQN":
            for target_param, local_param in zip(self.policy_target.parameters(), self.policy_local.parameters()):
                target_param.data.copy_(self.TAU * local_param.data + (1.0 - self.TAU) * target_param.data)
        elif self.agent_type == "DDPG":
            for target_param, local_param in zip(self.policy_target.actor.parameters(), self.policy_local.actor.parameters()):
                target_param.data.copy_(self.TAU * local_param.data + (1.0 - self.TAU) * target_param.data)
            for target_param, local_param in zip(self.policy_target.critic.parameters(), self.policy_local.critic.parameters()):
                target_param.data.copy_(self.TAU * local_param.data + (1.0 - self.TAU) * target_param.data)
        elif self.agent_type == "DQN":
            for target_param, local_param in zip(self.policy_target.parameters(), self.policy_local.parameters()):
                target_param.data.copy_(self.TAU * local_param.data + (1.0 - self.TAU) * target_param.data)
        elif self.agent_type == "SAC":
            for target_param, local_param in zip(self.policy_target.actor.parameters(), self.policy_local.actor.parameters()):
                target_param.data.copy_(self.TAU * local_param.data + (1.0 - self.TAU) * target_param.data)
            for target_param, local_param in zip(self.policy_target.critic_1.parameters(), self.policy_local.critic_1.parameters()):
                target_param.data.copy_(self.TAU * local_param.data + (1.0 - self.TAU) * target_param.data)
            for target_param, local_param in zip(self.policy_target.critic_2.parameters(), self.policy_local.critic_2.parameters()):
                target_param.data.copy_(self.TAU * local_param.data + (1.0 - self.TAU) * target_param.data)
        elif self.agent_type == "Rainbow":
            for target_param, local_param in zip(self.policy_target.parameters(), self.policy_local.parameters()):
                target_param.data.copy_(self.TAU * local_param.data + (1.0 - self.TAU) * target_param.data)
        else:
            raise RuntimeError("Agent type not implemented!")

    def save_latest_model(self,directory):
        self.policy_local.save(directory)

    def load_model(self,path,device="cpu"):
        if self.agent_type == "AC-IQN":
            self.policy_local = AC_IQN_Policy.load(path,device)
        elif self.agent_type == "IQN":
            self.policy_local = IQN_Policy.load(path,device)
        elif self.agent_type == "DDPG":
            self.policy_local = DDPG_Policy.load(path,device)
        elif self.agent_type == "DQN":
            self.policy_local = DQN_Policy.load(path,device)
        elif self.agent_type == "SAC":
            self.policy_local = SAC_Policy.load(path,device)
        elif self.agent_type == "Rainbow":
            self.policy_local = Rainbow_Policy.load(path,device)
        else:
            raise RuntimeError("Agent type not implemented!")


def calculate_huber_loss(td_errors, k=1.0):
    """
    Calculate huber loss element-wisely depending on kappa k.
    """
    loss = torch.where(td_errors.abs() <= k, 0.5 * td_errors.pow(2), k * (td_errors.abs() - 0.5 * k))
    assert loss.shape == (td_errors.shape[0], 8, 8), "huber loss has wrong shape"
    return loss