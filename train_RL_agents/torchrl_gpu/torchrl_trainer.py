import torch
import numpy as np
import os
import json
import copy
from torchrl_gpu.policy_wrapper import PolicyWrapper


class TorchRLTrainer:
    """
    TorchRL-based trainer using Collector and ReplayBuffer for GPU-accelerated sampling.
    Maintains compatibility with original training workflow for reproducibility.
    """
    
    def __init__(self,
                 train_env,
                 eval_env,
                 eval_schedule,
                 rl_agent,
                 device="cuda",
                 UPDATE_EVERY=4,
                 learning_starts=2000,
                 target_update_interval=10000,
                 exploration_fraction=0.25,
                 initial_eps=0.6,
                 final_eps=0.05,
                 buffer_size=1_000_000,
                 batch_size=64,
                 imitation=False,
                 il_agent=None):
        
        self.train_env = train_env
        self.eval_env = eval_env
        self.rl_agent = rl_agent
        self.device = torch.device(device)
        self.eval_config = []
        self.create_eval_configs(eval_schedule)
        
        self.UPDATE_EVERY = UPDATE_EVERY
        self.learning_starts = 0 if imitation else learning_starts
        self.target_update_interval = target_update_interval
        self.exploration_fraction = exploration_fraction
        self.initial_eps = initial_eps
        self.final_eps = final_eps
        self.batch_size = batch_size
        self.imitation = imitation
        self.il_agent = il_agent
        
        if self.imitation:
            assert self.il_agent is not None, "Imitation Learning agent not given!"
        
        self.policy_wrapper = PolicyWrapper(self.rl_agent, max_obj_num=5, epsilon=self.initial_eps)
        
        self.current_timestep = 0
        self.learning_timestep = 0
        
        self.eval_timesteps = []
        self.eval_observations = []
        self.eval_actions = []
        self.eval_trajectories = []
        self.eval_rewards = []
        self.eval_successes = []
        self.eval_times = []
        self.eval_energies = []
        self.eval_relations = []
    
    def create_eval_configs(self, eval_schedule):
        self.eval_config.clear()
        
        count = 0
        for i, num_episode in enumerate(eval_schedule["num_episodes"]):
            for _ in range(num_episode):
                self.eval_env._env.num_robots = eval_schedule["num_robots"][i]
                self.eval_env._env.num_cores = eval_schedule["num_cores"][i]
                self.eval_env._env.num_obs = eval_schedule["num_obstacles"][i]
                self.eval_env._env.min_start_goal_dis = eval_schedule["min_start_goal_dis"][i]
                
                self.eval_env._env.reset()
                
                self.eval_config.append(self.eval_env._env.episode_data())
                count += 1
    
    def save_eval_config(self, directory):
        file = os.path.join(directory, "eval_configs.json")
        with open(file, "w+") as f:
            json.dump(self.eval_config, f)
    
    def learn(self, total_timesteps, eval_freq, eval_log_path, verbose=True):
        td = self.train_env.reset()
        
        ep_rewards = np.zeros(len(self.train_env.robots))
        ep_deactivated_t = [-1] * len(self.train_env.robots)
        ep_length = 0
        ep_num = 0
        
        while self.current_timestep <= total_timesteps:
            if not self.imitation:
                eps = self.linear_eps(total_timesteps)
                self.policy_wrapper.set_epsilon(eps)
            
            td = self.policy_wrapper(td)
            
            td_next = self.train_env.step(td)
            
            if self.rl_agent.training:
                rewards_cpu = td_next["reward"].cpu().numpy().flatten()
                dones_cpu = td_next["done"].cpu().numpy().flatten()
                
                self_state_cpu = td["self_state"].cpu().numpy()
                obj_state_cpu = td["objects_state"].cpu().numpy()
                obj_mask_cpu = td["objects_mask"].cpu().numpy()
                
                next_self_state_cpu = td_next["self_state"].cpu().numpy()
                next_obj_state_cpu = td_next["objects_state"].cpu().numpy()
                next_obj_mask_cpu = td_next["objects_mask"].cpu().numpy()
                
                actions_cpu = td["action"].cpu().numpy()
            else:
                rewards_cpu = td_next["reward"].cpu().numpy().flatten()
                dones_cpu = td_next["done"].cpu().numpy().flatten()
            
            for i, rob in enumerate(self.train_env.robots):
                if rob.deactivated:
                    continue
                
                ep_rewards[i] += self.rl_agent.GAMMA ** ep_length * rewards_cpu[i]
                
                if self.rl_agent.training:
                    valid_mask = obj_mask_cpu[i] > 0.5
                    obj_list = obj_state_cpu[i][valid_mask].tolist()
                    state = (self_state_cpu[i].tolist(), obj_list)
                    
                    next_valid_mask = next_obj_mask_cpu[i] > 0.5
                    next_obj_list = next_obj_state_cpu[i][next_valid_mask].tolist()
                    next_state = (next_self_state_cpu[i].tolist(), next_obj_list)
                    
                    action = actions_cpu[i]
                    reward = rewards_cpu[i]
                    done = dones_cpu[i]
                    
                    if self.rl_agent.agent_type == "Rainbow":
                        self.rl_agent.memory.append(state, action, reward, done)
                    else:
                        self.rl_agent.memory.add((state, action, reward, next_state, done))
                
                if rob.collision or rob.reach_goal:
                    rob.deactivated = True
                    ep_deactivated_t[i] = ep_length
            
            end_episode = (ep_length >= 1000) or self.train_env.check_all_deactivated()
            
            if self.current_timestep >= self.learning_starts:
                if not self.rl_agent.training:
                    continue
                
                if self.current_timestep % self.UPDATE_EVERY == 0:
                    num_elements = self.rl_agent.memory.transitions.num_elements() if self.rl_agent.agent_type == \
                                   "Rainbow" else self.rl_agent.memory.size()
                    
                    if num_elements > self.rl_agent.BATCH_SIZE:
                        self.rl_agent.train()
                
                if self.current_timestep % self.target_update_interval == 0:
                    self.rl_agent.soft_update()
                
                if self.current_timestep == self.learning_starts or self.current_timestep % eval_freq == 0:
                    self.evaluation()
                    self.save_evaluation(eval_log_path)
                    
                    if not self.rl_agent.training:
                        continue
                    
                    self.rl_agent.save_latest_model(eval_log_path)
            
            if end_episode:
                ep_num += 1
                
                if verbose:
                    if self.imitation:
                        print("======== IL Episode Info ========")
                    else:
                        print("======== RL Episode Info ========")
                    
                    print("current ep_length: ", ep_length)
                    print("current ep_num: ", ep_num)
                    
                    if not self.imitation:
                        print("current exploration rate: ", eps)
                    
                    print("current timesteps: ", self.current_timestep)
                    print("total timesteps: ", total_timesteps)
                    print("======== Episode Info ========\n")
                    print("======== Robots Info ========")
                    for i, rob in enumerate(self.train_env.robots):
                        info_state = "normal"
                        if rob.collision:
                            info_state = "deactivated after collision"
                        elif rob.reach_goal:
                            info_state = "deactivated after reaching goal"
                        
                        if info_state != "normal":
                            print(f"Robot {i} ep reward: {ep_rewards[i]:.2f}, {info_state} at step {ep_deactivated_t[i]}")
                        else:
                            print(f"Robot {i} ep reward: {ep_rewards[i]:.2f}, {info_state}")
                    print("======== Robots Info ========\n")
                
                td = self.train_env.reset()
                
                ep_rewards = np.zeros(len(self.train_env.robots))
                ep_deactivated_t = [-1] * len(self.train_env.robots)
                ep_length = 0
            else:
                td = td_next
                ep_length += 1
            
            self.current_timestep += 1
    
    def linear_eps(self, total_timesteps):
        progress = self.current_timestep / total_timesteps
        if progress < self.exploration_fraction:
            r = progress / self.exploration_fraction
            return self.initial_eps + r * (self.final_eps - self.initial_eps)
        else:
            return self.final_eps
    
    def evaluation(self):
        observations_data = []
        actions_data = []
        trajectories_data = []
        rewards_data = []
        successes_data = []
        times_data = []
        energies_data = []
        relations_data = []
        
        self.policy_wrapper.set_training_mode(False)
        
        for idx, config in enumerate(self.eval_config):
            print(f"Evaluating episode {idx}")
            td = self.eval_env.reset_with_eval_config(config)
            
            rob_num = len(self.eval_env.robots)
            
            relations = [[] for _ in range(rob_num)]
            rewards = [0.0] * rob_num
            times = [0.0] * rob_num
            energies = [0.0] * rob_num
            end_episode = False
            length = 0
            
            while not end_episode:
                td = self.policy_wrapper(td)
                td_next = self.eval_env._step(td)
                
                reward_vals = td_next["reward"].cpu().numpy().flatten()
                
                for i, rob in enumerate(self.eval_env.robots):
                    if rob.deactivated:
                        continue
                    
                    rewards[i] += self.rl_agent.GAMMA ** length * reward_vals[i]
                    times[i] += rob.dt * rob.N
                    energies[i] += rob.compute_step_energy_cost()
                    
                    if rob.collision or rob.reach_goal:
                        rob.deactivated = True
                
                end_episode = (length >= 1000) or self.eval_env.check_any_collision() or self.eval_env.check_all_deactivated()
                length += 1
                td = td_next
            
            observations = []
            actions = []
            trajectories = []
            for rob in self.eval_env.robots:
                observations.append(copy.deepcopy(rob.observation_history))
                actions.append(copy.deepcopy(rob.action_history))
                trajectories.append(copy.deepcopy(rob.trajectory))
            
            success = True if self.eval_env.check_all_reach_goal() else False
            
            observations_data.append(observations)
            actions_data.append(actions)
            trajectories_data.append(trajectories)
            rewards_data.append(np.mean(rewards))
            successes_data.append(success)
            times_data.append(np.mean(times))
            energies_data.append(np.mean(energies))
            relations_data.append(relations)
        
        self.policy_wrapper.set_training_mode(True)
        
        avg_r = np.mean(rewards_data)
        success_rate = np.sum(successes_data) / len(successes_data)
        idx = np.where(np.array(successes_data) == 1)[0]
        avg_t = None if np.shape(idx)[0] == 0 else np.mean(np.array(times_data)[idx])
        avg_e = None if np.shape(idx)[0] == 0 else np.mean(np.array(energies_data)[idx])
        
        print(f"++++++++ Evaluation Info ++++++++")
        print(f"Avg cumulative reward: {avg_r:.2f}")
        print(f"Success rate: {success_rate:.2f}")
        if avg_t is not None:
            print(f"Avg time: {avg_t:.2f}")
            print(f"Avg energy: {avg_e:.2f}")
        print(f"++++++++ Evaluation Info ++++++++\n")
        
        self.eval_timesteps.append(self.current_timestep)
        self.eval_observations.append(observations_data)
        self.eval_actions.append(actions_data)
        self.eval_trajectories.append(trajectories_data)
        self.eval_rewards.append(rewards_data)
        self.eval_successes.append(successes_data)
        self.eval_times.append(times_data)
        self.eval_energies.append(energies_data)
        self.eval_relations.append(relations_data)
    
    def save_evaluation(self, eval_log_path):
        filename = "evaluations.npz"
        
        np.savez(
            os.path.join(eval_log_path, filename),
            timesteps=np.array(self.eval_timesteps, dtype=object),
            observations=np.array(self.eval_observations, dtype=object),
            actions=np.array(self.eval_actions, dtype=object),
            trajectories=np.array(self.eval_trajectories, dtype=object),
            rewards=np.array(self.eval_rewards, dtype=object),
            successes=np.array(self.eval_successes, dtype=object),
            times=np.array(self.eval_times, dtype=object),
            energies=np.array(self.eval_energies, dtype=object),
            relations=np.array(self.eval_relations, dtype=object),
        )
