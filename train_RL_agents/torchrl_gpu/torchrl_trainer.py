import torch
import numpy as np
import os
import json
import copy
from tensordict import TensorDict
from torchrl_gpu.policy_wrapper import PolicyWrapper
from policy.replay_buffer import TensorDictReplayBuffer


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
        
        # Initialize GPU replay buffer for non-Rainbow agents
        if self.rl_agent.training and self.rl_agent.agent_type != "Rainbow":
            self.rl_agent.memory = TensorDictReplayBuffer(capacity=self.rl_agent.BUFFER_SIZE, device=self.device)
        
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
                # Evaluation env is single (E=1)
                base_env = self.eval_env._envs[0]
                base_env.num_robots = eval_schedule["num_robots"][i]
                base_env.num_cores = eval_schedule["num_cores"][i]
                base_env.num_obs = eval_schedule["num_obstacles"][i]
                base_env.min_start_goal_dis = eval_schedule["min_start_goal_dis"][i]
                
                base_env.reset()
                
                self.eval_config.append(base_env.episode_data())
                count += 1
    
    def save_eval_config(self, directory):
        file = os.path.join(directory, "eval_configs.json")
        with open(file, "w+") as f:
            json.dump(self.eval_config, f)
    
    def learn(self, total_timesteps, eval_freq, eval_log_path, verbose=True):
        td = self.train_env.reset()

        # E×R dimensions
        E = td.batch_size[0] if len(td.batch_size) > 0 else 1
        R = self.train_env.num_robots

        def flatten_ER_obs(td_obs):
            # Only observation keys
            return TensorDict(
                {
                    "self_state": td_obs["self_state"].reshape(E * R, -1),
                    "objects_state": td_obs["objects_state"].reshape(E * R, td_obs["objects_state"].shape[2], td_obs["objects_state"].shape[3]),
                    "objects_mask": td_obs["objects_mask"].reshape(E * R, -1),
                },
                batch_size=[E * R],
                device=self.device,
            )

        def flatten_ER_transition(td_curr, td_next, actions_flat):
            B = E * R
            return TensorDict(
                {
                    "self_state": td_curr["self_state"].reshape(B, -1),
                    "objects_state": td_curr["objects_state"].reshape(B, td_curr["objects_state"].shape[2], td_curr["objects_state"].shape[3]),
                    "objects_mask": td_curr["objects_mask"].reshape(B, -1),
                    "action": actions_flat,
                    "reward": td_next["reward"].reshape(B, 1),
                    "done": td_next["done"].reshape(B, 1),
                    ("next", "self_state"): td_next["self_state"].reshape(B, -1),
                    ("next", "objects_state"): td_next["objects_state"].reshape(B, td_next["objects_state"].shape[2], td_next["objects_state"].shape[3]),
                    ("next", "objects_mask"): td_next["objects_mask"].reshape(B, -1),
                },
                batch_size=[B],
                device=self.device,
            )

        ep_length = 0
        ep_num = 0

        # Precompute stream ids for Rainbow n-step builder (env_id * R + robot_id)
        env_ids = torch.arange(E, device=self.device).view(E, 1).expand(E, R)
        robot_ids = torch.arange(R, device=self.device).view(1, R).expand(E, R)
        stream_ids = (env_ids * R + robot_ids).reshape(E * R)

        # Gradient accumulation / AMP settings for Rainbow
        train_batch_size_total = getattr(self, "train_batch_size_total", self.rl_agent.BATCH_SIZE)
        grad_accum_steps = getattr(self, "grad_accum_steps", 1)
        amp_enabled = getattr(self, "amp_enabled", (self.device.type == "cuda"))

        while self.current_timestep <= total_timesteps:
            if not self.imitation:
                eps = self.linear_eps(total_timesteps)
                self.policy_wrapper.set_epsilon(eps)

            # Flatten E×R for policy forward
            td_flat = flatten_ER_obs(td)
            td_actions = self.policy_wrapper(td_flat)  # sets "action"
            actions_flat = td_actions["action"]
            # Unflatten to (E, R, A)
            action_dim = actions_flat.shape[-1]
            actions_ER = actions_flat.view(E, R, action_dim)

            td_next = self.train_env.step(TensorDict({"action": actions_ER}, batch_size=[E], device=self.device))

            # Pack and push transitions to replay buffer(s)
            if self.rl_agent.training:
                # Determine action dtype per agent
                is_discrete = self.rl_agent.agent_type in ("IQN", "DQN", "Rainbow")
                if is_discrete:
                    actions_store = actions_flat.round().long().view(E * R, 1)
                else:
                    actions_store = actions_flat.to(dtype=torch.float32)

                transition_flat = flatten_ER_transition(td, td_next, actions_store)

                # Active mask to skip already-done slots
                active_mask = (~td_next["done"].squeeze(-1)).reshape(E * R)

                if self.rl_agent.agent_type == "Rainbow":
                    if active_mask.any():
                        self.rl_agent.memory.add_batch(transition_flat[active_mask], stream_ids[active_mask])
                else:
                    if active_mask.any():
                        self.rl_agent.memory.add(transition_flat[active_mask])

            # Episode control: keep global episode semantics (reset all envs together)
            end_episode = (ep_length >= 1000)

            if self.current_timestep >= self.learning_starts and self.rl_agent.training:
                if self.current_timestep % self.UPDATE_EVERY == 0:
                    num_elements = (
                        self.rl_agent.memory.transitions.num_elements()
                        if self.rl_agent.agent_type == "Rainbow"
                        else self.rl_agent.memory.size()
                    )

                    if num_elements > self.rl_agent.BATCH_SIZE:
                        if self.rl_agent.agent_type == "Rainbow":
                            # Large batch, gradient accumulation, AMP
                            self.rl_agent.optimizer.zero_grad()
                            micro_bs = max(1, train_batch_size_total // max(1, grad_accum_steps))
                            for _ in range(grad_accum_steps):
                                idxs, states, actions, returns, next_states, nonterminals, weights = self.rl_agent.memory.sample(micro_bs)
                                with torch.cuda.amp.autocast(enabled=amp_enabled):
                                    loss_per = self.rl_agent.rainbow_loss_from_batch(states, actions, returns, next_states, nonterminals)
                                    loss = (weights.to(self.device) * loss_per).mean()
                                (loss / max(1, grad_accum_steps)).backward()
                                # priority update per micro-batch
                                self.rl_agent.memory.update_priorities(idxs, loss_per.detach())
                            torch.nn.utils.clip_grad_norm_(self.rl_agent.policy_local.parameters(), 0.5)
                            self.rl_agent.optimizer.step()
                        else:
                            # Fallback to agents' original train routine
                            self.rl_agent.train()

                if self.current_timestep % self.target_update_interval == 0:
                    self.rl_agent.soft_update()

                if self.current_timestep == self.learning_starts or self.current_timestep % eval_freq == 0:
                    self.evaluation()
                    self.save_evaluation(eval_log_path)

                    if self.rl_agent.training:
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
                td = self.train_env.reset()
                # Reset counters
                ep_length = 0
                # Re-read in case schedule changed
                E = td.batch_size[0] if len(td.batch_size) > 0 else 1
                R = self.train_env.num_robots
                env_ids = torch.arange(E, device=self.device).view(E, 1).expand(E, R)
                robot_ids = torch.arange(R, device=self.device).view(1, R).expand(E, R)
                stream_ids = (env_ids * R + robot_ids).reshape(E * R)
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
            
            rob_num = td["self_state"].shape[1]
            
            relations = [[] for _ in range(rob_num)]
            rewards = [0.0] * rob_num
            times = [0.0] * rob_num
            energies = [0.0] * rob_num
            end_episode = False
            length = 0
            
            while not end_episode:
                # Flatten (E=1,R,...) -> (R,...) - get actual R from tensor shape
                actual_rob_num = td["self_state"].shape[1]
                obs_flat = TensorDict(
                    {
                        "self_state": td["self_state"].squeeze(0),
                        "objects_state": td["objects_state"].squeeze(0),
                        "objects_mask": td["objects_mask"].squeeze(0),
                    },
                    batch_size=[actual_rob_num],
                    device=self.device,
                )
                td_actions = self.policy_wrapper(obs_flat)
                actions = td_actions["action"].view(1, actual_rob_num, -1)
                td_next = self.eval_env._step(TensorDict({"action": actions}, batch_size=[1], device=self.device))
                
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
