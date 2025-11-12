import json
import numpy as np
import os
import copy
import torch

class TorchTrainer():
    """GPU-optimized trainer for TorchMarineNavEnv"""
    def __init__(self,
                 train_env,
                 eval_env,
                 eval_schedule,
                 rl_agent,
                 UPDATE_EVERY=4,
                 learning_starts=2000,
                 target_update_interval=10000,
                 exploration_fraction=0.25,
                 initial_eps=0.6,
                 final_eps=0.05,
                 imitation=False,
                 il_agent=None
                 ):
        
        self.train_env = train_env
        self.eval_env = eval_env
        self.rl_agent = rl_agent
        self.eval_config = []
        self.create_eval_configs(eval_schedule)

        self.UPDATE_EVERY = UPDATE_EVERY
        self.learning_starts = 0 if imitation else learning_starts
        self.target_update_interval = target_update_interval
        self.exploration_fraction = exploration_fraction
        self.initial_eps = initial_eps
        self.final_eps = final_eps
        self.imitation = imitation
        self.il_agent = il_agent

        if self.imitation:
            assert self.il_agent is not None, "Imitation Learning agent not given!"

        self.current_timestep = 0
        self.learning_timestep = 0

        # Evaluation data
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
        for i, num_episode in enumerate(eval_schedule["num_episodes"]):
            for _ in range(num_episode):
                config = {
                    "num_robots": eval_schedule["num_robots"][i],
                    "num_cores": eval_schedule["num_cores"][i],
                    "num_obstacles": eval_schedule["num_obstacles"][i],
                    "min_start_goal_dis": eval_schedule["min_start_goal_dis"][i]
                }
                self.eval_config.append(config)

    def save_eval_config(self, directory):
        file = os.path.join(directory, "eval_configs.json")
        with open(file, "w+") as f:
            json.dump(self.eval_config, f)

    def learn(self, total_timesteps, eval_freq, eval_log_path, verbose=True):
        # 重置环境
        self.train_env.reset()
        
        # 使用GPU张量版观测替代CPU states列表
        self_obs, obj_obs, obj_mask, _, _ = self.train_env.get_tensor_observation()
        
        # 获取环境批次大小和机器人数量
        B = self.train_env.B
        R = self.train_env.R
        
        # 将初始观测组装为states列表格式（展平B*R维度）
        states = []
        for b in range(B):
            for i in range(R):
                if obj_obs is not None and obj_obs.shape[2] > 0 and (obj_mask[b, i] > 0).any():
                    # 有障碍物观测 - 将GPU张量转为numpy以保持与原版格式一致
                    state_i = (self_obs[b, i].cpu().numpy(), obj_obs[b, i].cpu().numpy(), obj_mask[b, i].cpu().numpy())
                else:
                    # 无障碍物观测
                    state_i = (self_obs[b, i].cpu().numpy(), None, obj_mask[b, i].cpu().numpy())
                states.append(state_i)
        
        ep_rewards = np.zeros(B * R)
        ep_deactivated_t = [-1] * (B * R)
        ep_length = 0
        ep_num = 0
        
        while self.current_timestep <= total_timesteps:
            if not self.imitation:
                eps = self.linear_eps(total_timesteps)
            
            # 使用张量版观测（GPU常驻）
            self_obs, obj_obs, obj_mask, collisions, reach_goals = self.train_env.get_tensor_observation()
            
            # 批量张量版动作选择（GPU常驻）
            is_continuous_action = self.rl_agent.agent_type in ["AC-IQN", "DDPG", "SAC"]
            
            if self.imitation:
                # 模仿学习仍使用原有逐个处理方式
                actions = []
                for flat_idx in range(B * R):
                    b, i = flat_idx // R, flat_idx % R
                    if self.train_env.deactivated_flags[b, i]:
                        actions.append(None)
                        continue
                    state_i = states[flat_idx]
                    if state_i[0] is None:
                        actions.append(None)
                        continue
                    action = self.il_agent.act(state_i)
                    actions.append(action)
                
                if is_continuous_action:
                    actions_t = torch.zeros((B, R, 2), device=self.train_env.device, dtype=self.train_env.dtype)
                    for flat_idx, a in enumerate(actions):
                        b, i = flat_idx // R, flat_idx % R
                        if a is None:
                            actions_t[b, i, 0] = 0.0
                            actions_t[b, i, 1] = 0.0
                        else:
                            actions_t[b, i, 0] = float(a[0])
                            actions_t[b, i, 1] = float(a[1])
                else:
                    actions_t = torch.zeros((B, R, 1), device=self.train_env.device, dtype=torch.long)
                    zero_idx = 12
                    for flat_idx, a in enumerate(actions):
                        b, i = flat_idx // R, flat_idx % R
                        if a is None:
                            actions_t[b, i, 0] = zero_idx
                        else:
                            actions_t[b, i, 0] = int(a)
            else:
                # RL训练：批量张量版动作选择
                if self.rl_agent.agent_type == "AC-IQN":
                    actions_t = self.rl_agent.act_batch_tensor_ac_iqn(self_obs, obj_obs, obj_mask, eps, use_eval=False)
                elif self.rl_agent.agent_type == "IQN":
                    actions_t = self.rl_agent.act_batch_tensor_iqn(self_obs, obj_obs, obj_mask, eps, use_eval=False).unsqueeze(-1)
                elif self.rl_agent.agent_type == "DDPG":
                    actions_t = self.rl_agent.act_batch_tensor_ddpg(self_obs, obj_obs, obj_mask, eps, use_eval=False)
                elif self.rl_agent.agent_type == "DQN":
                    actions_t = self.rl_agent.act_batch_tensor_dqn(self_obs, obj_obs, obj_mask, eps, use_eval=False).unsqueeze(-1)
                elif self.rl_agent.agent_type == "SAC":
                    actions_t = self.rl_agent.act_batch_tensor_sac(self_obs, obj_obs, obj_mask, eps, use_eval=False)
                elif self.rl_agent.agent_type == "Rainbow":
                    actions_t = self.rl_agent.act_batch_tensor_rainbow(self_obs, obj_obs, obj_mask, eps, use_eval=False).unsqueeze(-1)
                else:
                    raise RuntimeError("Agent type not implemented!")
                
                # 对失活机器人的动作置零
                deactivated_mask = self.train_env.deactivated_flags
                if is_continuous_action:
                    actions_t[deactivated_mask] = 0.0
                else:
                    actions_t[deactivated_mask.unsqueeze(-1).expand_as(actions_t)] = 12  # zero action index

            # Execute actions using GPU tensor version (零CPU传输)
            next_self_obs, next_obj_obs, next_obj_mask, rewards, dones, collisions, reach_goals = \
                self.train_env.step_tensor(actions_t, is_continuous_action)
            
            # Generate infos from environment state (展平B*R维度)
            infos = []
            for b in range(B):
                for r in range(R):
                    if self.train_env.deactivated_flags[b, r]:
                        if self.train_env.collision_flags[b, r]:
                            infos.append({"state": "deactivated after collision"})
                        elif self.train_env.reach_goal_flags[b, r]:
                            infos.append({"state": "deactivated after reaching goal"})
                        else:
                            infos.append({"state": "deactivated"})
                    elif self.train_env.reach_goal_flags[b, r]:
                        infos.append({"state": "reach goal"})
                    else:
                        infos.append({"state": "normal"})
            
            # 组装next_states（展平B*R维度）
            next_states = []
            for b in range(B):
                for i in range(R):
                    if next_obj_obs is not None and next_obj_obs.shape[2] > 0 and (next_obj_mask[b, i] > 0).any():
                        # 有障碍物观测
                        next_state_i = (next_self_obs[b, i].cpu().numpy(), next_obj_obs[b, i].cpu().numpy(), next_obj_mask[b, i].cpu().numpy())
                    else:
                        # 无障碍物观测
                        next_state_i = (next_self_obs[b, i].cpu().numpy(), None, next_obj_mask[b, i].cpu().numpy())
                    next_states.append(next_state_i)
            
            # 使用TensorReplayBuffer进行批量GPU经验存储（零CPU传输）
            if self.rl_agent.training:
                # 展平B和R维度为(B*R, ...)形式统一处理
                flat_self_obs = self_obs.view(B * R, -1)  # [B*R, 7]
                flat_obj_obs = obj_obs.view(B * R, obj_obs.shape[2], -1) if obj_obs.shape[2] > 0 else torch.empty((B * R, 0, 5), device=self.train_env.device)  # [B*R, K, 5]
                flat_obj_mask = obj_mask.view(B * R, -1) if obj_mask.shape[2] > 0 else torch.empty((B * R, 0), device=self.train_env.device)  # [B*R, K]
                flat_actions = actions_t.view(B * R, -1)  # [B*R, action_dim]
                flat_rewards = rewards.view(B * R)  # [B*R]
                flat_next_self_obs = next_self_obs.view(B * R, -1)  # [B*R, 7]
                flat_next_obj_obs = next_obj_obs.view(B * R, next_obj_obs.shape[2], -1) if next_obj_obs.shape[2] > 0 else torch.empty((B * R, 0, 5), device=self.train_env.device)  # [B*R, K, 5]
                flat_next_obj_mask = next_obj_mask.view(B * R, -1) if next_obj_mask.shape[2] > 0 else torch.empty((B * R, 0), device=self.train_env.device)  # [B*R, K]
                flat_dones = dones.view(B * R)  # [B*R]
                flat_deactivated = self.train_env.deactivated_flags.view(B * R)  # [B*R]
                
                # 创建有效机器人掩码（排除失活机器人）
                valid_mask = ~flat_deactivated
                valid_indices = torch.where(valid_mask)[0]
                
                if len(valid_indices) > 0:
                    # 批量提取有效的经验数据（保持GPU张量）
                    batch_self_obs = flat_self_obs[valid_indices]      # [valid_count, 7]
                    batch_obj_obs = flat_obj_obs[valid_indices] if flat_obj_obs.shape[1] > 0 else torch.empty((len(valid_indices), 0, 5), device=self.train_env.device)
                    batch_obj_mask = flat_obj_mask[valid_indices] if flat_obj_mask.shape[1] > 0 else torch.empty((len(valid_indices), 0), device=self.train_env.device)
                    batch_actions = flat_actions[valid_indices]      # [valid_count, action_dim]
                    batch_rewards = flat_rewards[valid_indices]        # [valid_count]
                    batch_next_self_obs = flat_next_self_obs[valid_indices]
                    batch_next_obj_obs = flat_next_obj_obs[valid_indices] if flat_next_obj_obs.shape[1] > 0 else torch.empty((len(valid_indices), 0, 5), device=self.train_env.device)
                    batch_next_obj_mask = flat_next_obj_mask[valid_indices] if flat_next_obj_mask.shape[1] > 0 else torch.empty((len(valid_indices), 0), device=self.train_env.device)
                    batch_dones = flat_dones[valid_indices]            # [valid_count]
                    
                    # 批量添加到TensorReplayBuffer（GPU常驻）
                    # 为数据添加批次维度以匹配replay_buffer期望的[B, N, ...]格式
                    batch_self_obs_batch = batch_self_obs.unsqueeze(0)      # [1, valid_count, 7]
                    batch_obj_obs_batch = batch_obj_obs.unsqueeze(0)  # [1, valid_count, K, 5]
                    batch_obj_mask_batch = batch_obj_mask.unsqueeze(0)      # [1, valid_count, K]
                    batch_actions_batch = batch_actions.unsqueeze(0)      # [1, valid_count, action_dim]
                    batch_rewards_batch = batch_rewards.unsqueeze(0)      # [1, valid_count]
                    batch_next_self_obs_batch = batch_next_self_obs.unsqueeze(0)  # [1, valid_count, 7]
                    batch_next_obj_obs_batch = batch_next_obj_obs.unsqueeze(0)  # [1, valid_count, K, 5]
                    batch_next_obj_mask_batch = batch_next_obj_mask.unsqueeze(0)  # [1, valid_count, K]
                    batch_dones_batch = batch_dones.unsqueeze(0)            # [1, valid_count]
                    
                    self.rl_agent.memory.add_batch(
                        batch_self_obs_batch, batch_obj_obs_batch, batch_obj_mask_batch,
                        batch_actions_batch, batch_rewards_batch,
                        batch_next_self_obs_batch, batch_next_obj_obs_batch, batch_next_obj_mask_batch,
                        batch_dones_batch
                    )
            
            # 更新episode奖励（GPU张量计算，展平处理）
            flat_deactivated = self.train_env.deactivated_flags.view(B * R)
            flat_rewards = rewards.view(B * R)
            valid_rewards_mask = ~flat_deactivated
            ep_rewards[valid_rewards_mask.cpu().numpy()] += \
                (self.rl_agent.GAMMA ** ep_length) * flat_rewards[valid_rewards_mask].cpu().numpy()

            end_episode = (ep_length >= 1000) or bool(torch.all(self.train_env.deactivated_flags).item())
            
            # Learn and update
            if self.current_timestep >= self.learning_starts:
                if not self.rl_agent.training:
                    continue

                if self.current_timestep % self.UPDATE_EVERY == 0:
                    num_elements = self.rl_agent.memory.transitions.num_elements() if self.rl_agent.agent_type == "Rainbow" else self.rl_agent.memory.size
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
                    print("======== RL Episode Info ========" if not self.imitation else "======== IL Episode Info ========")
                    print("current ep_length: ", ep_length)
                    print("current ep_num: ", ep_num)
                    if not self.imitation:
                        print("current exploration rate: ", eps)
                    print("current timesteps: ", self.current_timestep)
                    print("total timesteps: ", total_timesteps)
                    print("======== Episode Info ========\n")
                    print("======== Robots Info ========")
                    for flat_idx in range(B * R):
                        b, i = flat_idx // R, flat_idx % R
                        info = infos[flat_idx]["state"]
                        if "deactivated" in info and ep_deactivated_t[flat_idx] >= 0:
                            print(f"Robot [B{b},R{i}] ep reward: {ep_rewards[flat_idx]:.2f}, {info} at step {ep_deactivated_t[flat_idx]}")
                        else:
                            print(f"Robot [B{b},R{i}] ep reward: {ep_rewards[flat_idx]:.2f}, {info}")
                    print("======== Robots Info ========\n")

                # 重置时使用GPU张量版观测
                self.train_env.reset()
                self_obs, obj_obs, obj_mask, _, _ = self.train_env.get_tensor_observation()
                
                # 重新组装states（展平B*R维度）
                states = []
                for b in range(B):
                    for i in range(R):
                        if obj_obs is not None and obj_obs.shape[2] > 0 and (obj_mask[b, i] > 0).any():
                            # 有障碍物观测
                            state_i = (self_obs[b, i].cpu().numpy(), obj_obs[b, i].cpu().numpy(), obj_mask[b, i].cpu().numpy())
                        else:
                            # 无障碍物观测
                            state_i = (self_obs[b, i].cpu().numpy(), None, obj_mask[b, i].cpu().numpy())
                        states.append(state_i)
                
                # 更新R（可能因schedule改变）
                R = self.train_env.R
                ep_rewards = np.zeros(B * R)
                ep_deactivated_t = [-1] * (B * R)
                ep_length = 0
            else:
                states = next_states
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
        
        # 获取评估环境的批次大小
        eval_B = self.eval_env.B
        
        for idx, config in enumerate(self.eval_config):
            print(f"Evaluating episode {idx}")
            
            # Reset eval env with config
            self.eval_env.R = config["num_robots"]
            self.eval_env.C = config["num_cores"]
            self.eval_env.O = config["num_obstacles"]
            self.eval_env.min_start_goal_dis = config["min_start_goal_dis"]
            
            state, _, _ = self.eval_env.reset()
            
            rob_num = self.eval_env.R
            # 为每个batch的每个robot准备统计数据
            rewards = [[0.0] * rob_num for _ in range(eval_B)]
            times = [[0.0] * rob_num for _ in range(eval_B)]
            energies = [[0.0] * rob_num for _ in range(eval_B)]
            relations = [[[] for _ in range(rob_num)] for _ in range(eval_B)]
            observations = [[[] for _ in range(rob_num)] for _ in range(eval_B)]
            actions_list = [[[] for _ in range(rob_num)] for _ in range(eval_B)]
            trajectories = [[[] for _ in range(rob_num)] for _ in range(eval_B)]
            
            end_episode = False
            length = 0
            
            while not end_episode:
                # 使用张量版观测（GPU常驻）
                self_obs, obj_obs, obj_mask, collisions, reach_goals = self.eval_env.get_tensor_observation()
                
                # 批量张量版动作选择
                is_continuous_action = self.rl_agent.agent_type in ["AC-IQN", "DDPG", "SAC"]
                
                if self.rl_agent.agent_type == "AC-IQN":
                    action_t = self.rl_agent.act_batch_tensor_ac_iqn(self_obs, obj_obs, obj_mask, eps=0.0, use_eval=True)
                elif self.rl_agent.agent_type == "IQN":
                    action_t = self.rl_agent.act_batch_tensor_iqn(self_obs, obj_obs, obj_mask, eps=0.0, use_eval=True).unsqueeze(-1)
                elif self.rl_agent.agent_type == "DDPG":
                    action_t = self.rl_agent.act_batch_tensor_ddpg(self_obs, obj_obs, obj_mask, eps=0.0, use_eval=True)
                elif self.rl_agent.agent_type == "DQN":
                    action_t = self.rl_agent.act_batch_tensor_dqn(self_obs, obj_obs, obj_mask, eps=0.0, use_eval=True).unsqueeze(-1)
                elif self.rl_agent.agent_type == "SAC":
                    action_t = self.rl_agent.act_batch_tensor_sac(self_obs, obj_obs, obj_mask, eps=0.0, use_eval=True)
                elif self.rl_agent.agent_type == "Rainbow":
                    action_t = self.rl_agent.act_batch_tensor_rainbow(self_obs, obj_obs, obj_mask, eps=0.0, use_eval=True).unsqueeze(-1)
                else:
                    raise RuntimeError("Agent type not implemented!")
                
                # 对失活机器人的动作置零
                deactivated_mask = self.eval_env.deactivated_flags
                if is_continuous_action:
                    action_t[deactivated_mask] = 0.0
                else:
                    action_t[deactivated_mask.unsqueeze(-1).expand_as(action_t)] = 12
                
                state, reward, done, info = self.eval_env.step(action_t, is_continuous_action)
                
                lt_step = self.eval_env.left_thrust.abs().detach().cpu().numpy()
                rt_step = self.eval_env.right_thrust.abs().detach().cpu().numpy()
                pos_step = self.eval_env.pos.detach().cpu().numpy()
                
                for b in range(eval_B):
                    for i in range(rob_num):
                        flat_idx = b * rob_num + i
                        if self.eval_env.deactivated_flags[b, i] and not done[flat_idx]:
                            continue
                        
                        rewards[b][i] += self.rl_agent.GAMMA ** length * reward[flat_idx]
                        times[b][i] += self.eval_env.dt * self.eval_env.N
                        
                        energies[b][i] += (lt_step[b, i] + rt_step[b, i]) * self.eval_env.dt
                        
                        trajectories[b][i].append(pos_step[b, i].tolist())
                        
                        # Store observations and actions
                        if state[flat_idx][0] is not None:
                            observations[b][i].append(copy.deepcopy(state[flat_idx]))
                        
                        # 提取单个机器人的动作
                        if is_continuous_action:
                            action_i = action_t[b, i, :].cpu().numpy().tolist()
                        else:
                            action_i = int(action_t[b, i, 0].item())
                        actions_list[b][i].append(copy.deepcopy(action_i))

                end_episode = (length >= 1000) or bool(torch.all(self.eval_env.deactivated_flags).item())
                length += 1

            # 对于评估，只记录第一个batch的结果（保持与单batch行为一致）
            success = bool(torch.all(self.eval_env.reach_goal_flags[0, :]).item())

            observations_data.append(observations[0])
            actions_data.append(actions_list[0])
            trajectories_data.append(trajectories[0])
            rewards_data.append(np.mean(rewards[0]))
            successes_data.append(success)
            times_data.append(np.mean(times[0]))
            energies_data.append(np.mean(energies[0]))
            relations_data.append(relations[0])
        
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
