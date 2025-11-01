#!/usr/bin/env python3
"""
GPU 并行化训练脚本 - 基于 Isaac Lab 的向量化环境

使用方法:
    python run_isaac_marine_nav.py --config cfg/marinenav_task.yaml --eval
    
训练参数:
    --num_envs: 并行环境数量 (默认: 1024)
    --num_robots: 每个环境的机器人数量 (默认: 6)
    --total_timesteps: 总训练步数 (默认: 1000000)
    --eval_freq: 评估频率 (默认: 50000)
    --device: 计算设备 (默认: cuda:0)
    --eval: 启用评估模式
"""

import argparse
import yaml
import torch
import numpy as np
import os
import sys

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from isaac_task.wrapper import IsaacMarineNavVecWrapper
from isaac_task.marine_nav_task import MarineNavTask, MarineNavTaskCfg


def load_config(config_path):
    """加载配置文件"""
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    return {}


def create_task_from_config(config, device="cuda:0"):
    """从配置文件创建任务"""
    cfg = MarineNavTaskCfg()
    
    # 覆盖默认配置
    cfg.num_envs = config.get('num_envs', 1024)
    cfg.num_robots = config.get('num_robots', 6)
    cfg.num_cores = config.get('num_cores', 8)
    cfg.num_obstacles = config.get('num_obstacles', 8)
    cfg.max_obj_num = config.get('max_obj_num', 5)
    
    cfg.map_width = config.get('map_width', 55.0)
    cfg.map_height = config.get('map_height', 55.0)
    cfg.core_r = config.get('core_r', 0.5)
    cfg.detect_range = config.get('detect_range', 20.0)
    cfg.detect_angle = config.get('detect_angle', 2 * np.pi)
    cfg.max_episode_len = config.get('max_episode_len', 1000)
    
    cfg.timestep_penalty = config.get('timestep_penalty', -0.1)
    cfg.colregs_penalty_scale = config.get('colregs_penalty_scale', 0.1)
    cfg.collision_penalty = config.get('collision_penalty', -5.0)
    cfg.goal_bonus = config.get('goal_bonus', 10.0)
    cfg.goal_distance_threshold = config.get('goal_distance_threshold', 2.0)
    
    cfg.dt = config.get('dt', 0.05)
    cfg.N = config.get('N', 10)
    
    # 创建任务
    task = MarineNavTask(cfg, device=device)
    return task


def create_environments(config, device="cuda:0"):
    """创建训练和评估环境"""
    # 训练环境
    train_config = config.copy()
    train_config['num_envs'] = config.get('num_envs', 1024)
    train_task = create_task_from_config(train_config, device)
    train_env = IsaacMarineNavVecWrapper(train_task, device=device)
    
    # 评估环境
    eval_config = config.copy()
    eval_config['num_envs'] = config.get('eval_num_envs', 64)
    eval_task = create_task_from_config(eval_config, device)
    eval_env = IsaacMarineNavVecWrapper(eval_task, device=device)
    
    return train_env, eval_env


def simple_train_loop(train_env, eval_env, total_timesteps=1000000, eval_freq=50000, device="cuda:0"):
    """简单的训练循环，用于测试GPU并行化"""
    
    print(f"开始训练...")
    print(f"训练环境: {train_env.E} 个环境 x {train_env.R} 机器人 = {train_env.E * train_env.R} 总机器人")
    print(f"评估环境: {eval_env.E} 个环境 x {eval_env.R} 机器人 = {eval_env.E * eval_env.R} 总机器人")
    
    # 重置环境
    observations = train_env.reset()
    print(f"观测形状: 自车状态 {observations[0][0].shape}, 目标观测 {observations[0][1].shape}")
    
    # 简单的随机策略用于测试
    action_dim = train_env.get_action_space_dimension()
    total_robots = train_env.get_num_robots()
    
    # 统计信息
    episode_rewards = []
    episode_lengths = []
    current_episode_reward = 0
    current_episode_length = 0
    
    for step in range(total_timesteps):
        # 生成随机动作
        actions = np.random.uniform(-1, 1, size=(total_robots, action_dim)).tolist()
        
        # 执行动作
        observations, rewards, dones, infos = train_env.step(actions)
        
        # 统计
        current_episode_reward += np.mean(rewards)
        current_episode_length += 1
        
        # 检查episode结束
        if any(dones):
            episode_rewards.append(current_episode_reward)
            episode_lengths.append(current_episode_length)
            
            if len(episode_rewards) % 10 == 0:
                avg_reward = np.mean(episode_rewards[-10:])
                avg_length = np.mean(episode_lengths[-10:])
                print(f"步骤 {step}: 平均奖励 = {avg_reward:.3f}, 平均长度 = {avg_length:.1f}")
            
            current_episode_reward = 0
            current_episode_length = 0
            
            # 重置环境
            observations = train_env.reset()
        
        # 定期评估
        if step > 0 and step % eval_freq == 0:
            print(f"\n=== 评估步骤 {step} ===")
            eval_rewards = evaluate_policy(eval_env, num_episodes=5, device=device)
            print(f"评估平均奖励: {np.mean(eval_rewards):.3f}")
            print("==================\n")


def evaluate_policy(eval_env, num_episodes=5, device="cuda:0"):
    """评估当前策略（随机策略）"""
    episode_rewards = []
    
    for episode in range(num_episodes):
        observations = eval_env.reset()
        total_reward = 0
        
        for step in range(1000):  # 最多1000步
            action_dim = eval_env.get_action_space_dimension()
            total_robots = eval_env.get_num_robots()
            
            # 随机动作
            actions = np.random.uniform(-1, 1, size=(total_robots, action_dim)).tolist()
            
            observations, rewards, dones, infos = eval_env.step(actions)
            total_reward += np.mean(rewards)
            
            if all(dones):
                break
        
        episode_rewards.append(total_reward)
        print(f"评估 episode {episode + 1}: 奖励 = {total_reward:.3f}")
    
    return episode_rewards


def benchmark_performance():
    """性能基准测试"""
    print("\n=== GPU 并行化性能基准测试 ===")
    
    # 测试不同环境数量
    env_configs = [
        (64, 6, "小规模测试"),
        (256, 6, "中规模测试"),
        (512, 6, "大规模测试"),
        (1024, 6, "最大规模测试")
    ]
    
    for num_envs, num_robots, desc in env_configs:
        print(f"\n测试: {desc} ({num_envs} 环境 x {num_robots} 机器人)")
        
        config = {
            'num_envs': num_envs,
            'num_robots': num_robots,
            'num_cores': 4,
            'num_obstacles': 4
        }
        
        # 创建环境
        train_env, eval_env = create_environments(config)
        
        # 性能测试
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)
        
        start_time.record()
        
        # 执行100步
        observations = train_env.reset()
        for step in range(100):
            action_dim = train_env.get_action_space_dimension()
            total_robots = train_env.get_num_robots()
            actions = np.random.uniform(-1, 1, size=(total_robots, action_dim)).tolist()
            observations, rewards, dones, infos = train_env.step(actions)
            
            if all(dones):
                observations = train_env.reset()
        
        end_time.record()
        torch.cuda.synchronize()
        
        elapsed_time = start_time.elapsed_time(end_time) / 1000.0  # 转换为秒
        steps_per_second = 100.0 / elapsed_time
        
        total_agents = num_envs * num_robots
        steps_per_second_per_agent = steps_per_second / total_agents
        
        print(f"  总机器人数量: {total_agents}")
        print(f"  100步耗时: {elapsed_time:.2f} 秒")
        print(f"  步/秒: {steps_per_second:.1f}")
        print(f"  步/秒/智能体: {steps_per_second_per_agent:.2f}")


def main():
    parser = argparse.ArgumentParser(description="GPU 并行化海洋导航训练")
    parser.add_argument("--config", type=str, default="cfg/marinenav_task.yaml",
                       help="配置文件路径")
    parser.add_argument("--num_envs", type=int, default=1024,
                       help="并行环境数量")
    parser.add_argument("--num_robots", type=int, default=6,
                       help="每个环境的机器人数量")
    parser.add_argument("--total_timesteps", type=int, default=100000,
                       help="总训练步数")
    parser.add_argument("--eval_freq", type=int, default=10000,
                       help="评估频率")
    parser.add_argument("--device", type=str, default="cuda:0",
                       help="计算设备")
    parser.add_argument("--eval", action="store_true",
                       help="启用评估模式")
    parser.add_argument("--benchmark", action="store_true",
                       help="运行性能基准测试")
    
    args = parser.parse_args()
    
    # 检查CUDA
    if not torch.cuda.is_available():
        print("警告: CUDA 不可用，使用CPU可能会很慢")
        args.device = "cpu"
    
    print(f"使用设备: {args.device}")
    
    # 加载配置
    config = load_config(args.config)
    
    # 覆盖配置参数
    config['num_envs'] = args.num_envs
    config['num_robots'] = args.num_robots
    
    if args.benchmark:
        benchmark_performance()
        return
    
    # 创建环境
    print("创建环境...")
    train_env, eval_env = create_environments(config, args.device)
    
    if args.eval:
        # 评估模式
        print("运行评估...")
        eval_rewards = evaluate_policy(eval_env, num_episodes=10, device=args.device)
        print(f"最终评估结果: 平均奖励 = {np.mean(eval_rewards):.3f}")
    else:
        # 训练模式
        simple_train_loop(
            train_env, eval_env,
            total_timesteps=args.total_timesteps,
            eval_freq=args.eval_freq,
            device=args.device
        )
    
    print("完成!")


if __name__ == "__main__":
    main()
