# TorchRL GPU-Accelerated Training Pipeline

This directory contains a refactored training pipeline using TorchRL's `EnvBase`, `Collector`, and `ReplayBuffer` for GPU-accelerated sampling and replay buffer management.

## Architecture Overview

### Components

1. **MarineNavTorchRLEnv** (`marinenav_env/envs/torchrl_gpu/marinenav_torchrl_env.py`)
   - TorchRL-compatible wrapper for `MarineNavEnv3`
   - Converts multi-robot observations to batched `TensorDict` format
   - Provides action, observation, reward, and done specifications
   - Maintains compatibility with the original environment's logic

2. **PolicyWrapper** (`policy_wrapper.py`)
   - Wraps existing `Agent` implementations (AC-IQN, IQN, DDPG, DQN, SAC, Rainbow)
   - Converts TensorDict observations to the format expected by existing agents
   - Handles epsilon-greedy exploration and training/evaluation modes
   - Ensures action format consistency

3. **TorchRLTrainer** (`torchrl_trainer.py`)
   - Replaces the original `Trainer` with TorchRL-based data collection
   - Uses TorchRL's `TensorDictReplayBuffer` with `LazyTensorStorage` for GPU replay
   - Maintains the same training loop structure as the original for reproducibility
   - Preserves evaluation workflow and metrics

4. **train_torchrl_gpu.py**
   - Entry point for training with the TorchRL pipeline
   - Compatible with the same config files as `train_RL_agents.py`
   - Supports all agent types: AC-IQN, IQN, DDPG, DQN, SAC, Rainbow

## Key Features

### GPU Acceleration
- TensorDict operations on GPU for efficient data transfer
- Replay buffer stored on GPU (configurable via `device` parameter)
- Reduced CPU-GPU data movement during sampling

### Reproducibility
- Maintains exact training loop logic from original implementation
- Same epsilon schedule, update frequency, and target network updates
- Compatible evaluation metrics and logging
- Preserves random seed behavior

### Backward Compatibility
- Works with existing config files
- Reuses existing Agent implementations
- Produces identical evaluation outputs
- Original environment files remain untouched

## Usage

### Basic Training Command

```bash
python -m torchrl_gpu.train_torchrl_gpu \
    -C config/your_config.json \
    -D cuda
```

### Config File Format

Same format as the original training script:

```json
{
    "seed": 0,
    "total_timesteps": 1000000,
    "eval_freq": 50000,
    "imitation_learning": false,
    "agent_type": "AC-IQN",
    "save_dir": "./logs",
    "training_schedule": {
        "timesteps": [0, 500000],
        "num_robots": [2, 4],
        "num_cores": [4, 8],
        "num_obstacles": [4, 8],
        "min_start_goal_dis": [20.0, 30.0]
    },
    "eval_schedule": {
        "num_episodes": [10, 10],
        "num_robots": [2, 4],
        "num_cores": [4, 8],
        "num_obstacles": [4, 8],
        "min_start_goal_dis": [20.0, 30.0]
    }
}
```

## Differences from Original Pipeline

### What Changed
- **Data Collection**: Uses TorchRL's TensorDict for batched operations
- **Replay Buffer**: Uses `TensorDictReplayBuffer` with GPU storage
- **Observation Format**: Internally uses TensorDict, converted to original format for agents

### What Stayed the Same
- Environment simulation logic (exact same `MarineNavEnv3`)
- Agent networks and training algorithms
- Epsilon-greedy exploration schedule
- Update frequency and target network synchronization
- Evaluation procedure and metrics
- Random seed handling

## Performance Considerations

### Memory
- GPU replay buffer can use significant VRAM for large buffer sizes
- Adjust `buffer_size` parameter if encountering OOM errors
- Consider using CPU device for replay buffer if needed

### Throughput
- GPU operations reduce data transfer overhead
- Most benefit seen with continuous action spaces (AC-IQN, DDPG, SAC)
- Discrete action spaces (IQN, DQN, Rainbow) also benefit from batched operations

## Future Extensions

The modular structure allows for easy extensions:

1. **Vectorized Environments**: Add multiple parallel environments
2. **Multi-GPU Training**: Distribute replay buffer and policy networks
3. **Prioritized Experience Replay**: Use TorchRL's PER implementations
4. **Advanced Collectors**: Integrate multi-step returns or off-policy corrections

## Troubleshooting

### Import Errors
Ensure TorchRL is installed:
```bash
pip install torchrl tensordict
```

### CUDA Out of Memory
- Reduce `buffer_size` in trainer initialization
- Use `device="cpu"` for replay buffer
- Reduce batch size

### Action Shape Mismatch
- Check that agent type matches config file
- Verify action space dimensions in environment wrapper

## Directory Structure

```
torchrl_gpu/
├── __init__.py
├── README.md
├── train_torchrl_gpu.py       # Training entry point
├── torchrl_trainer.py          # TorchRL-based trainer
└── policy_wrapper.py           # Agent wrapper for TorchRL

marinenav_env/envs/torchrl_gpu/
├── __init__.py
├── marinenav_torchrl_env.py   # TorchRL environment wrapper
└── field_ops.py               # GPU-vectorized field operations (future)
```

## Citation

If you use this TorchRL pipeline in your research, please cite both the original repository and TorchRL:

```
@article{torchrl2023,
  title={TorchRL: A Library for PyTorch Reinforcement Learning},
  author={Bou, Albert and others},
  journal={arXiv preprint arXiv:2306.00577},
  year={2023}
}
```
