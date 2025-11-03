# Quick Start Guide: TorchRL GPU Training Pipeline

This guide helps you quickly get started with the new TorchRL GPU-accelerated training pipeline.

## Prerequisites

```bash
pip install torch torchrl tensordict numpy scipy gymnasium
```

## Basic Usage

### 1. Prepare Your Config File

Use any existing config file from `train_RL_agents/config/`:

```bash
config/
├── ac_iqn.json
├── ddpg.json
├── dqn.json
├── iqn.json
├── rainbow.json
└── sac.json
```

### 2. Run Training

```bash
cd /path/to/project
python -m train_RL_agents.torchrl_gpu.train_torchrl_gpu \
    -C train_RL_agents/config/ac_iqn.json \
    -D cuda
```

Or for CPU:
```bash
python -m train_RL_agents.torchrl_gpu.train_torchrl_gpu \
    -C train_RL_agents/config/ac_iqn.json \
    -D cpu
```

## Example Config

Create `config/quickstart.json`:

```json
{
    "seed": [0],
    "total_timesteps": 100000,
    "eval_freq": 10000,
    "save_dir": "./training_data",
    "training_schedule": {
        "timesteps": [0, 50000],
        "num_robots": [2, 3],
        "num_cores": [0, 0],
        "num_obstacles": [0, 2],
        "min_start_goal_dis": [25.0, 30.0]
    },
    "eval_schedule": {
        "num_episodes": [5, 5],
        "num_robots": [2, 3],
        "num_cores": [0, 0],
        "num_obstacles": [0, 2],
        "min_start_goal_dis": [25.0, 30.0]
    },
    "imitation_learning": false,
    "agent_type": "AC-IQN"
}
```

Then run:
```bash
python -m train_RL_agents.torchrl_gpu.train_torchrl_gpu \
    -C config/quickstart.json \
    -D cuda
```

## Output Structure

Training will create:
```
training_data/
└── training_YYYY-MM-DD-HH-MM-SS/
    └── seed_0/
        ├── trial_config.json      # Training parameters
        ├── eval_configs.json      # Evaluation scenarios
        ├── evaluations.npz        # Evaluation metrics
        ├── latest_actor.pth       # Latest actor weights (AC-IQN/DDPG/SAC)
        └── latest_critic.pth      # Latest critic weights (AC-IQN/DDPG/SAC)
        # or
        └── latest.pth             # Latest model (IQN/DQN/Rainbow)
```

## Monitoring Progress

During training, you'll see output like:

```
======== Training Setup (TorchRL GPU) ======

seed:  [0]
total_timesteps:  100000
eval_freq:  10000
imitation_learning:  False
agent_type:  AC-IQN


======== training schedule ========
num of robots:  2
num of cores:  0
num of obstacles:  0
min start goal dis:  25.0
======== training schedule ========

======== RL Episode Info ========
current ep_length:  247
current ep_num:  1
current exploration rate:  0.588
current timesteps:  247
total timesteps:  100000
======== Episode Info ========

======== Robots Info ========
Robot 0 ep reward: 4.23, deactivated after reaching goal at step 234
Robot 1 ep reward: 3.87, deactivated after reaching goal at step 247
======== Robots Info ========

...

Evaluating episode 0
Evaluating episode 1
...
++++++++ Evaluation Info ++++++++
Avg cumulative reward: 4.56
Success rate: 0.80
Avg time: 23.45
Avg energy: 1234.56
++++++++ Evaluation Info ++++++++
```

## Supported Agent Types

All original agent types are supported:

- **AC-IQN**: Actor-Critic with Implicit Quantile Networks (continuous actions)
- **IQN**: Implicit Quantile Networks (discrete actions)
- **DDPG**: Deep Deterministic Policy Gradient (continuous actions)
- **DQN**: Deep Q-Network (discrete actions)
- **SAC**: Soft Actor-Critic (continuous actions)
- **Rainbow**: Rainbow DQN (discrete actions)

## GPU vs CPU

### GPU Training
```bash
python -m train_RL_agents.torchrl_gpu.train_torchrl_gpu -C config/ac_iqn.json -D cuda
```

Benefits:
- Faster tensor operations
- Batched observation processing
- Reduced data transfer overhead

### CPU Training
```bash
python -m train_RL_agents.torchrl_gpu.train_torchrl_gpu -C config/ac_iqn.json -D cpu
```

Use when:
- No GPU available
- Debugging
- Small-scale experiments

## Comparison with Original Pipeline

| Feature | Original | TorchRL GPU |
|---------|----------|-------------|
| Command | `train_RL_agents.py` | `torchrl_gpu.train_torchrl_gpu` |
| Environment | MarineNavEnv3 | MarineNavTorchRLEnv (wrapper) |
| Data Format | Tuples | TensorDict |
| GPU Support | Partial | Full |
| Config Format | Identical | Identical |
| Results | Identical | Identical |

## Troubleshooting

### ImportError: No module named 'torchrl'
```bash
pip install torchrl tensordict
```

### CUDA Out of Memory
- Reduce batch size in agent parameters
- Use smaller replay buffer
- Switch to CPU: `-D cpu`

### Different Results
- Ensure same random seed
- Check device consistency (all CPU or all GPU)
- Verify config file is identical

## Next Steps

1. **Run a short experiment** to verify setup:
   ```bash
   python -m train_RL_agents.torchrl_gpu.train_torchrl_gpu \
       -C config/quickstart.json -D cuda
   ```

2. **Monitor GPU usage**:
   ```bash
   watch -n 1 nvidia-smi
   ```

3. **Compare with original pipeline**:
   ```bash
   # Original
   python train_RL_agents.py -C config/quickstart.json -D cuda
   
   # TorchRL
   python -m train_RL_agents.torchrl_gpu.train_torchrl_gpu \
       -C config/quickstart.json -D cuda
   ```

4. **Analyze results**:
   ```python
   import numpy as np
   
   data = np.load('training_data/.../seed_0/evaluations.npz', allow_pickle=True)
   rewards = data['rewards']
   successes = data['successes']
   ```

## Documentation

- **README.md**: Overview and architecture
- **COMPARISON.md**: Detailed comparison with original pipeline
- **QUICKSTART.md**: This guide

## Support

For issues specific to:
- **Environment simulation**: See original `marinenav_env/`
- **Agent algorithms**: See `policy/`
- **TorchRL wrapper**: See `torchrl_gpu/`
