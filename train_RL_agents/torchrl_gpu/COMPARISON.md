# Comparison: Original vs TorchRL GPU Pipeline

This document compares the original training pipeline with the new TorchRL GPU-accelerated pipeline to ensure reproducibility and highlight improvements.

## Architecture Comparison

### Original Pipeline (train_RL_agents.py)
```
MarineNavEnv3 → Agent → Trainer → ReplayBuffer (CPU) → Train
```

### TorchRL Pipeline (train_torchrl_gpu.py)
```
MarineNavTorchRLEnv (wrapper) → PolicyWrapper → TorchRLTrainer → ReplayBuffer (CPU/GPU) → Train
                ↓
         MarineNavEnv3 (unchanged)
```

## Key Components Comparison

| Component | Original | TorchRL GPU | Status |
|-----------|----------|-------------|--------|
| Environment Simulation | MarineNavEnv3 (CPU) | MarineNavEnv3 (CPU, wrapped) | ✅ Identical |
| Observation Format | Tuple (self_state, objects) | TensorDict → Tuple | ✅ Identical |
| Action Selection | Agent.act_*() | PolicyWrapper → Agent.act_*() | ✅ Identical |
| Replay Buffer | ReplayBuffer (CPU) | Agent's ReplayBuffer (CPU) | ✅ Identical |
| Training Loop | Trainer.learn() | TorchRLTrainer.learn() | ✅ Identical |
| Epsilon Schedule | Linear decay | Linear decay | ✅ Identical |
| Target Network | Soft update | Soft update | ✅ Identical |
| Evaluation | Trainer.evaluation() | TorchRLTrainer.evaluation() | ✅ Identical |

## Data Flow Comparison

### Original Pipeline

1. **Reset Environment**
   ```python
   states, _, _ = env.reset()  # Returns list of tuples
   ```

2. **Action Selection**
   ```python
   for i, state in enumerate(states):
       action = agent.act_iqn(state, eps)
       actions.append(action)
   ```

3. **Environment Step**
   ```python
   next_states, rewards, dones, infos = env.step(actions)
   ```

4. **Store in Replay Buffer**
   ```python
   agent.memory.add((state, action, reward, next_state, done))
   ```

### TorchRL GPU Pipeline

1. **Reset Environment**
   ```python
   td = env.reset()  # Returns TensorDict on GPU
   # td = {
   #     "self_state": [N, 7],
   #     "objects_state": [N, 5, 5],
   #     "objects_mask": [N, 5]
   # }
   ```

2. **Action Selection**
   ```python
   td = policy_wrapper(td)  # Converts to original format internally
   # Calls agent.act_iqn() for each robot
   # Returns TensorDict with actions
   ```

3. **Environment Step**
   ```python
   td_next = env.step(td)  # Returns TensorDict with rewards, dones
   ```

4. **Store in Replay Buffer**
   ```python
   # Converts back to original format
   agent.memory.add((state, action, reward, next_state, done))
   ```

## Reproducibility Guarantees

### What Ensures Identical Results

1. **Same Environment Logic**
   - `MarineNavTorchRLEnv` is a wrapper around `MarineNavEnv3`
   - All simulation logic (dynamics, collisions, rewards) unchanged
   - Same random seed handling

2. **Same Agent Behavior**
   - No modifications to Agent networks or training algorithms
   - Epsilon-greedy exploration schedule identical
   - Replay buffer sampling identical

3. **Same Training Schedule**
   - Curriculum learning schedule preserved
   - Update frequency unchanged
   - Target network sync frequency unchanged

4. **Same Evaluation Protocol**
   - Uses same eval configs
   - Same metrics computed
   - Same logging format

### Critical Code Paths

#### Episode Reset
**Original:**
```python
states, _, _ = self.train_env.reset()
```

**TorchRL:**
```python
td = self.train_env.reset()  # Internally calls MarineNavEnv3.reset()
```

#### Action Execution
**Original:**
```python
action = self.rl_agent.act_iqn(states[i], eps, use_eval=False)
```

**TorchRL:**
```python
# In PolicyWrapper.forward()
action = self.agent.act_iqn(state, self.epsilon, use_eval=False)
```

#### Replay Buffer Storage
**Original:**
```python
self.rl_agent.memory.add((states[i], actions[i], rewards[i], next_states[i], dones[i]))
```

**TorchRL:**
```python
# Converts TensorDict back to original format
state = (self_state, obj_list)
next_state = (next_self_state, next_obj_list)
self.rl_agent.memory.add((state, action, reward, next_state, done))
```

## Performance Differences

### Computation Location

| Operation | Original | TorchRL GPU |
|-----------|----------|-------------|
| Environment Sim | CPU | CPU |
| Observation Building | CPU | CPU → GPU |
| Action Selection | CPU/GPU | GPU |
| Replay Buffer | CPU | CPU (can be GPU) |
| Network Forward | GPU | GPU |
| Network Training | GPU | GPU |

### Data Transfer

**Original:**
- numpy arrays → torch tensors (when needed)
- Individual robot observations processed separately

**TorchRL:**
- numpy arrays → torch tensors → TensorDict (batched)
- All robot observations batched together
- Reduced transfer overhead for multi-robot scenarios

### Memory Usage

**Original:**
- Replay buffer on CPU
- Individual transitions stored as tuples

**TorchRL:**
- Replay buffer on CPU (same as original)
- TensorDict intermediate representation on GPU
- Slightly higher GPU memory usage for batched operations

## Configuration Compatibility

### Config File Format
Both pipelines use identical JSON config format:

```json
{
    "seed": [0],
    "total_timesteps": 1000000,
    "eval_freq": 50000,
    "agent_type": "AC-IQN",
    "training_schedule": {...},
    "eval_schedule": {...}
}
```

### Command Line Usage

**Original:**
```bash
python train_RL_agents.py -C config/ac_iqn.json -D cuda
```

**TorchRL:**
```bash
python -m torchrl_gpu.train_torchrl_gpu -C config/ac_iqn.json -D cuda
```

## Migration Checklist

When switching from original to TorchRL pipeline:

- ✅ No config file changes needed
- ✅ No hyperparameter tuning required
- ✅ Same agent checkpoints work
- ✅ Same evaluation metrics
- ✅ Same logging format
- ✅ Same curriculum schedule

## Testing Reproducibility

To verify identical behavior:

1. **Train with same seed on both pipelines**
   ```bash
   python train_RL_agents.py -C config/test.json -D cpu
   python -m torchrl_gpu.train_torchrl_gpu -C config/test.json -D cpu
   ```

2. **Compare evaluation results**
   - Check `evaluations.npz` for identical rewards
   - Compare trajectories
   - Verify success rates

3. **Compare model checkpoints**
   - Network weights should be similar (minor floating point differences OK)
   - Loss curves should match

## Known Differences

### Acceptable Differences
- **Floating point precision**: Minor differences due to batched operations
- **GPU scheduling**: Non-deterministic CUDA operations may cause small variations
- **Memory layout**: TensorDict uses different memory layout than tuples

### Unacceptable Differences
- **Episode rewards**: Should match within 1%
- **Exploration schedule**: Must be identical
- **Update timing**: Must occur at same timesteps
- **Evaluation metrics**: Should match within measurement error

## Future Enhancements

The TorchRL pipeline enables future optimizations:

1. **Vectorized Environments**: Run multiple parallel environments
2. **GPU Replay Buffer**: Move entire replay buffer to GPU
3. **Distributed Training**: Multi-GPU and multi-node support
4. **Advanced Collectors**: Prioritized replay, n-step returns, etc.

All while maintaining backward compatibility with the original pipeline.
