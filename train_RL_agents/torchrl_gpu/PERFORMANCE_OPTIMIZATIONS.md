# GPU Performance Optimizations

## Overview

This document describes the performance optimizations made to eliminate GPU→CPU→GPU round-trip transfers during training, which were causing significant performance bottlenecks.

## Problem Description

The original implementation had several performance issues:

1. **PolicyWrapper.forward()** (lines 31-37): Looped through each sample in the batch, performing `.cpu().numpy()` on GPU tensors for each robot individually, building Python lists, and calling agent methods one at a time. This completely broke GPU batching and caused numerous CPU transfers.

2. **TorchRLTrainer.learn()** (experience insertion): Multiple `.cpu().numpy()` calls for each robot during training, with per-robot processing instead of batched operations.

3. **Environment step**: While the environment simulation remains on CPU (as designed), the data transfers were not optimized, causing additional overhead.

## Implemented Solutions

### 1. PolicyWrapper Batched GPU Inference

**File**: `train_RL_agents/torchrl_gpu/policy_wrapper.py`

**Changes**:
- Removed per-sample loop with individual `.cpu().numpy()` calls
- Implemented `_forward_continuous_batched()` for continuous action spaces (AC-IQN, DDPG, SAC)
- Implemented `_forward_discrete_batched()` for discrete action spaces (IQN, DQN, Rainbow)
- All operations now stay on GPU throughout the forward pass
- Epsilon-greedy exploration is now vectorized on GPU
- Action bounds are cached as buffers to avoid repeated tensor creation

**Benefits**:
- Eliminates N CPU transfers per batch (where N = batch_size)
- Fully utilizes GPU parallelism for action selection
- Reduces Python loop overhead
- Maintains compatibility with all agent types

### 2. TorchRLTrainer Experience Collection

**File**: `train_RL_agents/torchrl_gpu/torchrl_trainer.py`

**Changes** (lines 105-146):
- Batched all CPU transfers at once instead of per-robot
- Converted tensors to NumPy only once per timestep
- Optimized mask-based filtering using NumPy boolean indexing
- Only performs CPU transfers when actually training

**Benefits**:
- Reduces number of GPU→CPU synchronization points
- Minimizes memory copies
- Better cache locality for CPU operations

### 3. Code Quality Improvements

- Removed unused `numpy` import from `policy_wrapper.py`
- Added proper type handling for tensor devices and dtypes
- Improved code organization with separate methods for continuous/discrete actions

## Performance Impact

The optimizations provide the following improvements:

1. **Reduced latency**: Batch operations on GPU are much faster than sequential CPU operations
2. **Better GPU utilization**: Network forward passes now use full batch size
3. **Fewer synchronization points**: CPU-GPU transfers only occur when absolutely necessary
4. **Scalability**: Performance improvement scales with batch size

## Compatibility

All changes maintain full backward compatibility:
- Same API for PolicyWrapper
- Same training workflow in TorchRLTrainer
- Compatible with all agent types: AC-IQN, IQN, DDPG, DQN, SAC, Rainbow
- No changes to model architectures or training algorithms

## Future Optimization Opportunities

1. **GPU Environment Simulation**: The environment step still runs on CPU. Using Isaac Lab's GPU-based physics simulation could eliminate this bottleneck entirely.

2. **Replay Buffer on GPU**: Keep experience replay buffer on GPU to avoid additional transfers during sampling.

3. **Mixed Precision Training**: Use torch.amp for automatic mixed precision to further improve performance.

4. **Distributed Training**: Scale across multiple GPUs with DDP for larger batch sizes.

## Testing

The optimizations have been verified to:
- Compile without syntax errors
- Maintain correct tensor shapes and types
- Work with both continuous and discrete action spaces
- Handle epsilon-greedy exploration correctly
- Support both training and evaluation modes

## References

- Original issue: GPU→CPU→GPU roundtrips in action selection
- Related files:
  - `train_RL_agents/torchrl_gpu/policy_wrapper.py`
  - `train_RL_agents/torchrl_gpu/torchrl_trainer.py`
  - `train_RL_agents/marinenav_env/envs/torchrl_gpu/marinenav_torchrl_env.py`
