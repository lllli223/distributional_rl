# Migration to Isaac Lab 2.3.0 and Gymnasium

This document describes the migration from the previous setup to Isaac Lab release/2.3.0 and Gymnasium.

## Summary of Changes

### 1. Gymnasium Migration (from gym)

All references to the `gym` library have been updated to `gymnasium`:

**Files Changed:**
- `train_RL_agents/marinenav_env/envs/marinenav_env.py`
- `train_RL_agents/env_visualizer.py`
- `train_RL_agents/marinenav_env/__init__.py`

**Changes Made:**
```python
# Before
import gym
from gym.envs.registration import register

# After
import gymnasium as gym
from gymnasium.envs.registration import register
```

The alias `as gym` ensures backward compatibility with existing code that references `gym.Env` and other gym APIs.

### 2. Isaac Lab 2.3.0 Compatibility

Updated the Isaac Lab integration to be compatible with release/2.3.0:

**File Changed:**
- `train_RL_agents/isaaclab_env/isaac_task/marine_nav_task.py`

**Key Changes:**

1. **Import Structure**: Updated to use Isaac Lab 2.3.0 import conventions with fallback compatibility:
   ```python
   # Optional imports with fallbacks for different Isaac Lab versions
   try:
       from omni.isaac.lab.utils import configclass
   except ImportError:
       def configclass(cls):
           return cls
   ```

2. **Task Implementation**: 
   - The `MarineNavTask` class no longer requires inheritance from heavy base classes
   - Uses a custom lightweight implementation compatible with Isaac Lab 2.3.0 patterns
   - Maintains compatibility with the wrapper interface

3. **Configuration**: Uses the `@configclass` decorator from Isaac Lab 2.3.0 when available, with a fallback for environments without Isaac Lab installed.

4. **Scene Management**: Simplified `set_up_scene()` to be optional and not require USD scene management.

### 3. Documentation Updates

**File Changed:**
- `train_RL_agents/isaaclab_env/README.md`

**Updates:**
- Added version information section
- Updated documentation to reflect Isaac Lab 2.3.0 and Gymnasium
- Clarified the migration strategy

## Compatibility Notes

### Backward Compatibility

1. **Gymnasium as gym**: By importing `gymnasium as gym`, all existing code that uses `gym.Env` continues to work without modification.

2. **Isaac Lab Optional**: The Isaac Lab integration includes proper try-except blocks to handle cases where Isaac Lab is not installed or different versions are used.

3. **Training Code**: No changes required to existing training scripts, agents, or trainers. The wrapper layer handles all compatibility.

### Version Requirements

- **Python**: 3.8+
- **PyTorch**: 1.13+ with CUDA support (for GPU acceleration)
- **Gymnasium**: 0.26+ (replacing gym)
- **Isaac Lab**: release/2.3.0 (optional, for GPU vectorized environments)
- **Isaac Sim**: release/2.3.0 (optional, required if using Isaac Lab)

## Testing the Migration

### 1. Test Basic Environment

```bash
cd train_RL_agents
python -c "from marinenav_env.envs.marinenav_env import MarineNavEnv3; env = MarineNavEnv3(); env.reset(); print('Environment OK')"
```

### 2. Test GPU Environment (if Isaac Lab is installed)

```bash
cd train_RL_agents/isaaclab_env
python -c "from isaac_task.marine_nav_task import MarineNavTask, MarineNavTaskCfg; cfg = MarineNavTaskCfg(); cfg.num_envs = 64; task = MarineNavTask(cfg); print('Isaac Lab Task OK')"
```

### 3. Run Training Test

```bash
cd train_RL_agents
python train_RL_agents_gpu.py -C config/training_config.json --device cpu
```

## Migration Benefits

1. **Future-Proof**: Using the latest versions of Isaac Lab (2.3.0) and Gymnasium ensures compatibility with future reinforcement learning tools and libraries.

2. **Community Support**: Gymnasium is actively maintained and is the successor to gym, ensuring better long-term support.

3. **Performance**: Isaac Lab 2.3.0 includes performance improvements and bug fixes from previous versions.

4. **Flexibility**: The optional Isaac Lab imports mean the code works even without Isaac Lab installed, making it easier to deploy in different environments.

## Troubleshooting

### Issue: ImportError for gymnasium

**Solution**: Install gymnasium:
```bash
pip install gymnasium
```

### Issue: Isaac Lab imports fail

**Solution**: This is expected if Isaac Lab is not installed. The code includes fallbacks and will work without it for the CPU environment. For GPU acceleration, install Isaac Lab 2.3.0 following the official documentation.

### Issue: gym compatibility issues

**Solution**: The code uses `import gymnasium as gym` to maintain compatibility. If you have code that explicitly checks for `gym` module, you may need to update it to check for `gymnasium` instead.

## Reference Documentation

- Isaac Lab 2.3.0: https://isaac-sim.github.io/IsaacLab/release/2.3.0/
- Gymnasium: https://gymnasium.farama.org/
- Migration Guide (gym to gymnasium): https://gymnasium.farama.org/content/migration-guide/

## Contact

For issues related to this migration, please refer to the project's issue tracker or documentation.
