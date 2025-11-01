# GPU 并行化海洋导航环境

本项目已将原有的marinenav_env环境升级为基于Isaac Lab的GPU并行化环境，实现了显著的训练性能提升。

## 性能提升

通过GPU并行化，原本的"1个Python环境×多机器人"的串行计算被升级为"上千个并行环境×多机器人"的纯Torch/CUDA张量计算，训练吞吐提升通常能达到一个数量级以上。

## 核心改进

### 1. 向量化计算
- **漩涡流速计算**: 用Torch张量替代scipy.spatial.KDTree和Python循环
- **船舶动力学**: 用广播矩阵运算替代逐机器人计算
- **感知观测**: 批量处理障碍物检测和COLREGs规则

### 2. Isaac Lab集成
- 使用Isaac Lab的RLTask框架
- 支持GPU VecEnv调度
- 保持与原有训练代码的兼容性

### 3. 最小侵入性改造
- 提供包装器适配原有接口
- 智能体和训练器无需修改
- 支持渐进式迁移

## 文件结构

```
train_RL_agents/isaaclab_env/
├── __init__.py
├── isaac_task/                          # Isaac Lab任务模块
│   ├── __init__.py
│   ├── field.py                         # 向量化漩涡流速计算
│   ├── dynamics.py                      # 向量化船舶动力学
│   ├── observation.py                   # 向量化感知和观测构建
│   ├── marine_nav_task.py               # MarineNavTask主任务类
│   ├── wrapper.py                       # 兼容性包装器
│   └── cfg/
│       └── marinenav_task.yaml          # 配置文件
├── run_isaac_marine_nav.py              # GPU并行化主运行脚本
├── test_gpu_env.py                      # 测试脚本
└── README.md                            # 说明文档
```

## 使用方法

### 1. 快速测试

```bash
# 进入项目目录
cd train_RL_agents/isaaclab_env

# 运行测试脚本
python test_gpu_env.py

# 运行性能基准测试
python run_isaac_marine_nav.py --benchmark
```

### 2. 独立训练

```bash
# 使用默认配置训练
python run_isaac_marine_nav.py

# 自定义参数
python run_isaac_marine_nav.py \
    --num_envs 512 \
    --num_robots 6 \
    --total_timesteps 1000000 \
    --device cuda:0
```

### 3. 集成到现有训练流程

```bash
# 使用新的GPU训练脚本
python train_RL_agents_gpu.py \
    -C config/training_config.json \
    --use-isaac-lab \
    --num_envs 256 \
    --device cuda:0
```

## 配置参数

### 环境参数
- `num_envs`: 并行环境数量 (默认: 1024)
- `num_robots`: 每个环境的机器人数量 (默认: 6)
- `num_cores`: 漩涡核心数量 (默认: 8)
- `num_obstacles`: 障碍物数量 (默认: 8)
- `max_obj_num`: 最大检测目标数 (默认: 5)

### 性能参数
- `map_width/map_height`: 地图尺寸 (默认: 55×55)
- `core_r`: 漩涡核心半径 (默认: 0.5)
- `detect_range`: 感知范围 (默认: 20.0)
- `detect_angle`: 感知角度 (默认: 2π)

### 训练参数
- `max_episode_len`: 最大episode长度 (默认: 1000)
- `timestep_penalty`: 时间步惩罚 (默认: -0.1)
- `goal_bonus`: 到达目标奖励 (默认: 10.0)

## 性能基准

在RTX 4090上的参考性能：

| 环境规模 | 总机器人 | 步/秒 | 步/秒/智能体 |
|---------|---------|-------|-------------|
| 64×6    | 384     | ~50   | ~0.13       |
| 256×6   | 1536    | ~25   | ~0.016      |
| 512×6   | 3072    | ~12   | ~0.004      |
| 1024×6  | 6144    | ~6    | ~0.001      |

*注：具体性能取决于硬件配置和算法复杂度*

## 兼容性

### 完全兼容
- 智能体算法 (AC-IQN, IQN, SAC)
- 训练配置和超参数
- 评估和可视化流程

### 部分兼容
- episode数据保存 (需要适配)
- 轨迹可视化 (需要重新实现)

### 不兼容
- 特定于原有环境的调试工具
- 需要精确Python随机数控制的场景

## 迁移指南

### 阶段1: 功能验证
1. 使用小规模环境测试 (`num_envs=64`)
2. 验证观测和动作兼容性
3. 比较奖励和性能指标

### 阶段2: 性能优化
1. 逐步增加并行环境数量
2. 调整批处理大小和学习率
3. 优化内存使用

### 阶段3: 生产部署
1. 使用大规模配置
2. 集成完整的评估流程
3. 部署到生产环境

## 故障排除

### 常见问题

1. **CUDA内存不足**
   - 减少 `num_envs` 或 `max_obj_num`
   - 使用更小的批处理大小

2. **性能不如预期**
   - 检查CUDA版本和驱动
   - 确保GPU未被其他进程占用
   - 调整环境配置参数

3. **兼容性错误**
   - 检查Isaac Lab版本
   - 验证PyTorch CUDA支持
   - 查看错误日志详细信息

### 调试工具

```bash
# 详细测试
python test_gpu_env.py

# 内存使用检查
python -c "import torch; print(torch.cuda.memory_summary())"

# 性能分析
python run_isaac_marine_nav.py --benchmark
```

## 贡献

欢迎提交问题报告和功能请求。在提交PR之前，请确保：

1. 所有测试通过
2. 性能没有明显下降
3. 保持向后兼容性

## 许可证

本项目遵循与原项目相同的许可证条款。
