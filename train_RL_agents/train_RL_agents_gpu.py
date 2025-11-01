import os
import argparse
import itertools
from multiprocessing import Pool
import json
from datetime import datetime
import numpy as np
import torch

# 原有环境
from marinenav_env.envs.marinenav_env import MarineNavEnv3

# 新增GPU并行化环境
try:
    from isaaclab_env.isaac_task.wrapper import IsaacMarineNavVecWrapper
    ISAAC_LAB_AVAILABLE = True
except ImportError:
    print("警告: Isaac Lab 环境不可用，将使用原有环境")
    ISAAC_LAB_AVAILABLE = False

from policy.agent import Agent
from policy.trainer import Trainer

parser = argparse.ArgumentParser(description="训练IQN模型（支持GPU并行化）")

parser.add_argument(
    "-C",
    "--config-file",
    dest="config_file",
    type=open,
    required=True,
    help="configuration file for training parameters",
)
parser.add_argument(
    "-P",
    "--num-procs",
    dest="num_procs",
    type=int,
    default=1,
    help="number of subprocess workers to use for trial parallelization",
)
parser.add_argument(
    "-D",
    "--device",
    dest="device",
    type=str,
    default="cpu",
    help="device to run all subprocesses, could only specify 1 device in each run"
)
parser.add_argument(
    "--use-isaac-lab",
    action="store_true",
    help="使用Isaac Lab GPU并行化环境（实验性功能）"
)
parser.add_argument(
    "--num-envs",
    type=int,
    default=256,
    help="Isaac Lab环境的并行环境数量"
)
# 评估保持原CPU环境，不需要并行env数


def product(*args, repeat=1):
    # This function is a modified version of 
    # https://docs.python.org/3/library/itertools.html#itertools.product
    pools = [tuple(pool) for pool in args] * repeat
    result = [[]]
    for pool in pools:
        result = [x+[y] for x in result for y in pool]
    for prod in result:
        yield tuple(prod)

def trial_params(params):
    if isinstance(params,(str,int,float)):
        return [params]
    elif isinstance(params,list):
        return params
    elif isinstance(params, dict):
        keys, vals = zip(*params.items())
        mix_vals = []
        for val in vals:
            val = trial_params(val)
            mix_vals.append(val)
        return [dict(zip(keys, mix_val)) for mix_val in itertools.product(*mix_vals)]
    else:
        raise TypeError("Parameter type is incorrect.")

def params_dashboard(params):
    print("\n====== Training Setup ======\n")
    print("seed: ",params["seed"])
    print("total_timesteps: ",params["total_timesteps"])
    print("eval_freq: ",params["eval_freq"])
    print("imitation_learning: ",params["imitation_learning"])
    print("agent_type: ",params["agent_type"])
    if params.get('use_isaac_lab', False):
        print("使用Isaac Lab GPU并行化环境: 是")
        print(f"训练环境数量: {params.get('num_envs', 256)}")
        print("评估环境: 原CPU环境")
    else:
        print("使用原有串行环境: 是")
    print("\n")


def create_isaac_lab_train_env(params, device="cuda:0"):
    """仅创建训练用Isaac Lab环境；评估沿用原环境以保持Trainer兼容。"""
    if not ISAAC_LAB_AVAILABLE:
        raise ImportError("Isaac Lab环境不可用")
    from isaaclab_env.isaac_task.marine_nav_task import MarineNavTask, MarineNavTaskCfg
    cfg = MarineNavTaskCfg()
    cfg.num_envs = params.get('num_envs', 256)
    cfg.num_robots = params.get('num_robots', 6)
    cfg.num_cores = params.get('num_cores', 8)
    cfg.num_obstacles = params.get('num_obstacles', 8)
    cfg.max_obj_num = params.get('max_obj_num', 5)
    train_task = MarineNavTask(cfg, device=device)
    train_env = IsaacMarineNavVecWrapper(train_task, device=device)
    return train_env


def run_trial(device, params):
    exp_dir = os.path.join(params["save_dir"],
                           "training_"+params["training_time"],
                           "seed_"+str(params["seed"]))
    os.makedirs(exp_dir)

    param_file = os.path.join(exp_dir,"trial_config.json")
    with open(param_file, 'w+') as outfile:
        json.dump(params, outfile)
    
    # 创建环境
    if params.get('use_isaac_lab', False) and ISAAC_LAB_AVAILABLE:
        print("使用Isaac Lab GPU并行化训练环境 + 原CPU评估环境")
        train_env = create_isaac_lab_train_env(params, device)
        eval_env = MarineNavEnv3(seed=253, is_eval_env=True)
    else:
        print("使用原有串行环境")
        train_env = MarineNavEnv3(seed=params["seed"], schedule=params.get("training_schedule"))
        eval_env = MarineNavEnv3(seed=253, is_eval_env=True)

    # 创建RL agent
    rl_agent = Agent(device=device, seed=params["seed"]+100, agent_type=params["agent_type"]) 

    if "load_model" in params:
        rl_agent.load_model(params["load_model"], device)

    il_agent = None
    
    trainer = Trainer(train_env=train_env,
                    eval_env=eval_env,
                    eval_schedule=params["eval_schedule"],
                    rl_agent=rl_agent,
                    imitation=params["imitation_learning"],
                    il_agent=il_agent
                    )
    
    trainer.save_eval_config(exp_dir)

    # 训练
    trainer.learn(total_timesteps=params["total_timesteps"],
                  eval_freq=params["eval_freq"],
                  eval_log_path=exp_dir)

if __name__ == "__main__":
    args = parser.parse_args()
    params = json.load(args.config_file)
    
    # 添加GPU并行化相关参数
    params['use_isaac_lab'] = args.use_isaac_lab and ISAAC_LAB_AVAILABLE
    if params['use_isaac_lab']:
        params['num_envs'] = args.num_envs
        
        # 检查CUDA
        if not torch.cuda.is_available():
            print("警告: CUDA不可用，GPU并行化需要CUDA支持")
            print("回退到CPU环境")
            params['use_isaac_lab'] = False
    
    params_dashboard(params)
    training_schedule = params.pop("training_schedule")
    eval_schedule = params.pop("eval_schedule")
    
    trial_param_list = trial_params(params)

    dt = datetime.now()
    timestamp = dt.strftime("%Y-%m-%d-%H-%M-%S")

    if args.num_procs == 1:
        for param in trial_param_list:
            param["training_time"]=timestamp
            param["training_schedule"]=training_schedule
            param["eval_schedule"]=eval_schedule
            
            run_trial(args.device, param)
    else:
        with Pool(processes=args.num_procs) as pool:
            for param in trial_param_list:
                param["training_time"]=timestamp
                param["training_schedule"]=training_schedule
                param["eval_schedule"]=eval_schedule

                pool.apply_async(run_trial,(args.device,param))
            
            pool.close()
            pool.join()

