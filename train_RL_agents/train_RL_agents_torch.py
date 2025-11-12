import os
import argparse
import itertools
from multiprocessing import Pool
import json
from datetime import datetime
import numpy as np
import random
import torch
from marinenav_env.envs.torch_marinenav_env import TorchMarineNavEnv
from policy.agent import Agent
from policy.trainer_torch import TorchTrainer

parser = argparse.ArgumentParser(description="Train IQN model with GPU-accelerated environment")

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
    default="cuda",
    help="device to run all subprocesses (cuda or cpu)"
)

def product(*args, repeat=1):
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
    print("\n")

def run_trial(device,params):
    exp_dir = os.path.join(params["save_dir"],
                           "training_"+params["training_time"],
                           "seed_"+str(params["seed"]))
    os.makedirs(exp_dir)

    param_file = os.path.join(exp_dir,"trial_config.json")
    with open(param_file, 'w+') as outfile:
        json.dump(params, outfile)
    
    # Setup device
    if device == "cuda" and torch.cuda.is_available():
        torch_device = torch.device("cuda")
    else:
        torch_device = torch.device("cpu")
    np.random.seed(params["seed"])
    random.seed(params["seed"])
    torch.manual_seed(params["seed"])
    if torch_device.type == "cuda":
        torch.cuda.manual_seed_all(params["seed"])
    
    # Create GPU-accelerated training and evaluation environments
    # allow env batch size via config: env_batch_size and eval_env_batch_size (defaults 1)
    env_B = int(params.get("env_batch_size", 4))
    eval_env_B = int(params.get("eval_env_batch_size", 1))
    train_env = TorchMarineNavEnv(
        B=env_B,
        R=params["training_schedule"]["num_robots"][0],
        C=params["training_schedule"]["num_cores"][0],
        O=params["training_schedule"]["num_obstacles"][0],
        device=torch_device,
        seed=params["seed"],
        schedule=params["training_schedule"],
        is_eval_env=False
    )

    eval_env = TorchMarineNavEnv(
        B=eval_env_B,
        R=params["eval_schedule"]["num_robots"][0],
        C=params["eval_schedule"]["num_cores"][0],
        O=params["eval_schedule"]["num_obstacles"][0],
        device=torch_device,
        seed=253,
        schedule=None,
        is_eval_env=True
    )

    # Create RL agent
    rl_agent = Agent(device=device, seed=params["seed"]+100, agent_type=params["agent_type"])

    if "load_model" in params:
        rl_agent.load_model(params["load_model"], device)

    il_agent = None
    
    trainer = TorchTrainer(
        train_env=train_env,
        eval_env=eval_env,
        eval_schedule=params["eval_schedule"],
        rl_agent=rl_agent,
        imitation=params["imitation_learning"],
        il_agent=il_agent
    )
    
    trainer.save_eval_config(exp_dir)

    trainer.learn(
        total_timesteps=params["total_timesteps"],
        eval_freq=params["eval_freq"],
        eval_log_path=exp_dir
    )

if __name__ == "__main__":
    args = parser.parse_args()
    params = json.load(args.config_file)
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
            
            run_trial(args.device,param)
    else:
        with Pool(processes=args.num_procs) as pool:
            for param in trial_param_list:
                param["training_time"]=timestamp
                param["training_schedule"]=training_schedule
                param["eval_schedule"]=eval_schedule

                pool.apply_async(run_trial,(args.device,param))
            
            pool.close()
            pool.join()
