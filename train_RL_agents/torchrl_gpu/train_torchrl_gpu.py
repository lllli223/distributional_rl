import os
import argparse
import itertools
import json
from datetime import datetime
import torch

from marinenav_env.envs.torchrl_gpu.marinenav_torchrl_env import MarineNavTorchRLEnv
from policy.agent import Agent
from torchrl_gpu.torchrl_trainer import TorchRLTrainer


parser = argparse.ArgumentParser(description="Train IQN model with TorchRL GPU pipeline")

parser.add_argument(
    "-C",
    "--config-file",
    dest="config_file",
    type=open,
    required=True,
    help="configuration file for training parameters",
)
parser.add_argument(
    "-D",
    "--device",
    dest="device",
    type=str,
    default="cuda" if torch.cuda.is_available() else "cpu",
    help="device for training",
)


def product(*args, repeat=1):
    pools = [tuple(pool) for pool in args] * repeat
    result = [[]]
    for pool in pools:
        result = [x + [y] for x in result for y in pool]
    for prod in result:
        yield tuple(prod)


def trial_params(params):
    if isinstance(params, (str, int, float)):
        return [params]
    elif isinstance(params, list):
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
    print("\n====== Training Setup (TorchRL GPU) ======\n")
    print("seed: ", params["seed"])
    print("total_timesteps: ", params["total_timesteps"])
    print("eval_freq: ", params["eval_freq"])
    print("imitation_learning: ", params["imitation_learning"])
    print("agent_type: ", params["agent_type"])
    print("\n")


def run_trial(device, params):
    seed = params["seed"]

    exp_dir = os.path.join(
        params["save_dir"], "training_" + params["training_time"], "seed_" + str(seed)
    )
    os.makedirs(exp_dir, exist_ok=True)

    param_file = os.path.join(exp_dir, "trial_config.json")
    with open(param_file, "w+") as outfile:
        json.dump(params, outfile)

    # Optional explicit E parallelism
    env_batch_size = params.get("env_batch_size", 1)
    train_env = MarineNavTorchRLEnv(seed=seed, schedule=params["training_schedule"], device=device, env_batch_size=env_batch_size)
    eval_env = MarineNavTorchRLEnv(seed=253, is_eval_env=True, device=device, env_batch_size=1)

    rl_agent = Agent(device=device, seed=seed + 100, agent_type=params["agent_type"])

    if "load_model" in params:
        rl_agent.load_model(params["load_model"], device)

    trainer = TorchRLTrainer(
        train_env=train_env,
        eval_env=eval_env,
        eval_schedule=params["eval_schedule"],
        rl_agent=rl_agent,
        device=device,
        imitation=params["imitation_learning"],
        il_agent=None,
    )

    # Optional large-batch training controls
    if "train_batch_size_total" in params:
        trainer.train_batch_size_total = params["train_batch_size_total"]
    if "grad_accum_steps" in params:
        trainer.grad_accum_steps = params["grad_accum_steps"]
    if "amp_enabled" in params:
        trainer.amp_enabled = params["amp_enabled"]

    trainer.save_eval_config(exp_dir)

    trainer.learn(
        total_timesteps=params["total_timesteps"],
        eval_freq=params["eval_freq"],
        eval_log_path=exp_dir,
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

    for param in trial_param_list:
        param["training_time"] = timestamp
        param["training_schedule"] = training_schedule
        param["eval_schedule"] = eval_schedule

        run_trial(args.device, param)
