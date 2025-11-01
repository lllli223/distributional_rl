import json
import sys
sys.path.insert(0,"../")
import marinenav_env.envs.marinenav_env as marinenav_env
import os

if __name__ == "__main__":
    # This is an example of generating and saving a vrx experiment config file
    seed = 6
    env = marinenav_env.MarineNavEnv3(seed = seed)
    
    env.num_robots = 5
    env.num_cores = 0
    env.num_obs = 3
    env.min_start_goal_dis = 40.0
    env.reset()

    ep_data = env.episode_data()

    save_dir = "vrx/episode/results/save/directory"
    file = os.path.join(save_dir,"vrx_exp_config.json")
    with open(file, "w+") as f:
        json.dump(ep_data, f)