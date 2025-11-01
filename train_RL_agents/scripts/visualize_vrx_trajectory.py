import json
import sys
sys.path.insert(0,"../")
import env_visualizer
import copy
import os

if __name__ == "__main__":    
    episode_dir = "vrx/episode/results/save/directory"
    config_file = "vrx_exp_config.json"
    traj_file = "vrx_exp_traj.json"

    with open(os.path.join(episode_dir,config_file),"r") as f:
        episode_config = json.load(f)
    with open(os.path.join(episode_dir,traj_file),"r") as f:
        vrx_traj = json.load(f)

    ev = env_visualizer.EnvVisualizer(draw_vrx_traj=True)

    # moments of visualizing poses (list indices)
    plot_steps = [100,200,300]

    # convert coordinates in the VRX simulator to that in env_visualizer 
    for i in range(len(episode_config["robots"]["start"])):
        s_x = episode_config["robots"]["start"][i][0]
        episode_config["robots"]["start"][i][0] = -1.0 * s_x + ev.env.height
        g_x = episode_config["robots"]["goal"][i][0]
        episode_config["robots"]["goal"][i][0] = -1.0 * g_x + ev.env.width
    
    for i in range(len(episode_config["env"]["obstacles"]["positions"])):
        o_x = episode_config["env"]["obstacles"]["positions"][i][0]
        episode_config["env"]["obstacles"]["positions"][i][0] = -1.0 * o_x + ev.env.height

    ev.env.reset_with_eval_config(episode_config)

    colors = ['r','lime','cyan','tab:orange','y']
    vrx_center = [-480,240]

    ev.init_visualize()
    ev.play_vrx_episode(vrx_traj["timestamps"],
                        vrx_traj["poses"],
                        vrx_traj["velocities"],
                        vrx_center,
                        colors,
                        plot_steps)

