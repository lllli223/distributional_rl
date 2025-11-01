import sys
sys.path.insert(0,"../")
import env_visualizer

if __name__ == "__main__":
    dir = "training/evaluation/results/save/directory"

    eval_configs = "eval_configs.json"
    evaluations = "evaluations.npz"

    ev = env_visualizer.EnvVisualizer()

    eval_id = -1
    episode_id = -1

    colors = ['r','g','b','y','gray','tab:orange']
    ev.load_eval_config_and_episode(dir+eval_configs,dir+evaluations)
    ev.play_eval_episode(eval_id,episode_id,colors)