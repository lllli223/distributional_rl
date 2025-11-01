import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as ticker
import os
from matplotlib.ticker import ScalarFormatter

if __name__ == "__main__":
    seeds = ["random/seed(s)/used/in/training"]
    eval_agents = ["RL/agent(s)"]
    colors = ["plotting/color(s)"]
    data_dirs = ["RL/agent(s)/training/evaluation/file/directory"]


    fig, (ax_rewards,ax_success_rate,ax_times) = plt.subplots(1,3,figsize=(24,6))

    for idx, eval_agent in enumerate(eval_agents):
        all_rewards = []
        all_success_rates = []
        all_times = []
        all_energies = []

        final_success_rate = []
        print(f"===== Plotting {eval_agent} results =====")
        for seed in seeds:
            seed_dir = "seed_"+str(seed)
            eval_data = np.load(os.path.join(data_dirs[idx],seed_dir,"evaluations.npz"),allow_pickle=True)

            timesteps = np.array(eval_data['timesteps'],dtype=np.float64)
            rewards = np.mean(eval_data['rewards'],axis=1)
            success_rates = []
            avg_ts = []
            avg_es = []
            for i in range(len(eval_data['timesteps'])):
                successes = eval_data['successes'][i]
                success_rates.append(np.sum(successes)/len(successes))
                s_idx = np.where(np.array(successes) == 1)[0]
                
                times = eval_data['times'][i]
                energies = eval_data['energies'][i]
                avg_t = -1 if np.shape(s_idx)[0] == 0 else np.mean(np.array(times)[s_idx])
                avg_e = -1 if np.shape(s_idx)[0] == 0 else np.mean(np.array(energies)[s_idx])
                avg_ts.append(avg_t)
                avg_es.append(avg_e)

            final_success_rate.append(success_rates[-1])

            all_rewards.append(rewards.tolist())
            all_success_rates.append(success_rates)
            all_times.append(avg_ts)
            all_energies.append(avg_es)

        all_rewards_mean = np.mean(all_rewards,axis=0)
        all_rewards_std = np.std(all_rewards,axis=0)/np.sqrt(np.shape(all_rewards)[0])
        all_success_rates_mean = np.mean(all_success_rates,axis=0)
        all_success_rates_std = np.std(all_success_rates,axis=0)/np.sqrt(np.shape(all_success_rates)[0])
        all_times_mean = np.mean(all_times,axis=0)
        all_times_std = np.std(all_times,axis=0)/np.sqrt(np.shape(all_times)[0])
        all_energies_mean = np.mean(all_energies,axis=0)
        all_energies_std = np.std(all_energies,axis=0)

        ##### plot reward #####
        mpl.rcParams["font.size"]=20
        [x.set_linewidth(1.5) for x in ax_rewards.spines.values()]
        ax_rewards.tick_params(axis="x", labelsize=22)
        ax_rewards.tick_params(axis="y", labelsize=22)
        ax_rewards.plot(timesteps,all_rewards_mean,linewidth=3,label=eval_agent,c=colors[idx],zorder=5-idx)
        ax_rewards.fill_between(timesteps,all_rewards_mean+all_rewards_std,all_rewards_mean-all_rewards_std,alpha=0.2,color=colors[idx],zorder=5-idx)
        ax_rewards.xaxis.set_ticks(np.arange(0,eval_data['timesteps'][-1]+1,1000000))
        scale_x = 1e-5
        ticks_x = ticker.FuncFormatter(lambda x, pos:'{0:g}'.format(x*scale_x))
        ax_rewards.xaxis.set_major_formatter(ticks_x)
        ax_rewards.set_xlabel("Timestep(x10^5)",fontsize=25)
        ax_rewards.yaxis.set_ticks(np.arange(-20,31,10))
        ax_rewards.set_title("Cumulative Reward",fontsize=25,fontweight='bold')
        ax_rewards.legend(loc="lower right",bbox_to_anchor=(1, 0),ncol=2)

        ##### plot success rate #####
        [x.set_linewidth(1.5) for x in ax_success_rate.spines.values()]
        ax_success_rate.tick_params(axis="x", labelsize=22)
        ax_success_rate.tick_params(axis="y", labelsize=22)
        ax_success_rate.plot(timesteps,all_success_rates_mean,linewidth=3,label=eval_agent,c=colors[idx],zorder=5-idx)
        ax_success_rate.fill_between(timesteps,all_success_rates_mean+all_success_rates_std,all_success_rates_mean-all_success_rates_std,alpha=0.2,color=colors[idx],zorder=5-idx)
        ax_success_rate.xaxis.set_ticks(np.arange(0,eval_data['timesteps'][-1]+1,1000000))
        scale_x = 1e-5
        ticks_x = ticker.FuncFormatter(lambda x, pos:'{0:g}'.format(x*scale_x))
        ax_success_rate.xaxis.set_major_formatter(ticks_x)
        ax_success_rate.set_xlabel("Timestep(x10^5)",fontsize=25)
        ax_success_rate.yaxis.set_ticks(np.arange(0,1.1,0.2))
        ax_success_rate.set_title("Success Rate",fontsize=25,fontweight='bold')

        ##### plot time #####
        [x.set_linewidth(1.5) for x in ax_times.spines.values()]
        ax_times.tick_params(axis="x", labelsize=22)
        ax_times.tick_params(axis="y", labelsize=22)
        ax_times.plot(timesteps,all_times_mean,linewidth=3,label=eval_agent,c=colors[idx],zorder=5-idx)
        ax_times.xaxis.set_ticks(np.arange(0,eval_data['timesteps'][-1]+1,1000000))
        scale_x = 1e-5
        ticks_x = ticker.FuncFormatter(lambda x, pos:'{0:g}'.format(x*scale_x))
        ax_times.xaxis.set_major_formatter(ticks_x)
        ax_times.set_xlabel("Timestep(x10^5)",fontsize=25)
        ax_times.set_title("Average Travel Time (s)",fontsize=25,fontweight='bold')
        ax_times.yaxis.set_major_formatter(ScalarFormatter())
        ax_times.minorticks_off()

        print(f"{eval_agent} final success rates of all models: {final_success_rate}")

    fig.tight_layout()
    fig.savefig("learning_curves.png")
