import marinenav_env.envs.marinenav_env as marinenav_env
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.animation as animation
from matplotlib.patches import Rectangle, Polygon
from matplotlib.transforms import Affine2D
import copy
import scipy.spatial
import gym
import json
import os

class EnvVisualizer:

    def __init__(self, 
                 seed:int=0, 
                 draw_envs:bool=False, # Mode 2: plot the envrionment
                 draw_vrx_traj:bool=False, # Mode 3: plot trajectories from VRX env
                 ): 
        self.env = marinenav_env.MarineNavEnv3(seed)
        self.env.reset()
        self.fig = None # figure for visualization
        self.axis_graph = None # sub figure for the map
        self.robot_components = {}
        self.robots_plot = []
        self.robots_plot_counts = [] # number of elements added at each step
        self.robots_last_pos = []
        self.robots_traj_plot = []
        self.axis_perception = None # sub figure for perception output

        self.draw_envs = draw_envs # draw only the envs
        self.draw_vrx_traj = draw_vrx_traj # draw trajectories from VRX env
        
        self.configs = None # evaluation configs
        self.episodes = None # evaluation episodes to visualize 

    def init_visualize(self,
                       env_configs=None, # used in Mode 2,
                       colors=None, # used in Mode 5
                       ):
        
        # initialize subplot for the map, robot state and sensor measurments
        if self.draw_envs:
            # Mode 2: plot the envrionment
            if env_configs is None:
                self.fig, self.axis_graph = plt.subplots(1,1,figsize=(8,8))
            else:
                num = len(env_configs)
                if num == 1:
                    self.fig, self.axis_graph = plt.subplots(1,1,figsize=(16,16))
                elif num % 3 == 0:
                    self.fig, self.axis_graph = plt.subplots(int(num/3),3,figsize=(8*3,8*int(num/3))) 
                else:
                    self.fig, self.axis_graph = plt.subplots(1,num,figsize=(8*num,8))
        elif self.draw_vrx_traj:
            # Mode 3: draw trajectories from VRX env
            self.fig, self.axis_graph = plt.subplots(1,1,figsize=(16,16))
        else:
            # Mode 1 (default): Display an episode
            self.fig = plt.figure(figsize=(32,16))
            spec = self.fig.add_gridspec(5,3)
            self.axis_graph = self.fig.add_subplot(spec[:,:2])
            self.axis_perception = self.fig.add_subplot(spec[:,2])

        if self.draw_envs and env_configs is not None:
            for i,env_config in enumerate(env_configs):
                self.load_env_config(env_config)
                if len(env_configs) == 1:
                    self.plot_graph(self.axis_graph)
                elif len(env_configs) % 3 == 0: 
                    self.plot_graph(self.axis_graph[i],colors=colors)
                    [x.set_linewidth(3) for x in self.axis_graph[i].spines.values()]
                else:
                    self.plot_graph(self.axis_graph[i])
        elif self.draw_vrx_traj:
            self.plot_graph(self.axis_graph,colors=colors)
        else:
            self.plot_graph(self.axis_graph)

    def plot_graph(self,axis,plot_current=False,colors=None):
        # x-y-z (+): Forward-Starboard-Down (robot frame), North-East-Down (world frame) 
        # yaw (+): clockwise
        
        # compute current velocity in the map
        x_pos = list(np.linspace(0.0,self.env.width,100))
        y_pos = list(np.linspace(0.0,self.env.height,100))

        pos_x = []
        pos_y = []
        arrow_x = []
        arrow_y = []
        speeds = np.zeros((len(x_pos),len(y_pos)))
        for m,x in enumerate(x_pos):
            for n,y in enumerate(y_pos):
                v = self.env.get_velocity(x,y)
                speed = np.clip(np.linalg.norm(v),0.1,self.env.v_range[1])
                pos_x.append(x)
                pos_y.append(y)
                arrow_x.append(v[0])
                arrow_y.append(v[1])
                speeds[m,n] = np.log(speed)


        self.cmap = cm.Blues(np.linspace(0,1,20))
        cmap = mpl.colors.ListedColormap(self.cmap[13:18,:-1])

        if not self.draw_envs:
            axis.contourf(y_pos,x_pos,speeds,cmap=cmap)
        
        if plot_current:
            axis.quiver(pos_y,pos_x,arrow_y,arrow_x,width=0.001,scale_units='xy',scale=1.2)

        # plot obstacles in the map
        for obs in self.env.obstacles:
            c = 'k' if self.draw_envs else 'r'
            axis.add_patch(mpl.patches.Circle((obs.y,obs.x),radius=obs.r,color=c))

        axis.set_aspect('equal')
        axis.set_xlim([0.0,self.env.width])
        axis.set_ylim([0.0,self.env.height])
        axis.set_xticks([])
        axis.set_yticks([])

        # plot start and goal state of each robot
        for idx,robot in enumerate(self.env.robots):
            if not self.draw_envs:
                axis.scatter(robot.start[1],robot.start[0],marker="o",color="yellow",s=200,zorder=6)
                axis.text(robot.start[1]+1.5,robot.start[0]-1.5,str(idx),color="yellow",fontsize=30,zorder=6)
                axis.scatter(robot.goal[1],robot.goal[0],marker="*",color="yellow",s=650,zorder=6)
                axis.text(robot.goal[1]+1.5,robot.goal[0]-1.5,str(idx),color="yellow",fontsize=30,zorder=6)
            else:
                axis.scatter(robot.goal[1],robot.goal[0],marker="*",color="g",s=650,zorder=6)
                axis.text(robot.goal[1]+1.5,robot.goal[0]-1.5,str(idx),color="g",fontsize=30,zorder=6)
            self.robots_last_pos.append([])
            self.robots_traj_plot.append([])

        self.create_robot_components(colors)
        self.plot_robots(axis)

    def remove_last_robot_plots(self):
        last_count = self.robots_plot_counts[-1]
        for robot_plot in self.robots_plot[-last_count:]:
            robot_plot.remove()
        self.robots_plot = self.robots_plot[:-last_count]
        self.robots_plot_counts = self.robots_plot_counts[:-1]
    
    def plot_robots(self,axis,traj_color=None,remove_last_plots=True,times=None):
        if not self.draw_envs:
            if remove_last_plots and len(self.robots_plot_counts) > 0:
                self.remove_last_robot_plots()

        self.robots_plot_counts.append(0)
        for i,robot in enumerate(self.env.robots):
            if robot.deactivated:
                continue
            
            c = 'lime'
            # draw robot velocity
            v_mag = np.linalg.norm(robot.velocity[:2])
            robot.velocity *= (max(v_mag,1.0)/v_mag)
            self.robots_plot.append(axis.quiver(robot.y,robot.x,robot.velocity[1],robot.velocity[0], \
                                                color="r" if traj_color is None else traj_color[i], \
                                                width=0.007 if self.draw_envs else 0.005, \
                                                zorder=15 if self.draw_envs else 11,headlength=5,headwidth=3, \
                                                scale=4.2,scale_units='inches'))
            self.robots_plot_counts[-1] += 1

            # draw robot
            self.plot_robot_pose(axis,i)

            label_pos = robot.r + 0.5
            if times is None:
                c = "b" if self.draw_envs else "yellow"
                # mark robot id
                self.robots_plot.append(axis.text(robot.y-label_pos,robot.x-label_pos,str(i),color=c,fontsize=25,zorder=10))
            else:
                # mark timestamp
                self.robots_plot.append(axis.text(robot.y+label_pos,robot.x+label_pos,f'{times[i]:.1f}',color="yellow" if traj_color is None else traj_color[i],fontsize=25,zorder=10))

            self.robots_plot_counts[-1] += 1

            if not self.draw_envs:
                if self.robots_last_pos[i] != []:
                    h = axis.plot((self.robots_last_pos[i][1],robot.y),
                                (self.robots_last_pos[i][0],robot.x),
                                color='tab:orange' if traj_color is None else traj_color[i],
                                linewidth=3.0)
                    self.robots_traj_plot[i].append(h)
                
                self.robots_last_pos[i] = [robot.x, robot.y]

    def create_robot_components(self,colors=None):
        self.robot_components['hull_tip_left'] = []
        self.robot_components['hull_mid_left'] = []
        self.robot_components['hull_rear_left'] = []
        self.robot_components['thruster_left'] = []
        self.robot_components['beam_base_left'] = []
        self.robot_components['rear_beam'] = []
        self.robot_components['platform'] = []
        self.robot_components['thruster_left_pivot'] = []
        self.robot_components['thruster_right_pivot'] = []

        for i,rob in enumerate(self.env.robots):
            color_hull = "darkgray" if colors is None else colors[i]
            color_beam = "k" if colors is None else colors[i]
            color_platform = "k" if colors is None else colors[i]
            color_thruster = "tab:orange" if colors is None else colors[i]
            
            # compute left hull component base points (lower left points in body frame (x-y (+): Forward-Starboard))
            thruster_tip_left_px = -0.5*rob.length
            thruster_rear_left_px = thruster_tip_left_px + rob.thruster_length
            hull_rear_left_px = thruster_rear_left_px + rob.thruster_gap
            hull_mid_left_px = hull_rear_left_px + rob.hull_rear_length
            hull_tip_left_px = 0.5*rob.length - rob.hull_tip_length
            beam_base_left_px = -0.5*rob.beam_base_length
            rear_beam_px = -0.5*rob.beam_distance - 0.5*rob.beam_width

            hull_tip_left_py = -0.5*rob.width
            hull_mid_left_py = -0.5*rob.width        
            hull_rear_left_py = hull_mid_left_py + 0.5*(rob.hull_width - rob.hull_rear_width)
            thruster_rear_left_py = hull_rear_left_py + 0.5*(rob.hull_rear_width - rob.thruster_rear_width)
            thruster_tip_left_py = hull_rear_left_py + 0.5*(rob.hull_rear_width - rob.thruster_tip_width)
            beam_base_left_py = -0.5*rob.width + 0.5*(rob.hull_width - rob.beam_base_width)
            rear_beam_py = -0.5*rob.width + 0.5*rob.hull_width

            # create left hull components
            hull_tip_points = np.array([[hull_tip_left_py,hull_tip_left_px],
                                        [hull_tip_left_py+0.5*(rob.hull_width-rob.hull_tip_width),0.5*rob.length],
                                        [hull_tip_left_py+0.5*(rob.hull_width-rob.hull_tip_width)+rob.hull_tip_width,0.5*rob.length],
                                        [hull_tip_left_py+rob.hull_width,hull_tip_left_px]])
            hull_tip_left = Polygon(hull_tip_points,closed=True,facecolor=color_hull,label='hull_tip',zorder=8)
            self.robot_components['hull_tip_left'].append(hull_tip_left)

            hull_mid_left = Rectangle((hull_mid_left_py,hull_mid_left_px),rob.hull_width,hull_tip_left_px-hull_mid_left_px,facecolor=color_hull,label='hull_mid',zorder=8)
            self.robot_components['hull_mid_left'].append(hull_mid_left)

            hull_rear_points = np.array([[hull_rear_left_py,hull_rear_left_px],
                                        [hull_mid_left_py,hull_mid_left_px],
                                        [hull_mid_left_py+rob.hull_width,hull_mid_left_px],
                                        [hull_rear_left_py+rob.hull_rear_width,hull_rear_left_px]])
            hull_rear_left = Polygon(hull_rear_points,closed=True,facecolor=color_hull,label='hull_rear',zorder=8)
            self.robot_components['hull_rear_left'].append(hull_rear_left)

            thruster_points = np.array([[thruster_tip_left_py,thruster_tip_left_px],
                                        [thruster_rear_left_py,thruster_rear_left_px],
                                        [thruster_rear_left_py+rob.thruster_rear_width,thruster_rear_left_px],
                                        [thruster_tip_left_py+rob.thruster_tip_width,thruster_tip_left_px]])
            thruster_left = Polygon(thruster_points,closed=True,facecolor=color_thruster,label='thruster',zorder=8)
            self.robot_components['thruster_left'].append(thruster_left)

            beam_base_left = Rectangle((beam_base_left_py,beam_base_left_px),rob.beam_base_width,rob.beam_base_length,facecolor=color_beam,label='beam_base',zorder=9)
            self.robot_components['beam_base_left'].append(beam_base_left)

            rear_beam = Rectangle((rear_beam_py,rear_beam_px),rob.beam_length,rob.beam_width,facecolor=color_beam,label='beam',zorder=9)
            self.robot_components['rear_beam'].append(rear_beam)

            platform = Rectangle((-0.5*rob.platform_width,-0.5*rob.platform_length),rob.platform_width,rob.platform_length,facecolor=color_platform,label='platform',zorder=10)
            self.robot_components['platform'].append(platform)

            # compute thruster pivots
            thruster_left_pivot_px = -0.5*rob.length + rob.thruster_length
            thruster_left_pivot_py = -0.5*rob.width + 0.5*rob.hull_width
            thruster_right_pivot_px = thruster_left_pivot_px
            thruster_right_pivot_py = -thruster_left_pivot_py
            self.robot_components['thruster_left_pivot'].append([thruster_left_pivot_px,thruster_left_pivot_py])
            self.robot_components['thruster_right_pivot'].append([thruster_right_pivot_px,thruster_right_pivot_py])

    def plot_robot_pose(self,axis,robot_id,in_world_frame=True):
        robot = self.env.robots[robot_id]

        # compute left to right transform
        trans_left_to_right = Affine2D().translate(robot.width-robot.hull_width,0)
        trans_rear_to_front = Affine2D().translate(0,robot.beam_distance)
        
        # compute thruster pos transform
        thruster_left_pivot_cp = copy.deepcopy(self.robot_components['thruster_left_pivot'][robot_id])
        thruster_left_rot = Affine2D().rotate_around(thruster_left_pivot_cp[1],thruster_left_pivot_cp[0],-robot.left_pos)

        thruster_right_pivot_cp = copy.deepcopy(self.robot_components['thruster_right_pivot'][robot_id])
        thruster_right_rot = Affine2D().rotate_around(thruster_right_pivot_cp[1],thruster_right_pivot_cp[0],-robot.right_pos)

        # move robot to the current pose
        if in_world_frame:
            robot_pose = Affine2D().rotate(-robot.theta).translate(robot.y,robot.x)
        else:
            robot_pose = Affine2D()
        
        for component in self.robot_components.keys():
            if component == 'thruster_left_pivot' or component == 'thruster_right_pivot':
                continue
            elif component == 'thruster_left':
                component_cp = copy.deepcopy(self.robot_components[component][robot_id])
                component_cp.set_transform(thruster_left_rot+robot_pose+axis.transData)
                self.robots_plot.append(axis.add_patch(component_cp))
                self.robots_plot_counts[-1] += 1
                
                component_cp_2 = copy.deepcopy(self.robot_components[component][robot_id])
                component_cp_2.set_transform(trans_left_to_right+thruster_right_rot+robot_pose+axis.transData)
                self.robots_plot.append(axis.add_patch(component_cp_2))
                self.robots_plot_counts[-1] += 1
            else:
                component_cp = copy.deepcopy(self.robot_components[component][robot_id])
                component_cp.set_transform(robot_pose+axis.transData)
                self.robots_plot.append(axis.add_patch(component_cp))
                self.robots_plot_counts[-1] += 1
                
                if component != 'platform':
                    component_cp_2 = copy.deepcopy(self.robot_components[component][robot_id])

                    if component == 'rear_beam':
                        component_cp_2.set_transform(trans_rear_to_front+robot_pose+axis.transData)
                    else:
                        component_cp_2.set_transform(trans_left_to_right+robot_pose+axis.transData)
                    
                    self.robots_plot.append(axis.add_patch(component_cp_2))
                    self.robots_plot_counts[-1] += 1

    def plot_measurements(self,robot_idx,observation,R_matrix=None):
        self.axis_perception.clear()
        # self.axis_observation.clear()
        # self.axis_goal.clear()

        rob = self.env.robots[robot_idx]

        # if rob.reach_goal:
        #     print(f"robot {robot_idx} reached goal, no measurements are available!")
        #     return

        legend_size = 12
        font_size = 15

        # plot detected objects in the robot frame
        self.axis_perception.add_patch(mpl.patches.Circle((0,0), \
                                       rob.perception.range, color='g', alpha=0.2))
        
        # plot self velocity
        abs_velocity_r = observation[0][2:]
        self.axis_perception.quiver(0.0,0.0,abs_velocity_r[1],abs_velocity_r[0], \
                                   color='r',width=0.008,headlength=5,headwidth=3,zorder=11)
        
        # plot robot
        self.plot_robot_pose(self.axis_perception,robot_idx,in_world_frame=False)

        x_pos = 0
        y_pos = 0
        relation_pos = [[0.0,0.0]]

        # for i,obs in enumerate(rob.perception.observation["objects"]): 
        #     self.axis_perception.add_patch(mpl.patches.Circle((obs[0],obs[1]), \
        #                                    obs[2], color="m"))
        #     relation_pos.append([-obs[1],obs[0]])
            # include into observation info
            # self.axis_observation.text(x_pos,y_pos,f"position: ({obs[0]:.2f},{obs[1]:.2f}), radius: {obs[2]:.2f}")
            # y_pos += 1

        # self.axis_observation.text(x_pos,y_pos,"Static obstacles",fontweight="bold",fontsize=15)
        # y_pos += 2

        # for i,obj_history in enumerate(rob.perception.observation["dynamic"].values()):
        for i,obj in enumerate(observation[1]):
            # plot the current position
            # pos = obj_history[-1][:2]

            # plot velocity
            self.axis_perception.quiver(obj[1],obj[0],obj[3],obj[2],color="r", \
                                        width=0.008,headlength=5,headwidth=3,zorder=11)
            
            # plot position
            object = mpl.patches.Circle((obj[1],obj[0]), obj[4], edgecolor="m")
            object.set_facecolor((0, 0, 0, 0))
            self.axis_perception.add_patch(object)
            relation_pos.append([obj[0],obj[1]])
            
            # include history into observation info
            # self.axis_observation.text(x_pos,y_pos,f"position: ({obj[0]:.2f},{obj[1]:.2f}), velocity: ({obj[2]:.2f},{obj[3]:.2f})")
            # y_pos += 1
        
        # self.axis_observation.text(x_pos,y_pos,"Other Robots",fontweight="bold",fontsize=15)
        # y_pos += 2
        

        if R_matrix is not None:
            # plot relation matrix
            length = len(R_matrix)
            assert len(relation_pos) == length, "The number of objects do not match size of the relation matrix"
            for i in range(length):
                for j in range(length):
                    self.axis_perception.plot([relation_pos[i][0],relation_pos[j][0]], \
                                              [relation_pos[i][1],relation_pos[j][1]],
                                              linewidth=2*R_matrix[i][j],color='k',zorder=0)

        self.axis_perception.set_xlim([-rob.perception.range-1,rob.perception.range+1])
        self.axis_perception.set_ylim([-rob.perception.range-1,rob.perception.range+1])
        self.axis_perception.set_aspect('equal')
        self.axis_perception.set_title(f'Robot {robot_idx}',fontsize=25)

        self.axis_perception.set_xticks([])
        self.axis_perception.set_yticks([])
        self.axis_perception.spines["left"].set_visible(False)
        self.axis_perception.spines["top"].set_visible(False)
        self.axis_perception.spines["right"].set_visible(False)
        self.axis_perception.spines["bottom"].set_visible(False)

    def load_env_config(self,episode_dict):
        episode = copy.deepcopy(episode_dict)
        self.env.reset_with_eval_config(episode)

    def load_eval_config_and_episode(self,config_file,eval_file):
        with open(config_file,"r") as f:
            self.configs = json.load(f)

        self.episodes = np.load(eval_file,allow_pickle=True)

    def play_eval_episode(self,eval_id,episode_id,colors,robot_ids=None):
        self.env.reset_with_eval_config(self.configs[episode_id])
        self.init_visualize()
        
        # relations = None
        # if draw_relation:
        #     relations = self.episodes["relations"][eval_id][episode_id]
        # actions = self.episodes["actions"][eval_id][episode_id]
        trajectories = self.episodes["trajectories"][eval_id][episode_id]
        observations = self.episodes["observations"][eval_id][episode_id]
        
        self.play_episode(trajectories,observations,colors,robot_ids)

    def play_episode(self,
                     trajectories,
                     observations,
                     colors,
                     robot_ids=None,
                     max_steps=None,
                     start_step=0):
        
        # sort robots according to trajectory lengths
        all_robots = []
        for i,traj in enumerate(trajectories):
            plot_observation = False if robot_ids is None else i in robot_ids
            all_robots.append({"id":i,"traj_len":len(traj),"plot_observation":plot_observation})
        all_robots = sorted(all_robots, key=lambda x: x["traj_len"])
        all_robots[-1]["plot_observation"] = True

        if max_steps is None:
            max_steps = all_robots[-1]["traj_len"]-1

        robots = []
        for robot in all_robots:
            if robot["plot_observation"] is True:
                robots.append(robot)

        idx = 0
        current_robot_step = 0
        for i in range(max_steps):
            if i >= robots[idx]["traj_len"]:
                current_robot_step = 0
                idx += 1

            for j,rob in enumerate(self.env.robots):
                if rob.deactivated:
                    continue
                rob.x = trajectories[j][i][0]
                rob.y = trajectories[j][i][1]
                rob.theta = trajectories[j][i][2]
                rob.velocity_r = np.array(trajectories[j][i][3:6])
                rob.velocity = np.array(trajectories[j][i][6:9])
                rob.left_pos = trajectories[j][i][9]
                rob.right_pos = trajectories[j][i][10]
                rob.left_thrust = trajectories[j][i][11]
                rob.right_thrust = trajectories[j][i][12]  

            self.plot_robots(self.axis_graph,colors)
            self.plot_measurements(robots[idx]["id"],observations[robots[idx]["id"]][i])
            # action = [actions[j][i] for j in range(len(self.env.robots))]
            # self.env.step(action)

            plt.pause(0.1)

            for j,rob in enumerate(self.env.robots):
                if i == len(trajectories[j])-1:
                    rob.deactivated = True

            current_robot_step += 1

    def quaternion_to_euler_angle(self,quaternion):
        # Extract individual components of the quaternion
        x = quaternion[0] 
        y = quaternion[1] 
        z = quaternion[2]
        w = quaternion[3]

        # Convert quaternion to yaw angle
        yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y**2 + z**2))

        return (-yaw+np.pi)
    
    def project_vrx_velocity_to_world_frame(self,theta,vrx_velocity):
        R_wr = np.matrix([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
        
        velocity = [vrx_velocity[0],-1.0*vrx_velocity[1]]
        v = np.matrix(velocity).transpose()
        v = R_wr * v
        return np.squeeze(np.array(v))

    def play_vrx_episode(self,
                         timestamps,
                         poses,
                         velocities,
                         vrx_center,
                         colors,
                         plot_steps
                         ):
        
        ep_length = np.inf
        for v in timestamps.values():
            if len(v) < ep_length:
                ep_length = len(v)

        for i in range(ep_length):
            
            if i > np.max(plot_steps):
                break

            for j,rob in enumerate(self.env.robots):
                
                # flip x coordinate and yaw angle about the middle line of the map,
                rob.x = -1.0*(poses[f'wamv{j+1}'][i][0] - vrx_center[0]) + self.env.height/2
                rob.y = poses[f'wamv{j+1}'][i][1] - vrx_center[1] + self.env.width/2
                rob.theta = self.quaternion_to_euler_angle(poses[f'wamv{j+1}'][i][2:])
                rob.velocity = np.zeros(3)
                rob.velocity[:2] = self.project_vrx_velocity_to_world_frame(rob.theta,velocities[f'wamv{j+1}'][i])

            remove_last_plots = True
            if i in plot_steps:
                remove_last_plots = False

            times = [timestamps[f'wamv{j+1}'][i]-timestamps[f'wamv{j+1}'][0] for j in range(len(self.env.robots))]

            self.plot_robots(self.axis_graph,colors,remove_last_plots,times)

        if (ep_length-1) not in plot_steps:
            self.remove_last_robot_plots()

        self.fig.savefig("vrx_episode_section.png",bbox_inches="tight")

