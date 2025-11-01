import launch
import launch_ros.actions
import rclpy
import time
from std_msgs.msg import Empty
from std_msgs.msg import Bool
from robot_info_msg.msg import RobotInfo
import threading
import numpy as np
import math
from functools import partial
import subprocess
import sys
sys.path.insert(0,"../train_RL_agents")
import marinenav_env.envs.marinenav_env as marinenav_env

from launch.actions import RegisterEventHandler
from launch.event_handlers import OnProcessExit
from launch.actions import ExecuteProcess, EmitEvent
from launch.events import Shutdown

import copy
from datetime import datetime
import os

import xml.etree.ElementTree as ET
import shutil
import json

def add_buoy_to_sdf(input_file, output_file, buoy_poses):
    shutil.copyfile(input_file, output_file)
    tree = ET.parse(output_file)
    root = tree.getroot()

    # Find the <world name="sydney_regatta"> tag
    world_tag = root.find(".//world[@name='sydney_regatta']")
    if world_tag is None:
        print("Error: Could not find <world name='sydney_regatta'> tag in SDF file.")
        return
    
    poses = []
    if len(buoy_poses)>0: 
        poses = [[float(p) for p in pos.split(',')] for pos in buoy_poses.split(';')]
    
    assert len(poses) <= 4, "Error: Currently do not support more than 4 buoys"
    buoys = ["robotx_light_buoy_rgb","robotx_light_buoy_rgy","robotx_light_buoy_ybr",
             "robotx_light_buoy_yrg"]

    for i,pose in enumerate(poses):
        include_elem = ET.Element('include')

        name_elem = ET.Element('name')
        name_elem.text = buoys[i]
        include_elem.append(name_elem)

        # Add pose element
        pose_elem = ET.Element('pose')
        pose_elem.text = f"{pose[0]} {pose[1]} 0.32 0 0 0"  # Set the fixed orientation
        include_elem.append(pose_elem)

        # Add uri element
        uri_elem = ET.Element('uri')
        uri_elem.text = buoys[i]
        include_elem.append(uri_elem)

        # Create the plugin element
        plugin_elem = ET.Element('plugin')
        plugin_elem.set('name', 'vrx::PolyhedraBuoyancyDrag')
        plugin_elem.set('filename', 'libPolyhedraBuoyancyDrag.so')

        # Add plugin sub-elements
        fluid_density_elem = ET.Element('fluid_density')
        fluid_density_elem.text = '500'
        plugin_elem.append(fluid_density_elem)

        fluid_level_elem = ET.Element('fluid_level')
        fluid_level_elem.text = '0.0'
        plugin_elem.append(fluid_level_elem)

        linear_drag_elem = ET.Element('linear_drag')
        linear_drag_elem.text = '10.0'
        plugin_elem.append(linear_drag_elem)

        angular_drag_elem = ET.Element('angular_drag')
        angular_drag_elem.text = '0.0'
        plugin_elem.append(angular_drag_elem)

        # Add buoyancy elements
        buoyancy_base_elem = ET.Element('buoyancy', name='buoyancy_base')
        buoyancy_base_link_name_elem = ET.Element('link_name')
        buoyancy_base_link_name_elem.text = 'base_link'
        buoyancy_base_elem.append(buoyancy_base_link_name_elem)
        buoyancy_base_pose_elem = ET.Element('pose')
        buoyancy_base_pose_elem.text = '0 0 -0.1 0 0 0'
        buoyancy_base_elem.append(buoyancy_base_pose_elem)
        buoyancy_base_geometry_elem = ET.Element('geometry')
        buoyancy_base_box_elem = ET.Element('box')
        buoyancy_base_box_size_elem = ET.Element('size')
        buoyancy_base_box_size_elem.text = '1.5 1.5 0.4'
        buoyancy_base_box_elem.append(buoyancy_base_box_size_elem)
        buoyancy_base_geometry_elem.append(buoyancy_base_box_elem)
        buoyancy_base_elem.append(buoyancy_base_geometry_elem)
        plugin_elem.append(buoyancy_base_elem)

        buoyancy_body_elem = ET.Element('buoyancy', name='buoyancy_body')
        buoyancy_body_link_name_elem = ET.Element('link_name')
        buoyancy_body_link_name_elem.text = 'base_link'
        buoyancy_body_elem.append(buoyancy_body_link_name_elem)
        buoyancy_body_pose_elem = ET.Element('pose')
        buoyancy_body_pose_elem.text = '0 0 0.8 0 0 0'
        buoyancy_body_elem.append(buoyancy_body_pose_elem)
        buoyancy_body_geometry_elem = ET.Element('geometry')
        buoyancy_body_cylinder_elem = ET.Element('cylinder')
        buoyancy_body_cylinder_radius_elem = ET.Element('radius')
        buoyancy_body_cylinder_radius_elem.text = '0.2'
        buoyancy_body_cylinder_elem.append(buoyancy_body_cylinder_radius_elem)
        buoyancy_body_cylinder_length_elem = ET.Element('length')
        buoyancy_body_cylinder_length_elem.text = '0.15'
        buoyancy_body_cylinder_elem.append(buoyancy_body_cylinder_length_elem)
        buoyancy_body_geometry_elem.append(buoyancy_body_cylinder_elem)
        buoyancy_body_elem.append(buoyancy_body_geometry_elem)
        plugin_elem.append(buoyancy_body_elem)

        wavefield_elem = ET.Element('wavefield')
        wavefield_topic_elem = ET.Element('topic')
        wavefield_topic_elem.text = '/vrx/wavefield/parameters'
        wavefield_elem.append(wavefield_topic_elem)
        plugin_elem.append(wavefield_elem)

        include_elem.append(plugin_elem)

        # Append the include element to the world_tag
        world_tag.append(include_elem)

    # Write the modified tree back to the file
    tree.write(output_file)

def exp_setup(test_env,eval_schedule,i):
    test_env.num_robots = eval_schedule["num_robots"][i]
    test_env.num_cores = eval_schedule["num_cores"][i]
    test_env.num_obs = eval_schedule["num_obstacles"][i]
    test_env.min_start_goal_dis = eval_schedule["min_start_goal_dis"][i]
    test_env.reset()

    ep_data = test_env.episode_data()

    return read_exp_setup(ep_data)

def read_exp_setup(ep_data):

    test_env_center = [27.5,27.5]
    vrx_center = [-480.0,240.0]
    
    init_poses = ""
    goals = ""
    for i in range(len(ep_data["robots"]["start"])):
        start = ep_data["robots"]["start"][i]
        goal = ep_data["robots"]["goal"][i]
        init_theta = ep_data["robots"]["init_theta"][i]

        start_x = start[0] - test_env_center[0] + vrx_center[0]
        start_y = start[1] - test_env_center[1] + vrx_center[1]

        goal_x = goal[0] - test_env_center[0] + vrx_center[0]
        goal_y = goal[1] - test_env_center[1] + vrx_center[1]

        pose = str(start_x)+","+str(start_y)+","+str(init_theta)+";"
        goal = str(goal_x)+","+str(goal_y)+";"

        init_poses += pose
        goals += goal

    buoy_poses = ""
    for i in range(len(ep_data["env"]["obstacles"]["positions"])):
        pos = ep_data["env"]["obstacles"]["positions"][i]

        pos_x = pos[0] - test_env_center[0] + vrx_center[0]
        pos_y = pos[1] - test_env_center[1] + vrx_center[1]
        r = 0.75

        buoy_poses += str(pos_x)+","+str(pos_y)+","+str(r)+";"

    return init_poses[:-1], goals[:-1], buoy_poses[:-1]

class ExperimentManager:
    def __init__(self,num_robots,successes_data,travel_times_data,save_traj=False):
        self.ls = None
        self.lock = threading.Lock()
        self.unpause_signal_node = rclpy.create_node('unpause_signal_node')
        self.unpause_signal_publishers = []
        
        self.robot_r = 2.8
        self.goal_dis = 2.0
        self.num_robots = num_robots

        self.collision = False
        self.exp_success = False

        self.save_traj = save_traj
        self.timestamp_data = {}
        self.pose_data = {}
        self.velocity_data = {}

        for i in range(self.num_robots):
            self.unpause_signal_publishers.append(self.unpause_signal_node.create_publisher(Empty, 
                                                  f'/wamv{i+1}/unpause_signal', 10))
            self.timestamp_data[f'wamv{i+1}'] = []
            self.pose_data[f'wamv{i+1}'] = []
            self.velocity_data[f'wamv{i+1}'] = []

        # self.goals = {}
        
        # for i,goal in enumerate(goals.split(';')):
        #     self.goals[f'wamv{i+1}'] = [float(g) for g in goal.split(',')]
            
        self.successes_data = copy.deepcopy(successes_data)
        self.travel_times_data = copy.deepcopy(travel_times_data)

    def launch_simulation(self, init_poses, goals, buoy_poses, method, agent_type, model_path, world_name):
        # Create launch description
        ld = launch.LaunchDescription()

        # Add action to launch competition environment
        competition_launch_file = launch.actions.ExecuteProcess(
            cmd=['ros2', 'launch', 'vrx_gz', 'competition.launch.py', 
                 "init_poses:="+init_poses, "goals:="+goals,
                 "buoy_poses:="+buoy_poses, "method:="+method,
                 "agent_type:="+agent_type, "model_path:="+model_path,
                 "world:="+world_name]
                 if len(buoy_poses)>0 else
                 ['ros2', 'launch', 'vrx_gz', 'competition.launch.py', 
                 "init_poses:="+init_poses, "goals:="+goals, "method:="+method,
                 "agent_type:="+agent_type, "model_path:="+model_path,
                 "world:="+world_name],
            output='screen'
        )

        vrx_exit_event_handler = RegisterEventHandler(
            OnProcessExit(
                target_action=competition_launch_file,\
                on_exit=[
                    EmitEvent(event=Shutdown(reason='VRX Sim Ended'))
                ]
                
                )
        )

        ld.add_action(competition_launch_file)
        ld.add_action(vrx_exit_event_handler)

        self.unpause_signal_thread = threading.Thread(target=self.start_unpause_signal_thread)
        self.unpause_signal_thread.start()

        self.robot_info_thread = threading.Thread(target=self.start_robot_info_subscribers)
        self.robot_info_thread.start()

        self.experiment_monitoring_thread = threading.Thread(target=self.experiment_monitoring)
        self.experiment_monitoring_thread.start()
        
        # Launch simulation
        ls = launch.LaunchService()
        ls.include_launch_description(ld)

        self.ls = ls.run()

        self.unpause_signal_thread.join()
        self.robot_info_thread.join()
        self.experiment_monitoring_thread.join()

    def start_robot_info_subscribers(self):
        self.robot_info_node = rclpy.create_node('robot_info_node')
        self.robot_info_subscribers = []
        self.robot_info = {}

        self.robot_info_subscribers.append(self.robot_info_node.create_subscription(
                                           Bool, '/collision_detection', self.collision_detection_callback,10))

        for i in range(self.num_robots):
            self.robot_info_subscribers.append(self.robot_info_node.create_subscription(
                                               RobotInfo, f'/wamv{i+1}/robot_info', self.robot_info_callback,10))
            self.robot_info[f'wamv{i+1}'] = {}

        executor = rclpy.executors.MultiThreadedExecutor(num_threads=1)
        executor.add_node(self.robot_info_node)
        executor.spin()

    def start_unpause_signal_thread(self):
        # Send unpause signal 10 seconds after the launch
        time.sleep(20)
        self.start_time = time.time()
        msg = Empty()
        while rclpy.ok():
            for publisher in self.unpause_signal_publishers:
                publisher.publish(msg)
            time.sleep(0.05)

    def collision_detection_callback(self,msg):
        with self.lock:
            self.collision = msg.data

    def robot_info_callback(self,msg):
        assert msg.robot_name.data in self.robot_info.keys(), "robot name in reach goal callback is wrong!"

        with self.lock:
            self.robot_info[msg.robot_name.data]["reach_goal"] = msg.reach_goal.data
            self.robot_info[msg.robot_name.data]["travel_time"] = msg.travel_time.data
            
            self.timestamp_data[msg.robot_name.data].append(msg.header)
            self.pose_data[msg.robot_name.data].append(msg.pose)
            self.velocity_data[msg.robot_name.data].append(msg.velocity)

    def experiment_monitoring(self):
        time.sleep(20.5)
        while rclpy.ok():
            with self.lock:
                if self.collision or self.check_all_reach_goals_or_over_time():
                    self.end_simulation()
            time.sleep(0.05)

    def check_all_reach_goals_or_over_time(self):
        for robot_name in self.robot_info.keys():
            if self.robot_info[robot_name]["travel_time"] > 90:
                self.exp_success = False
                return True
        
        for robot_name in self.robot_info.keys():
            if not self.robot_info[robot_name]["reach_goal"]:
                return False
        
        self.exp_success = True
        return True
    
    def end_simulation(self):
        self.successes_data[-1].append(self.exp_success)
        if self.exp_success:
            for robot_name in self.robot_info.keys():
                self.travel_times_data[-1].append(self.robot_info[robot_name]["travel_time"])

        print("\n\nShutdown simulation\n\n")
        _process = subprocess.Popen(['pkill', '-f', 'gz sim'])
        _process.communicate()
        rclpy.shutdown()


if __name__ == '__main__':
    
    method = "the/method/used/in/vrx/experiments"
    
    if method == "RL":
        agent_type = "RL/agent/used/in/vrx/experiments"
        model_path = "corresponding/torch/script/of/the/RL/model"
    elif method == "APF":
        agent_type = "APF"
        model_path = " "
    elif method == "MPC":
        agent_type = "MPC"
        model_path = " "
    else:
        raise RuntimeError("Agent type not implemented!")


    # vrx envrionment configuration file
    sdf_file_dir = "install/share/vrx_gz/worlds"
    
    input_world = "sydney_regatta_original"
    input_sdf_file = f"{sdf_file_dir}/{input_world}.sdf"
    
    output_world = "sydney_regatta"
    output_sdf_file = f"{sdf_file_dir}/{output_world}.sdf"


    init_poses_data = []
    goals_data = []
    buoy_poses_data = []
    robot_num_data = []
    buoy_num_data = []
    successes_data = []
    travel_times_data = []

    
    dt = datetime.now()
    timestamp = dt.strftime("%Y-%m-%d-%H-%M-%S")
    exp_result_file_dir = "vrx/experiments/results/save/directory"
    result_file_dir = os.path.join(exp_result_file_dir,f"{agent_type}/{timestamp}")
    os.makedirs(result_file_dir)


    run_with_exp_config = True
    if run_with_exp_config:
        ##### run an experiment with specified config ##### 
        
        print(f"\n\n\nRunning {agent_type} experiment with given config settings \n\n\n")
        
        exp_config_file = "path/to/vrx/experiment/episode/config/file"

        successes_data.append([])
        travel_times_data.append([])

        with open(exp_config_file,"r") as f:
            episode_setup = json.load(f)
        
        init_poses,goals,buoy_poses = read_exp_setup(episode_setup)

        add_buoy_to_sdf(input_sdf_file,output_sdf_file,buoy_poses)

        rclpy.init()

        exp_manager = ExperimentManager(len(episode_setup["robots"]["start"]),successes_data,travel_times_data,save_traj=True)
        exp_manager.launch_simulation(init_poses,goals,buoy_poses,method,agent_type,model_path,output_world)

        # save trajectory data
        timestamps = {}
        poses = {}
        velocities = {}
        for name in exp_manager.timestamp_data.keys():
            timestamps[name] = [round(t.stamp.sec+t.stamp.nanosec * 1e-9,2) for t in exp_manager.timestamp_data[name]]
            poses[name] = [[p.position.x,p.position.y,p.orientation.x,p.orientation.y,p.orientation.z,p.orientation.w] 
                           for p in exp_manager.pose_data[name]]
            velocities[name] = [[v.linear.x,v.linear.y] for v in exp_manager.velocity_data[name]]

            print(f"\ntimes: {len(timestamps[name])}, poses: {len(poses[name])}, velocities: {len(velocities[name])}\n")
        
        traj_data = dict(timestamps=timestamps,
                         poses=poses,
                         velocities=velocities)

        result_file = "vrx_exp_traj.json"
        with open(os.path.join(result_file_dir,result_file),"w") as file:
            json.dump(traj_data,file)
        
        config_file = "vrx_exp_config.json"
        with open(os.path.join(result_file_dir,config_file),"w") as file:
            json.dump(episode_setup,file)
    else:
        ##### run multiple experiments with randomly generated configs #####

        # vrx experiment settings file 
        # note: running over 10 episodes in one trial is not recommended
        #       since Gazebo is more likely to crash after frequent killing
        #       and relaunching simulations in this program   
        eval_schedules = dict(num_episodes=[10],
                            num_robots=[5],
                            num_cores=[0],
                            num_obstacles=[4],
                            min_start_goal_dis=[40.0]
                            )

        seed = 0
        test_env = marinenav_env.MarineNavEnv3(seed = seed)
        
        result_file = "vrx_exp_results.npz"

        for idx,count in enumerate(eval_schedules["num_episodes"]):
            init_poses_data.append([])
            goals_data.append([])
            buoy_poses_data.append([])
            robot_num_data.append(eval_schedules["num_robots"][idx])
            buoy_num_data.append(eval_schedules["num_obstacles"][idx])
            successes_data.append([])
            travel_times_data.append([])
            for i in range(count):
                print(f"\n\n\nRunning {agent_type} experiment {i} of schedule {idx}\n\n\n")

                init_poses,goals,buoy_poses = exp_setup(test_env,eval_schedules,idx)
                
                init_poses_data[-1].append(copy.deepcopy(init_poses))
                goals_data[-1].append(copy.deepcopy(goals))
                buoy_poses_data[-1].append(copy.deepcopy(buoy_poses))

                add_buoy_to_sdf(input_sdf_file,output_sdf_file,buoy_poses)

                rclpy.init()

                exp_manager = ExperimentManager(eval_schedules["num_robots"][idx],successes_data,travel_times_data)
                exp_manager.launch_simulation(init_poses,goals,buoy_poses,method,agent_type,model_path,output_world)

                successes_data = copy.deepcopy(exp_manager.successes_data)
                travel_times_data = copy.deepcopy(exp_manager.travel_times_data)

                print("\n\n\n====== Experiment Result ======")
                print("Number of robots: ",exp_manager.num_robots)
                print("Current episode success: ","True" if exp_manager.exp_success else "False")
                print("Current episode travel times:",exp_manager.robot_info)
                print("All episodes success rate: ",np.sum(successes_data[-1])/len(successes_data[-1]))
                print("All episodes avg travel time: ",np.mean(travel_times_data[-1]) if len(travel_times_data[-1])>0 else "NaN","\n\n\n")

                np.savez(os.path.join(result_file_dir,result_file),
                        agent_type=agent_type,
                        seed=seed,
                        init_poses=init_poses_data,
                        goals=goals_data,
                        buoy_poses=buoy_poses_data,
                        robot_num=robot_num_data,
                        buoy_num=buoy_num_data,
                        successes=successes_data,
                        travel_times=travel_times_data)

