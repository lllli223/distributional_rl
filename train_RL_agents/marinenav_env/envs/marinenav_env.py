import numpy as np
import scipy.spatial
import marinenav_env.envs.utils.robot as robot
import gym
import json
import copy

class Core:

    def __init__(self, x:float, y:float, clockwise:bool, Gamma:float):

        self.x = x  # x coordinate of the vortex core  
        self.y = y  # y coordinate of the vortex core
        self.clockwise = clockwise # if the vortex direction is clockwise
        self.Gamma = Gamma  # circulation strength of the vortex core

class Obstacle:

    def __init__(self, x:float, y:float, r:float):

        self.x = x # x coordinate of the obstacle center
        self.y = y # y coordinate of the obstacle center
        self.r = r # radius of the obstacle    

class MarineNavEnv3(gym.Env):

    def __init__(self, seed:int=0, schedule:dict=None, is_eval_env:bool=False):

        self.seed = seed
        self.rd = np.random.RandomState(seed) # PRNG
        self.is_eval_env = is_eval_env 
        
        # parameter initialization
        self.width = 55 # x coordinate dimension of the map
        self.height = 55 # y coordinate dimension of the map
        self.r = 0.5  # radius of vortex core
        self.v_rel_max = 1.0 # max allowable speed when two currents flowing towards each other
        self.p = 0.8 # max allowable relative speed at another vortex core
        self.v_range = [3,3] # speed range of the vortex (at the edge of core)
        self.obs_r_range = [1,1] # radius range of the obstacle 
        self.clear_r = 10.0 # radius of area centered at the start(goal) of each robot,
                           # where no vortex cores, static obstacles, or the start(goal) of other robots exist
        self.angular_speed_max = np.pi/2 # penalty will be applied if angular speed exceeds the threshold
        self.angular_speed_penalty = -1.0
        self.steering_reward_angle_max = np.pi/3 # reward will be given if the difference between velocity and steering angle is smaller that the threshold
        self.steering_reward_speed_min = 1.0
        self.steering_reward = 0.5
        self.timestep_penalty = -0.1
        self.COLREGs_penalty = -0.1
        self.collision_penalty = -5.0
        self.goal_reward = 10.0
        self.num_cores = 8 # number of vortices
        self.num_obs = 8 # number of static obstacles
        self.min_start_goal_dis = 30.0 # minimum distance between start and goal
        self.num_robots = 6 # number of robots

        self.robots = [] # list of robots
        for _ in range(self.num_robots):
            self.robots.append(robot.Robot(seed=self.rd.randint(0,5*self.num_robots)))
        
        self.cores = [] # list of vortex cores
        self.obstacles = [] # list of static obstacles

        self.schedule = schedule # schedule for curriculum learning
        self.episode_timesteps = 0 # current episode timesteps
        self.total_timesteps = 0 # learning timesteps

        self.observation_in_robot_frame = True # return observation in robot frame
    
    def get_action_space_dimension(self):
        return self.robots[0].compute_actions_dimension()

    def reset(self):
        # reset the environment

        if self.schedule is not None:
            steps = self.schedule["timesteps"]
            diffs = np.array(steps) - self.total_timesteps
            
            # find the interval the current timestep falls into
            idx = len(diffs[diffs<=0])-1

            self.num_robots = self.schedule["num_robots"][idx]
            assert self.num_robots > 0, "Number of robots is 0!"

            self.num_cores = self.schedule["num_cores"][idx]
            self.num_obs = self.schedule["num_obstacles"][idx]
            self.min_start_goal_dis = self.schedule["min_start_goal_dis"][idx]

            print("\n======== training schedule ========")
            print("num of robots: ",self.num_robots)
            print("num of cores: ",self.num_cores)
            print("num of obstacles: ",self.num_obs)
            print("min start goal dis: ",self.min_start_goal_dis)
            print("======== training schedule ========\n") 
        
        self.episode_timesteps = 0

        self.cores.clear()
        self.obstacles.clear()
        self.robots.clear()

        num_cores = self.num_cores
        num_obs = self.num_obs

        ##### generate robots with randomly generated start and goal 
        num_robots = 0
        iteration = 500
        while True:
            start = self.rd.uniform(low = 2.0*np.ones(2), high = np.array([self.width-2.0,self.height-2.0]))
            goal = self.rd.uniform(low = 2.0*np.ones(2), high = np.array([self.width-2.0,self.height-2.0]))
            iteration -= 1
            if self.check_start_and_goal(start,goal):
                rob = robot.Robot(seed=self.rd.randint(0,5*self.num_robots))
                rob.start = start
                rob.goal = goal
                self.reset_robot(rob)
                self.robots.append(rob)
                num_robots += 1
            if iteration == 0 or num_robots == self.num_robots:
                break

        ##### generate vortex with random position, spinning direction and strength
        if num_cores > 0:
            iteration = 500
            while True:
                center = self.rd.uniform(low = np.zeros(2), high = np.array([self.width,self.height]))
                direction = self.rd.binomial(1,0.5)
                v_edge = self.rd.uniform(low = self.v_range[0], high = self.v_range[1])
                Gamma = 2 * np.pi * self.r * v_edge
                core = Core(center[0],center[1],direction,Gamma)
                iteration -= 1
                if self.check_core(core):
                    self.cores.append(core)
                    num_cores -= 1
                if iteration == 0 or num_cores == 0:
                    break
        
        centers = None
        for core in self.cores:
            if centers is None:
                centers = np.array([[core.x,core.y]])
            else:
                c = np.array([[core.x,core.y]])
                centers = np.vstack((centers,c))
        
        # KDTree storing vortex core center positions
        if centers is not None:
            self.core_centers = scipy.spatial.KDTree(centers)

        ##### generate static obstacles with random position and size
        if num_obs > 0:
            iteration = 500
            while True:
                center = self.rd.uniform(low = 5.0*np.ones(2), high = np.array([self.width-5.0,self.height-5.0]))
                r = self.rd.uniform(low = self.obs_r_range[0], high = self.obs_r_range[1])
                obs = Obstacle(center[0],center[1],r)
                iteration -= 1
                if self.check_obstacle(obs):
                    self.obstacles.append(obs)
                    num_obs -= 1
                if iteration == 0 or num_obs == 0:
                    break
        
        return self.get_observations()

    def reset_robot(self,rob):
        # reset robot state
        rob.reach_goal = False
        rob.collision = False
        rob.deactivated = False
        rob.init_theta = self.rd.uniform(low = 0.0, high = 2*np.pi)
        rob.init_velocity_r = np.array([0.0,0.0,0.0])
        rob.init_pos = 0.0
        rob.init_thrust = 0.0
        current_v = self.get_velocity(rob.start[0],rob.start[1])
        rob.reset_state(current_velocity=current_v)

    def check_all_deactivated(self):
        res = True
        for rob in self.robots:
            if not rob.deactivated:
                res = False
                break
        return res

    def check_all_reach_goal(self):
        res = True
        for rob in self.robots:
            if not rob.reach_goal:
                res = False
                break
        return res
    
    def check_any_collision(self):
        res = False
        for rob in self.robots:
            if rob.collision:
                res = True
                break
        return res

    # def compute_angular_penalty(self, rob):
    #     abs_angular_speed = np.abs(rob.velocity[2])

    #     penalty = 0.0
    #     if abs_angular_speed > self.angular_speed_max:
    #         penalty += self.angular_speed_penalty

    #     return penalty
    
    # def compute_steering_reward(self, rob):
    #     abs_velocity_r = rob.project_to_robot_frame(rob.velocity[:2])
    #     # v_angle = np.abs(np.arctan2(abs_velocity_r[1],abs_velocity_r[0]))

    #     reward = 0.0
    #     # steer_well = False
    #     # if v_angle <= self.steering_reward_angle_max and abs_velocity_r[0] >= self.steering_reward_speed_min:
    #     if abs_velocity_r[0] >= self.steering_reward_speed_min:
    #         reward += self.steering_reward
    #         # steer_well = True

    #     return reward

    # def compute_danger_penalty(self, rob):
    #     penalty = 0.0
        
    #     if rob.closest_dist_to_obj < rob.safe_dis:
    #         penalty += self.danger_penalty * (rob.safe_dis - rob.closest_dist_to_obj)

    #     return penalty 

    def compute_COLREGs_penalty(self,rob):
        penalty = 0.0
        
        if rob.apply_COLREGs:
            penalty += self.COLREGs_penalty * rob.phi

        return penalty

    def step(self, actions, is_continuous_action=True):

        rewards = [0]*len(self.robots)

        assert len(actions) == len(self.robots), "Number of actions not equal number of robots!"
        assert self.check_all_reach_goal() is not True, "All robots reach goals, not actions are available!"
        # Execute actions for all robots
        for i,action in enumerate(actions):
            rob = self.robots[i]

            if rob.deactivated:
                # This robot is in the deactivate state
                continue

            dis_before = rob.dist_to_goal()

            # update robot state after executing the action    
            for idx in range(rob.N):
                current_velocity = self.get_velocity(rob.x, rob.y)
                is_new_action = True if idx == 0 else False
                rob.update_state(action,current_velocity,is_new_action,is_continuous_action)
            
            if self.is_eval_env:
                # save action to history
                rob.action_history.append(action)
                
                # save robot state
                rob.trajectory.append([rob.x,rob.y,rob.theta,rob.velocity_r[0],rob.velocity_r[1],rob.velocity_r[2], \
                                      rob.velocity[0],rob.velocity[1],rob.velocity[2],rob.left_pos,rob.right_pos, \
                                      rob.left_thrust,rob.right_thrust])

            dis_after = rob.dist_to_goal()

            # constant penalty applied at every time step
            rewards[i] += self.timestep_penalty

            # reward agent for getting closer to the goal
            rewards[i] += dis_before - dis_after

            # penalize agent for having large angular speed
            # rewards[i] += self.compute_angular_penalty(rob)

            # reward agent for moving facing forward
            # steering_reward = self.compute_steering_reward(rob)
            # rewards[i] += steering_reward    
        

        # Get observation for all robots
        observations, collisions, reach_goals = self.get_observations()

        dones = [False]*len(self.robots)
        infos = [{"state":"normal"}]*len(self.robots)
        
        # end_episode = False
        for idx,rob in enumerate(self.robots):
            if rob.deactivated:
                # This robot is in the deactivate state
                dones[idx] = True
                if rob.collision:
                    infos[idx] = {"state":"deactivated after collision"}
                elif rob.reach_goal:
                    infos[idx] = {"state":"deactivated after reaching goal"}
                else:
                    raise RuntimeError("Robot being deactived can only be caused by collsion or reaching goal!")
                continue
            
            # penalize agent for getting close to other objects
            # rewards[idx] += self.compute_danger_penalty(rob)

            # penalize agent for not following COLREGs
            rewards[idx] += self.compute_COLREGs_penalty(rob)

            if self.episode_timesteps >= 1000:
                # end_episode = True
                dones[idx] = True
                infos[idx] = {"state":"too long episode"}
            elif collisions[idx]:
                rewards[idx] += self.collision_penalty
                # end_episode = True
                dones[idx] = True
                infos[idx] = {"state":"collision"}
            elif reach_goals[idx]:
                # if steer_well:
                rewards[idx] += self.goal_reward
                dones[idx] = True
                infos[idx] = {"state":"reach goal"}
            else:
                dones[idx] = False
                infos[idx] = {"state":"normal"}

        self.episode_timesteps += 1
        self.total_timesteps += 1

        return observations, rewards, dones, infos

    def out_of_boundary(self):
        # only used when boundary is set
        x_out = self.robot.x < 0.0 or self.robot.x > self.width
        y_out = self.robot.y < 0.0 or self.robot.y > self.height
        return x_out or y_out

    def get_observations(self):
        observations = []
        collisions = []
        reach_goals = []
        for robot in self.robots:
            observation, collision, reach_goal = robot.perception_output(self.obstacles,self.robots,self.observation_in_robot_frame)
            
            if self.is_eval_env:
                # save observation to history
                self_obs, objects_obs = observation
                robot.observation_history.append([self_obs,objects_obs])
            
            observations.append(observation)
            collisions.append(collision)
            reach_goals.append(reach_goal)
        return observations, collisions, reach_goals
    
    def check_start_and_goal(self,start,goal):
        
        # The start and goal point is far enough
        if np.linalg.norm(goal-start) < self.min_start_goal_dis:
            return False
        
        for robot in self.robots:
            
            dis_s = robot.start - start
            # Start point not too close to that of existing robots
            if np.linalg.norm(dis_s) <= self.clear_r:
                return False
            
            dis_g = robot.goal - goal
            # Goal point not too close to that of existing robots
            if np.linalg.norm(dis_g) <= self.clear_r:
                return False
        
        return True
    
    def check_core(self,core_j):

        # Within the range of the map
        if core_j.x - self.r < 0.0 or core_j.x + self.r > self.width:
            return False
        if core_j.y - self.r < 0.0 or core_j.y + self.r > self.width:
            return False

        for robot in self.robots:
            # Not too close to start and goal point of each robot
            core_pos = np.array([core_j.x,core_j.y])
            dis_s = core_pos - robot.start
            if np.linalg.norm(dis_s) < self.r + self.clear_r:
                return False
            dis_g = core_pos - robot.goal
            if np.linalg.norm(dis_g) < self.r + self.clear_r:
                return False

        for core_i in self.cores:
            dx = core_i.x - core_j.x
            dy = core_i.y - core_j.y
            dis = np.sqrt(dx*dx+dy*dy)

            if core_i.clockwise == core_j.clockwise:
                # i and j rotate in the same direction, their currents run towards each other at boundary
                # The currents speed at boundary need to be lower than threshold  
                boundary_i = core_i.Gamma / (2*np.pi*self.v_rel_max)
                boundary_j = core_j.Gamma / (2*np.pi*self.v_rel_max)
                if dis < boundary_i + boundary_j:
                    return False
            else:
                # i and j rotate in the opposite direction, their currents join at boundary
                # The relative current speed of the stronger vortex at boundary need to be lower than threshold 
                Gamma_l = max(core_i.Gamma, core_j.Gamma)
                Gamma_s = min(core_i.Gamma, core_j.Gamma)
                v_1 = Gamma_l / (2*np.pi*(dis-2*self.r))
                v_2 = Gamma_s / (2*np.pi*self.r)
                if v_1 > self.p * v_2:
                    return False

        return True

    def check_obstacle(self,obs):

        # Within the range of the map
        if obs.x - obs.r < 0.0 or obs.x + obs.r > self.width:
            return False
        if obs.y - obs.r < 0.0 or obs.y + obs.r > self.height:
            return False

        for robot in self.robots:
            # Not too close to start and goal point
            obs_pos = np.array([obs.x,obs.y])
            dis_s = obs_pos - robot.start
            if np.linalg.norm(dis_s) < obs.r + self.clear_r:
                return False
            dis_g = obs_pos - robot.goal
            if np.linalg.norm(dis_g) < obs.r + self.clear_r:
                return False

        # Not too close to vortex cores
        for core in self.cores:
            dx = core.x - obs.x
            dy = core.y - obs.y
            dis = np.sqrt(dx*dx + dy*dy)

            if dis <= self.r + obs.r:
                return False
        
        # Not collide with other obstacles
        for obstacle in self.obstacles:
            dx = obstacle.x - obs.x
            dy = obstacle.y - obs.y
            dis = np.sqrt(dx*dx + dy*dy)

            if dis <= obstacle.r + obs.r:
                return False
        
        return True

    def get_velocity(self,x:float, y:float):
        if len(self.cores) == 0:
            return np.zeros(3)
        
        # sort the vortices according to their distance to the query point
        d, idx = self.core_centers.query(np.array([x,y]),k=len(self.cores))
        if isinstance(idx,np.int64):
            idx = [idx]

        v_radial_set = []
        v_velocity = np.zeros((2,1))
        for i in list(idx): 
            core = self.cores[i]
            v_radial = np.matrix([[core.x-x],[core.y-y]])

            for v in v_radial_set:
                project = np.transpose(v) * v_radial
                if project[0,0] > 0:
                    # if the core is in the outter area of a checked core (wrt the query position),
                    # assume that it has no influence the velocity of the query position
                    continue
            
            v_radial_set.append(v_radial)
            dis = np.linalg.norm(v_radial)
            v_radial /= dis
            if core.clockwise:
                rotation = np.matrix([[0., -1.],[1., 0]])
            else:
                rotation = np.matrix([[0., 1.],[-1., 0]])
            v_tangent = rotation * v_radial
            speed = self.compute_speed(core.Gamma,dis)
            v_velocity += v_tangent * speed
        
        return np.array([v_velocity[0,0], v_velocity[1,0], 0.0])

    def get_velocity_test(self,x:float, y:float):
        v = np.ones(2)
        return v / np.linalg.norm(v)

    def compute_speed(self, Gamma:float, d:float):
        if d <= self.r:
            return Gamma / (2*np.pi*self.r*self.r) * d
        else:
            return Gamma / (2*np.pi*d)

    def reset_with_eval_config(self,eval_config):
        self.episode_timesteps = 0
        
        # load env config
        self.seed = eval_config["env"]["seed"]
        self.rd = np.random.RandomState(self.seed)
        self.is_eval_env = eval_config["env"]["is_eval_env"]
        self.width = eval_config["env"]["width"]
        self.height = eval_config["env"]["height"]
        self.r = eval_config["env"]["r"]
        self.v_rel_max = eval_config["env"]["v_rel_max"]
        self.p = eval_config["env"]["p"]
        self.v_range = copy.deepcopy(eval_config["env"]["v_range"])
        self.obs_r_range = copy.deepcopy(eval_config["env"]["obs_r_range"])
        self.clear_r = eval_config["env"]["clear_r"]
        self.angular_speed_max = eval_config["env"]["angular_speed_max"]
        self.angular_speed_penalty = eval_config["env"]["angular_speed_penalty"]
        self.timestep_penalty = eval_config["env"]["timestep_penalty"]
        self.COLREGs_penalty = eval_config["env"]["COLREGs_penalty"]
        self.collision_penalty = eval_config["env"]["collision_penalty"]
        self.goal_reward = eval_config["env"]["goal_reward"]

        # load vortex cores
        self.cores.clear()
        centers = None
        for i in range(len(eval_config["env"]["cores"]["positions"])):
            center = eval_config["env"]["cores"]["positions"][i]
            clockwise = eval_config["env"]["cores"]["clockwise"][i]
            Gamma = eval_config["env"]["cores"]["Gamma"][i]
            core = Core(center[0],center[1],clockwise,Gamma)
            self.cores.append(core)
            if centers is None:
                centers = np.array([[core.x,core.y]])
            else:
                c = np.array([[core.x,core.y]])
                centers = np.vstack((centers,c))
        
        if centers is not None:
            self.core_centers = scipy.spatial.KDTree(centers)

        # load obstacles
        self.obstacles.clear()
        for i in range(len(eval_config["env"]["obstacles"]["positions"])):
            center = eval_config["env"]["obstacles"]["positions"][i]
            r = eval_config["env"]["obstacles"]["r"][i]
            obs = Obstacle(center[0],center[1],r)
            self.obstacles.append(obs)

        # load robot config
        self.robots.clear()
        for i in range(eval_config["env"]["num_robots"]):
            rob = robot.Robot(seed=self.rd.randint(0,5*self.num_robots))
            rob.dt = eval_config["robots"]["dt"][i]
            rob.N = eval_config["robots"]["N"][i]
            rob.length = eval_config["robots"]["length"][i]
            rob.width = eval_config["robots"]["width"][i]
            rob.detect_r = eval_config["robots"]["detect_r"][i]
            rob.r = eval_config["robots"]["r"][i]
            rob.goal_dis = eval_config["robots"]["goal_dis"][i]
            rob.power_coefficient = eval_config["robots"]["power_coefficient"][i]
            rob.min_thrust = eval_config["robots"]["min_thrust"][i]
            rob.max_thrust = eval_config["robots"]["max_thrust"][i]
            rob.left_thrust_change = np.array(eval_config["robots"]["left_thrust_change"][i])
            rob.right_thrust_change = np.array(eval_config["robots"]["right_thrust_change"][i])
            rob.compute_actions()

            rob.m = eval_config["robots"]["m"][i]
            rob.Izz = eval_config["robots"]["Izz"][i]
            rob.xDotU = eval_config["robots"]["xDotU"][i]
            rob.yDotV = eval_config["robots"]["yDotV"][i]
            rob.yDotR = eval_config["robots"]["yDotR"][i]
            rob.nDotR = eval_config["robots"]["nDotR"][i]
            rob.nDotV = eval_config["robots"]["nDotV"][i]
            rob.xU = eval_config["robots"]["xU"][i]
            rob.xUU = eval_config["robots"]["xUU"][i]
            rob.yV = eval_config["robots"]["yV"][i]
            rob.yVV = eval_config["robots"]["yVV"][i]
            rob.yR = eval_config["robots"]["yR"][i]
            rob.yRV = eval_config["robots"]["yRV"][i]
            rob.yVR = eval_config["robots"]["yVR"][i]
            rob.yRR = eval_config["robots"]["yRR"][i]
            rob.nR = eval_config["robots"]["nR"][i]
            rob.nRR = eval_config["robots"]["nRR"][i]
            rob.nV = eval_config["robots"]["nV"][i]
            rob.nVV = eval_config["robots"]["nVV"][i]
            rob.nRV = eval_config["robots"]["nRV"][i]
            rob.nVR = eval_config["robots"]["nVR"][i]
            rob.compute_constant_matrices()

            rob.start = np.array(eval_config["robots"]["start"][i])
            rob.goal = np.array(eval_config["robots"]["goal"][i])
            rob.init_theta = eval_config["robots"]["init_theta"][i]
            rob.init_velocity_r = np.array(eval_config["robots"]["init_velocity_r"][i])
            rob.init_left_pos = eval_config["robots"]["init_left_pos"][i]
            rob.init_right_pos = eval_config["robots"]["init_right_pos"][i]
            rob.init_left_thrust = eval_config["robots"]["init_left_thrust"][i]
            rob.init_right_thrust = eval_config["robots"]["init_right_thrust"][i]

            rob.perception.range = eval_config["robots"]["perception"]["range"][i]
            rob.perception.angle = eval_config["robots"]["perception"]["angle"][i]
            rob.perception.max_obj_num = eval_config["robots"]["perception"]["max_obj_num"][i]
            rob.perception.pos_std = eval_config["robots"]["perception"]["pos_std"][i]
            rob.perception.vel_std = eval_config["robots"]["perception"]["vel_std"][i]
            rob.perception.r_kappa = eval_config["robots"]["perception"]["r_kappa"][i]
            rob.perception.r_mean_ratio = eval_config["robots"]["perception"]["r_mean_ratio"][i]

            current_v = self.get_velocity(rob.start[0],rob.start[1])
            rob.reset_state(current_velocity=current_v)
            
            self.robots.append(rob)

        return self.get_observations()          

    def episode_data(self):
        episode = {}

        # save environment config
        episode["env"] = {}
        episode["env"]["seed"] = self.seed
        episode["env"]["is_eval_env"] = self.is_eval_env
        episode["env"]["width"] = self.width
        episode["env"]["height"] = self.height
        episode["env"]["r"] = self.r
        episode["env"]["v_rel_max"] = self.v_rel_max
        episode["env"]["p"] = self.p
        episode["env"]["v_range"] = copy.deepcopy(self.v_range) 
        episode["env"]["obs_r_range"] = copy.deepcopy(self.obs_r_range)
        episode["env"]["clear_r"] = self.clear_r
        episode["env"]["angular_speed_max"] = self.angular_speed_max
        episode["env"]["angular_speed_penalty"] = self.angular_speed_penalty
        episode["env"]["timestep_penalty"] = self.timestep_penalty
        episode["env"]["COLREGs_penalty"] = self.COLREGs_penalty
        episode["env"]["collision_penalty"] = self.collision_penalty
        episode["env"]["goal_reward"] = self.goal_reward
        episode["env"]["num_robots"] = self.num_robots

        # save vortex cores information
        episode["env"]["cores"] = {}
        episode["env"]["cores"]["positions"] = []
        episode["env"]["cores"]["clockwise"] = []
        episode["env"]["cores"]["Gamma"] = []
        for core in self.cores:
            episode["env"]["cores"]["positions"].append([core.x,core.y])
            episode["env"]["cores"]["clockwise"].append(core.clockwise)
            episode["env"]["cores"]["Gamma"].append(core.Gamma)

        # save obstacles information
        episode["env"]["obstacles"] = {}
        episode["env"]["obstacles"]["positions"] = []
        episode["env"]["obstacles"]["r"] = []
        for obs in self.obstacles:
            episode["env"]["obstacles"]["positions"].append([obs.x,obs.y])
            episode["env"]["obstacles"]["r"].append(obs.r)

        # save robots information
        episode["robots"] = {}
        episode["robots"]["dt"] = []
        episode["robots"]["N"] = []
        episode["robots"]["length"] = []
        episode["robots"]["width"] = []
        episode["robots"]["detect_r"] = []
        episode["robots"]["r"] = []
        episode["robots"]["goal_dis"] = []
        episode["robots"]["power_coefficient"] = []
        episode["robots"]["min_thrust"] = []
        episode["robots"]["max_thrust"] = []
        episode["robots"]["left_thrust_change"] = []
        episode["robots"]["right_thrust_change"] = []
        
        episode["robots"]["m"] = []
        episode["robots"]["Izz"] = []
        episode["robots"]["xDotU"] = []
        episode["robots"]["yDotV"] = []
        episode["robots"]["yDotR"] = []
        episode["robots"]["nDotR"] = []
        episode["robots"]["nDotV"] = []
        episode["robots"]["xU"] = []
        episode["robots"]["xUU"] = []
        episode["robots"]["yV"] = []
        episode["robots"]["yVV"] = []
        episode["robots"]["yR"] = []
        episode["robots"]["yRV"] = []
        episode["robots"]["yVR"] = []
        episode["robots"]["yRR"] = []
        episode["robots"]["nR"] = []
        episode["robots"]["nRR"] = []
        episode["robots"]["nV"] = []
        episode["robots"]["nVV"] = []
        episode["robots"]["nRV"] = []
        episode["robots"]["nVR"] = []
        
        episode["robots"]["start"] = []
        episode["robots"]["goal"] = []
        episode["robots"]["init_theta"] = []
        episode["robots"]["init_velocity_r"] = []
        episode["robots"]["init_left_pos"] = []
        episode["robots"]["init_right_pos"] = []
        episode["robots"]["init_left_thrust"] = []
        episode["robots"]["init_right_thrust"] = []

        episode["robots"]["perception"] = {}
        episode["robots"]["perception"]["range"] = []
        episode["robots"]["perception"]["angle"] = []
        episode["robots"]["perception"]["max_obj_num"] = []
        episode["robots"]["perception"]["pos_std"] = []
        episode["robots"]["perception"]["vel_std"] = []
        episode["robots"]["perception"]["r_kappa"] = []
        episode["robots"]["perception"]["r_mean_ratio"] = []

        episode["robots"]["observation_history"] = []
        episode["robots"]["action_history"] = []
        episode["robots"]["trajectory"] = []

        for rob in self.robots:
            episode["robots"]["dt"].append(rob.dt)
            episode["robots"]["N"].append(rob.N)
            episode["robots"]["length"].append(rob.length)
            episode["robots"]["width"].append(rob.width)
            episode["robots"]["detect_r"].append(rob.detect_r)
            episode["robots"]["r"].append(rob.r)
            episode["robots"]["goal_dis"].append(rob.goal_dis)
            episode["robots"]["power_coefficient"].append(rob.power_coefficient)
            episode["robots"]["min_thrust"].append(rob.min_thrust)
            episode["robots"]["max_thrust"].append(rob.max_thrust)
            episode["robots"]["left_thrust_change"].append(list(rob.left_thrust_change))
            episode["robots"]["right_thrust_change"].append(list(rob.right_thrust_change))
            
            episode["robots"]["m"].append(rob.m)
            episode["robots"]["Izz"].append(rob.Izz)
            episode["robots"]["xDotU"].append(rob.xDotU)
            episode["robots"]["yDotV"].append(rob.yDotV)
            episode["robots"]["yDotR"].append(rob.yDotR)
            episode["robots"]["nDotR"].append(rob.nDotR)
            episode["robots"]["nDotV"].append(rob.nDotV)
            episode["robots"]["xU"].append(rob.xU)
            episode["robots"]["xUU"].append(rob.xUU)
            episode["robots"]["yV"].append(rob.yV)
            episode["robots"]["yVV"].append(rob.yVV)
            episode["robots"]["yR"].append(rob.yR)
            episode["robots"]["yRV"].append(rob.yRV)
            episode["robots"]["yVR"].append(rob.yVR)
            episode["robots"]["yRR"].append(rob.yRR)
            episode["robots"]["nR"].append(rob.nR)
            episode["robots"]["nRR"].append(rob.nRR)
            episode["robots"]["nV"].append(rob.nV)
            episode["robots"]["nVV"].append(rob.nVV)
            episode["robots"]["nRV"].append(rob.nRV)
            episode["robots"]["nVR"].append(rob.nVR)
            
            episode["robots"]["start"].append(list(rob.start))
            episode["robots"]["goal"].append(list(rob.goal))
            episode["robots"]["init_theta"].append(rob.init_theta)
            episode["robots"]["init_velocity_r"].append(list(rob.init_velocity_r))
            episode["robots"]["init_left_pos"].append(rob.init_left_pos)
            episode["robots"]["init_right_pos"].append(rob.init_right_pos)
            episode["robots"]["init_left_thrust"].append(rob.init_left_thrust)
            episode["robots"]["init_right_thrust"].append(rob.init_right_thrust)

            episode["robots"]["perception"]["range"].append(rob.perception.range)
            episode["robots"]["perception"]["angle"].append(rob.perception.angle)
            episode["robots"]["perception"]["max_obj_num"].append(rob.perception.max_obj_num)
            episode["robots"]["perception"]["pos_std"].append(rob.perception.pos_std)
            episode["robots"]["perception"]["vel_std"].append(rob.perception.vel_std)
            episode["robots"]["perception"]["r_kappa"].append(rob.perception.r_kappa)
            episode["robots"]["perception"]["r_mean_ratio"].append(rob.perception.r_mean_ratio)

            episode["robots"]["observation_history"].append(copy.deepcopy(rob.observation_history))
            episode["robots"]["action_history"].append(copy.deepcopy(rob.action_history))
            episode["robots"]["trajectory"].append(copy.deepcopy(rob.trajectory))

        return episode

    def save_episode(self,filename):
        episode = self.episode_data()
        with open(filename,"w") as file:
            json.dump(episode,file)
