import numpy as np
import copy

class Perception:

    def __init__(self,seed:int=0):
        self.seed = seed
        self.rd = np.random.RandomState(seed) # PRNG
        
        # 2D LiDAR model with detection area as a sector
        self.range = 20.0 # range of beams (meter)
        self.angle = 2 * np.pi # detection angle range
        self.max_obj_num = 5 # the maximum number of objects to be considered
        self.observation = dict(self=[],objects=[]) # format: {"self": [goal,velocity(linear and angular)], 
                                                    #          "objects":[[px_1,py_1,vx_1,vy_1,r_1],...
                                                    #                     [px_n,py_n,vx_n,vy_n,r_n]]
        self.observed_obs = [] # indices of observed static obstacles
        self.observed_objs = [] # indiced of observed dynamic objects

        # perception noise
        self.pos_std = 0.05 # position
        self.vel_std = 0.05 # velocity
        self.r_kappa = 1.0 # radius
        self.r_mean_ratio = 0.8 # percentage of radius that is the mean of noisy observation

    def pos_observation(self,px,py):
        obs_px = px + self.rd.normal(0,self.pos_std)
        obs_py = py + self.rd.normal(0,self.pos_std)
        return obs_px, obs_py
    
    def vel_observation(self,vx,vy):
        obs_vx = vx + self.rd.normal(0,self.vel_std)
        obs_vy = vy + self.rd.normal(0,self.vel_std)
        return obs_vx, obs_vy
    
    def r_observation(self,r):
        r_mean = self.r_mean_ratio * r
        r_noise = (1-self.r_mean_ratio) * self.rd.vonmises(0,self.r_kappa)/np.pi * r
        return (r_mean + r_noise)


class Robot:

    def __init__(self,seed:int=0):
        self.dt = 0.05 # discretized time step (second)
        self.N = 10 # number of time step per action
        self.perception = Perception(seed)
        
        # WAM-V 16 simulation model
        self.length = 5.0 
        self.width = 2.5
        self.detect_r = 0.5*np.sqrt(self.length**2+self.width**2) # detection range
        self.r = self.detect_r # collision range
        self.hull_width = 0.2 * self.width
        self.hull_tip_length = 0.25 * self.length
        self.hull_tip_width = 0.5 * self.hull_width
        self.hull_rear_length = 0.2 * self.length
        self.hull_rear_width = 0.8 * self.hull_width
        self.thruster_gap = 0.02 * self.length
        self.thruster_length = 0.1 * self.length
        self.thruster_tip_width = 0.6 * self.hull_rear_width
        self.thruster_rear_width = 0.75 * self.hull_rear_width
        self.beam_length = self.width - self.hull_width
        self.beam_width = 0.2 * self.hull_width
        self.beam_distance = 0.3 * self.length
        self.beam_base_length = 0.5 * self.length
        self.beam_base_width = 0.5 * self.hull_width
        self.platform_length = 0.4 * self.length
        self.platform_width = 0.45 * self.width 

        # COLREGs zone dimension (for the closest vehicle)
        self.head_on_zone_x_dim = 17.0
        self.head_on_zone_y_dim = 9.0
        self.left_crossing_zone_x_dim = np.array([-9.0,12.0])
        self.left_crossing_zone_y_dim_front = np.array([-17.0,-7.0]) 
        
        self.safe_dis = 10.0 # min distance to other objects to be considered safe
        self.goal_dis = 2.0 # max distance to goal considered as reached
        self.goal_angluar_speed = np.pi/12 # max angular speed at goal to be considered as reach 
        self.max_angular_speed = np.pi/3 # max angular speed allowed 

        self.power_coefficient = 1.0 # assume linear relation between power and thrust
        self.min_thrust = -500.0 # min thrust force
        self.max_thrust = 1000.0 # max thrust force
        self.left_thrust_change = np.array([0.0,-500.0,-1000.0,500.0,1000.0]) # For IQN: left thrust force change per second (action 1)
        self.right_thrust_change = np.array([0.0,-500.0,-1000.0,500.0,1000.0]) # For IQN: right thrust force change per second (action 2)
        self.compute_actions() # list of actions

        # x-y-z (+): Forward-Starboard-Down (robot frame), North-East-Down (world frame) 
        # yaw (+): clockwise
        self.x = None # x coordinate
        self.y = None # y coordinate
        self.theta = None # yaw angle
        self.velocity_r = None # velocity wrt to current in world frame
        self.velocity = None # velocity wrt sea floor in world frame
        
        self.left_pos = None # left thruster angle (rad)
        self.right_pos = None # right thruster angle (rad)
        self.left_thrust = None # left thruster force (N)
        self.right_thrust = None # right thruster force (N)

        self.m = 400 # WAM-V weight when fully loaded (kg)
        self.Izz = 450 # moment of inertia Izz
        
        # hydrodynamic derivatives
        self.xDotU = 20
        self.yDotV = 0
        self.yDotR = 0
        self.nDotR = -980
        self.nDotV = 0
        self.xU = -100
        self.xUU = -150
        self.yV = -100
        self.yVV = -150
        self.yR = 0
        self.yRV = 0
        self.yVR = 0
        self.yRR = 0
        self.nR = -980
        self.nRR = -950
        self.nV = 0
        self.nVV = 0
        self.nRV = 0
        self.nVR = 0
        self.compute_constant_matrices() # ship maneuvering model matrices that are constant

        self.start = None # start position
        self.goal = None # goal position
        self.collision = False
        self.reach_goal = False
        self.deactivated = False # deactivate the robot if it collides with any objects or reaches the goal

        self.init_theta = 0.0 # theta at initial position
        self.init_velocity_r = np.array([0.0,0.0,0.0]) # relative velocity at initial position
        
        self.init_left_pos = 0.0 # left thruster angle at initial position
        self.init_right_pos = 0.0 # right thruster angle at initial position
        self.init_left_thrust = 0.0 # left thrust at initial position
        self.init_right_thrust = 0.0 # right thrust at initial position

        self.observation_history = [] # history of noisy observations in one episode
        self.action_history = [] # history of action commands in one episode
        self.trajectory = [] # trajectory in one episode 
        self.apply_COLREGs = False

    def compute_actions(self):
        self.actions = [(l,r) for l in self.left_thrust_change for r in self.right_thrust_change]

    def compute_actions_dimension(self):
        return len(self.actions)
    
    def compute_constant_matrices(self):
        self.M_RB = np.array([[self.m, 0.0, 0.0],
                              [0.0, self.m, 0.0],
                              [0.0, 0.0, self.Izz]], dtype=float)

        self.M_A = -1.0 * np.array([[self.xDotU, 0.0, 0.0],
                                    [0.0, self.yDotV, self.yDotR],
                                    [0.0, self.nDotV, self.nDotR]], dtype=float)

        self.D = -1.0 * np.array([[self.xU, 0.0, 0.0],
                                  [0.0, self.yV, self.yR],
                                  [0.0, self.nV, self.nR]], dtype=float)
        
        self.A_const = self.M_RB + self.M_A
        try:
            self._A_chol = np.linalg.cholesky(self.A_const)
        except np.linalg.LinAlgError:
            self._A_chol = None

    def compute_step_energy_cost(self):
        # TODO: Revise energy computation
        l = self.power_coefficient * np.abs(self.left_thrust) * self.dt * self.N
        r = self.power_coefficient * np.abs(self.right_thrust) * self.dt * self.N
        return (l+r)
    
    def dist_to_goal(self):
        return np.linalg.norm(self.goal - np.array([self.x,self.y]))

    def check_reach_goal(self):
        if self.dist_to_goal() <= self.goal_dis:
            self.reach_goal = True

    def check_over_spin(self):
        return (np.abs(self.velocity[2]) > self.max_angular_speed)

    def reset_state(self,current_velocity=np.zeros(3)):
        # only called when resetting the environment
        self.observation_history.clear()
        self.action_history.clear()
        self.trajectory.clear()
        self.x = self.start[0]
        self.y = self.start[1]
        self.theta = self.init_theta 
        self.velocity_r = self.init_velocity_r
        self.update_velocity(current_velocity)
        self.left_pos = self.init_left_pos
        self.right_pos = self.init_right_pos
        self.left_thrust = self.init_left_thrust
        self.right_thrust = self.init_right_thrust
        self.trajectory.append([self.x,self.y,self.theta,self.velocity_r[0],self.velocity_r[1],self.velocity_r[2], \
                                self.velocity[0],self.velocity[1],self.velocity[2],self.left_pos,self.right_pos, \
                                self.left_thrust,self.right_thrust])

    def get_robot_transform(self):
        # compute rotation (robot -> world) and translation in world frame using ndarrays
        R_wr = np.array([[np.cos(self.theta), -np.sin(self.theta)],
                         [np.sin(self.theta),  np.cos(self.theta)]], dtype=float)
        t_wr = np.array([self.x, self.y], dtype=float)
        return R_wr, t_wr

    def update_velocity(self,current_velocity=np.zeros(3)):
        self.velocity = self.velocity_r + current_velocity

    def update_state(self,action,current_velocity=np.zeros(3),is_new_action=False,is_continuous_action=True):
        # update robot pose in one time step
        self.update_velocity(current_velocity)
        dis = self.velocity * self.dt
        self.x += dis[0]
        self.y += dis[1]
        self.theta += dis[2]

        # wrap theta to [0,2*pi)
        while self.theta < 0.0:
            self.theta += 2 * np.pi
        while self.theta >= 2 * np.pi:
            self.theta -= 2 * np.pi

        if is_new_action:
            # update thruster thrust
            if is_continuous_action:
                # actor output value in (-1.0,1.0), map to (-1000.0,1000.0)
                l = action[0] * 1000.0
                r = action[1] * 1000.0
            else:
                l,r = self.actions[action]
            self.left_thrust += l * self.dt * self.N
            self.left_thrust = np.clip(self.left_thrust,self.min_thrust,self.max_thrust)
            self.right_thrust += r * self.dt * self.N
            self.right_thrust = np.clip(self.right_thrust,self.min_thrust,self.max_thrust)

        self.compute_motion()

    def compute_motion(self):
        # use 3 DOF ship maneuvering model from chapter 6.5 in Fossen's book
        velocity_r_b = self.project_to_robot_frame(self.velocity_r[:2])
        velocity_b = self.project_to_robot_frame(self.velocity[:2])

        u_r = float(velocity_r_b[0])
        v_r = float(velocity_r_b[1])
        u = float(velocity_b[0])
        v = float(velocity_b[1])
        r = float(self.velocity[2])

        C_RB = np.array([[0.0, -self.m * r, 0.0],
                         [self.m * r,  0.0,  0.0],
                         [0.0,         0.0,  0.0]], dtype=float)
        C_A = np.array([[0.0, 0.0, self.yDotV * v_r + self.yDotR * r],
                        [0.0, 0.0, -self.xDotU * u_r],
                        [-self.yDotV * v_r - self.yDotR * r, self.xDotU * u_r, 0.0]], dtype=float)
        D_n = -1.0 * np.array([[self.xUU * np.abs(u_r), 0.0, 0.0],
                               [0.0, self.yVV * np.abs(v_r) + self.yRV * np.abs(r), self.yVR * np.abs(v_r) + self.yRR * np.abs(r)],
                               [0.0, self.nVV * np.abs(v_r) + self.nRV * np.abs(r), self.nVR * np.abs(v_r) + self.nRR * np.abs(r)]], dtype=float)
        N = C_A + self.D + D_n

        # compute propulsion forces and moment
        F_x_left = self.left_thrust * np.cos(self.left_pos)
        F_y_left = self.left_thrust * np.sin(self.left_pos)
        M_x_left = F_x_left * self.width / 2
        M_y_left = -F_y_left * self.length / 2

        F_x_right = self.right_thrust * np.cos(self.right_pos)
        F_y_right = self.right_thrust * np.sin(self.right_pos)
        M_x_right = -F_x_right * self.width / 2
        M_y_right = -F_y_right * self.length / 2

        F_x = F_x_left + F_x_right
        F_y = F_y_left + F_y_right
        M_n = M_x_left + M_y_left + M_x_right + M_y_right
        tau_p = np.array([F_x, F_y, M_n], dtype=float)

        # compute accelerations (use same formula as original)
        A = self.M_RB + self.M_A
        V = np.array([u, v, r], dtype=float)
        V_r = np.array([u_r, v_r, r], dtype=float)
        b = -C_RB @ V - N @ V_r + tau_p
        
        acc = np.linalg.inv(A.T @ A) @ A.T @ b

        # apply accelerations to velocity
        V_r = V_r + acc * self.dt

        # project velocity to the world frame
        R_wr, _ = self.get_robot_transform()
        V_r[:2] = R_wr @ V_r[:2]
        self.velocity_r = V_r

    def check_collision(self,obj_x,obj_y,obj_r):
        d = self.compute_distance(obj_x,obj_y,obj_r)
        if d <= 0.0:
            self.collision = True

    def compute_distance(self,x,y,r,in_robot_frame=False):
        if in_robot_frame:
            d = np.sqrt(x**2+y**2) - r - self.r
        else:
            d = np.sqrt((self.x-x)**2+(self.y-y)**2) - r - self.r
        return d

    def check_detection(self,obj_x,obj_y,obj_r):
        proj_pos = self.project_to_robot_frame(np.array([obj_x,obj_y]),False)
        
        if np.linalg.norm(proj_pos) > self.perception.range + obj_r:
            return False
        
        angle = np.arctan2(proj_pos[1],proj_pos[0])
        if angle < -0.5*self.perception.angle or angle > 0.5*self.perception.angle:
            return False
        
        return True

    def project_to_robot_frame(self,x,is_vector=True):
        assert isinstance(x, np.ndarray), "the input needs to be an numpy array"
        assert x.shape == (2,)

        R_wr, t_wr = self.get_robot_transform()
        R_rw = R_wr.T
        t_rw = -R_rw @ t_wr

        if is_vector:
            x_r = R_rw @ x
        else:
            x_r = R_rw @ x + t_rw

        return x_r
    
    def project_ego_to_vehicle_frame(self,vehicle):
        vehicle_p = np.array(vehicle[:2], dtype=float)
        vehicle_v = np.array(vehicle[2:4], dtype=float)

        vehicle_v_angle = np.arctan2(vehicle_v[1], vehicle_v[0])
        R = np.array([[np.cos(vehicle_v_angle), -np.sin(vehicle_v_angle)],
                      [np.sin(vehicle_v_angle),  np.cos(vehicle_v_angle)]], dtype=float)
        t = vehicle_p

        # project ego position to vehicle_frame
        ego_p_proj = -(R.T @ t)

        # project ego velocity to vehicle_frame
        ego_v = self.project_to_robot_frame(self.velocity[:2])
        ego_v_proj = R.T @ ego_v

        return np.array(ego_p_proj), np.array(ego_v_proj)
    
    def check_in_left_crossing_zone(self,ego_p_proj,ego_v_proj):
        x_in_range = ((ego_p_proj[0] >= self.left_crossing_zone_x_dim[0]) and (ego_p_proj[0] <= self.left_crossing_zone_x_dim[1]))
        y_in_range = ((ego_p_proj[1] >= self.left_crossing_zone_y_dim_front[0]) and (ego_p_proj[1] <= 0.0))

        x_diff = ego_p_proj[0] - self.left_crossing_zone_x_dim[1]
        y_diff = ego_p_proj[1] - self.left_crossing_zone_y_dim_front[1]
        grad = self.left_crossing_zone_y_dim_front[1] / self.left_crossing_zone_x_dim[1]
        in_trangle_area = (y_diff > grad * x_diff)

        pos_in_left_crossing_zone = (x_in_range and y_in_range and not in_trangle_area)

        ego_v_angle = np.arctan2(ego_v_proj[1],ego_v_proj[0])

        angle_in_left_crossing_zone = ((ego_v_angle >= np.pi/4) and (ego_v_angle <= 3*np.pi/4))

        if pos_in_left_crossing_zone and angle_in_left_crossing_zone:
            return True
        
        return False
    
    def check_in_head_on_zone(self,ego_p_proj,ego_v_proj):
        x_in_range = ((ego_p_proj[0] >= 0.0) and (ego_p_proj[0] <= self.head_on_zone_x_dim))
        y_in_range = ((ego_p_proj[1] >= -0.5 * self.head_on_zone_y_dim) and (ego_p_proj[1] <= 0.5 * self.head_on_zone_y_dim))

        pos_in_head_on_zone = (x_in_range and y_in_range)

        ego_v_angle = np.arctan2(ego_v_proj[1],ego_v_proj[0])

        angle_in_head_on_zone = (np.abs(ego_v_angle) > 3*np.pi/4)

        if pos_in_head_on_zone and angle_in_head_on_zone:
            return True
        
        return False
    
    def compute_COLREGs_turn_angle(self,obj):
        obj_p = np.array(obj[:2])
        ego_v = self.project_to_robot_frame(self.velocity[:2])

        ego_v_angle = np.arctan2(ego_v[1],ego_v[0])
        obj_p_angle = np.arctan2(obj_p[1],obj_p[0])

        base_1 = obj[4] + 1.0
        dist = np.linalg.norm(obj_p)
        add_angle_1 = np.arcsin(base_1/dist)

        tangent_len = np.sqrt(dist**2-base_1**2)
        add_angle_2 = np.arctan2(self.r,tangent_len)

        # desired velocity direction according to COLREGs
        desired_dir = self.wrap_to_pi(obj_p_angle + add_angle_1 + add_angle_2)

        self.phi = self.wrap_to_pi(desired_dir - ego_v_angle)
    
    def check_apply_COLREGs(self,obj):
        obj_v = np.array(obj[2:4])
        if np.linalg.norm(obj_v) < 0.5:
            # considered as a static object
            return False
        
        ego_v = self.project_to_robot_frame(self.velocity[:2])
        if np.linalg.norm(ego_v) < 0.5:
            # ego vehile moves too slow 
            return False
        
        # project position and velocity of ego vehicle to the frame of checking vehicle
        ego_p_proj, ego_v_proj = self.project_ego_to_vehicle_frame(obj)

        # check if ego vehicle is in a CORLEGs relationship with the checking vehicle 
        in_left_crossing_zone = self.check_in_left_crossing_zone(ego_p_proj, ego_v_proj)
        in_head_on_zone = self.check_in_head_on_zone(ego_p_proj, ego_v_proj)

        if in_left_crossing_zone or in_head_on_zone:
            # compute the desired velocity direction
            self.compute_COLREGs_turn_angle(obj)

            # apply COLREGs if need to turn right to reach desired direction
            return True if self.phi > 0 else False
        
        return False

    def wrap_to_pi(self, angle_in):
        angle = angle_in
        
        # wrap angle to [-pi,pi)
        while angle < -np.pi:
            angle += 2 * np.pi
        while angle >= np.pi:
            angle -= 2 * np.pi
        
        return angle

    def perception_output(self,obstacles,robots,in_robot_frame=True):
        self.apply_COLREGs = False
        if self.deactivated:
            return (None,None), self.collision, self.reach_goal
        
        self.perception.observation["objects"].clear()

        ##### self observation #####
        if in_robot_frame:
            abs_velocity_r = self.project_to_robot_frame(self.velocity[:2])
            goal_r = self.project_to_robot_frame(self.goal,False)
            self.perception.observation["self"] = list(np.concatenate((goal_r,abs_velocity_r)))
            self.perception.observation["self"].append(self.velocity[2])
            self.perception.observation["self"].append(self.left_thrust)
            self.perception.observation["self"].append(self.right_thrust)
        else:
            self.perception.observation["self"] = [self.x,self.y,self.velocity[0],self.velocity[1],self.velocity[2], \
                                                   self.left_thrust,self.right_thrust,self.goal[0],self.goal[1]]

        self.perception.observed_obs.clear()
        self.perception.observed_objs.clear()
        self.check_reach_goal()

        # Count active objects
        active_robot_indices = [j for j, r in enumerate(robots) if r is not self and not r.deactivated]
        n_obstacles = len(obstacles)
        n_robots = len(active_robot_indices)
        n_objects = n_obstacles + n_robots
        
        if n_objects == 0:
            return (copy.deepcopy(self.perception.observation["self"]), []), self.collision, self.reach_goal
        
        # Preallocate arrays for all objects
        obj_positions = np.zeros((n_objects, 2), dtype=float)
        obj_velocities = np.zeros((n_objects, 2), dtype=float)
        obj_radii = np.zeros(n_objects, dtype=float)
        obj_types = np.zeros(n_objects, dtype=int)  # 0=obstacle, 1=robot
        obj_indices = np.zeros(n_objects, dtype=int)
        
        # Gather obstacle data
        idx = 0
        for i, obs in enumerate(obstacles):
            obj_positions[idx] = [obs.x, obs.y]
            obj_velocities[idx] = [0.0, 0.0]
            obj_radii[idx] = obs.r
            obj_types[idx] = 0
            obj_indices[idx] = i
            idx += 1
        
        # Gather robot data
        for j_idx in active_robot_indices:
            robot = robots[j_idx]
            obj_positions[idx] = [robot.x, robot.y]
            obj_velocities[idx] = [robot.velocity[0], robot.velocity[1]]
            obj_radii[idx] = robot.r
            obj_types[idx] = 1
            obj_indices[idx] = j_idx
            idx += 1
        
        # Vectorized noise generation
        pos_noise = self.perception.rd.normal(0, self.perception.pos_std, size=(n_objects, 2))
        vel_noise = self.perception.rd.normal(0, self.perception.vel_std, size=(n_objects, 2))
        r_noise_angles = self.perception.rd.vonmises(0, self.perception.r_kappa, size=n_objects)
        r_noise = (1 - self.perception.r_mean_ratio) * (r_noise_angles / np.pi) * obj_radii
        
        # Apply noise
        noisy_positions = obj_positions + pos_noise
        noisy_velocities = obj_velocities + vel_noise
        noisy_radii = self.perception.r_mean_ratio * obj_radii + r_noise
        
        # Get robot transform
        R_wr, t_wr = self.get_robot_transform()
        R_rw = R_wr.T
        t_rw = -R_rw @ t_wr
        
        # Transform noisy positions to robot frame for detection
        noisy_pos_robot = (R_rw @ noisy_positions.T).T + t_rw
        
        # Vectorized circular sector detection
        distances_from_origin = np.linalg.norm(noisy_pos_robot, axis=1)
        in_range = distances_from_origin <= (self.perception.range + noisy_radii)
        
        angles_robot = np.arctan2(noisy_pos_robot[:, 1], noisy_pos_robot[:, 0])
        half_fov = 0.5 * self.perception.angle
        in_fov = (angles_robot >= -half_fov) & (angles_robot <= half_fov)
        
        detected = in_range & in_fov
        
        # Update observed lists
        obstacle_mask = (obj_types == 0) & detected
        robot_mask = (obj_types == 1) & detected
        self.perception.observed_obs = obj_indices[obstacle_mask].astype(int).tolist()
        self.perception.observed_objs = obj_indices[robot_mask].astype(int).tolist()
        
        # Vectorized collision check (using true positions)
        if not self.collision:
            collision_distances = np.sqrt((obj_positions[:, 0] - self.x)**2 + 
                                         (obj_positions[:, 1] - self.y)**2) - obj_radii - self.r
            if np.any(collision_distances <= 0.0):
                self.collision = True
        
        # Select k-nearest detected objects
        detected_indices = np.where(detected)[0]
        n_detected = len(detected_indices)
        
        if n_detected == 0:
            return (copy.deepcopy(self.perception.observation["self"]), []), self.collision, self.reach_goal
        
        # Compute distances for k-nearest selection using np.argpartition
        distances_for_sorting = np.linalg.norm(noisy_pos_robot[detected_indices], axis=1) - \
                               noisy_radii[detected_indices] - self.r
        
        k = min(self.perception.max_obj_num, n_detected)
        if k < n_detected:
            partition_indices = np.argpartition(distances_for_sorting, k-1)[:k]
        else:
            partition_indices = np.arange(n_detected)
        
        # Build object observations
        object_observations = []
        if in_robot_frame:
            noisy_vel_robot = (R_rw @ noisy_velocities.T).T
            for idx in partition_indices:
                global_idx = detected_indices[idx]
                obj_obs = [
                    noisy_pos_robot[global_idx, 0],
                    noisy_pos_robot[global_idx, 1],
                    noisy_vel_robot[global_idx, 0],
                    noisy_vel_robot[global_idx, 1],
                    noisy_radii[global_idx]
                ]
                object_observations.append(obj_obs)
        else:
            for idx in partition_indices:
                global_idx = detected_indices[idx]
                obj_obs = [
                    noisy_positions[global_idx, 0],
                    noisy_positions[global_idx, 1],
                    noisy_velocities[global_idx, 0],
                    noisy_velocities[global_idx, 1],
                    noisy_radii[global_idx]
                ]
                object_observations.append(obj_obs)
        
        self.apply_COLREGs = False
        for obj in object_observations:
            if self.check_apply_COLREGs(obj):
                self.apply_COLREGs = True
                break
        
        return (copy.deepcopy(self.perception.observation["self"]), object_observations), self.collision, self.reach_goal       
