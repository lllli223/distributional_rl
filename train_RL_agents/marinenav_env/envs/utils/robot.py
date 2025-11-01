import numpy as np
import copy
import heapq

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

    def compute_actions(self):
        self.actions = [(l,r) for l in self.left_thrust_change for r in self.right_thrust_change]

    def compute_actions_dimension(self):
        return len(self.actions)
    
    def compute_constant_matrices(self):
        self.M_RB = np.matrix([[self.m,0.0,0.0],[0.0,self.m,0.0],[0.0,0.0,self.Izz]])

        self.M_A = -1.0 * np.matrix([[self.xDotU,0.0,0.0],[0.0,self.yDotV,self.yDotR],
                                     [0.0,self.nDotV,self.nDotR]])
        
        self.D = -1.0 * np.matrix([[self.xU,0.0,0.0],[0.0,self.yV,self.yR],
                                   [0.0,self.nV,self.nR]])

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
        # compute transformation from world frame to robot frame
        R_wr = np.matrix([[np.cos(self.theta),-np.sin(self.theta)],[np.sin(self.theta),np.cos(self.theta)]])
        t_wr = np.matrix([[self.x],[self.y]])
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
        u_r = velocity_r_b[0]
        v_r = velocity_r_b[1]
        u = velocity_b[0]
        v = velocity_b[1]
        r = self.velocity[2]
        C_RB = np.matrix([[0.0,-self.m*r,0.0],[self.m*r,0.0,0.0],[0.0,0.0,0.0]])
        C_A = np.matrix([[0.0,0.0,self.yDotV*v_r+self.yDotR*r],[0.0,0.0,-self.xDotU*u_r],
                         [-self.yDotV*v_r-self.yDotR*r,self.xDotU*u_r,0.0]])
        D_n = -1.0 * np.matrix([[self.xUU*np.abs(u_r),0.0,0.0],
                                [0.0,self.yVV*np.abs(v_r)+self.yRV*np.abs(r),self.yVR*np.abs(v_r)+self.yRR*np.abs(r)],
                                [0.0,self.nVV*np.abs(v_r)+self.nRV*np.abs(r),self.nVR*np.abs(v_r)+self.nRR*np.abs(r)]])
        N = C_A + self.D + D_n

        # compute propulsion forces and moment
        F_x_left = self.left_thrust * np.cos(self.left_pos)
        F_y_left = self.left_thrust * np.sin(self.left_pos)
        M_x_left = F_x_left * self.width/2
        M_y_left = -F_y_left * self.length/2
        
        F_x_right = self.right_thrust * np.cos(self.right_pos)
        F_y_right = self.right_thrust * np.sin(self.right_pos)
        M_x_right = -F_x_right * self.width/2
        M_y_right = -F_y_right * self.length/2
        
        F_x = F_x_left + F_x_right
        F_y = F_y_left + F_y_right
        M_n = M_x_left + M_y_left + M_x_right + M_y_right
        tau_p = np.matrix([[F_x],[F_y],[M_n]])

        # compute accelerations
        A = self.M_RB + self.M_A
        V = np.matrix([[u,v,r]]).transpose()
        V_r = np.matrix([[u_r,v_r,r]]).transpose()
        b = -C_RB*V - N*V_r + tau_p
        acc = np.linalg.inv(A.transpose()*A)*A.transpose()*b

        # apply accelerations to velocity
        V_r += acc * self.dt
        
        # project velocity to the world frame
        R_wr,_ = self.get_robot_transform()
        V_r[:2,:] = R_wr * V_r[:2,:]
        self.velocity_r = np.squeeze(np.array(V_r))

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
        assert isinstance(x,np.ndarray), "the input needs to be an numpy array"
        assert np.shape(x) == (2,)

        x_r = np.reshape(x,(2,1))

        R_wr, t_wr = self.get_robot_transform()

        R_rw = np.transpose(R_wr)
        t_rw = -R_rw * t_wr 

        if is_vector:
            x_r = R_rw * x_r
        else:
            x_r = R_rw * x_r + t_rw

        x_r.resize((2,))
        return np.array(x_r)
    
    def project_ego_to_vehicle_frame(self,vehicle):
        vehicle_p = np.array(vehicle[:2])
        vehicle_v = np.array(vehicle[2:4])

        vehicle_v_angle = np.arctan2(vehicle_v[1],vehicle_v[0])
        R = np.matrix([[np.cos(vehicle_v_angle),-np.sin(vehicle_v_angle)], \
                       [np.sin(vehicle_v_angle),np.cos(vehicle_v_angle)]])
        t = np.matrix([[vehicle_p[0]],[vehicle_p[1]]])

        # project ego position to vehicle_frame
        ego_p_proj = -np.transpose(R) * t
        ego_p_proj.resize((2,))

        # project ego velocity to vehicle_frame
        ego_v = self.project_to_robot_frame(self.velocity[:2])
        ego_v_proj = np.transpose(R) * np.matrix([[ego_v[0]],[ego_v[1]]])
        ego_v_proj.resize((2,))

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
        if self.deactivated:
            return (None,None), self.collision, self.reach_goal
        
        self.perception.observation["objects"].clear()

        ##### self observation #####
        if in_robot_frame:
            # vehicle velocity wrt seafloor in self frame
            abs_velocity_r = self.project_to_robot_frame(self.velocity[:2])

            # goal position in self frame
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

        ##### static objects #####
        for i,obs in enumerate(obstacles):            
            obs_px, obs_py = self.perception.pos_observation(obs.x,obs.y)
            obs_vx, obs_vy = self.perception.vel_observation(0.0,0.0)
            obs_r = self.perception.r_observation(obs.r)

            if not self.check_detection(obs_px,obs_py,obs_r):
                continue

            self.perception.observed_obs.append(i)

            if not self.collision:
                self.check_collision(obs.x,obs.y,obs.r)

            if in_robot_frame:
                pos_r = self.project_to_robot_frame(np.array([obs_px,obs_py]),False)
                vel_r = self.project_to_robot_frame(np.array([obs_vx,obs_vy]))
                self.perception.observation["objects"].append([pos_r[0],pos_r[1],vel_r[0],vel_r[1],obs_r])
            else:
                self.perception.observation["objects"].append([obs_px,obs_py,obs_vx,obs_vy,obs_r])

        ##### robots #####
        for j,robot in enumerate(robots):
            if robot is self:
                continue
            if robot.deactivated:
                # This robot is in the deactivate state, and abscent from the current map
                continue
            
            rob_px, rob_py = self.perception.pos_observation(robot.x,robot.y)
            rob_vx, rob_vy = self.perception.vel_observation(robot.velocity[0],robot.velocity[1])
            rob_r = self.perception.r_observation(robot.r)
            
            if not self.check_detection(rob_px,rob_py,rob_r):
                continue

            self.perception.observed_objs.append(j)
            
            if not self.collision:
                self.check_collision(robot.x,robot.y,robot.r)

            if in_robot_frame:
                pos_r = self.project_to_robot_frame(np.array([rob_px,rob_py]),False)
                vel_r = self.project_to_robot_frame(np.array([rob_vx,rob_vy]))
                self.perception.observation["objects"].append([pos_r[0],pos_r[1],vel_r[0],vel_r[1],rob_r])
            else:
                self.perception.observation["objects"].append([rob_px,rob_py,rob_vx,rob_vy,rob_r])

        self_state = copy.deepcopy(self.perception.observation["self"])
        object_observations = copy.deepcopy(heapq.nsmallest(self.perception.max_obj_num,
                                            self.perception.observation["objects"],
                                            key=lambda obj:self.compute_distance(obj[0],obj[1],obj[4],True)))

        self.apply_COLREGs = False
        for obj in object_observations:
            if self.check_apply_COLREGs(obj):
                self.apply_COLREGs = True
                break

        # object_states = []
        # for object in object_observations:
        #     object_states += object
        # object_states += [0.0,0.0,0.0]*(self.perception.max_obj_num-len(object_observations))

        # return self_state+object_states, self.collision, self.reach_goal
        return (self_state,object_observations), self.collision, self.reach_goal       
