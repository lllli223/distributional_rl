import torch
import torch.nn.functional as F


def build_observation(pos_er, theta_er, vel_er, vel_r_er,
                     goal_er, left_thrust_er, right_thrust_er,
                     obs_xy_eo, obs_r_eo, robots_xy_er, robots_r_er,
                     detect_range, detect_angle, max_obj_num, device):
    """
    向量化构建观测，替代robot.perception_output方法
    
    参数:
    - pos_er: [E,R,2] 机器人位置
    - theta_er: [E,R] 机器人姿态角
    - vel_er: [E,R,3] 机器人绝对速度
    - vel_r_er: [E,R,3] 机器人相对速度
    - goal_er: [E,R,2] 目标位置
    - left_thrust_er: [E,R] 左推力
    - right_thrust_er: [E,R] 右推力
    - obs_xy_eo: [E,O,2] 障碍物位置
    - obs_r_eo: [E,O] 障碍物半径
    - robots_xy_er: [E,R,2] 其他机器人位置
    - robots_r_er: [E,R] 其他机器人半径
    - detect_range: 检测范围
    - detect_angle: 检测角度
    - max_obj_num: 最大目标数量
    - device: 计算设备
    """
    
    E, R, _ = pos_er.shape
    O = obs_xy_eo.shape[1]
    
    # 1. 自身状态观测
    # 计算机器人坐标转换矩阵
    cos_theta = torch.cos(theta_er)
    sin_theta = torch.sin(theta_er)
    
    # 构造旋转矩阵 [E,R,2,2]
    R_wr = torch.stack([
        torch.stack([cos_theta, -sin_theta], dim=-1),
        torch.stack([sin_theta,  cos_theta], dim=-1)
    ], dim=-2)

    # 目标在自车坐标系的位置
    goal_vec = goal_er - pos_er
    goal_r = torch.matmul(R_wr.transpose(-2, -1), goal_vec.unsqueeze(-1)).squeeze(-1)

    # 速度在自车坐标系
    vel_abs_2d = vel_er[..., :2]
    vel_r_2d = torch.matmul(R_wr.transpose(-2, -1), vel_abs_2d.unsqueeze(-1)).squeeze(-1)

    # 自身状态 [E,R,7]
    self_state = torch.stack([
        goal_r[..., 0], goal_r[..., 1],
        vel_r_2d[..., 0], vel_r_2d[..., 1],
        vel_er[..., 2], left_thrust_er, right_thrust_er
    ], dim=-1)

    # 2. 感知其他对象（障碍物和其他机器人）
    # 构造所有对象的位置列表 [E, R, O+ (R-1), 2]
    # 先处理障碍物
    obstacles_expanded = obs_xy_eo.unsqueeze(1).expand(-1, R, -1, -1)

    # 处理其他机器人
    # 为每个机器人构造其他机器人位置矩阵
    robots_xy_expanded = robots_xy_er.unsqueeze(2).expand(-1, -1, R, -1)
    # 排除自身
    eye_mask = torch.eye(R, device=device).unsqueeze(0).expand(E, -1, -1)
    robots_xy_filtered = robots_xy_expanded.clone()
    robots_xy_filtered[eye_mask.bool()] = 0

    # 其他机器人速度（用于对象速度）
    robots_vel_expanded = vel_er[..., :2].unsqueeze(2).expand(-1, -1, R, -1).clone()
    robots_vel_expanded[eye_mask.bool().unsqueeze(-1)] = 0

    # 半径：障碍 [E,R,O] + 机器人 [E,R,R]（自身半径清零以避免被选中）
    robots_r_cube = robots_r_er.unsqueeze(2).expand(-1, -1, R).clone()  # [E,R,R]
    self_mask = eye_mask.bool()  # [E,R,R] 自身位置mask
    robots_r_cube[self_mask] = 0

    # 合并障碍物和其他机器人
    # 对象数量：O + (R-1)
    other_robots_count = R - 1
    if other_robots_count > 0:
        objects_xy = torch.cat([obstacles_expanded, robots_xy_filtered], dim=2)
        objects_r = torch.cat([obs_r_eo.unsqueeze(1).expand(-1, R, -1), robots_r_cube], dim=2)  # [E,R,O+R]
        # 对象速度：障碍为0，机器人为其速度
        zeros_obs_vel = torch.zeros(E, R, O, 2, device=device)
        objects_vel = torch.cat([zeros_obs_vel, robots_vel_expanded], dim=2)  # [E,R,O+R,2]
        # 自身排除mask: 障碍物全False, 自身机器人位置True
        self_exclusion_mask = torch.cat([
            torch.zeros(E, R, O, dtype=torch.bool, device=device),
            self_mask  # [E,R,R]
        ], dim=-1)  # [E,R,O+R]
    else:
        objects_xy = obstacles_expanded
        objects_r = obs_r_eo
        objects_vel = torch.zeros(E, R, O, 2, device=device)
        self_exclusion_mask = torch.zeros(E, R, O, dtype=torch.bool, device=device)

    # 3. 相对位置计算和坐标转换
    # 相对位置 [E,R,num_objects,2]
    rel_pos = objects_xy - pos_er.unsqueeze(2)
    rel_pos_robot = torch.matmul(R_wr.transpose(-2, -1).unsqueeze(2), rel_pos.unsqueeze(-1)).squeeze(-1)
    # 速度也转到自车坐标
    rel_vel_robot = torch.matmul(R_wr.transpose(-2, -1).unsqueeze(2), objects_vel.unsqueeze(-1)).squeeze(-1)

    # 4. 可见性筛选（距离和角度）
    # 距离筛选
    distances = torch.norm(rel_pos_robot, dim=-1)
    # 角度筛选
    angles = torch.atan2(rel_pos_robot[..., 1], rel_pos_robot[..., 0])

    # 可见性mask (排除自身位置)
    distance_mask = distances <= (detect_range + objects_r)
    angle_mask = torch.abs(angles) <= (detect_angle / 2)
    visible_mask = distance_mask & angle_mask & (~self_exclusion_mask)

    # 5. 碰撞检测
    collision_distances = distances - objects_r
    collision_mask = collision_distances <= 0

    # 6. 选择最近的目标（不超过max_obj_num）
    num_objects = distances.shape[-1]
    # 为每个机器人选择最近的目标
    if num_objects > 0:
        # 计算可见目标的距离，不可见的设为无穷大
        visible_distances = torch.where(visible_mask, distances, float('inf'))
        
        # 选择最近的K个目标
        K = min(max_obj_num, num_objects)
        topk_distances, topk_indices = torch.topk(visible_distances, k=K, largest=False, dim=-1)
        
        # 构建最终的目标观测 [E,R,max_obj_num,5]
        objects_obs = torch.zeros(E, R, max_obj_num, 5, device=device)
        objects_mask = torch.zeros(E, R, max_obj_num, dtype=torch.bool, device=device)
        
        # 使用完全向量化的方式填充实际检测到的目标
        if K > 0:
            # 批量索引 - 使用 gather 提取选中的目标
            # 扩展索引以匹配维度
            idx_pos = topk_indices.unsqueeze(-1).expand(-1, -1, -1, 2)
            selected_rel_pos = torch.gather(rel_pos_robot, 2, idx_pos)
            selected_rel_vel = torch.gather(rel_vel_robot, 2, idx_pos)
            idx_r = topk_indices.unsqueeze(-1)
            selected_r = torch.gather(objects_r.unsqueeze(-1), 2, idx_r).squeeze(-1)

            objects_obs[:, :, :K, :2] = selected_rel_pos
            objects_obs[:, :, :K, 2:4] = selected_rel_vel
            objects_obs[:, :, :K, 4] = selected_r

            # 设置有效性mask（距离不是无穷大的才是有效的）
            valid_mask = topk_distances < float('inf')
            objects_mask[:, :, :K] = valid_mask
    else:
        objects_obs = torch.zeros(E, R, max_obj_num, 5, device=device)
        objects_mask = torch.zeros(E, R, max_obj_num, dtype=torch.bool, device=device)
    
    return self_state, objects_obs, objects_mask, collision_mask.any(dim=-1)


def add_perception_noise(self_state, objects_obs, objects_mask, device, noise_std=0.05):
    """
    添加感知噪声，模拟真实的传感器噪声
    """
    # 自身状态噪声
    self_state_noisy = self_state + torch.randn_like(self_state) * noise_std
    
    # 对象观测噪声
    objects_obs_noisy = objects_obs.clone()
    # 只对位置和速度添加噪声，半径保持不变
    objects_obs_noisy[..., :2] += torch.randn_like(objects_obs[..., :2]) * noise_std
    objects_obs_noisy[..., 2:4] += torch.randn_like(objects_obs[..., 2:4]) * noise_std
    
    return self_state_noisy, objects_obs_noisy


def compute_colregs_penalty(robots_pos, robots_vel, robots_r, theta_er, detect_angle, device):
    """
    完全向量化的 COLREGs 规则检测和惩罚计算
    
    参数:
    - robots_pos: [E,R,2] 机器人位置
    - robots_vel: [E,R,3] 机器人速度
    - robots_r: [E,R] 机器人半径
    - theta_er: [E,R] 机器人姿态角
    - detect_angle: 检测角度
    - device: 计算设备
    
    返回:
    - penalty: [E,R] 每个机器人的 COLREGs 惩罚
    """
    E, R, _ = robots_pos.shape
    pos_i = robots_pos.unsqueeze(2)
    pos_j = robots_pos.unsqueeze(1)
    rel_pos = pos_j - pos_i
    cos_theta = torch.cos(theta_er)
    sin_theta = torch.sin(theta_er)
    R_rot = torch.stack([
        torch.stack([cos_theta, sin_theta], dim=-1),
        torch.stack([-sin_theta, cos_theta], dim=-1)
    ], dim=-2)
    R_rot_expanded = R_rot.unsqueeze(2)
    rel_pos_robot = torch.matmul(R_rot_expanded, rel_pos.unsqueeze(-1)).squeeze(-1)

    head_on_x_in_range = (rel_pos_robot[..., 0] >= 0) & (rel_pos_robot[..., 0] <= 17.0)
    head_on_y_in_range = torch.abs(rel_pos_robot[..., 1]) <= 4.5
    in_head_on = head_on_x_in_range & head_on_y_in_range

    left_cross_x_in_range = (rel_pos_robot[..., 0] >= -9.0) & (rel_pos_robot[..., 0] <= 12.0)
    left_cross_y_in_range = (rel_pos_robot[..., 1] >= -17.0) & (rel_pos_robot[..., 1] <= -7.0)
    in_left_cross = left_cross_x_in_range & left_cross_y_in_range

    colregs_violation = (in_head_on | in_left_cross).float()
    eye_mask = torch.eye(R, device=device).unsqueeze(0).expand(E, -1, -1).bool()
    colregs_violation = colregs_violation.masked_fill(eye_mask, 0.0)
    penalty = colregs_violation.sum(dim=2)
    return penalty
