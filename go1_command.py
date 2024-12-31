#!/usr/bin/python

import sys
import numpy as np
from apf_vom_vector_minima_pract import calculate_distance

sys.path.append('../lib/python/arm64')
import robot_interface as sdk # type: ignore

def init_udp():
    HIGHLEVEL = 0xee
    # udp = sdk.UDP(HIGHLEVEL, 8080, "192.168.12.1", 8082)
    udp = sdk.UDP(HIGHLEVEL, 8080, "192.168.1.170", 8082)
    # udp = sdk.UDP(HIGHLEVEL, 8080, "192.168.1.172", 8082)

    cmd = sdk.HighCmd()
    state = sdk.HighState()
    udp.InitCmdData(cmd)
    return udp, cmd, state

def gait_velocity(current, next_point, dist_to_goal, Kp = [1,1], clip = [1,1]):
    # Kp_v = 0.1 * Kp
    # Kp_w = 0.01 * Kp
    vx = Kp[0] * (dist_to_goal)
    vy = 0
    # next_point = [100,100]

    delta = [next_point[0] - current[0], next_point[1] - current[1]]

    # Calculate the angle in radians
    theta_required = np.degrees(np.arctan2(delta[1], delta[0]))
    theta_current =  90
    theta_error = theta_required - theta_current
    if theta_error<-180:
        theta_error+=360
    elif theta_error>180:
        theta_error-=360
    # print("theta_required", theta_required , theta_error)
    if abs(theta_error) > 30:
        vx = 0
    w = Kp[1] * (theta_error)
    # print("thetaerror", theta_error, theta_current, theta_required)
    print("theta and w, v",theta_error, w, theta_required, delta)
    vx = np.clip(vx, -clip[0], clip[0])
    vy = np.clip(vy, -clip[0], clip[0])
    w = np.clip(w, -clip[1], clip[1])
    vx = round(vx,2)
    w = round(w,2)

    return vx , vy, w

def gait_command(udp, cmd, state, initial_yaw, current, next_point, dist_to_goal, terrain_type, Kp, vel_clip):
    # vel_clip = [0.2,0.2]
    
    udp.Recv()
    udp.GetRecv(state)
    
    quadruped_angles = state.imu.rpy
    if initial_yaw == 0 :
        initial_yaw = quadruped_angles[2]
        # print(initial_yaw, quadruped_angles[2])
    quadruped_yaw = quadruped_angles[2] - initial_yaw
    
    if quadruped_yaw > np.pi:
        quadruped_yaw = -2*np.pi + quadruped_yaw
    elif quadruped_yaw < -np.pi:
        quadruped_yaw = 2*np.pi + quadruped_yaw

    cmd.mode = 0      # 0:idle, default stand      1:forced stand     2:walk continuously
    cmd.gaitType = 0  # 0 -> idle, 1 -> trot, 2 -> trot running, 3 -> stair climbing, 4 -> trot obstacle
    cmd.speedLevel = 0
    cmd.footRaiseHeight = 0
    cmd.bodyHeight = 0
    cmd.euler = [0, 0, 0]
    cmd.velocity = [0, 0]
    cmd.yawSpeed = 0.0
    cmd.reserve = 0

    if terrain_type ==  0:  # sand
        vx, vy, w = gait_velocity(current, next_point, dist_to_goal, Kp, clip=vel_clip)
        cmd.mode = 2
        cmd.gaitType = 1
        cmd.velocity = [vx, vy] # -1  ~ +1
        cmd.yawSpeed = w
        cmd.footRaiseHeight = 1.0
        cmd.bodyHeight = -0.3

    elif terrain_type == 2: # stairs
        vx, vy, w = gait_velocity(current, next_point, dist_to_goal, Kp, clip=vel_clip)
        cmd.mode = 2
        cmd.gaitType = 3
        cmd.velocity = [vx, vy] # -1  ~ +1
        cmd.yawSpeed = w

    elif terrain_type == 1:  # stones
        vx, vy, w = gait_velocity(current, next_point, dist_to_goal, Kp, clip=vel_clip)
        cmd.mode = 2
        cmd.gaitType = 1
        cmd.velocity = [vx, vy] # -1  ~ +1
        cmd.yawSpeed = w
        cmd.footRaiseHeight = 1.0
        cmd.bodyHeight = -0.3

    elif terrain_type == 3:  # plane
        vx, vy, w = gait_velocity(current, next_point, dist_to_goal, Kp, clip=vel_clip)
        cmd.mode = 2
        cmd.gaitType = 1
        cmd.velocity = [vx, vy] # -1  ~ +1
        cmd.yawSpeed = w
        cmd.footRaiseHeight = 0.0
        cmd.bodyHeight = 0.0
    

    udp.SetSend(cmd)
    udp.Send()

    return quadruped_yaw, [vx,vy,w], initial_yaw, quadruped_angles

def stair_inv_command_vel(udp, cmd, state, initial_yaw, current, next_point, dist_to_goal, Kp, clip):
    vx = - Kp[0] * (dist_to_goal)
    vy = 0
    # next_point = [100,100]

    delta = [next_point[0] - current[0], next_point[1] - current[1]]

    # Calculate the angle in radians
    theta_required = np.degrees(np.arctan2(delta[1], delta[0]))
    theta_current =  90
    theta_error = theta_required - theta_current + 180
    if theta_error<-180:
        theta_error+=360
    elif theta_error>180:
        theta_error-=360
    # print("theta_required", theta_required , theta_error)
    if abs(theta_error) > 30:
        vx = 0
    w = Kp[1] * (theta_error)
    # print("thetaerror", theta_error, theta_current, theta_required)
    print("theta and w, v",theta_error, w, theta_required, delta)
    vx = np.clip(vx, -clip[0], clip[0])
    vy = np.clip(vy, -clip[0], clip[0])
    w = np.clip(w, -clip[1], clip[1])
    vx = round(vx,2)
    w = round(w,2)

    return vx , vy, w


def stair_command_vel(udp, cmd, state, initial_yaw, vel):
    # vel_clip = [0.2,0.2]
    vx = vel[0]
    vy = vel[1]
    w = vel[2]
    udp.Recv()
    udp.GetRecv(state)
    
    quadruped_angles = state.imu.rpy
    if initial_yaw == 0 :
        initial_yaw = quadruped_angles[2]
        # print(initial_yaw, quadruped_angles[2])
    quadruped_yaw = quadruped_angles[2] - initial_yaw
    
    if quadruped_yaw > np.pi:
        quadruped_yaw = -2*np.pi + quadruped_yaw
    elif quadruped_yaw < -np.pi:
        quadruped_yaw = 2*np.pi + quadruped_yaw

    cmd.mode = 0      # 0:idle, default stand      1:forced stand     2:walk continuously
    cmd.gaitType = 0  # 0 -> idle, 1 -> trot, 2 -> trot running, 3 -> stair climbing, 4 -> trot obstacle
    cmd.reserve = 0

    cmd.mode = 2
    cmd.gaitType = 3
    cmd.velocity = [vx, vy] # -1  ~ +1
    cmd.yawSpeed = w


    udp.SetSend(cmd)
    udp.Send()

    return quadruped_yaw, initial_yaw

# def get_quadruped_angles(udp, cmd, state):

#     udp.Recv()
#     udp.GetRecv(state)
#     quadruped_angles = state.imu.rpy
#     quadruped_yaw = quadruped_angles[2]
#     # print("qudruped yaw", quadruped_yaw)
#     return quadruped_yaw

def center_from_corners(input_corners = [[1,1],[2,2],[2,3],[4,5]]):
    # center = [(input_corners[0] + input_corners[2])/2, (input_corners[1] + input_corners[3])/2]
    center = [(input_corners[0][0] + input_corners[1][0] + input_corners[2][0] + input_corners[3][0] )/4, 
                        (input_corners[0][1] + input_corners[1][1] + input_corners[2][1] + input_corners[3][1] )/4]
    return center

def midpoint(p1 = [1,1] ,p2 = [2,2]):
    return np.array([(p1[0]+p2[0])/2,(p1[1]+p2[1])/2])

def adaptive_gait_selection(quadruped, sand, stair, stones, reached, completed, i, initial_goal, next_goal, proximity_dist = 20, goal_offset_dist = 30):
    coord = [stones, sand, stair]
    start = np.array(center_from_corners(quadruped))
    goal_points = coord [i]
    # goal_midpoint = np.array(center_from_corners(goal_points))

    goal_l1 = calculate_distance(goal_points[0], goal_points[1])
    goal_l2 = calculate_distance(goal_points[1], goal_points[2])
    center_rect = center_from_corners(goal_points)
    if goal_l1 > goal_l2:
        point1_rect = midpoint(goal_points[1], goal_points[2])
        point2_rect = midpoint(goal_points[0], goal_points[3])
        dir1 = point1_rect - center_rect
        dir2 = point2_rect - center_rect
        dir1 = dir1/np.linalg.norm(dir1)
        dir2 = dir2/np.linalg.norm(dir2)
        point1 = point1_rect + dir1*goal_offset_dist
        point2 = point2_rect + dir2*goal_offset_dist

    else:
        point1_rect = midpoint(goal_points[0], goal_points[1])
        point2_rect = midpoint(goal_points[2], goal_points[3])
        dir1 = point1_rect - center_rect
        dir2 = point2_rect - center_rect
        dir1 = dir1/np.linalg.norm(dir1)
        dir2 = dir2/np.linalg.norm(dir2)
        point1 = point1_rect + dir1*goal_offset_dist
        point2 = point2_rect + dir2*goal_offset_dist
    dist1 = calculate_distance(start, point1)
    dist2 = calculate_distance(start, point2)
    terrain_type = 3
    
    if completed[i] == 0:
        if reached[i] == 0:
            if dist1 < dist2:
                initial_goal = point1
                next_goal = point2
            else:
                initial_goal = point2
                next_goal = point1
            # print(goal_points, initial_goal, next_goal)
            terrain_type = 3
            goal = initial_goal
            dist_to_goal = calculate_distance(start, goal)
            # print("reached", reached[i], initial_goal, next_goal)
            if dist_to_goal < proximity_dist:
                reached[i] = 1
        else:
            terrain_type = i
            distp1 = calculate_distance(next_goal, point1)
            distp2 = calculate_distance(next_goal, point2)
            next_goal = point1 if distp1<distp2 else point2
            goal = next_goal
            dist_to_goal = calculate_distance(start, goal)
            if dist_to_goal < proximity_dist:
                completed[i] = 1
    else:
        i+=1
        goal = next_goal
    return start, goal, terrain_type, reached, completed, i, initial_goal, next_goal, goal_points

def stair_mode_gait(quadruped, stair, reached, completed, initial_goal, next_goal, mid_reached, oriented, proximity_dist = 20, goal_offset_dist = 30):
    start = np.array(center_from_corners(quadruped))
    goal_points = stair
    i = 2
    vel = [0, 0, 0]
    # goal_midpoint = np.array(center_from_corners(goal_points))

    goal_l1 = calculate_distance(goal_points[0], goal_points[1])
    goal_l2 = calculate_distance(goal_points[1], goal_points[2])
    center_rect = center_from_corners(goal_points)
    if goal_l1 > goal_l2:
        point1_rect = midpoint(goal_points[1], goal_points[2])
        point2_rect = midpoint(goal_points[0], goal_points[3])
        dir1 = point1_rect - center_rect
        dir2 = point2_rect - center_rect
        dir1 = dir1/np.linalg.norm(dir1)
        dir2 = dir2/np.linalg.norm(dir2)
        point1 = point1_rect + dir1*goal_offset_dist
        point2 = point2_rect + dir2*goal_offset_dist

    else:
        point1_rect = midpoint(goal_points[0], goal_points[1])
        point2_rect = midpoint(goal_points[2], goal_points[3])
        dir1 = point1_rect - center_rect
        dir2 = point2_rect - center_rect
        dir1 = dir1/np.linalg.norm(dir1)
        dir2 = dir2/np.linalg.norm(dir2)
        point1 = point1_rect + dir1*goal_offset_dist
        point2 = point2_rect + dir2*goal_offset_dist
    dist1 = calculate_distance(start, point1)
    dist2 = calculate_distance(start, point2)
    terrain_type = 3
    
    if completed[i] == 0:
        if reached[i] == 0:
            if dist1 < dist2:
                initial_goal = point1
                next_goal = point2
            else:
                initial_goal = point2
                next_goal = point1
            # print(goal_points, initial_goal, next_goal)
            terrain_type = 3
            goal = initial_goal
            dist_to_goal = calculate_distance(start, goal)
            # print("reached", reached[i], initial_goal, next_goal)
            if dist_to_goal < proximity_dist:
                reached[i] = 1
        else:
            terrain_type = 2
            distp1 = calculate_distance(next_goal, point1)
            distp2 = calculate_distance(next_goal, point2)
            if distp1<distp2:
                next_goal = point1
                initial_goal = point2
            else:
                next_goal = point2
                initial_goal = point1
            center_point = midpoint(point1, point2)
            if mid_reached == 0:
                goal = center_point
                print("mid reaching")
                dist_to_goal = calculate_distance(start, goal)
                if dist_to_goal < proximity_dist/2:
                    mid_reached = 1
            else:

                if oriented == 0:
                    print("oreintation achieveing")
                    goal = center_point
                    Kp = [0.1,0.01]
                    vx = Kp[0] * (center_point[0]-start[0])
                    vy = -Kp[0] * (center_point[1]-start[1])
                    # next_point = [100,100]

                    delta = [initial_goal[0] - start[0], initial_goal[1] - start[1]]

                    # Calculate the angle in radians
                    theta_required = np.degrees(np.arctan2(delta[1], delta[0]))
                    theta_current =  90
                    theta_error = theta_required - theta_current
                    if theta_error<-180:
                        theta_error+=360
                    elif theta_error>180:
                        theta_error-=360
                    w = Kp[1] * (theta_error)
                    clip = [0.111,0.111]
                    vx = np.clip(vx, -clip[0], clip[0])
                    vy = np.clip(vy, -clip[0], clip[0])
                    w = np.clip(w, -clip[1], clip[1])
                    vel = [vx,vy,w]
                    print("orientation_ velocity w/o clip",vx,vy,w, delta, theta_error)
                else:
                    print("back side going")
                    goal = next_goal
                    dist_to_goal = calculate_distance(start, goal)
                    if dist_to_goal < proximity_dist:
                        completed[2] = 1



    else:
        i+=1
        goal = next_goal
    return start, goal, terrain_type, reached, completed, initial_goal, next_goal, goal_points, mid_reached, oriented, vel


def tf_g_i(input_point = [1,1], image_angle = 3.14/4):
    tf = np.array([[np.cos(image_angle), -np.sin(image_angle), 0],
               [np.sin(image_angle), np.cos(image_angle), 0],
               [0, 0, 1]])
    input_point_3d = [input_point[0],input_point[1], 0]
    # point = np.array([ input_point[0], input_point[1], input_point[2]])
    point_t = np.transpose(input_point_3d)

    transformed_point = np.matmul(tf,point_t)
    return [transformed_point[0], transformed_point[1]]

def tf_g_i2(input_point = [1,1,2,2], image_angle = 3.14/4):
    point1 = [input_point[0],input_point[1]]
    point2 = [input_point[2],input_point[1]]
    point3 = [input_point[2], input_point[3]]
    point4 = [input_point[0],input_point[3]]
    point1_t = tf_g_i(input_point=point1, image_angle=image_angle)
    point2_t = tf_g_i(input_point=point2, image_angle=image_angle)
    point3_t = tf_g_i(input_point=point3, image_angle=image_angle)
    point4_t = tf_g_i(input_point=point4, image_angle=image_angle)

    final_point = [ point1_t, point2_t, point3_t, point4_t]
    return final_point


def tf_q_g(input_point = [5,5],quadruped_xy = [5,3] , quadruped_angle = 0):
    tf = np.array([[np.cos(quadruped_angle), np.sin(quadruped_angle), 0, (-quadruped_xy[0] * np.cos(quadruped_angle)) - (quadruped_xy[1] * np.sin(quadruped_angle))],
               [-np.sin(quadruped_angle), np.cos(quadruped_angle), 0, (quadruped_xy[0] * np.sin(quadruped_angle)) - (quadruped_xy[1] * np.cos(quadruped_angle))],
               [0, 0, 1, 0],
               [0, 0, 0, 1]])
    point = np.array([ input_point[0], input_point[1], 0, 1])
    point_t = np.transpose(point)

    transformed_vector = np.matmul(tf,point_t)
    transformed_point = np.array([ transformed_vector[0] + 640/2, transformed_vector[1] - 350])
    return transformed_point



def tf_q_g2(input_point = [[1,1],[2,2],[3,1],[2,0]],quadruped_xy = [5,3] , quadruped_angle = np.pi/2):
    point1 = input_point[0]
    point2 = input_point[1]
    point3 = input_point[2]
    point4 = input_point[3]
    point1_t = tf_q_g(input_point=point1, quadruped_xy = quadruped_xy , quadruped_angle =quadruped_angle)
    point2_t = tf_q_g(input_point=point2, quadruped_xy = quadruped_xy , quadruped_angle =quadruped_angle)
    point3_t = tf_q_g(input_point=point3, quadruped_xy = quadruped_xy , quadruped_angle =quadruped_angle)
    point4_t = tf_q_g(input_point=point4, quadruped_xy = quadruped_xy , quadruped_angle =quadruped_angle)
    point = np.array([ point1_t, point2_t, point3_t, point4_t])
    # final_point = np.array([np.min(point[:,0]),np.max(point[:,1]),np.max(point[:,0]),np.min(point[:,1])])
    final_point = point
    return final_point

# print(center_from_corners())