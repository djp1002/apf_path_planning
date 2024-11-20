#!/usr/bin/python

import sys
import numpy as np
from apf_vom_vector_minima_pract import calculate_distance

sys.path.append('../lib/python/arm64')
import robot_interface as sdk

def init_udp():
    HIGHLEVEL = 0xee
    # udp = sdk.UDP(HIGHLEVEL, 8080, "192.168.123.161", 8082)
    udp = sdk.UDP(HIGHLEVEL, 8080, "192.168.1.170", 8082)

    cmd = sdk.HighCmd()
    state = sdk.HighState()
    udp.InitCmdData(cmd)
    return udp, cmd, state

def gait_velocity(current, next_point, dist_to_goal, quadruped_angles, Kp = 1, clip = [1,1]):
    Kp_v = 0.01 * Kp
    Kp_w = 0.01 * Kp
    vx = Kp_v * (dist_to_goal)
    vy = 0
    # next_point = [100,100]

    delta = [next_point[0] - current[0], next_point[1] - current[1]]

    # Calculate the angle in radians
    theta_required = np.degrees(np.arctan2(delta[1], delta[0]))
    theta_current =  np.degrees(quadruped_angles[2])+75
    theta_error = theta_required - theta_current
    # print("theta_required",theta_error)
    if abs(theta_error) > 60:
        vx = 0
    w = Kp_w * (theta_error)
    # print("thetaerror", theta_error, theta_current, theta_required)
    print(current, next_point, theta_required)
    vx = np.clip(vx, -clip[0], clip[0])
    vy = np.clip(vy, -clip[0], clip[0])
    w = np.clip(w, -clip[1], clip[1])
    return vx , vy, w

def gait_command(udp, cmd, state, current, next_point, dist_to_goal, terrain_type):
    vel_clip = [0.2,0.2]
    
    udp.Recv()
    udp.GetRecv(state)
    
    quadruped_angles = state.imu.rpy

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
        vx, vy, w = gait_velocity(current, next_point, dist_to_goal, quadruped_angles, Kp=1, clip=vel_clip)
        cmd.mode = 2
        cmd.gaitType = 1
        cmd.velocity = [vx, vy] # -1  ~ +1
        cmd.yawSpeed = w
        cmd.footRaiseHeight = 0.0
        cmd.bodyHeight = 0.0

    elif terrain_type == 1: # stairs
        stair_commands()

    elif terrain_type == 2:  # stones
        vx, vy, w = gait_velocity(current, next_point, dist_to_goal, quadruped_angles, Kp=1, clip=vel_clip)
        cmd.mode = 2
        cmd.gaitType = 1
        cmd.velocity = [vx, vy] # -1  ~ +1
        cmd.yawSpeed = w
        cmd.footRaiseHeight = 0.0
        cmd.bodyHeight = 0.0

    elif terrain_type == 3:  # plane
        vx, vy, w = gait_velocity(current, next_point, dist_to_goal, quadruped_angles, Kp=1, clip=vel_clip)
        cmd.mode = 2
        cmd.gaitType = 1
        cmd.velocity = [vx, vy] # -1  ~ +1
        cmd.yawSpeed = w
        cmd.footRaiseHeight = 0.0
        cmd.bodyHeight = 0.0
    

    udp.SetSend(cmd)
    udp.Send()

    return quadruped_angles, [vx,vy,w]


def stair_commands():
    return None

def get_quadruped_angles(udp, cmd, state):

    udp.Recv()
    udp.GetRecv(state)
    quadruped_angles = state.imu.rpy
    return quadruped_angles


def adaptive_gait_selection(quadruped, sand, stair, stones, reached, completed, i, proximity_dist = 2):
    coord = [sand, stair, stones]
    start = np.array([(quadruped[0]+quadruped[2])/2,-(quadruped[1]+quadruped[3])/2])
    goal_points = coord [i]
    goal_midpoint = np.array([(goal_points[0]+goal_points[2])/2,-(goal_points[1]+goal_points[3])/2])
    l1 = abs(goal_points[0] - goal_points[2])
    l2 = abs(goal_points[1] - goal_points[3])
    if l1 > l2:
        point1 = np.array([goal_midpoint[0] - l1/2 - proximity_dist, goal_midpoint[1]])
        point2 = np.array([goal_midpoint[0] + l1/2 + proximity_dist, goal_midpoint[1]])
    else:
        point1 = np.array([goal_midpoint[0],goal_midpoint[1] - l1/2 - proximity_dist])
        point2 = np.array([goal_midpoint[0],goal_midpoint[1] + l1/2 + proximity_dist])
    dist1 = calculate_distance(start, point1)
    dist2 = calculate_distance(start, point2)
    terrain_type = 0
    
    if completed[i] == 0:
        if reached[i] == 0:
            if dist1 < dist2:
                initial_goal = point1
            else:
                initial_goal = point2
            terrain_type = 3
            goal = initial_goal
            dist_to_goal = calculate_distance(start, goal)
            if dist_to_goal < proximity_dist:
                reached[i] = 1
        else:
            terrain_type = i
            goal = point1 if initial_goal == point2 else point2
            dist_to_goal = calculate_distance(start, goal)
            if dist_to_goal < proximity_dist:
                completed[i] = 1
    else:
        i+=1
    return start, goal, terrain_type, reached, completed, i
