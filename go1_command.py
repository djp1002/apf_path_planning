#!/usr/bin/python

import sys
import numpy as np

sys.path.append('../lib/python/amd64')
import robot_interface as sdk

def gait_velocity(current, next_point, dist_to_goal):
    Kp_v = 0.01
    Kp_w = 0.01
    vx = Kp_v * (dist_to_goal)
    vy = 0
    delta = [next_point[0] - current[0], next_point[1] - current[1]]

    # Calculate the angle in radians
    theta_required = np.degrees(np.arctan2(delta[1], delta[0]))
    theta_current = 0
    theta_error = theta_required - theta_current
    if abs(theta_error) > 60:
        vx = 0
    w = Kp_w * (theta_error)

    w = np.clip(w, -1, 1)
    vx = np.clip(vx, -1, 1)
    vy = np.clip(vy, -1, 1)
    return vx , vy, w

def func(current, next_point, dist_to_goal, gait_type):
    if gait_type ==  0:
        vx, vy, w = gait_velocity(current, next_point, dist_to_goal)
        gait_command(vx, vy, w, gait_type)
    elif gait_type == 1:
        vx, vy, w = gait_velocity(current, next_point, dist_to_goal)
        gait_command(vx, vy, w, gait_type)
    elif gait_type == 2:
        vx, vy, w = gait_velocity(current, next_point, dist_to_goal)
    return None
def gait_command(vx, vy, w, gait_type):

    HIGHLEVEL = 0xee

    udp = sdk.UDP(HIGHLEVEL, 8080, "192.168.123.161", 8082)

    cmd = sdk.HighCmd()
    state = sdk.HighState()
    udp.InitCmdData(cmd)
    udp.Recv()
    udp.GetRecv(state)
    
    quad_angles = state.imu.rpy
    # print(motiontime)
    # print(state.imu.rpy[0])
    # print(motiontime, state.motorState[0].q, state.motorState[1].q, state.motorState[2].q)
    # print(state.imu.rpy[0])

    cmd.mode = 0      # 0:idle, default stand      1:forced stand     2:walk continuously
    cmd.gaitType = 0
    cmd.speedLevel = 0
    cmd.footRaiseHeight = 0
    cmd.bodyHeight = 0
    cmd.euler = [0, 0, 0]
    cmd.velocity = [0, 0]
    cmd.yawSpeed = 0.0
    cmd.reserve = 0

    # cmd.mode = 2
    # cmd.gaitType = 1
    # # cmd.position = [1, 0]
    # # cmd.position[0] = 2
    # cmd.velocity = [-0.2, 0] # -1  ~ +1
    # cmd.yawSpeed = 0
    # cmd.bodyHeight = 0.1
    
    cmd.mode = 2
    cmd.gaitType = 2
    cmd.velocity = [0.4, 0] # -1  ~ +1
    cmd.yawSpeed = 2
    cmd.footRaiseHeight = 0.1
        
    udp.SetSend(cmd)
    udp.Send()
    return quad_angles
