#!/usr/bin/python

import sys
import time
import math

sys.path.append('../lib/python/arm64')
import robot_interface as sdk


if __name__ == '__main__':

    HIGHLEVEL = 0xee
    LOWLEVEL  = 0xff

    # udp = sdk.UDP(HIGHLEVEL, 8080, "192.168.123.161", 8082)
    udp = sdk.UDP(HIGHLEVEL, 8080, "192.168.1.170", 8082)


    cmd = sdk.HighCmd()
    state = sdk.HighState()
    udp.InitCmdData(cmd)

    motiontime = 0
    while True:
        time.sleep(0.002)
        motiontime = motiontime + 1

        udp.Recv()
        udp.GetRecv(state)
        
        # print(motiontime)
        # print(state.imu.rpy[0])
        # print(motiontime, state.motorState[0].q, state.motorState[1].q, state.motorState[2].q)
        # print(state.imu.rpy[0])

        cmd.mode = 0      # 0:idle, default stand      1:forced stand     2:walk continuously
        cmd.gaitType = 0  # 0 -> idle, 1 -> trot, 2 -> trot running, 3 -> stair climbing, 4 -> trot obstacle
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

        if(motiontime > 0 and motiontime <= 800):
            cmd.mode = 2
            cmd.gaitType = 1
            cmd.velocity = [0.111, 0.111] # -1  ~ +1
            cmd.yawSpeed = 1
            cmd.footRaiseHeight = -0.5
            cmd.bodyHeight = -0.2
            print("stair",motiontime)
        
        if(motiontime > 4000 and motiontime < 6000):
            cmd.mode = 0
            cmd.velocity = [0, 0]
            print("idle", motiontime)
        
        if(motiontime > 6000 and motiontime < 8000):
            cmd.mode = 2
            cmd.gaitType = 1
            cmd.velocity = [0.0, 0] # -1  ~ +1
            # cmd.bodyHeight = 0.0
            print("walk", motiontime)
            

        udp.SetSend(cmd)
        udp.Send()
