import cv2
import time
import numpy as np
from filter_box_lowpass import find_near_matches_2d
from detect_quad_v11 import yolo_detection, class_colors, class_names
from depth_detection_refined import initialize_realsense,initialize_filters, process_frames, detect_boxes
from apf_vom_vector_minima_pract import apf_path
from go1_command import gait_command, get_quadruped_angles, adaptive_gait_selection, init_udp, tf_q_g2, tf_g_i2
from tf.transformations import euler_from_quaternion
import rospy
from sensor_msgs.msg import Imu
from mavros_msgs.msg import RCIn

global uav_yaw, rc_ch
uav_yaw = 0.0
rc_ch = 0
def main():
    global uav_yaw, rc_ch
    pipeline, config = initialize_realsense()
    align, colorizer, decimation, spatial, temporal, hole_filling, depth_to_disparity, disparity_to_depth = initialize_filters()    

    rospy.init_node('quadruped_node', anonymous=True) 
    rospy.Subscriber("/mavros/imu/data", Imu, imu_callback)
    rospy.Subscriber("/mavros/rc/in", RCIn, rc_callback)
    # fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    # writer_depth_box = cv2.VideoWriter('output_videos/depth_box_1.avi', fourcc, 30, (640, 480))
    # writer_depth_raw = cv2.VideoWriter('output_videos/depth_raw_1.avi', fourcc, 30, (640, 480))
    # writer_rgb = cv2.VideoWriter('output_videos/rgb_22_04_45.avi', fourcc, 30, (640, 480))
    rect_filtered =[]
    pipeline.start(config)
    start = np.array([40, -440])
    goal = np.array([300, -140])
    prev_path = []
    best_path = None
    # Path planning parameters
    magnitude = 50  # Starting magnitude

    best_magnitude = magnitude
    min_path_length = float('inf')

    reached = [0,0,0]
    completed = [0,0,0]
    terrain_type = 0
    goal_index = 0
    udp, cmd, state =init_udp()
    quadruped_yaw = get_quadruped_angles(udp, cmd, state)
    quadruped_xy, sand_xy, stair_xy, stones_xy = [100,-200,110,-210], [600,-200,600,-200], [0,0,0,0], [0,0,0,0]
    try:
        while True:
            obstacle_points = []
            start_time = time.time()
            
            colorized_depth_filtered, color_image = process_frames(
                pipeline, align, colorizer, decimation, spatial, temporal, 
                hole_filling, depth_to_disparity, disparity_to_depth
            )
            if colorized_depth_filtered is None:
                continue
            
            color_image, yolo_results, quadruped_xy, sand_xy, stair_xy, stones_xy = yolo_detection(color_image, quadruped_xy, sand_xy, stair_xy, stones_xy)
            
            # important transformations  -------------------------------------------------------------------------->>>
            quadruped_xy_g = tf_g_i2(quadruped_xy, uav_yaw)
            quadruped_center_g = [(quadruped_xy_g[0]+quadruped_xy_g[2])/2, (quadruped_xy_g[1]+quadruped_xy_g[3])/2]

            # transformations to quadruped frame ------------------------------------------------------------------>>>
            quadruped_xy_q = tf_q_g2( tf_g_i2(quadruped_xy, uav_yaw), quadruped_center_g, quadruped_yaw)
            sand_xy_q = tf_q_g2( tf_g_i2(sand_xy, uav_yaw), quadruped_center_g, quadruped_yaw)
            stair_xy_q = tf_q_g2( tf_g_i2(stair_xy, uav_yaw), quadruped_center_g, quadruped_yaw)
            stones_xy_q = tf_q_g2( tf_g_i2(stones_xy, uav_yaw), quadruped_center_g, quadruped_yaw)

            # print("tranformation_ check", quadruped_xy, uav_yaw)
            # print(quadruped_xy,quadruped_center_i)
            image_bounding_box, raw_image, rect_raw = detect_boxes(colorized_depth_filtered)

            if len(rect_filtered) == 0:rect_filtered = [[2000, 2000, 2010, 2010, 0]]
            rect_filtered = find_near_matches_2d(rect_raw, rect_filtered, dim_inc=10, k_low=0.25, tolerance=400, delete_value=0.05)
            # print(len(rect_filtered),rect_filtered)

            if (len(rect_filtered)>0):
                for rect in rect_filtered:
                    x1, y1, x2, y2, weight = rect
                    if weight>0.4:
                        cv2.rectangle(raw_image, (x1, y1), (x2, y2), (255, 255, 255), 4)
                        obstacle_i = [x1, -y1, x2, -y2]
                        obstacle_q = tf_q_g2( tf_g_i2(obstacle_i, uav_yaw), quadruped_center_g, quadruped_yaw)
                        obstacle_points.append(obstacle_q)
            # obstacle_points = [[100, -100, 200, -200], [150, -350, 250, -450]]             
            cv2.imshow('Step 6: Bounding Boxes', raw_image)
            # print(obstacle_points)
            
            
            # cv2.imshow('Final Result with detection', image_bounding_box)
            # cv2.imshow('Raw depth filtered', raw_image)

            if goal_index < 3:
                start, goal, terrain_type, reached, completed, goal_index = adaptive_gait_selection(quadruped_xy_q, sand_xy_q, stair_xy_q, stones_xy_q, reached, completed, goal_index)
                magnitude, best_path, prev_path, best_magnitude, min_path_length = apf_path(start, goal,obstacle_points,magnitude,best_path,prev_path,best_magnitude, min_path_length)
                next_point = prev_path[1]
                # print(start, goal, next_point)
                total_left_dist = len(best_path)
                
                vel_clip = [0.2,0.2]
                if rc_ch<1600:
                    vel_clip = [0 , 0]

                quadruped_yaw, quadruped_velocity = gait_command(udp, cmd, state, start, next_point, total_left_dist, terrain_type, vel_clip)
                print("quadruped yaw and velocity", quadruped_yaw, quadruped_velocity, terrain_type, uav_yaw)
                # print(start, goal)
                # print("drone yaw angle", uav_yaw)
            else:
                print("course, complete")
                break
            # print("velocity ", quadruped_velocity, terrain_type, quadruped_angles)
            # print("quadruped angles", quadruped_angles)

            
            # if gait_type == 1:
            #     stair_mode()
            end_time = time.time()
            elapsed_time = end_time - start_time
            fps = 1/ elapsed_time
            start_time = end_time

            cv2.putText(color_image, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow('RealSense RGB', color_image)

            # writer_depth_box.write(image_bounding_box)
            # writer_depth_raw.write(raw_image)
            # writer_rgb.write(color_image)
            if cv2.waitKey(1) == 27:  # ESC key
                break

    finally:
        pipeline.stop()
        # writer_depth_box.release()
        # writer_depth_raw.release()
        # writer_rgb.release()
        cv2.destroyAllWindows()

def imu_callback(data):
    global uav_yaw
    quat = [data.orientation.w, data.orientation.x, data.orientation.y, data.orientation.z]
    angles = euler_from_quaternion(quat)
    uav_yaw = -angles[0] - (150 * np.pi / 180)
    if uav_yaw > np.pi:
        uav_yaw = - 2*np.pi + uav_yaw

    elif uav_yaw < -np.pi:
        uav_yaw = 2*np.pi + uav_yaw

    # print(an,uav_yaw)

def rc_callback(data):
    global rc_ch
    ch = data.channels
    rc_ch = ch[5]

if __name__ == "__main__":
    main()