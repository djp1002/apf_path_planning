import cv2
import time
import numpy as np
from filter_box_lowpass import find_near_matches_2d
from detect_quad_v11 import yolo_detection, class_colors, class_names
from depth_detection_refined import initialize_realsense,initialize_filters, process_frames, detect_boxes
from apf_vom_vector_minima_pract import apf_path, calculate_distance
from go1_command import gait_command, adaptive_gait_selection, init_udp, tf_q_g2, tf_g_i2
from tf.transformations import euler_from_quaternion
from custom_msg_python.msg import custom
from geometry_msgs.msg import TwistStamped
import rospy
from sensor_msgs.msg import Imu
from mavros_msgs.msg import RCIn

# cd legged_ws/src/unitree_legged_sdk/example_py/mera_bag/
# rosbag record /custom_topic
global uav_yaw, initial_uav_yaw, uav_yaw_set, rc_ch, uav_raw_yaw
uav_yaw = 0.0
rc_ch = [1500,1500,1500,1500,1500,1100]
initial_uav_yaw = 0.0
uav_yaw_set = 0
uav_raw_yaw = 0
def main():
    global uav_yaw, rc_ch, initial_uav_yaw, uav_raw_yaw
    pipeline, config = initialize_realsense()
    align, colorizer, decimation, spatial, temporal, hole_filling, depth_to_disparity, disparity_to_depth = initialize_filters()    

    rospy.init_node('quadruped_node', anonymous=True) 
    rospy.Subscriber("/mavros/imu/data", Imu, imu_callback)
    rospy.Subscriber("/mavros/rc/in", RCIn, rc_callback)
    vel_pub = rospy.Publisher('/mavros/setpoint_velocity/cmd_vel', TwistStamped, queue_size=10)
    custom_pub=rospy.Publisher("/custom_topic",custom,queue_size=10)
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    # writer_depth_box = cv2.VideoWriter('output_videos/depth_box_1.avi', fourcc, 30, (640, 480))
    # writer_depth_raw = cv2.VideoWriter('output_videos/depth_raw_1.avi', fourcc, 30, (640, 480))
    writer_rgb = cv2.VideoWriter('rgb_depth.avi', fourcc, 30, (1920, 480))
    apf_image = np.zeros((480, 640, 3), dtype=np.uint8)
    rect_filtered =[]
    pipeline.start(config)
    start = np.array([40, -440])
    goal = np.array([300, -140])
    prev_path = []
    best_path = None
    # Path planning parameters
    magnitude = 50  # Starting magnitude

    dist_tolerance_obstacles = 30


    best_magnitude = magnitude
    min_path_length = float('inf')

    reached = [0,0,0]
    completed = [0,0,0]
    # reached = [1,1,0]
    # completed = [1,1,0]
    mid_reached = 0
    oriented = 0
    terrain_type = 0
    goal_index = 0
    udp, cmd, state =init_udp()
    # quadruped_yaw_initial = get_quadruped_angles(udp, cmd, state)
    quadruped_yaw_initial = 0
    quadruped_yaw = 0
    quadruped_xy, sand_xy, stair_xy, stones_xy = [310,-230,330,-250], [400,-200,420,-220],  [600,-200,620,-220], [200,-30,240,-90]
    initial_goal =[]
    next_goal = []
    quadruped_raw_angles = [0,0,0]
    kp_v = 0.1
    kp_w = 0.02
    pid_gains = [0.005,-0.000]
    xpid_gains = ypid_gains = zpid_gains = pid_gains
    image_center = [640/2,-480/2]
    prev_err_x = prev_err_y = prev_err_z = 0
    plot_data=custom()
    plot_data.header.frame_id="map"
    
    quadruped_raw_angles = [0,0,0]
    quadruped_velocity = [0,0,0]
    total_left_dist = 100000
    terrain_type = 10
    curr_err_x = curr_err_y = 0
    raw_length = filter_length = weight_d = 0

    try:
        while True:
            obstacle_points = []
            start_time = time.time()
            
            colorized_depth_filtered, color_image, average_depth = process_frames(
                pipeline, align, colorizer, decimation, spatial, temporal, 
                hole_filling, depth_to_disparity, disparity_to_depth
            )
            if colorized_depth_filtered is None:
                continue
            
            color_image, yolo_results, quadruped_xy, sand_xy, stair_xy, stones_xy = yolo_detection(
                                            color_image, quadruped_xy, sand_xy, stair_xy, stones_xy)
            
            
            # important transformations  -------------------------------------------------------------------------->>>
            quadruped_xy_g = tf_g_i2(quadruped_xy, uav_yaw)
            quadruped_center_g = [(quadruped_xy_g[0][0] + quadruped_xy_g[1][0] + quadruped_xy_g[2][0] + quadruped_xy_g[3][0] )/4, 
                                  (quadruped_xy_g[0][1] + quadruped_xy_g[1][1] + quadruped_xy_g[2][1] + quadruped_xy_g[3][1] )/4]

            # transformations to quadruped frame ------------------------------------------------------------------>>>
            quadruped_xy_q = tf_q_g2( tf_g_i2(quadruped_xy, uav_yaw), quadruped_center_g, quadruped_yaw)
            sand_xy_q = tf_q_g2( tf_g_i2(sand_xy, uav_yaw), quadruped_center_g, quadruped_yaw)
            stair_xy_q = tf_q_g2( tf_g_i2(stair_xy, uav_yaw), quadruped_center_g, quadruped_yaw)
            stones_xy_q = tf_q_g2( tf_g_i2(stones_xy, uav_yaw), quadruped_center_g, quadruped_yaw)

            # quadruped_xy_q2 = np.array([np.min(quadruped_xy_q[:,0]),np.max(quadruped_xy_q[:,1]),np.max(quadruped_xy_q[:,0]),np.min(quadruped_xy_q[:,1])])

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
                        obstacle_center_i = [(obstacle_i[0] + obstacle_i[2])/2,(obstacle_i[1] + obstacle_i[3])/2]
                        quadruped_center_i = [(quadruped_xy[0] + quadruped_xy[2])/2,(quadruped_xy[1] + quadruped_xy[3])/2]
                        stair_center_i = [(stair_xy[0] + stair_xy[2])/2,(stair_xy[1] + stair_xy[3])/2]
                        dist_stair = calculate_distance(stair_center_i, obstacle_center_i)
                        dist_quadruped = calculate_distance(quadruped_center_i, obstacle_center_i)
                        if dist_quadruped > dist_tolerance_obstacles and dist_stair > dist_tolerance_obstacles:
                            obstacle_q = tf_q_g2( tf_g_i2(obstacle_i, uav_yaw), quadruped_center_g, quadruped_yaw)
                            obstacle_points.append(obstacle_q)
            # obstacle_points = [[100, -100, 200, -200], [150, -350, 250, -450]]             
            # cv2.imshow('Step 6: Bounding Boxes', raw_image)
            # print(obstacle_points)
            
            
            # cv2.imshow('Final Result with detection', image_bounding_box)
            # cv2.imshow('Raw depth filtered', raw_image)
            # print(rc_ch)
            if rc_ch[5] > 1700:
                # print(rc_ch)
                if goal_index < 3:
                    start, goal, terrain_type, reached, completed, goal_index, initial_goal, next_goal, goal_points = adaptive_gait_selection(
                        quadruped_xy_q, sand_xy_q, stair_xy_q, stones_xy_q, reached, completed, goal_index, initial_goal, next_goal, proximity_dist=20)
                    
                    apf_image, magnitude, best_path, prev_path, best_magnitude, min_path_length = apf_path(
                        quadruped_xy_q, goal,obstacle_points,magnitude,best_path,prev_path,best_magnitude, min_path_length, goal_points, next_goal, proximity_dist=10)
                    next_point = prev_path[1] if prev_path else start
                    # print(start, goal, next_point)
                    total_left_dist = len(prev_path)
                    
                    vel_clip = [0.2,3]
                    # if rc_ch<1600:
                    #     vel_clip = [0 , 0]

                    quadruped_yaw, quadruped_velocity, quadruped_yaw_initial, quadruped_raw_angles = gait_command(udp, cmd, state, quadruped_yaw_initial, start, next_point, total_left_dist, terrain_type, Kp=[kp_v,kp_w], vel_clip = vel_clip)
                    print("quadruped---------------------------------------------------------->", "terrain type",terrain_type, quadruped_velocity)
                    # print("goals", initial_goal, next_goal, goal_points, completed, reached)
                    # print("drone yaw angle", quadruped_velocity, quadruped_yaw, quadruped_yaw_initial)
                # elif goal_index == 2:
                #     start, goal, terrain_type, reached, completed, initial_goal, next_goal, goal_points, mid_reached, oriented, vel = stair_mode_gait(quadruped_xy_q, stair_xy_q, reached, completed, initial_goal, next_goal, mid_reached, oriented, proximity_dist = 20, goal_offset_dist = 30)
                #     print("quadruped---------------------------------------------------------->", "terrain type",terrain_type, mid_reached, oriented)

                #     if mid_reached == 0:
                #         apf_image, magnitude, best_path, prev_path, best_magnitude, min_path_length = apf_path(
                #         quadruped_xy_q, goal,obstacle_points,magnitude,best_path,prev_path,best_magnitude, min_path_length, goal_points, next_goal, proximity_dist=10)
                #         next_point = prev_path[1] if prev_path else start
                #         # print(start, goal, next_point)
                #         total_left_dist = len(prev_path)
                #         vel_clip = [0.4,1.0]
                #         quadruped_yaw, quadruped_velocity, quadruped_yaw_initial = gait_command(udp, cmd, state, quadruped_yaw_initial, start, next_point, total_left_dist, terrain_type, Kp=[kp_v,kp_w], vel_clip = vel_clip)

                #     elif mid_reached == 1 and oriented == 0:
                #         quadruped_yaw, quadruped_yaw_initial = stair_command_vel(udp, cmd, state, quadruped_yaw_initial, vel)
                #         apf_image, magnitude, best_path, prev_path, best_magnitude, min_path_length = apf_path(
                #         quadruped_xy_q, goal,obstacle_points,magnitude,best_path,prev_path,best_magnitude, min_path_length, goal_points, next_goal, proximity_dist=10)
                #         next_point = prev_path[1] if prev_path else start
                #     elif mid_reached ==1 and oriented == 1:
                #         apf_image, magnitude, best_path, prev_path, best_magnitude, min_path_length = apf_path(
                #         quadruped_xy_q, goal,obstacle_points,magnitude,best_path,prev_path,best_magnitude, min_path_length, goal_points, next_goal, proximity_dist=10)
                #         next_point = prev_path[1] if prev_path else start
                #         # print(start, goal, next_point)
                #         total_left_dist = len(prev_path)
                #         vel_clip = [0.4,1.0]
                #         quadruped_yaw, quadruped_velocity, quadruped_yaw_initial = stair_inv_command_vel(udp, cmd, state, quadruped_yaw_initial, start, next_point, total_left_dist, terrain_type, Kp=[kp_v,kp_w], vel_clip = vel_clip)



                else:
                    print("course, complete")
                    break
            # print("velocity ", quadruped_velocity, terrain_type, quadruped_angles)
            # print("quadruped angles", quadruped_angles)

            
            # if gait_type == 1:
            #     stair_mode()
            curr_err_x = image_center[0] - (quadruped_xy[0] + quadruped_xy[2])/2
            curr_err_y = image_center[1] - (quadruped_xy[1] + quadruped_xy[3])/2
            curr_err_z = -(average_depth - 0.06)*3000
            # print("error z", curr_err_z)
            if abs(curr_err_x) < 20: curr_err_x = 0
            if abs(curr_err_y) < 20: curr_err_y = 0
            vy_b = pid(curr_err_x, prev_err_x, xpid_gains)
    
            vx_b = -pid(curr_err_y, prev_err_y, ypid_gains)

            vz_b = pid(curr_err_z, prev_err_z, zpid_gains)
            # print(vz_b)
            prev_err_x = curr_err_x
            prev_err_y = curr_err_y
            prev_err_z = curr_err_z
            # yaw_rate =-0.1*(initial_psi-psi)
            limit = 0.0
            vx_b=np.clip(vx_b,-limit,limit)
            vy_b=np.clip(vy_b,-limit,limit)
            vz_b=np.clip(vz_b,-limit,limit)
            vy_rc = -(rc_ch[0] - 1500)/500
            vx_rc = (rc_ch[1] - 1500)/500
            vz_rc = (rc_ch[2] - 1500)/500
            w_rc = -(rc_ch[3] - 1500)/500
            vx_b_comb = vx_rc + vx_b
            vy_b_comb = vy_rc + vy_b
            vz_b = vz_rc + vz_b
            vx_g = np.cos(uav_raw_yaw)*vx_b_comb - np.sin(uav_raw_yaw)*vy_b_comb
            vy_g = np.sin(uav_raw_yaw)*vx_b_comb + np.cos(uav_raw_yaw)*vy_b_comb
            vz_g = vz_b

            vel_msg = TwistStamped()
            vel_msg.twist.linear.x = vx_g
            vel_msg.twist.linear.y = vy_g
            vel_msg.twist.linear.z = vz_g
            vel_msg.twist.angular.z = w_rc
            vel_pub.publish(vel_msg)

            end_time = time.time()
            elapsed_time = end_time - start_time
            fps = 1/ elapsed_time
            start_time = end_time


            plot_data.header.stamp = rospy.Time.now()
            plot_data.uav_yaw = raw_length
            plot_data.uav_vx = filter_length
            plot_data.uav_vy = weight_d
            plot_data.quad_roll = quadruped_raw_angles[0]
            plot_data.quad_pitch = quadruped_raw_angles[1]
            plot_data.quad_yaw = quadruped_raw_angles[2]
            plot_data.quad_vx = quadruped_velocity[0]
            plot_data.quad_vy = quadruped_velocity[1]
            plot_data.quad_w = quadruped_velocity[2]
            plot_data.d_goal = total_left_dist
            plot_data.terrain_id = terrain_type
            plot_data.apf_magnitude = magnitude
            plot_data.rc_ch = rc_ch[5]
            plot_data.quad_uav_er_x = curr_err_x
            plot_data.quad_uav_er_y = curr_err_y
            custom_pub.publish(plot_data)

            # print(quadruped_xy[0]/2 + quadruped_xy[2]/2, quadruped_xy[1]/2 + quadruped_xy[3]/2,image_center, vx_b, vy_b,"fps", fps)
            # print("average_depth", average_depth, "vz", vz_b)
            raw_length = len(rect_raw)
            filter_length = len(rect_filtered)
            if filter_length>0:
                weight_d = rect_filtered[0][-1]
            else:
                weight_d = 0
            print(raw_length,filter_length,weight_d)

            cv2.putText(color_image, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            rgbd_image = cv2.hconcat([raw_image, color_image])
            final_image = cv2.hconcat([rgbd_image, apf_image]) 
            cv2.imshow('RealSense RGB', final_image)

            # writer_depth_box.write(image_bounding_box)
            # writer_depth_raw.write(raw_image)
            writer_rgb.write(final_image)
            if cv2.waitKey(1) == 27:  # ESC key
                break

    finally:
        pipeline.stop()
        # writer_depth_box.release()
        # writer_depth_raw.release()
        writer_rgb.release()
        cv2.destroyAllWindows()

def imu_callback(data):
    global uav_yaw, uav_yaw_set, initial_uav_yaw, uav_raw_yaw
    quat = [data.orientation.w, data.orientation.x, data.orientation.y, data.orientation.z]
    angles = euler_from_quaternion(quat)
    quat_raw = [data.orientation.x, data.orientation.y, data.orientation.z,data.orientation.w]
    angles_raw = euler_from_quaternion(quat_raw)
    uav_raw_yaw = angles_raw[2]
    uav_yaw = -angles[0] - initial_uav_yaw
    if uav_yaw > np.pi:
        uav_yaw = - 2*np.pi + uav_yaw

    elif uav_yaw < -np.pi:
        uav_yaw = 2*np.pi + uav_yaw

    if uav_yaw_set == 0:
        initial_uav_yaw = uav_yaw
        uav_yaw_set = 1

    # print(an,uav_yaw)
def pid(current_error,previous_error,pid_gain):
    
    p_value=pid_gain[0]*(current_error)
    d_value=pid_gain[1]*(previous_error - current_error)
    velocity =  p_value + d_value
    # print("yyyyyy",e_ly[-1],e_ly[-2])
    return velocity

def rc_callback(data):
    global rc_ch
    ch = data.channels
    rc_ch = ch

if __name__ == "__main__":
    main()