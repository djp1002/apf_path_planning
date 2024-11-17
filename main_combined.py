import cv2
import time
import numpy as np
from filter_box_lowpass import find_near_matches_2d
from detect_quad_v11 import yolo_detection, class_colors, class_names
from depth_detection_refined import initialize_realsense,initialize_filters, process_frames, detect_boxes
from gait_pattern import adaptive_gait_selection
from apf_vom_vector_minima_pract import apf_path
from go1_command import gait_command



def main():
    pipeline, config = initialize_realsense()
    align, colorizer, decimation, spatial, temporal, hole_filling, depth_to_disparity, disparity_to_depth = initialize_filters()    
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
    gait_type = 0
    goal_index = 0
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

            image_bounding_box, raw_image, rect_raw = detect_boxes(colorized_depth_filtered)

            if len(rect_filtered) == 0:rect_filtered = [[800, 800, 900, 900, 0]]
            rect_filtered = find_near_matches_2d(rect_raw, rect_filtered,k_low=0.25, tolerance=400,delete_value=0.05)
            # print(len(rect_filtered),rect_filtered)

            if (len(rect_filtered)>0):
                for rect in rect_filtered:
                    x1, y1, x2, y2, weight = rect
                    if weight>0.4:
                        cv2.rectangle(raw_image, (x1, y1), (x2, y2), (255, 255, 255), 4)
                        obstacle_points.append([x1, -y1, x2, -y2])
            # obstacle_points = [[100, -100, 200, -200], [150, -350, 250, -450]]             
            cv2.imshow('Step 6: Bounding Boxes', raw_image)
            print(len(obstacle_points),obstacle_points)
            color_image, yolo_results, quadruped_xy, sand_xy, stair_xy, stones_xy = yolo_detection(color_image)
            
            

            # cv2.imshow('Final Result with detection', image_bounding_box)
            # cv2.imshow('Raw depth filtered', raw_image)

            start, goal, gait_type, reached, completed, goal_index = adaptive_gait_selection(quadruped_xy, sand_xy, stair_xy, stones_xy, reached, completed, goal_index)

            magnitude, best_path,prev_path,best_magnitude, min_path_length = apf_path(start, goal,obstacle_points,magnitude,best_path,prev_path,best_magnitude, min_path_length)
            next_point = best_path[1]
            total_left_dist = len(best_path)

            
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

if __name__ == "__main__":
    main()