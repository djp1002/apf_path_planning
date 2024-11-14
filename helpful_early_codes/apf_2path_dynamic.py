#!/usr/bin/python3
import cv2
import numpy as np
import time

def to_cv2_point(point):
    """Convert NumPy point to CV2 coordinate system"""
    return (int(point[0]), -int(point[1]))

def calculate_force(p1, p2, charge):
    """Calculate force vector between two points using NumPy operations"""
    p1 = np.array(p1)
    p2 = np.array(p2)
    displacement = p2 - p1
    distance = max(np.linalg.norm(displacement), 1)
    direction = displacement / distance
    force_magnitude = 100000 * charge / distance**2
    return force_magnitude * direction

def get_closest_point_on_rectangle(point, rect_center, rect_dims):
    """Find closest point on rectangle to given point using NumPy"""
    point = np.array(point)
    rect_center = np.array(rect_center)
    rect_dims = np.array(rect_dims)
    half_dims = rect_dims 
    mins = rect_center - half_dims
    maxs = rect_center + half_dims
    closest = np.clip(point, mins, maxs)
    return closest

def calculate_next_position(current, goal, obstacles, magnitude):
    """Calculate next position using vector operations"""
    current = np.array(current)
    charge_goal = 10.0 * (1 + magnitude/100)
    charge_obstacle = -2.0 * (1 - min(magnitude, 90)/100)
    
    f_goal = calculate_force(current, goal, charge_goal)
    f_obstacles = [calculate_force(current, obstacle, charge_obstacle) for obstacle in obstacles]
    f_total = f_goal + sum(f_obstacles)
    
    direction = f_total / np.linalg.norm(f_total)
    shift = 5.0 * direction
    new_position = current + np.ceil(shift)
    return new_position

def calculate_distance(p1, p2):
    """Calculate Euclidean distance using NumPy"""
    return np.linalg.norm(np.array(p1) - np.array(p2))

def generate_path(start, goal, obstacles, obstacle_dims, magnitude, prev_path,prev_dir_path_unit):
    """Generate path from start to goal avoiding obstacles"""
    path = [np.array(start)]
    current = np.array(start)
    goal = np.array(goal)
    
    closest_points = []  # Store closest points for visualization
    
    while calculate_distance(current, goal) > 20.0:
        closest_obstacle_points = [get_closest_point_on_rectangle(current, obstacle, obstacle_dims) for obstacle in obstacles]
        closest_points.append(closest_obstacle_points)  # Store closest points
        # current = calculate_next_position(current, goal, closest_obstacle_points, magnitude)
        next_1_pos = calculate_next_position(current, goal, closest_obstacle_points, magnitude)
        next_2_pos = calculate_next_position(next_1_pos, goal, closest_obstacle_points, magnitude)
        prev_pos = current
        current = next_1_pos
        current_len = len(path)
        prev_len = len(prev_path)
        dir_path_unit = prev_dir_path_unit
        # print(current)
        if(current_len>3):
            dist_next_3 = calculate_distance(next_2_pos,prev_pos)
            print(prev_len,current_len)
            if dist_next_3 < 5:
                if(current_len > prev_len):
                    index = prev_len - 1
                    
                else:
                    index = current_len -1
                point_on_prev_path = prev_path[index]
                dir_vector_path = point_on_prev_path - prev_pos
                # print(point_on_prev_path, prev_pos)
                if np.linalg.norm(dir_vector_path) == 0:
                    dir_path_unit = prev_dir_path_unit
                    print("0 mag dir")
                else:
                    dir_path_unit = dir_vector_path / np.linalg.norm(dir_vector_path)
                    prev_dir_path_unit = dir_path_unit
                shift = 5.0 * dir_path_unit
                current = prev_pos + np.ceil(shift)

                print("stuck>>>>>>>>>>>>>>>>>>>>>")
            else:
               print("not stuck")
        
        path.append(current)
    
    return path, closest_points, dir_path_unit

def draw_scene(image, start, goal, obstacles, obstacle_dims, og_paths, cl_points):
    """Draw the current state"""
    # Draw obstacles
    for obstacle_center in obstacles:
        rect_width, rect_height = obstacle_dims
        top_left = np.array([obstacle_center[0] - rect_width, obstacle_center[1] + rect_height])
        bottom_right = np.array([obstacle_center[0] + rect_width, obstacle_center[1] - rect_height])
        cv2.rectangle(image, to_cv2_point(top_left), to_cv2_point(bottom_right), color=(0, 0, 255), thickness=-1)
    
    # Draw start and goal
    cv2.circle(image, to_cv2_point(start), radius=20, color=(50, 50, 0), thickness=1)
    cv2.circle(image, to_cv2_point(goal), radius=20, color=(0, 255, 0), thickness=-1)
    
    # Draw paths and closest points with different colors
    colors = [(100, 100, 255), (255, 255, 0)]  # Two colors for two paths
    for path, closest_points, color in zip(og_paths,cl_points, colors):
        if len(path) > 1:
            # Draw path
            for i in range(len(path)-1):
                cv2.line(image, to_cv2_point(path[i]), to_cv2_point(path[i+1]), color=color, thickness=2)
            
            # Draw closest points
            for obstacle_closest_points in closest_points:
                for point in obstacle_closest_points:
                    cv2.circle(image, to_cv2_point(point), radius=1, color=(0, 255, 255), thickness=-1)
                    # Draw line from path point to closest point
                    # path_point = path[closest_points.index(point)]
                    # cv2.line(image, to_cv2_point(path_point), to_cv2_point(point), 
                    #         color=(0, 255, 255), thickness=1, lineType=cv2.LINE_AA)

def on_trackbar_change(x):
    """Dummy function for trackbar callback"""
    pass

def main():
    # Initialize points and obstacles
    start = np.array([40, -440])
    goal = np.array([300, -140])
    prev_path = []
    
    window_name = 'APF Path Planning'
    cv2.namedWindow(window_name)
    
    # Create trackbars for obstacle positions
    for i in range(1, 6):
        cv2.createTrackbar(f'Obstacle {i} X', window_name, 101 + (i-1)*50, 640, on_trackbar_change)
        cv2.createTrackbar(f'Obstacle {i} Y', window_name, 325 - (i-1)*15, 480, on_trackbar_change)
    
    # Path planning parameters
    magnitude = 50  # Starting magnitude
    mag_diff = 20   # Difference between path magnitudes
    obstacle_dims = np.array([20, 40])
    prev_path = []
    prev_dir_path_unit = np.array([0,5])
    while True:
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        # paths = []
        path_lengths = []
        # Get obstacle positions from trackbars
        obstacles = []
        for i in range(1, 6):
            obstacle_x = cv2.getTrackbarPos(f'Obstacle {i} X', window_name)
            obstacle_y = cv2.getTrackbarPos(f'Obstacle {i} Y', window_name)
            obstacle_center = np.array([obstacle_x, -obstacle_y])  # Negative Y for CV2 coordinate system
            obstacles.append(obstacle_center)
        
        # Generate two paths with different magnitudes
        magnitudes = [magnitude, magnitude + mag_diff]
        og_paths, closest_points, prev_dir_path_unit_list = zip(*[generate_path(start, goal, obstacles, obstacle_dims, mag, prev_path,prev_dir_path_unit) 
                          for mag in magnitudes])
        
        
        # Compare path lengths
        # path_lengths = [len(path) for path, _ in paths_and_points]
        
        path_lengths = [len(og_paths[0]),len(og_paths[1])]
        # path_lengths[1] = len(og_paths[1])

        
        
        if path_lengths[0] < path_lengths[1]:
            magnitude -= mag_diff
            prev_path = og_paths[0]
            prev_dir_path_unit = prev_dir_path_unit_list[0]
    
        else:
            magnitude += mag_diff
            prev_path = og_paths[1]
            prev_dir_path_unit = prev_dir_path_unit_list[1]

        # print(len(prev_path))

        # Visualize
        draw_scene(image, start, goal, obstacles, obstacle_dims, og_paths,closest_points)
        
        cv2.imshow(window_name, image)
        if cv2.waitKey(1) & 0xFF == 27:
            break
        # time.sleep(0.2)
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()