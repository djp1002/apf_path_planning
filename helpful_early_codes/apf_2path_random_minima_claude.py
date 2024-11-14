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

# First, add these new functions after your existing helper functions:
def place_virtual_obstacle(stuck_position, goal, obstacles, obstacle_dims, prev_path):
    """Place virtual obstacle considering local environment and previous path"""
    # Find nearby obstacles
    nearby_obstacles = []
    for obs in obstacles:
        if calculate_distance(stuck_position, obs) < 50.0:  # Detection radius
            closest_point = get_closest_point_on_rectangle(stuck_position, obs, obstacle_dims)
            nearby_obstacles.append(closest_point)
    
    if nearby_obstacles:
        # Original logic for obstacle-nearby case
        avg_obstacle = np.mean(nearby_obstacles, axis=0)
        to_obstacles = avg_obstacle - stuck_position
        to_obstacles = to_obstacles / np.linalg.norm(to_obstacles)
        virtual_pos = stuck_position - to_obstacles * 30.0
    else:
        # Improved open space case using previous path
        to_goal = goal - stuck_position
        to_goal = to_goal / np.linalg.norm(to_goal)
        
        # Get perpendicular directions (both options)
        perp_right = np.array([-to_goal[1], to_goal[0]])
        perp_left = np.array([to_goal[1], -to_goal[0]])
        
        # If we have previous path points, use them to decide direction
        if len(prev_path) > 0:
            # Find recent previous positions (last 5 points)
            recent_positions = prev_path[-5:] if len(prev_path) >= 5 else prev_path
            avg_prev_pos = np.mean(recent_positions, axis=0)
            
            # Vector from average previous position to current position
            prev_to_current = stuck_position - avg_prev_pos
            
            # Choose direction based on which perpendicular better matches movement history
            dot_right = np.dot(prev_to_current, perp_right)
            dot_left = np.dot(prev_to_current, perp_left)
            
            # Choose the perpendicular direction that aligns better with previous movement
            perpendicular = perp_right if dot_right >= dot_left else perp_left
            print(f"Chose {'right' if dot_right >= dot_left else 'left'} based on previous path")
        else:
            # If no previous path, choose randomly but consistently
            random_seed = int(np.sum(stuck_position) * 100)  # Create seed from position
            np.random.seed(random_seed)
            perpendicular = perp_right if np.random.rand() > 0.5 else perp_left
            
        virtual_pos = stuck_position + perpendicular * 30.0
        
    return virtual_pos

def manage_virtual_obstacles(virtual_obstacles, current_pos, max_obstacles=3):
    """Remove distant virtual obstacles"""
    if not virtual_obstacles:
        return []
    
    # Keep only obstacles within certain distance
    updated = [obs for obs in virtual_obstacles 
              if calculate_distance(current_pos, obs) < 100.0]
    
    # Keep only most recent obstacles if too many
    if len(updated) > max_obstacles:
        updated = updated[-max_obstacles:]
    
    return updated

# Then modify your generate_path function:
def generate_path(start, goal, obstacles, obstacle_dims, magnitude, prev_path, prev_dir_path_unit):
    """Generate path from start to goal avoiding obstacles"""
    path = [np.array(start)]
    current = np.array(start)
    goal = np.array(goal)
    closest_points = []
    virtual_obstacles = []  # List to store virtual obstacles
    stuck_counter = 0
    last_positions = []
    stuck_positions = []  # Track positions where we get stuck
    dir_path_unit = prev_dir_path_unit
    
    while calculate_distance(current, goal) > 20.0:
        # Get closest points for real obstacles
        closest_obstacle_points = [get_closest_point_on_rectangle(current, obstacle, obstacle_dims)
                                 for obstacle in obstacles]
        
        # Add virtual obstacles to repulsion calculation
        all_repulsion_points = closest_obstacle_points + virtual_obstacles
        closest_points.append(closest_obstacle_points)  # Keep original points for visualization
        
        next_1_pos = calculate_next_position(current, goal, all_repulsion_points, magnitude)
        next_2_pos = calculate_next_position(next_1_pos, goal, all_repulsion_points, magnitude)
        prev_pos = current
        current = next_1_pos
        
        # Detect if stuck
        if len(path) > 3:
            dist_next_3 = calculate_distance(next_2_pos, prev_pos)
            
            # Store recent positions
            last_positions.append(np.copy(current))
            if len(last_positions) > 5:
                last_positions.pop(0)
            
            # Check if stuck
            stuck = False
            if len(last_positions) >= 3:
                center = np.mean(last_positions, axis=0)
                if all(calculate_distance(pos, center) < 5.0 for pos in last_positions[-3:]):
                    stuck = True
            
            if stuck or dist_next_3 < 5:
                stuck_counter += 1
                stuck_positions.append(np.copy(current))
                
                if stuck_counter > 5:  # If stuck for multiple iterations
                    # Calculate adaptive magnitude limit
                    max_magnitude = calculate_adaptive_magnitude(current, goal, obstacles, obstacle_dims, magnitude)
                    
                    # Calculate appropriate magnitude reduction
                    magnitude_reduction = calculate_magnitude_reduction(current, stuck_positions, obstacles, obstacle_dims)
                    
                    # Place and add new virtual obstacle
                    virtual_pos = place_virtual_obstacle(current, goal, obstacles, obstacle_dims, prev_path)
                    virtual_obstacles.append(virtual_pos)
                    
                    # Apply magnitude adjustments
                    if magnitude > max_magnitude:
                        magnitude = max_magnitude
                    magnitude = max(20, magnitude - magnitude_reduction)  # Ensure magnitude doesn't go too low
                    
                    stuck_counter = 0
                    print(f"Added virtual obstacle. Magnitude: {magnitude:.1f}, Reduction: {magnitude_reduction:.1f}")
                    
                    # Keep stuck_positions list manageable
                    if len(stuck_positions) > 10:
                        stuck_positions = stuck_positions[-10:]
            else:
                stuck_counter = 0
            
            # Manage virtual obstacles
            virtual_obstacles = manage_virtual_obstacles(virtual_obstacles, current)
            
        path.append(current)
        
        # Update direction unit
        movement = current - prev_pos
        if np.linalg.norm(movement) > 0:
            dir_path_unit = movement / np.linalg.norm(movement)
            
        print(f"Current magnitude: {magnitude:.1f}")
    
    return path, closest_points, dir_path_unit, virtual_obstacles

def calculate_adaptive_magnitude(current_pos, goal, obstacles, obstacle_dims, base_magnitude):
    """Calculate adaptive magnitude based on environment"""
    # Get distances to obstacles and goal
    goal_dist = calculate_distance(current_pos, goal)
    
    # Get closest obstacle distance
    obstacle_distances = []
    for obs in obstacles:
        closest_point = get_closest_point_on_rectangle(current_pos, obs, obstacle_dims)
        dist = calculate_distance(current_pos, closest_point)
        obstacle_distances.append(dist)
    
    min_obstacle_dist = min(obstacle_distances) if obstacle_distances else float('inf')
    
    # Calculate safe magnitude limit based on distances
    # When close to obstacles, need smaller magnitude to maintain control
    # When far from obstacles, can have larger magnitude
    distance_factor = min(1.0, min_obstacle_dist / 100.0)  # Normalize to 1
    goal_factor = min(1.0, goal_dist / 200.0)  # Consider distance to goal
    
    max_magnitude = base_magnitude * (1 + 2 * distance_factor * goal_factor)
    
    return max_magnitude

def calculate_magnitude_reduction(current_pos, stuck_positions, obstacles, obstacle_dims):
    """Calculate how much to reduce magnitude when stuck"""
    # Calculate density of stuck positions
    if len(stuck_positions) < 2:
        return 30  # Default reduction if not enough history
        
    # Calculate average distance between stuck positions
    distances = []
    for i in range(len(stuck_positions)-1):
        dist = calculate_distance(stuck_positions[i], stuck_positions[i+1])
        distances.append(dist)
    
    avg_stuck_distance = np.mean(distances)
    
    # Get closest obstacle distance
    obstacle_distances = []
    for obs in obstacles:
        closest_point = get_closest_point_on_rectangle(current_pos, obs, obstacle_dims)
        dist = calculate_distance(current_pos, closest_point)
        obstacle_distances.append(dist)
    
    min_obstacle_dist = min(obstacle_distances) if obstacle_distances else float('inf')
    
    # Calculate reduction based on:
    # 1. How tightly we're stuck (smaller distances = more reduction)
    # 2. Proximity to obstacles (closer = more reduction)
    # 3. Base the reduction on the current force magnitudes
    
    stuck_factor = max(0.2, min(1.0, avg_stuck_distance / 20.0))
    obstacle_factor = max(0.2, min(1.0, min_obstacle_dist / 50.0))
    
    # Calculate adaptive reduction
    base_reduction = 50  # Base reduction value
    reduction = base_reduction * (1 - stuck_factor * obstacle_factor)
    
    return reduction


# Finally, modify your draw_scene function to visualize virtual obstacles:

def draw_scene(image, start, goal, obstacles, obstacle_dims, og_paths, cl_points, virtual_obstacles_list):
    """Draw the current state"""
    # Original obstacle drawing
    for obstacle_center in obstacles:
        rect_width, rect_height = obstacle_dims
        top_left = np.array([obstacle_center[0] - rect_width, obstacle_center[1] + rect_height])
        bottom_right = np.array([obstacle_center[0] + rect_width, obstacle_center[1] - rect_height])
        cv2.rectangle(image, to_cv2_point(top_left), to_cv2_point(bottom_right), 
                     color=(0, 0, 255), thickness=-1)
    
    # Draw virtual obstacles
    for v_obstacles in virtual_obstacles_list:
        for v_obs in v_obstacles:
            cv2.circle(image, to_cv2_point(v_obs), radius=10, 
                      color=(0, 150, 150), thickness=-1)
    
    # Draw start and goal
    cv2.circle(image, to_cv2_point(start), radius=20, color=(50, 50, 0), thickness=1)
    cv2.circle(image, to_cv2_point(goal), radius=20, color=(0, 255, 0), thickness=-1)
    
    # Draw paths
    colors = [(100, 100, 255), (255, 255, 0)]
    for path, closest_points, color in zip(og_paths, cl_points, colors):
        if len(path) > 1:
            for i in range(len(path)-1):
                cv2.line(image, to_cv2_point(path[i]), to_cv2_point(path[i+1]), 
                        color=color, thickness=2)
            
            for obstacle_closest_points in closest_points:
                for point in obstacle_closest_points:
                    cv2.circle(image, to_cv2_point(point), radius=3, 
                             color=(0, 255, 255), thickness=-1)
    return None

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
        # og_paths, closest_points, prev_dir_path_unit_list = zip(*[generate_path(start, goal, obstacles, obstacle_dims, mag, prev_path,prev_dir_path_unit) 
        #                   for mag in magnitudes])
        results = [generate_path(start, goal, obstacles, obstacle_dims, mag, prev_path, prev_dir_path_unit) 
              for mag in magnitudes]
        og_paths, closest_points, prev_dir_path_unit_list, virtual_obstacles_list = zip(*results)
        
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
        if (magnitude >= 300): magnitude = 300

        # print(len(prev_path))

        # Visualize
        # draw_scene(image, start, goal, obstacles, obstacle_dims, og_paths,closest_points)
        draw_scene(image, start, goal, obstacles, obstacle_dims, og_paths, 
              closest_points, virtual_obstacles_list)
        
        cv2.imshow(window_name, image)
        if cv2.waitKey(1) & 0xFF == 27:
            break
        # time.sleep(0.2)
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()