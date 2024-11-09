#!/usr/bin/python3
import cv2
import numpy as np
import time


def calculate_path_direction_vector(current_pos, prev_path, search_radius=50.0):
    """Calculate direction vector based on nearest point in previous path"""
    if not len(prev_path):
        return None, None
        
    prev_path = np.array(prev_path)
    current_pos = np.array(current_pos)
    distances = np.linalg.norm(prev_path - current_pos, axis=1)
    
    near_points_mask = distances < search_radius
    if not np.any(near_points_mask):
        return None, None
        
    nearest_idx = np.argmin(distances[near_points_mask])
    actual_indices = np.where(near_points_mask)[0]
    nearest_idx = actual_indices[nearest_idx]
    
    if nearest_idx < len(prev_path) - 1:
        path_direction = prev_path[nearest_idx + 1] - prev_path[nearest_idx]
        if np.linalg.norm(path_direction) > 0:
            path_direction = path_direction / np.linalg.norm(path_direction)
            return path_direction, prev_path[nearest_idx]
    
    return None, None

def analyze_forces(current, goal, obstacles, obstacle_dims, magnitude):
    """
    Analyze attractive and repulsive forces to predict local minima
    Returns force analysis including magnitudes and directions
    """
    # Calculate attractive force from goal
    f_att = calculate_force(current, goal, charge=10.0* (1 + magnitude/100))
    mag_att = np.linalg.norm(f_att)
    
    # Calculate total repulsive force from obstacles
    f_rep_total = np.zeros(2)
    closest_points = []
    
    for obstacle in obstacles:
        closest_point = get_closest_point_on_rectangle(current, obstacle, obstacle_dims)
        closest_points.append(closest_point)
        f_rep = calculate_force(current, closest_point, charge=-2.0* (1 - min(magnitude, 90)/100))
        f_rep_total += f_rep
    
    mag_rep = np.linalg.norm(f_rep_total)
    
    # Normalize forces for direction analysis
    f_att_norm = f_att / mag_att if mag_att > 0 else f_att
    f_rep_norm = f_rep_total / mag_rep if mag_rep > 0 else f_rep_total
    
    # Calculate dot product for direction analysis
    dot_product = np.dot(f_att_norm, -f_rep_norm)
    
    # Calculate resultant force
    f_total = f_att + f_rep_total
    mag_total = np.linalg.norm(f_total)
    
    return {
        'f_att': f_att,
        'f_rep': f_rep_total,
        'mag_att': mag_att,
        'mag_rep': mag_rep,
        'mag_total': mag_total,
        'dot_product': dot_product,
        'closest_points': closest_points
    }

def predict_local_minima(force_analysis, threshold=1):
    """
    Predict potential local minima based on force analysis
    """
    # Check force alignment (near opposing forces)
    high_opposition = force_analysis['dot_product'] > threshold
    
    # Check force magnitudes (similar magnitudes indicate potential cancellation)
    similar_magnitudes = (abs(force_analysis['mag_att'] - force_analysis['mag_rep']) < 
                         0.1 * max(force_analysis['mag_att'], force_analysis['mag_rep']))
    
    # Check if resultant force is small compared to individual forces
    small_resultant = (force_analysis['mag_total'] < 
                      0.1 * (force_analysis['mag_att'] + force_analysis['mag_rep']))
    
    return high_opposition and similar_magnitudes and small_resultant


def place_vom_proactively(current, force_analysis, virtual_obstacles, prev_path, stuck_count, safe_distance=30.0):
    """Modified VOM placement with hybrid strategy"""
    f_att = force_analysis['f_att']
    f_rep = force_analysis['f_rep']
    
    # Get basic direction (perpendicular to total force)
    total_force = f_att + f_rep
    if np.linalg.norm(total_force) > 0:
        total_direction = total_force / np.linalg.norm(total_force)
        perp_direction = np.array([-total_direction[1], total_direction[0]])
        
        # Get direction from previous path
        path_dir, nearest_point = calculate_path_direction_vector(current, prev_path)
        
        if path_dir is not None and nearest_point is not None:
            # Calculate weights
            stuck_factor = min(stuck_count / 10.0, 1.0)
            dist_to_path = np.linalg.norm(current - nearest_point)
            path_weight = max(0, 1 - (dist_to_path / 50.0)) * (1 - stuck_factor)
            perp_weight = stuck_factor + 0.5
            
            # Combine directions
            combined_dir = perp_weight * perp_direction + path_weight * path_dir
            if np.linalg.norm(combined_dir) > 0:
                combined_dir = combined_dir / np.linalg.norm(combined_dir)
                vom_pos = current + combined_dir * safe_distance
                
                # Check VOM placement validity
                if not virtual_obstacles:
                    return vom_pos
                elif len(virtual_obstacles) == 1:
                    if calculate_distance(vom_pos, virtual_obstacles[0]) > 80.0:
                        return vom_pos
                elif len(virtual_obstacles) == 2:
                    distances = [calculate_distance(vom_pos, vom) for vom in virtual_obstacles]
                    if min(distances) > 100.0 and stuck_count > 20:
                        return vom_pos
                        
        return current + perp_direction * safe_distance
    
    return None


def is_path_cycling(path, window=10):
    """Check if path is cycling through similar positions"""
    if len(path) < window:
        return False
        
    recent_positions = path[-window:]
    center = np.mean(recent_positions, axis=0)
    radius = np.max([calculate_distance(pos, center) for pos in recent_positions])
    
    return radius < 20.0 
def generate_path(start, goal, obstacles, obstacle_dims, magnitude, prev_path, prev_dir_path_unit):
    """Modified path generation with enhanced analysis and VOM placement"""
    path = [np.array(start)]
    current = np.array(start)
    virtual_obstacles = []
    closest_points = []
    dir_path_unit = prev_dir_path_unit
    stuck_counter = 0
    last_positions = []
    vom_attempts = 0
    previous_stuck_pos = None
    last_progress_pos = current  # Track position where we last made progress
    
    def analyze_stuck_situation(current_pos, force_data, voms, recent_positions):
        """Analyze current stuck situation"""
        print("\nStuck Situation Analysis:")
        print(f"Current Position: {current_pos}")
        print(f"Distance to goal: {calculate_distance(current_pos, goal)}")
        print(f"Number of active VOMs: {len(voms)}")
        print(f"Force Analysis:")
        print(f"  Attractive force magnitude: {force_data['mag_att']:.2f}")
        print(f"  Repulsive force magnitude: {force_data['mag_rep']:.2f}")
        print(f"  Resultant force magnitude: {force_data['mag_total']:.2f}")
        print(f"  Force alignment (dot): {force_data['dot_product']:.2f}")
        
        if voms:
            vom_distances = [calculate_distance(current_pos, vom) for vom in voms]
            print(f"Distances to VOMs: {[f'{d:.2f}' for d in vom_distances]}")
        
        if len(recent_positions) > 2:
            recent_movement = calculate_distance(recent_positions[-1], recent_positions[-2])
            print(f"Recent movement magnitude: {recent_movement:.2f}")
    
    def is_making_progress(current_pos, last_progress):
        """Check if we're making progress towards goal"""
        current_to_goal = calculate_distance(current_pos, goal)
        last_to_goal = calculate_distance(last_progress, goal)
        return current_to_goal < last_to_goal
    
    while calculate_distance(current, goal) > 20.0:
        # Get closest points for obstacle avoidance
        closest_obstacle_points = [get_closest_point_on_rectangle(current, obstacle, obstacle_dims)
                                 for obstacle in obstacles]
        
        # Force analysis
        force_analysis = analyze_forces(current, goal, obstacles, obstacle_dims, magnitude)
        closest_points.append(closest_obstacle_points)
        
        # Store position history
        last_positions.append(np.copy(current))
        if len(last_positions) > 5:
            last_positions.pop(0)
        
        # Stuck detection with detailed analysis
        stuck = False
        if len(last_positions) >= 3:
            center = np.mean(last_positions, axis=0)
            stuck_threshold = 5.0 * (1 + magnitude/200)  # Adaptive threshold
            
            if all(calculate_distance(pos, center) < stuck_threshold for pos in last_positions[-3:]):
                stuck = True
                stuck_counter += 1
                
                # Analyze situation when stuck
                if stuck_counter % 5 == 0:  # Print analysis every 5 stuck iterations
                    analyze_stuck_situation(current, force_analysis, virtual_obstacles, last_positions)
                
                # Update stuck attempt tracking
                if previous_stuck_pos is None or calculate_distance(center, previous_stuck_pos) > 20.0:
                    vom_attempts = 0  # Reset attempts in new stuck location
                else:
                    vom_attempts = min(vom_attempts + 1, 3)  # Cap attempts
                
                previous_stuck_pos = np.copy(center)
                print(f"Stuck detected! Attempt: {vom_attempts}")
        
        # VOM placement logic
        # VOM placement logic
        potential_trap = predict_local_minima(force_analysis)

        if (potential_trap or stuck_counter > 5) and len(virtual_obstacles) < 3:
            safe_distance = 30.0 * (1 + min(vom_attempts, 2) * 0.5)
            vom_pos = place_vom_proactively(current, force_analysis, virtual_obstacles, 
                                        prev_path, stuck_counter, safe_distance)
            
            if vom_pos is not None and (not virtual_obstacles or 
                all(calculate_distance(vom_pos, vom) > 25.0 for vom in virtual_obstacles)):
                virtual_obstacles.append(vom_pos)
                print(f"Placed VOM - {'trap predicted' if potential_trap else 'stuck'} - Attempt {vom_attempts}")
                stuck_counter = 0
        
        # Calculate next position including VOMs
        all_repulsion_points = closest_obstacle_points + virtual_obstacles
        next_pos = calculate_next_position(current, goal, all_repulsion_points, magnitude)
        
        # Update positions
        prev_pos = current
        current = next_pos
        path.append(current)
        
        # Progress tracking and VOM management
        if is_making_progress(current, last_progress_pos):
            last_progress_pos = current
            # Remove distant VOMs if making progress
            virtual_obstacles = [vom for vom in virtual_obstacles 
                               if calculate_distance(current, vom) < 80.0]
            if not stuck:
                stuck_counter = max(0, stuck_counter - 1)
                vom_attempts = max(0, vom_attempts - 1)
        
        # Update direction unit
        movement = current - prev_pos
        if np.linalg.norm(movement) > 0:
            dir_path_unit = movement / np.linalg.norm(movement)
            
        # Break if stuck too long without progress
        if stuck_counter > 50:  # Arbitrary limit
            print("Failed to escape - stuck too long")
            break
    
    return path, closest_points, dir_path_unit, virtual_obstacles

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
        print(magnitude)
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
    