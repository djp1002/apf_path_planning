#!/usr/bin/python3
import cv2
import numpy as np
import time

# ---------------------- Utility Functions ----------------------
def calculate_force(p1, p2, charge):
    """Calculate force vector between two points"""
    p1 = np.array(p1)
    p2 = np.array(p2)
    displacement = p2 - p1
    distance = max(np.linalg.norm(displacement), 1)
    direction = displacement / distance
    force_magnitude = 100000 * charge / distance**2
    return force_magnitude * direction

def calculate_distance(p1, p2):
    """Calculate Euclidean distance"""
    return np.linalg.norm(np.array(p1) - np.array(p2))

def get_closest_point_on_rectangle(point, rect_center, rect_dims):
    """
    Find closest point on rectangle boundary to given point
    point: current position
    rect_center: center of obstacle rectangle 
    rect_dims: [width, height] of rectangle
    """
    point = np.array(point)
    rect_center = np.array(rect_center)
    rect_dims = np.array(rect_dims)
    
    # Calculate rectangle bounds
    top_left = rect_center - rect_dims
    bottom_right = rect_center + rect_dims
    
    # First clamp point to rectangle bounds
    closest = np.array([
        max(top_left[0], min(point[0], bottom_right[0])),
        max(top_left[1], min(point[1], bottom_right[1]))
    ])
    
    # If point is inside rectangle, find nearest edge
    if np.all(closest == point):
        # Calculate distances to edges
        dist_to_edges = np.array([
            abs(point[0] - top_left[0]),     # Distance to left edge
            abs(point[0] - bottom_right[0]),  # Distance to right edge
            abs(point[1] - top_left[1]),      # Distance to top edge
            abs(point[1] - bottom_right[1])   # Distance to bottom edge
        ])
        
        # Find nearest edge
        min_dist_idx = np.argmin(dist_to_edges)
        
        # Project point to nearest edge
        if min_dist_idx == 0:    # Left edge
            closest[0] = top_left[0]
        elif min_dist_idx == 1:  # Right edge
            closest[0] = bottom_right[0]
        elif min_dist_idx == 2:  # Top edge
            closest[1] = top_left[1]
        else:                    # Bottom edge
            closest[1] = bottom_right[1]
            
    return closest

# ---------------------- Force Analysis Functions ----------------------
def calculate_force_metrics(current_pos, goal, obstacles, obstacle_dims, magnitude, prev_metrics=None):
    """
    Calculate comprehensive force metrics using closest points on obstacles
    """
    # Calculate attractive force to goal
    f_att = calculate_force(current_pos, goal, charge=10.0 * (1 + magnitude/100))
    mag_att = np.linalg.norm(f_att)
    dir_att = f_att / mag_att if mag_att > 0 else np.zeros(2)
    
    # Calculate repulsive forces from closest points on obstacles
    f_rep_total = np.zeros(2)
    closest_points = []
    
    # Influence range for forces
    p0 = 100.0
    
    for obstacle in obstacles:
        # Find closest point on rectangle boundary
        closest_point = get_closest_point_on_rectangle(current_pos, obstacle, obstacle_dims)
        closest_points.append(closest_point)
        
        # Calculate distance to closest point
        dist = calculate_distance(current_pos, closest_point)
        
        # Only apply force if within influence range
        if dist < p0:
            f_rep = calculate_force(current_pos, closest_point, 
                                  charge=-2.0 * (1 - min(magnitude, 90)/100))
            
            # Scale force based on distance (stronger when closer)
            scale = (1 - dist/p0) * (1 - dist/p0)  # Quadratic scaling
            f_rep_total += f_rep * scale
    
    mag_rep = np.linalg.norm(f_rep_total)
    dir_rep = f_rep_total / mag_rep if mag_rep > 0 else np.zeros(2)
    
    # Calculate force alignment
    force_alignment = np.dot(dir_att, -dir_rep)
    
    # Calculate resultant force
    f_total = f_att + f_rep_total
    mag_total = np.linalg.norm(f_total)
    dir_total = f_total / mag_total if mag_total > 0 else np.zeros(2)
    
    # Calculate force ratio and its trend
    force_ratio = mag_att / mag_rep if mag_rep > 0 else float('inf')
    ratio_trend = 0
    if prev_metrics is not None:
        prev_ratio = prev_metrics.get('force_ratio', force_ratio)
        ratio_trend = force_ratio - prev_ratio
    
    return {
        'f_att': f_att,
        'f_rep': f_rep_total,
        'f_total': f_total,
        'mag_att': mag_att,
        'mag_rep': mag_rep,
        'mag_total': mag_total,
        'dir_att': dir_att,
        'dir_rep': dir_rep,
        'dir_total': dir_total,
        'force_alignment': force_alignment,
        'force_ratio': force_ratio,
        'ratio_trend': ratio_trend,
        'closest_points': closest_points
    }

def detect_local_minima_region(force_metrics, history_window=5):
    """
    Proactively detect if robot is entering a local minima region
    Returns (is_in_local_minima, confidence_score)
    """
    # Force alignment check (near opposing forces)
    alignment_score = max(0, (force_metrics['force_alignment'] - 0.7) / 0.3)
    
    # Force magnitude ratio check (attractive ≈ repulsive)
    ratio = force_metrics['force_ratio']
    ratio_score = max(0, 1 - abs(ratio - 1) / 0.5)
    
    # Trend analysis (ratio approaching 1)
    trend_score = 0
    if abs(force_metrics['ratio_trend']) > 0.1:
        trend_score = 1 if force_metrics['ratio_trend'] < 0 else 0
    
    # Resultant force check (small compared to individual forces)
    force_cancellation = force_metrics['mag_total'] / (force_metrics['mag_att'] + force_metrics['mag_rep'])
    cancellation_score = max(0, 1 - force_cancellation / 0.3)
    
    # Combine scores with weights
    weights = [0.4, 0.3, 0.2, 0.1]  # Alignment, ratio, trend, cancellation
    confidence = (weights[0] * alignment_score + 
                 weights[1] * ratio_score +
                 weights[2] * trend_score +
                 weights[3] * cancellation_score)
    
    return confidence > 0.6, confidence

def calculate_path_direction(path, current_pos, window=5):
    """Calculate recent path direction from history"""
    if len(path) < 2:
        return None
        
    # Get recent path segment
    end = len(path)
    start = max(0, end - window)
    recent_path = path[start:end]
    
    # Calculate average direction
    directions = []
    for i in range(len(recent_path)-1):
        segment = recent_path[i+1] - recent_path[i]
        if np.linalg.norm(segment) > 0:
            directions.append(segment / np.linalg.norm(segment))
    
    if not directions:
        return None
        
    avg_direction = np.mean(directions, axis=0)
    if np.linalg.norm(avg_direction) > 0:
        return avg_direction / np.linalg.norm(avg_direction)
    return None

# ---------------------- VOM Placement and Force Adjustment ----------------------
def calculate_vom_placement(current_pos, force_metrics, prev_path, safe_distance=30.0):
    """
    Calculate VOM placement using 45° shift method
    Returns (vom_position, desired_force_direction)
    """
    local_minima_vector = -force_metrics['dir_total']  # Direction to local minimum
    
    # Determine previous path direction
    path_dir = calculate_path_direction(prev_path, current_pos)
    if path_dir is None:
        # No previous path - use perpendicular to local_minima_vector
        path_dir = np.array([-local_minima_vector[1], local_minima_vector[0]])
    
    # Determine which side prev_path is relative to local_minima_vector
    cross_product = np.cross(np.append(local_minima_vector, 0), 
                           np.append(path_dir, 0))[2]
    on_right = cross_product > 0
    
    # Calculate 45° shift in appropriate direction
    angle = -np.pi/4 if on_right else np.pi/4  # Shift opposite to prev_path
    cos_theta, sin_theta = np.cos(angle), np.sin(angle)
    rotation_matrix = np.array([[cos_theta, -sin_theta],
                              [sin_theta, cos_theta]])
    
    # Calculate VOM position and desired force direction
    vom_direction = rotation_matrix @ local_minima_vector
    vom_position = current_pos + safe_distance * vom_direction
    
    # Desired force direction (45° same side as prev_path)
    desired_angle = np.pi/4 if on_right else -np.pi/4
    cos_theta, sin_theta = np.cos(desired_angle), np.sin(desired_angle)
    rotation_matrix = np.array([[cos_theta, -sin_theta],
                              [sin_theta, cos_theta]])
    desired_force_dir = rotation_matrix @ (-local_minima_vector)
    
    return vom_position, desired_force_dir

def adjust_vom_forces(current_pos, voms, force_metrics, desired_direction):
    """
    Adjust VOM force magnitudes based on escape progress
    Returns updated VOM charges
    """
    base_charge = -2.0  # Base repulsive charge
    charges = []
    
    for vom in voms:
        # Calculate current movement direction
        movement_dir = force_metrics['dir_total']
        
        # Calculate alignment with desired direction
        alignment = np.dot(movement_dir, desired_direction)
        
        # Adjust charge based on alignment
        if alignment < 0.7:  # Not moving in desired direction
            charge = base_charge * (1.5 - alignment)  # Increase magnitude
        else:
            charge = base_charge  # Maintain base magnitude
            
        charges.append(min(charge, -4.0))  # Limit maximum charge
        
    return charges

# second part ----------------------------------------------------------------------------------------------------------------------------------------->

# ---------------------- Path Generation Functions ----------------------
def calculate_next_position_with_voms(current, goal, obstacles, obstacle_dims, 
                                    voms, vom_charges, magnitude):
    """Calculate next position using closest points on rectangles"""
    # Attractive force to goal
    f_total = calculate_force(current, goal, charge=10.0 * (1 + magnitude/100))
    
    # Repulsive forces from obstacles with consistent p0
    p0 = 100.0  # Same influence range as in force_metrics
    for obstacle in obstacles:
        closest = get_closest_point_on_rectangle(current, obstacle, obstacle_dims)
        dist = calculate_distance(current, closest)
        
        if dist < p0:
            f_rep = calculate_force(current, closest, 
                                  charge=-2.0 * (1 - min(magnitude, 90)/100))
            # Scale force based on distance
            scale = (1 - dist/p0) * (1 - dist/p0)
            f_total += f_rep * scale
    
    # Add VOM forces with same scaling
    for vom, charge in zip(voms, vom_charges):
        dist = calculate_distance(current, vom)
        if dist < p0:
            f_vom = calculate_force(current, vom, charge=charge)
            scale = (1 - dist/p0) * (1 - dist/p0)
            f_total += f_vom * scale
    
    # Calculate movement
    if np.linalg.norm(f_total) > 0:
        direction = f_total / np.linalg.norm(f_total)
        shift = 5.0 * direction  # Fixed step size
        return current + np.ceil(shift)
    return current

def is_making_progress(current, last_progress, goal, threshold=5.0):
    """
    Check if making progress towards goal
    Returns True if moving closer to goal
    """
    curr_to_goal = calculate_distance(current, goal)
    last_to_goal = calculate_distance(last_progress, goal)
    return curr_to_goal < last_to_goal - threshold

def generate_enhanced_path(start, goal, obstacles, obstacle_dims, magnitude, prev_path):
    """Generate path with enhanced local minima avoidance"""
    path = [np.array(start)]
    current = np.array(start)
    virtual_obstacles = []
    vom_charges = []
    desired_direction = None
    
    # State tracking
    prev_metrics = None
    stuck_counter = 0
    last_progress = current
    escape_attempts = 0
    max_iterations = 1000
    
    print(f"\nAttempting path with magnitude {magnitude:.1f}")
    
    for iteration in range(max_iterations):
        # Calculate force metrics
        force_metrics = calculate_force_metrics(current, goal, obstacles, 
                                             obstacle_dims, magnitude, prev_metrics)
        
        # Check for local minima region
        in_local_minima, confidence = detect_local_minima_region(force_metrics)
        
        # Progress tracking
        making_progress = is_making_progress(current, last_progress, goal)
        if making_progress:
            stuck_counter = max(0, stuck_counter - 1)
            last_progress = current.copy()
            if stuck_counter == 0 and len(virtual_obstacles) > 0:
                print(f"Progress: {calculate_distance(current, goal):.1f} units to goal")
        else:
            stuck_counter += 1
            
        # Debug output
        if stuck_counter > 0 and stuck_counter % 10 == 0:
            print(f"Stuck for {stuck_counter} iterations")
            print(f"Force alignment: {force_metrics['force_alignment']:.2f}")
            print(f"Current position: {current}")
        
        # VOM management
        if (in_local_minima or stuck_counter > 10) and len(virtual_obstacles) < 3:
            vom_pos, new_desired_dir = calculate_vom_placement(
                current, force_metrics, prev_path)
            
            # Validate VOM placement
            valid_placement = True
            if virtual_obstacles:
                min_dist = min(calculate_distance(vom_pos, v) for v in virtual_obstacles)
                valid_placement = min_dist > 50.0
                
            if valid_placement:
                virtual_obstacles.append(vom_pos)
                vom_charges.append(-2.0)
                desired_direction = new_desired_dir
                escape_attempts += 1
                stuck_counter = 0
                print(f"Added VOM {len(virtual_obstacles)} at {vom_pos}")
        
        # Adjust VOM forces if needed
        if virtual_obstacles and desired_direction is not None:
            vom_charges = adjust_vom_forces(current, virtual_obstacles, 
                                         force_metrics, desired_direction)
            
        # Calculate next position including VOMs
        next_pos = calculate_next_position_with_voms(
            current, goal, obstacles, obstacle_dims, 
            virtual_obstacles, vom_charges, magnitude)
        
        # Update position
        current = next_pos
        path.append(current)
        
        # Clean up distant VOMs
        if virtual_obstacles:
            valid_voms = []
            valid_charges = []
            for vom, charge in zip(virtual_obstacles, vom_charges):
                if calculate_distance(current, vom) < 100.0:
                    valid_voms.append(vom)
                    valid_charges.append(charge)
            if len(valid_voms) != len(virtual_obstacles):
                print(f"Cleaned up {len(virtual_obstacles) - len(valid_voms)} distant VOMs")
            virtual_obstacles = valid_voms
            vom_charges = valid_charges
        
        # Check if goal reached
        dist_to_goal = calculate_distance(current, goal)
        if dist_to_goal < 20.0:
            print(f"Goal reached in {len(path)} steps!")
            return path, force_metrics['closest_points'], virtual_obstacles, len(path)
        
        # Store metrics for next iteration
        prev_metrics = force_metrics
            
        # Break conditions
        if stuck_counter > 50 or escape_attempts > 5:
            print(f"Path failed - Stuck: {stuck_counter}, Attempts: {escape_attempts}")
            print(f"Final distance to goal: {dist_to_goal:.1f}")
            return path, force_metrics['closest_points'], virtual_obstacles, float('inf')
            
    return path, force_metrics['closest_points'], virtual_obstacles, float('inf')

# ---------------------- Visualization Functions ----------------------
def draw_scene(image, start, goal, obstacles, obstacle_dims, paths_list, closest_points_list, voms_list):
    """Draw the scene with enhanced visualization"""
    def to_cv2_point(point):
        return (int(point[0]), -int(point[1]))
    
    # Clear image
    image.fill(0)
    
    # Draw obstacles
    for obstacle_center in obstacles:
        rect_width, rect_height = obstacle_dims
        top_left = np.array([obstacle_center[0] - rect_width, 
                           obstacle_center[1] + rect_height])
        bottom_right = np.array([obstacle_center[0] + rect_width, 
                               obstacle_center[1] - rect_height])
        cv2.rectangle(image, to_cv2_point(top_left), to_cv2_point(bottom_right), 
                     color=(0, 0, 255), thickness=-1)
    
    # Draw paths with different colors
    colors = [(100, 100, 255), (255, 255, 0)]  # Blue for test path, Yellow for best path
    
    for paths, voms, color in zip(paths_list, voms_list, colors):
        # Draw path with gradient
        if len(paths) > 1:
            for i in range(len(paths)-1):
                progress = i / max(1, len(paths)-1)
                # Interpolate color based on progress
                path_color = (
                    int(color[0] * (1-progress/2)),
                    int(color[1] * (progress/2)),
                    int(color[2] * (1-progress/2))
                )
                cv2.line(image, to_cv2_point(paths[i]), 
                        to_cv2_point(paths[i+1]), 
                        color=path_color, thickness=2)
        
        # Draw VOMs with influence radius
        for vom in voms:
            cv2.circle(image, to_cv2_point(vom), radius=10, 
                      color=(0, 150, 150), thickness=-1)
            cv2.circle(image, to_cv2_point(vom), radius=50, 
                      color=(0, 100, 100), thickness=1)
            
            # Draw direction indicators
            cv2.line(image, to_cv2_point(vom), 
                    to_cv2_point(vom + np.array([20, 0])),
                    color=(0, 200, 200), thickness=1)
    
    # Draw start and goal with labels
    cv2.circle(image, to_cv2_point(start), radius=20, 
              color=(50, 50, 0), thickness=1)
    cv2.putText(image, 'S', to_cv2_point(start - np.array([5, -5])),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 50, 0), 1)
    
    cv2.circle(image, to_cv2_point(goal), radius=20, 
              color=(0, 255, 0), thickness=-1)
    cv2.putText(image, 'G', to_cv2_point(goal - np.array([5, -5])),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

# ---------------------- Main Program ----------------------
def main():
    """Main program loop with dual path optimization"""
    # Initialize points and obstacles
    start = np.array([40, -440])
    goal = np.array([300, -140])
    prev_path = []
    
    window_name = 'Enhanced APF Path Planning'
    cv2.namedWindow(window_name)
    
    # Create trackbars for obstacles
    for i in range(1, 6):
        cv2.createTrackbar(f'Obstacle {i} X', window_name, 101 + (i-1)*50, 640, lambda x: None)
        cv2.createTrackbar(f'Obstacle {i} Y', window_name, 325 - (i-1)*15, 480, lambda x: None)
    
    # Path planning parameters
    magnitude = 50  # Starting magnitude
    mag_diff = 20   # Magnitude difference between paths
    mag_increment = 10  # How much to change magnitude between iterations
    best_magnitude = magnitude
    min_path_length = float('inf')
    obstacle_dims = np.array([20, 40])
    
    print("Starting Enhanced APF Path Planning")
    print("Controls: ESC - Exit, R - Reset")
    
    while True:
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Get obstacle positions
        obstacles = []
        for i in range(1, 6):
            x = cv2.getTrackbarPos(f'Obstacle {i} X', window_name)
            y = cv2.getTrackbarPos(f'Obstacle {i} Y', window_name)
            obstacles.append(np.array([x, -y]))
        
        # Generate two paths with different magnitudes
        paths = []
        closest_points = []
        voms = []
        path_lengths = []
        
        # Current best magnitude path
        path1, cp1, v1, len1 = generate_enhanced_path(
            start, goal, obstacles, obstacle_dims, magnitude, prev_path)
        
        # Test path with different magnitude
        test_magnitude = magnitude + mag_diff
        path2, cp2, v2, len2 = generate_enhanced_path(
            start, goal, obstacles, obstacle_dims, test_magnitude, prev_path)
        
        paths = [path1, path2]
        closest_points = [cp1, cp2]
        voms = [v1, v2]
        path_lengths = [len1, len2]
        
        # Update magnitude based on path lengths
        if path_lengths[0] < path_lengths[1]:
            magnitude = max(10, magnitude - mag_increment)
            prev_path = path1
            if path_lengths[0] < min_path_length:
                min_path_length = path_lengths[0]
                best_magnitude = magnitude
        else:
            magnitude = min(300, magnitude + mag_increment)
            prev_path = path2
            if path_lengths[1] < min_path_length:
                min_path_length = path_lengths[1]
                best_magnitude = magnitude + mag_diff
        
        # Visualization
        draw_scene(image, start, goal, obstacles, obstacle_dims, 
                  paths, closest_points, voms)
        
        # Display info
        info_text = f"Magnitude: {magnitude:.1f} | Best: {best_magnitude:.1f} | Min Length: {min_path_length}"
        cv2.putText(image, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, (255, 255, 255), 1)
        
        cv2.imshow(window_name, image)
        
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            print("\nExiting program")
            break
        elif key == ord('r'):  # Reset
            print("\nResetting path planning")
            magnitude = 50
            prev_path = []
            min_path_length = float('inf')
        
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()