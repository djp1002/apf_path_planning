#!/usr/bin/python3
import cv2
import numpy as np
# from go1_command import center_from_corners

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

def center_from_corners(input_corners = [[1,1],[2,2],[2,3],[4,5]]):
    # center = [(input_corners[0] + input_corners[2])/2, (input_corners[1] + input_corners[3])/2]
    center = [(input_corners[0][0] + input_corners[1][0] + input_corners[2][0] + input_corners[3][0] )/4, 
                        (input_corners[0][1] + input_corners[1][1] + input_corners[2][1] + input_corners[3][1] )/4]
    return center

# def get_closest_point_on_rectangle(point, rect_points):
#     """
#     Find closest point on rectangle boundary to given point
#     point: current position
#     rect_center: center of obstacle rectangle 
#     rect_dims: [width, height] of rectangle
#     """
#     point = np.array(point)
#     rect_points = np.array(rect_points)

#     top_left = np.array([rect_points[0], rect_points[1]])       # [x1,y1]
#     bottom_right = np.array([rect_points[2], rect_points[3]])   # [x2,y2]

#     closest = np.array([
#         np.clip(point[0], top_left[0], bottom_right[0]),    # clip x between x1 and x2
#         np.clip(point[1], bottom_right[1], top_left[1])     # clip y between y2 and y1 (note order due to negative y)
#     ])

#     return closest

# def get_closest_point_on_rectangle(point, rect_points):
#     """
#     Find closest point on rectangle boundary to given point using line intersection
#     point: [x,y] current position 
#     rect_points: [x1,y1,x2,y2] rectangle points (top-left, bottom-right)
#     """
    
#     # Convert inputs to numpy arrays
#     point = np.array(point)
#     rect = np.array(rect_points)
    
#     # Get rectangle corners and center
#     top_left = rect[:2]
#     bottom_right = rect[2:]
#     center = (top_left + bottom_right) / 2
    
#     # Rectangle sides as line segments
#     sides = [
#         [top_left, [bottom_right[0], top_left[1]]],     # top
#         [[bottom_right[0], top_left[1]], bottom_right], # right
#         [[top_left[0], bottom_right[1]], bottom_right], # bottom
#         [top_left, [top_left[0], bottom_right[1]]]      # left
#     ]
def get_closest_point_on_rectangle(point, rect_points):
    """
    Find closest point on oriented rectangle boundary
    point: [x,y] current position
    rect_points: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]] four corners in order
    """
    point = np.array(point)
    rect = np.array(rect_points)  # now a 4x2 array
    
    # Center is average of all corners
    center = np.mean(rect, axis=0)
    
    # Rectangle sides are now just consecutive points
    sides = [
        [rect[0], rect[1]],  # first side
        [rect[1], rect[2]],  # second side
        [rect[2], rect[3]],  # third side
        [rect[3], rect[0]]   # fourth side (back to start)
    ]
    # Vector from center to point
    dir_vector = point - center
    
    # Find intersections with all sides
    intersections = []
    for p1, p2 in sides:
        # Line intersection formula
        p1, p2 = np.array(p1), np.array(p2)
        side_vector = p2 - p1
        
        # Skip if parallel
        cross_product = np.cross(dir_vector, side_vector)
        if abs(cross_product) < 1e-10: continue
            
        # Get intersection parameter
        t = np.cross(p1 - center, side_vector) / cross_product
        s = np.cross(p1 - center, dir_vector) / cross_product
        
        # Check if intersection is valid
        if 0 <= s <= 1:
            intersection = p1 + s * side_vector
            intersections.append(intersection)
    
    # Return closest valid intersection
    if intersections:
        distances = [np.linalg.norm(p - point) for p in intersections]
        return intersections[np.argmin(distances)]
    
    # Fallback to center if no valid intersection
    return center

# ---------------------- Force Analysis Functions ----------------------
def calculate_force_metrics(current_pos_xy, goal, obstacles_xy, magnitude, prev_metrics=None):
    """
    Calculate comprehensive force metrics using closest points on obstacles
    """
    # current_pos = [(current_pos_xy[0]+current_pos_xy[2])/2,(current_pos_xy[1]+current_pos_xy[3])/2]
    current_pos = center_from_corners(current_pos_xy)
    # Calculate attractive force to goal
    f_att = calculate_force(current_pos, goal, charge=10.0 * (1 + magnitude/100))
    mag_att = np.linalg.norm(f_att)
    dir_att = f_att / mag_att if mag_att > 0 else np.zeros(2)
    
    # Calculate repulsive forces from closest points on obstacles
    f_rep_total = np.zeros(2)
    closest_points = []
    agent_closest_points = []
    
    for obstacle_xy in obstacles_xy:
        # Find closest point on rectangle boundary
        closest_point = get_closest_point_on_rectangle(current_pos, obstacle_xy)
        closest_points.append(closest_point)
        agent_closest_point = get_closest_point_on_rectangle(closest_point, current_pos_xy)
        agent_closest_points.append(agent_closest_point)
        f_rep = calculate_force(agent_closest_point, closest_point, charge=-2.0 * (1 - min(magnitude, 90)/100))
        f_rep_total += f_rep
    
    mag_rep = np.linalg.norm(f_rep_total)
    dir_rep = f_rep_total / mag_rep if mag_rep > 0 else np.zeros(2)
    
    # Calculate force alignment
    force_alignment = np.dot(dir_att, dir_rep)
    
    # Calculate resultant force
    f_total = f_att + f_rep_total
    mag_total = np.linalg.norm(f_total)
    dir_total = f_total / mag_total if mag_total > 0 else np.zeros(2)
    
    # Calculate force ratio and its trend
    force_ratio = mag_att / mag_rep if mag_rep > 0 else float('inf')
    ratio_trend = 0
    
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
        'closest_points': closest_points,
        'agent_closest_points': agent_closest_points
    }

def detect_local_minima_region(force_metrics,prev_force_metrics):
    """
    Proactively detect if robot is approaching a local minima region
    Returns (is_approaching_minima, confidence_score, approach_vector)
    """
    # Calculate force opposition (dot product should be approaching -1)
    force_angle = force_metrics['force_alignment']
    force_angle_score = max(0, -force_angle) 
    
    # Check if forces are becoming equal (ratio approaching 1)
    force_mag_ratio = force_metrics['mag_att'] / force_metrics['mag_rep'] if force_metrics['mag_rep'] > 0 else float('inf')
    force_mag_score = max(0, 1 - abs(force_mag_ratio - 1))
    
    if prev_force_metrics is not None:
        prev_force_mag_ratio = prev_force_metrics['mag_att'] / prev_force_metrics['mag_rep'] if prev_force_metrics['mag_rep'] > 0 else float('inf')
        prev_force_mag_score = max(0, 1 - abs(prev_force_mag_ratio - 1))
        mag_diff_force = force_mag_score - prev_force_mag_score

        prev_force_angle = prev_force_metrics['force_alignment']
        prev_force_angle_score = max(0, -prev_force_angle)
        mag_diff_angle = force_angle_score - prev_force_angle_score
        trend_score = 1.0 if mag_diff_force>0 and mag_diff_angle>0 else 0.0
        steps_to_minima = (1 - force_mag_score)/mag_diff_force if mag_diff_force>0 else 15
        
    else:
        trend_score = 0.0
        steps_to_minima = 15

    # Early warning score - higher when we're approaching but not yet trapped
    warning_score = (force_angle_score * force_mag_score * trend_score)
    
    # We want to detect earlier, so lower threshold
    is_approaching = warning_score > 0.98 # More sensitive than previous 0.6
    # if is_approaching:
    #    print("steps_to_minima", steps_to_minima)

    return is_approaching, warning_score, steps_to_minima


def calculate_prev_path_direction(path, current_pos):   #************************change whole logic
    """Calculate recent path direction from history"""
    # Example array of 10 points (each row is a [x, y] coordinate)
    # Calculate Euclidean distances from each point to the current point
    zero_dir = np.array([0,0])
    if len(path)>1:
        # print("debud_lists", current_pos, path)
        path = np.array(path)                    # convert list to numpy array
        current_pos = np.array(current_pos)
        distances = np.linalg.norm(path - current_pos, axis=1)

        # Find the index of the minimum distance
        nearest_index = np.argmin(distances)
        # Get the closest point
        nearest_point = path[nearest_index]
        direction = nearest_point - current_pos
        dir_mag = np.linalg.norm(direction)
        if dir_mag > 0:
            return direction / dir_mag
        else:
            return zero_dir
    else:
        return zero_dir

# ---------------------- VOM Placement and Force Adjustment ----------------------
def calculate_vom_placement(current_pos_xy, force_metrics, prev_path, best_path, prev_vom, steps):
    """
    Calculate VOM placement using 45° shift method based on approach vector
    Uses best path as reference for direction
    """
    # Get attractive force direction (pure direction towards goal)
    local_minima_vector = force_metrics['dir_total']
    # current_pos = [(current_pos_xy[0]+current_pos_xy[2])/2,(current_pos_xy[1]+current_pos_xy[3])/2]
    current_pos = center_from_corners(current_pos_xy)
    
    
    # Get the path direction - first try best path, then prev path
    reference_path = best_path if best_path is not None else prev_path
    prev_path_dir = calculate_prev_path_direction(reference_path, current_pos)  # Increased window  # need to change this logic
    
    if prev_path_dir is None:
        # If no reference path, use perpendicular to attractive force
        prev_path_dir = np.array([-local_minima_vector[1], local_minima_vector[0]])
        prev_path_dir = prev_path_dir / np.linalg.norm(prev_path_dir)
    
    # Determine which side of local minima vector the path lies
    cross_product = np.cross(np.append(local_minima_vector, 0), 
                           np.append(prev_path_dir, 0))[2]
    on_right = cross_product > 0
    
    # Calculate 45° shift OPPOSITE to prev_path side
    angle = -np.pi/4 if on_right else np.pi/4
    cos_theta, sin_theta = np.cos(angle), np.sin(angle)
    rotation_matrix = np.array([[cos_theta, -sin_theta],
                              [sin_theta, cos_theta]])
    
    # Calculate VOM position shifted from local minima vector
    vom_direction = rotation_matrix @ local_minima_vector
    vom_direction = vom_direction/np.linalg.norm(vom_direction)
    vom_new_pos = current_pos + (vom_direction*50)
    if prev_vom is not None:
        if np.linalg.norm(vom_new_pos - prev_vom) < 100:
            vom_new_pos = prev_vom


    force_total_mag = force_metrics['mag_total']
    theta = np.pi*3/4
    force_vom_mag = force_total_mag/(np.sin(theta)-np.cos(theta))
    dist_to_vom = calculate_distance(vom_new_pos,current_pos)
    charge = round((force_vom_mag * (dist_to_vom**2))/100000)
    # print("current and vom pos, steps",current_pos,vom_new_pos,steps)

    return vom_new_pos, charge

# second part ----------------------------------------------------------------------------------------------------------------------------------------->

# ---------------------- Path Generation Functions ----------------------
def calculate_next_position_with_voms(current_pos_xy,force_metrics,voms, vom_charges):
    """Calculate next position using closest points on rectangles"""
    f_total = force_metrics['f_total']
    # current_pos = [(current_pos_xy[0]+current_pos_xy[2])/2,(current_pos_xy[1]+current_pos_xy[3])/2]
    current_pos = center_from_corners(current_pos_xy)
    voms = []
    # Add VOM forces
    if voms is not None:
        for vom, charge in zip(voms, vom_charges):
            if vom is not None:
                agent_closest_point = get_closest_point_on_rectangle(vom, current_pos_xy)
                f_vom = calculate_force(agent_closest_point, vom, charge=charge) * 5
                f_total += f_vom
    
    # Calculate movement
    if np.linalg.norm(f_total) > 0:
        direction = f_total / np.linalg.norm(f_total)
        shift = 5.0 * direction  # Fixed step size
        # print("shift magnbitude---------------------->",shift)
        return current_pos + np.ceil(shift)
    return current_pos

def generate_enhanced_path(start_xy, goal, obstacles_xy, magnitude, prev_path,  proximity_dist, best_path=None):
    """Generate path with proactive local minima avoidance using best path reference"""
    # start = [(start_xy[0]+start_xy[2])/2,(start_xy[1]+start_xy[3])/2]
    start = center_from_corners(start_xy)
    path = [np.array(start)]
    current_xy = start_xy
    virtual_obstacles = []
    vom_charges = []

    # State tracking
    prev_metrics = None
    max_iterations = 70
    iteration = 0
    dist_to_goal = float('inf')
    prev_vom_pos = None
    reached = False
    closest_points = []
    agent_closest_points = []
    # print(f"\nAttempting path with magnitude {magnitude:.1f}")
    
    while dist_to_goal>proximity_dist and iteration < max_iterations:
        # Calculate force metrics
        force_metrics = calculate_force_metrics(current_xy, goal, obstacles_xy, magnitude, prev_metrics)
        closest_points.append(force_metrics['closest_points'])
        agent_closest_points.append(force_metrics['agent_closest_points'])
        
        # Proactive local minima detection
        approaching_minima, confidence, steps_to_minima = detect_local_minima_region(force_metrics,prev_metrics)
        
        # VOM management - now more proactive and using best_path
        # if approaching_minima:
        #     # print(f"Potential local minimum detected! Confidence: {confidence:.2f}")
            
        #     # Use both initial direction and best path for better VOM placement
        #     new_vom_pos, new_vom_charge = calculate_vom_placement(current_xy, force_metrics, prev_path, best_path, prev_vom_pos,steps_to_minima)
        #     if prev_vom_pos is None :
        #         virtual_obstacles.append(new_vom_pos)
        #         vom_charges.append(new_vom_charge)
        #     else:
        #         dist_diff = np.linalg.norm(prev_vom_pos-new_vom_pos)
        #         if dist_diff > 100:
        #             virtual_obstacles.append(new_vom_pos)
        #             vom_charges.append(new_vom_charge)
                    
        next_pos = calculate_next_position_with_voms(current_xy, force_metrics, virtual_obstacles, vom_charges)
        # print("current and next position",current,next_pos,"charge",vom_charges)
        # current = next_pos
        # x2 = (current_xy[2] - current_xy[0])/2
        # y2 = (current_xy[3] - current_xy[1])/2
        # current_xy = [current[0]-x2, current[1]-y2, current[0]+x2, current[1]+y2 ]
        
        
        current = np.mean(current_xy, axis=0)
        shift = next_pos - current
        current_xy = current_xy + shift
        current = next_pos


        path.append(current)
        
        # Check if goal reached
        dist_to_goal = calculate_distance(current, goal)
        if dist_to_goal <= 20.0:
            # print(f"Goal reached in {len(path)} steps!")
            reached = True
        
        # Store metrics for next iteration
        prev_metrics = force_metrics
        iteration += 1
    if iteration > max_iterations:
        print("max iteration")
            
            
    return path, closest_points, agent_closest_points, virtual_obstacles, iteration, reached

# ---------------------- Visualization Functions ----------------------
def draw_scene(image, start_xy, goal, obstacles_xy, paths_list, closest_points_list, agent_closest_points_list, voms_list, goal_points, next_goal):
    """Draw the scene with enhanced visualization"""
    def to_cv2_point(point):
        return (int(point[0]), -int(point[1]))
    
    # Clear image
    image.fill(0)

    # Draw obstacles
    for obstacle_xy in obstacles_xy:
        # top_left = np.array([obstacle_xy[0], obstacle_xy[1]])
        # bottom_right = np.array([obstacle_xy[2], obstacle_xy[3]])
        # cv2.rectangle(image, to_cv2_point(top_left), to_cv2_point(bottom_right), 
        #              color=(0, 0, 255), thickness=-1)
        obstacle_xy_d = obstacle_xy.copy()            # Make actual copy
        obstacle_xy_d[:,1] = -obstacle_xy_d[:,1]      # Flip y coordinates
        obstacle_xy_d = obstacle_xy_d.astype(np.int32) # Convert to integers for OpenCV
        obstacle_xy_d = obstacle_xy_d.reshape((-1,4,2))  # Reshape to (1,4,2)
        # print(obstacle_xy_d)
        if len(obstacle_xy_d) > 0:
            # cv2.polylines(image, obstacle_xy_d, isClosed=True, color=(0, 0, 255), thickness=-1)
            cv2.fillPoly(image, obstacle_xy_d, color=(0, 0, 255))
    
    # Draw paths with different colors
    colors = [(100, 100, 255), (255, 255, 0)]  # Blue for test path, Yellow for best path
    
    for paths, voms, closest_points, agent_closest_points, color in zip(paths_list, voms_list, closest_points_list, agent_closest_points_list, colors):
        
        # Draw VOMs with influence radius
        for vom in voms:
            if vom is not None:
                cv2.circle(image, to_cv2_point(vom), radius=5, color=(0, 150, 150), thickness=-1)
            
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
        # for step_points in closest_points:  # For each step's points
        #     for point in step_points:       # For each obstacle's closest point
        #         cv2.circle(image, to_cv2_point(point), radius=3, 
        #                   color=(0, 255, 255), thickness=-1)
        # for step_points in agent_closest_points:  # For each step's points
        #     for point in step_points:       # For each obstacle's closest point
        #         cv2.circle(image, to_cv2_point(point), radius=3, 
        #                   color=(0, 255, 255), thickness=-1)
    
    # Draw start and goal with labels
    # cv2.circle(image, to_cv2_point(start), radius=20, 
    #           color=(50, 50, 0), thickness=1)
    # cv2.putText(image, 'S', to_cv2_point(start - np.array([5, -5])),
    #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 50, 0), 1)
    # top_left_start = np.array([start_xy[0], start_xy[1]])
    # bottom_right_start = np.array([start_xy[2], start_xy[3]])
    # cv2.rectangle(image, to_cv2_point(top_left_start), to_cv2_point(bottom_right_start), 
    #                 color=(0, 255, 255), thickness=-1)
    
    # top_left_start = np.array([goal_points[0], goal_points[1]])
    # bottom_right_start = np.array([goal_points[2], goal_points[3]])
    # cv2.rectangle(image, to_cv2_point(top_left_start), to_cv2_point(bottom_right_start), 
    #                 color=(255, 255, 255), thickness=2)
    
    # goal rectangle corner points draw
    goal_xy_d = goal_points.copy()            # Make actual copy
    goal_xy_d[:,1] = -goal_xy_d[:,1]      # Flip y coordinates
    goal_xy_d = goal_xy_d.astype(np.int32) # Convert to integers for OpenCV
    goal_xy_d = goal_xy_d.reshape((-1,4,2))  # Reshape to (1,4,2)
    if len(goal_xy_d) > 0:
        cv2.polylines(image, goal_xy_d, isClosed=True, color=(255,0,255), thickness=2)


    # quadruped rectangle corner points draw
    start_xy_d = start_xy.copy()            # Make actual copy
    start_xy_d[:,1] = -start_xy_d[:,1]      # Flip y coordinates
    start_xy_d = start_xy_d.astype(np.int32) # Convert to integers for OpenCV
    start_xy_d = start_xy_d.reshape((-1,4,2))  # Reshape to (1,4,2)
    if len(start_xy_d) > 0:
        cv2.polylines(image, start_xy_d, isClosed=True, color=(0,255,255), thickness=2)
    
    cv2.circle(image, to_cv2_point(goal), radius=5, 
              color=(0, 255, 0), thickness=2)
    cv2.circle(image, to_cv2_point(next_goal), radius=5, 
              color=(255, 255, 0), thickness=2)
    cv2.line(image, to_cv2_point(goal), 
                        to_cv2_point(next_goal), 
                        color=(200,200,100), thickness=1)
    cv2.putText(image, 'G', to_cv2_point(goal - np.array([5, -5])),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    return image

# ---------------------- Main Program ----------------------
def apf_path(start_xy, goal,obstacles_points,magnitude,best_path, prev_path, best_magnitude, min_path_length, goal_points, next_goal, proximity_dist = 20):

    mag_diff = 20   # Magnitude difference between paths
    mag_increment = 10  # How much to change magnitude between iterations
    # image = np.zeros((960, 1280, 3), dtype=np.uint8)
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    window_name = 'Enhanced APF Path Planning'
    # cv2.namedWindow(window_name)
    
    # Generate two paths with different magnitudes
    closest_points = []
    voms = []
    path_lengths = []
    
    # Current best magnitude path
    # start = [(start_xy[0]+start_xy[2])/2,(start_xy[1]+start_xy[3])/2]
    path1, cp1, acp1, v1, len1, reached1 = generate_enhanced_path(start_xy, goal, obstacles_points, magnitude, prev_path,proximity_dist, best_path)

    # Test path with different magnitude
    magnitude2 = magnitude + mag_diff
    path2, cp2, acp2, v2, len2, reached2 = generate_enhanced_path(start_xy, goal, obstacles_points, magnitude2, prev_path,proximity_dist, best_path)

    closest_points = [cp1, cp2]
    agent_closest_points = [acp1, acp2]
    voms = [v1, v2]
    path_lengths = [len1, len2]

    # Update magnitude based on path lengths
    if reached1 and reached2:
        if path_lengths[0] < path_lengths[1]:
            magnitude = max(10, magnitude - mag_increment)
            prev_path = path1 if len1 != float('inf') else prev_path
            if len1 <= min_path_length and len1 != float('inf'):
                min_path_length = len1
                best_magnitude = magnitude
                best_path = path1.copy()
        else:
            magnitude = min(200, magnitude + mag_increment)
            prev_path = path2 if len2 != float('inf') else prev_path
            if len2 <= min_path_length and len2 != float('inf'):
                min_path_length = len2
                best_magnitude = magnitude + mag_diff
                best_path = path2.copy()
    elif reached1:
        prev_path = path1
    elif reached2:
        prev_path = path2
    else:
        magnitude = best_magnitude

    if path_lengths[0] < path_lengths[1]:
        prev_path = path1
    else:
        prev_path = path2
    # Visualization - also show best path if it exists
    paths_to_draw = [path1, path2]
    if best_path is not None:
        paths_to_draw.append(best_path)
        voms.append([])  # Add empty voms for best path visualization
    
    image = draw_scene(image, start_xy, goal, obstacles_points,
                paths_to_draw, closest_points, agent_closest_points, voms, goal_points, next_goal)
    
    # Display info
    info_text = f"Magnitude: {magnitude:.1f} | Best: {best_magnitude:.1f} | Min Length: {min_path_length}"
    cv2.putText(image, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # cv2.imshow(window_name, image)

    return image, magnitude,best_path,prev_path,best_magnitude, min_path_length