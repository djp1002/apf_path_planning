import numpy as np
from apf_vom_vector_minima_pract import calculate_distance


def adaptive_gait_selection(quadruped, sand, stair, stones, reached, completed, i, proximity_dist = 20):
    coord = [sand, stair, stones]
    start = np.array([(quadruped[0]+quadruped[2])/2,-(quadruped[1]+quadruped[3])/2])
    goal_points = coord [i]
    goal_midpoint = np.array([(goal_points[0]+goal_points[2])/2,-(goal_points[1]+goal_points[3])/2])
    l1 = abs(goal_points[0] - goal_points[2])
    l2 = abs(goal_points[1] - goal_points[3])
    if l1 > l2:
        point1 = np.array([goal_midpoint - l1/2 - proximity_dist, goal_midpoint[1]])
        point2 = np.array([goal_midpoint + l1/2 + proximity_dist, goal_midpoint[1]])
    else:
        point1 = np.array([goal_midpoint[0],goal_midpoint - l1/2 - proximity_dist])
        point2 = np.array([goal_midpoint[0],goal_midpoint + l1/2 + proximity_dist])
    dist1 = calculate_distance(start, point1)
    dist2 = calculate_distance(start, point2)
    gait_type = 0
    
    if completed[i] == 0:
        if reached[i] == 0:
            if dist1 < dist2:
                initial_goal = point1
            else:
                initial_goal = point2
            gait_type = 0
            goal = initial_goal
            dist_to_goal = calculate_distance(start, goal)
            if dist_to_goal < proximity_dist:
                reached[i] = 1
        else:
            gait_type = i
            goal = point1 if initial_goal == point2 else point2
            dist_to_goal = calculate_distance(start, goal)
            if dist_to_goal < proximity_dist:
                completed[i] = 1
    else:
        i+=1
    return start, goal, gait_type, reached, completed, i


def stair_mode_velocity():

    return None