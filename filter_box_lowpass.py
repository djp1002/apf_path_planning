import math

def calculate_centroid(box):
    x1, y1, x2, y2 = box[:4]
    return ((x1 + x2) / 2, (y1 + y2) / 2)

def distance(point1, point2):
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def find_near_matches_2d(list1, list2, dim_inc = 10, tolerance=1, k_low=0.5, delete_value=0.2, detailed_print=0):
    # Convert bounding boxes to centroids for sorting and comparison
    # Sorted lists are tuple contains bounding box and centroid
    
    sized_list1 = [[x1 - dim_inc, y1 - dim_inc, x2 + dim_inc, y2 + dim_inc, w] for x1, y1, x2, y2, w in list1]
    sized_list2 = [[x1          , y1          , x2          , y2          , w] for x1, y1, x2, y2, w in list2]
    sorted1 = sorted([(calculate_centroid(box), box) for box in sized_list1], key=lambda p: p[0])
    sorted2 = sorted([(calculate_centroid(box), box) for box in sized_list2], key=lambda p: p[0])
    
    if detailed_print:print("lists input1", sorted1, "input2", sorted2)
    
    near_matches = []
    i, j = 0, 0
    last_ij = [0, 0]
    
    while i < len(sorted1) and j < len(sorted2):
        centroid1, box1 = sorted1[i]
        centroid2, box2 = sorted2[j]
        dist = distance(centroid1, centroid2)
        
        if detailed_print:print("inside loop start i,j", i, j)
        
        if dist <= tolerance:
            weight = k_low + ((1-k_low) * box2[4])
            near_matches.append(box1[:4] + [weight])
            if detailed_print:print("match_found")
            i += 1
            j += 1
        elif centroid1[0] < centroid2[0] or (centroid1[0] == centroid2[0] and centroid1[1] < centroid2[1]):
            near_matches.append(box1[:4] + [k_low])
            if detailed_print:print("first_list_element_added", len(sorted1)-1, len(sorted2)-1, i, j, i == len(sorted1)-1 and j == len(sorted2)-1)
            i += 1
        else:
            if box2[4] > delete_value:
                near_matches.append(box2[:4] + [(1-k_low)*box2[4]])
            j += 1
        
        last_ij = [i, j]
        if detailed_print:print("inside loop end i,j", last_ij,"\n")
    
    if detailed_print:print("outside loop i,j", last_ij)
    
    # Handle remaining boxes if left
    for k in range(max(len(sorted1)-last_ij[0], len(sorted2)-last_ij[1])):
        if detailed_print:print("inside for loop start last i,j", last_ij)
        if last_ij[0] < len(sorted1):
            near_matches.append(sorted1[last_ij[0]][1][:4] + [k_low])
            last_ij[0] += 1
            if detailed_print:print("append from 1st list")
        if last_ij[1] < len(sorted2) and sorted2[last_ij[1]][1][4] > delete_value:
            box2 = sorted2[last_ij[1]][1]
            near_matches.append(box2[:4] + [(1-k_low)*box2[4]])
            last_ij[1] += 1
            if detailed_print:print("append from 2nd list")
    
    return near_matches

# Example usage
# list1 = [
#     [[1, 1, 3, 3, 0], [3, 3, 5, 5, 0], [5, 5, 7, 7, 0]],
#     [[1, 1, 3, 3, 0], [4, 4, 6, 6, 0], [3, 3, 5, 5, 0], [5, 5, 7, 7, 0]],
#     [[1, 1, 3, 3, 0], [2, 2, 4, 4, 0], [3, 3, 5, 5, 0], [4, 4, 6, 6.2, 0], [5, 5, 7, 7, 0]],
#     [[1, 1, 3, 3, 0], [2, 2, 4, 4, 0], [3, 3, 5, 5, 0], [4.1, 4.7, 6.1, 6.7, 0], [5, 5, 7, 7, 0]],
#     [[1, 1, 3, 3, 0], [4.3, 3.8, 6.3, 5.8, 0], [5, 5, 7, 7, 0]]
# ]

# near_matches = [[40, 40, 42, 42, 1]]
# for boxes in list1:
#     if len(near_matches) == 0:near_matches = [[0, 0, 2, 2, 0]]
#     near_matches = find_near_matches_2d(boxes, near_matches, tolerance=0.5)
#     print("output", near_matches)