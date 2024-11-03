import numpy as np
from scipy.spatial import cKDTree

def calculate_centroids(boxes):
    return np.mean(boxes[:, [0, 2, 1, 3]].reshape(-1, 2, 2), axis=1)

def find_near_matches_2d(boxes1, boxes2, tolerance=1, k_low=0.5, delete_value=0.2, detailed_print=0):
    centroids1 = calculate_centroids(boxes1)
    centroids2 = calculate_centroids(boxes2)
    
    tree = cKDTree(centroids2)
    distances, indices = tree.query(centroids1, distance_upper_bound=tolerance)
    
    near_matches = []
    used_indices = set()

    for i, (distance, j) in enumerate(zip(distances, indices)):
        if distance <= tolerance:
            weight = k_low + ((1-k_low) * boxes2[j, 4])
            near_matches.append(np.append(boxes2[j, :4], weight))
            used_indices.add(j)
        else:
            near_matches.append(np.append(boxes1[i, :4], k_low))
    
    # Add remaining boxes from boxes2 that weren't matched
    for j in range(len(boxes2)):
        if j not in used_indices and boxes2[j, 4] > delete_value:
            near_matches.append(np.append(boxes2[j, :4], (1-k_low)*boxes2[j, 4]))
    
    return np.array(near_matches)

# Example usage
list1 = np.array([
    [[1, 1, 3, 3, 0], [3, 3, 5, 5, 0], [5, 5, 7, 7, 0]],
    [[1, 1, 3, 3, 0], [4, 4, 6, 6, 0], [3, 3, 5, 5, 0], [5, 5, 7, 7, 0]],
    [[1, 1, 3, 3, 0], [2, 2, 4, 4, 0], [3, 3, 5, 5, 0], [4, 4, 6, 6.2, 0], [5, 5, 7, 7, 0]],
    [[1, 1, 3, 3, 0], [2, 2, 4, 4, 0], [3, 3, 5, 5, 0], [4.1, 4.7, 6.1, 6.7, 0], [5, 5, 7, 7, 0]],
    [[1, 1, 3, 3, 0], [4.3, 3.8, 6.3, 5.8, 0], [5, 5, 7, 7, 0]]
])

near_matches = np.array([[40, 40, 42, 42, 0.5]])

for boxes in list1:
    if len(near_matches) == 0:
        near_matches = np.array([[0, 0, 2, 2, 0]])
    near_matches = find_near_matches_2d(boxes, near_matches, tolerance=0.5)
    print("output", near_matches)