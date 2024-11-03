import numpy as np
from collections import defaultdict

def calculate_centroid(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2, (y1 + y2) / 2)

def filter_detections(all_detections, proximity_threshold=50, min_occurrences=2):
    object_occurrences = defaultdict(int)
    filtered_boxes = []

    for frame_detections in all_detections:
        current_centroids = [calculate_centroid(box) for box in frame_detections]
        
        for i, centroid in enumerate(current_centroids):
            matched = False
            for j, filtered_box in enumerate(filtered_boxes):
                filtered_centroid = calculate_centroid(filtered_box)
                if np.linalg.norm(np.array(centroid) - np.array(filtered_centroid)) < proximity_threshold:
                    object_occurrences[j] += 1
                    filtered_boxes[j] = frame_detections[i]  # Update the position
                    matched = True
                    break
            
            if not matched:
                filtered_boxes.append(frame_detections[i])
                object_occurrences[len(filtered_boxes) - 1] = 1

    # Filter boxes based on minimum occurrences
    return [box for i, box in enumerate(filtered_boxes) if object_occurrences[i] >= min_occurrences]

# Example usage
# def process_frames():
#     # This is a mock function to simulate your frame processing
#     return [
#         [(10, 10, 20, 20), (100, 100, 110, 110)],
#         [(12, 12, 22, 22), (101, 101, 111, 111), (200, 200, 210, 210)],
#         [(11, 11, 21, 21), (102, 102, 112, 112),(200, 200, 210, 210)],
#         [(13, 13, 23, 23), (103, 103, 113, 113)],
#         [(12, 12, 22, 22), (101, 101, 111, 111), (200, 200, 210, 210)]
#     ]

# Main execution
# all_detections = process_frames()
# filtered_detections = filter_detections(all_detections)
# print("Filtered detections:", filtered_detections)