import cv2
import os

def extract_frames(video_path, output_dir, frame_interval=1):
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Loop through each frame and save it as an image at specified interval
    frame_count = 0
    cc = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Save frame as an image if it's time for the next frame capture
        if frame_count % frame_interval == 0:
            frame_path = os.path.join(output_dir, f"frame_{cc}.jpg")
            cv2.imwrite(frame_path, frame)
            cc += 1

        frame_count += 1

    # Release the video capture object
    cap.release()

    print(f"Frames extracted successfully from {video_path} to {output_dir}")

# Example usage
video_path = "/home/chitti/legged_ws/src/unitree_legged_sdk/example_py/output_videos/rgb_21_5_20.avi"
output_dir = "/home/chitti/legged_ws/src/unitree_legged_sdk/example_py/output_frames_extracted"
frame_interval = 100  # Capture every 5th frame
extract_frames(video_path, output_dir, frame_interval)
