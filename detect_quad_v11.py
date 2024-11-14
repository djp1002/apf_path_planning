import cv2
import time
from ultralytics import YOLO

# class_names = ["quadruped", "sand", "stairs", "stones"]

# Initialize YOLO model
model = YOLO('model_weights/best_v11_E100_default.engine', task="detect")


# class_colors = {
#     "quadruped": (0, 0, 255),#bgr
#     "stones": (255, 0, 0),
#     "sand": (0, 255, 255),
#     "stairs": (0, 255, 0)
# }

confidence_format = "{:.2f}"  # Format confidence score to two decimal places

def yolo_detection( frame):
    # thickness = 3
    # start_time = time.time()
    # frame_count = 0


    # ret, frame = cap.read()
    # frame = cv2.resize(frame, (640, 480)) # 640 * 480

    # Perform object detection
    results = model(source=frame, show=False, conf=0.6, save=False)



    # # out.write(final[:,:,::-1])  # Uncomment if writing video
    # end_time = time.time()
    # elapsed_time = end_time - start_time
    # frame_count += 1

    # if elapsed_time > 1:
    #     fps = frame_count / elapsed_time
    #     start_time = end_time
    #     frame_count = 0

    # if 'final' in locals():
    #     # Display FPS on the frame
    #     cv2.putText(final, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)


    # cv2.imshow("Object Detection", frame)
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break
    return frame, results