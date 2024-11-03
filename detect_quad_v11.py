import cv2
import time
from ultralytics import YOLO

class_names = ["quadruped", "sand", "stairs", "stones"]

# Initialize YOLO model
model = YOLO('model_weights/best_v11_E100_default.engine', task="detect")


class_colors = {
    "quadruped": (0, 0, 255),#bgr
    "stones": (255, 0, 0),
    "sand": (0, 255, 255),
    "stairs": (0, 255, 0)
}

confidence_format = "{:.2f}"  # Format confidence score to two decimal places

def yolo_detection( frame):
    thickness = 3
    # start_time = time.time()
    # frame_count = 0


    # ret, frame = cap.read()
    # frame = cv2.resize(frame, (640, 480)) # 640 * 480

    # Perform object detection
    results = model(source=frame, show=False, conf=0.6, save=False)

    for result in results:
        if len(result.boxes) > 0:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = box.conf[0].item()
                cls = int(box.cls[0].item())
                class_name = class_names[cls]

                color = class_colors[class_name]
                

                # Draw bounding box and display class
                final = cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
                text = f"{class_name}: {confidence_format.format(conf)}"
                cv2.putText(final, text, (int(x1) + 10, int(y1) + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

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
    return(frame)