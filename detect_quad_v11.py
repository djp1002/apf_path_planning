import cv2
import time
from ultralytics import YOLO
class_names = ["quadruped", "sand", "stairs", "stones"]

class_colors = {
    "quadruped": (0, 0, 255),#bgr
    "sand": (0, 255, 255),
    "stairs": (0, 255, 0),
    "stones": (255, 0, 0)
}
# Initialize YOLO model
model = YOLO('model_weights/best_v11_E100_default.engine', task="detect")

confidence_format = "{:.2f}"  # Format confidence score to two decimal places

def yolo_detection( frame ):
    # thickness = 3
    # start_time = time.time()
    # frame_count = 0


    # ret, frame = cap.read()
    # frame = cv2.resize(frame, (640, 480)) # 640 * 480

    # Perform object detection
    results = model(source=frame, show=False, conf=0.6, save=False)
    quadruped_xy, sand_xy, stair_xy, stones_xy = [100,200,100,200], [0,0,0,0], [0,0,0,0], [600,200,600,200]

    for result in results:
                if len(result.boxes) > 0:
                    for box in result.boxes:
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        conf = box.conf[0].item()
                        cls = int(box.cls[0].item())
                        class_name = class_names[cls]

                        if(class_name == "quadruped"):
                            quadruped_xy = [x1, y1, x2, y2]
                        elif(class_name =="sand"):
                            sand_xy = [x1, y1, x2, y2]
                        elif(class_name =="stairs"):
                            stair_xy = [x1, y1, x2, y2]
                        elif(class_name =="stones"):
                            stones_xy = [x1, y1, x2, y2]
                        color = class_colors[class_name]
                        
                        # Draw bounding box and display class
                        final = cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness=2)
                        text = f"{class_name}: {confidence_format.format(conf)}"
                        cv2.putText(final, text, (int(x1) + 10, int(y1) + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    return frame, results, quadruped_xy, sand_xy, stair_xy, stones_xy