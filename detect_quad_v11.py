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

def yolo_detection( frame, quadruped_xy, sand_xy, stair_xy, stones_xy ):
    # thickness = 3
    # start_time = time.time()
    # frame_count = 0


    # ret, frame = cap.read()
    # frame = cv2.resize(frame, (640, 480)) # 640 * 480

    # Perform object detection
    results = model(source=frame, verbose = False, show=False, conf=0.6, save=False)
    # quadruped_xy, sand_xy, stair_xy, stones_xy = [100,200,100,200], [0,0,0,0], [0,0,0,0], [600,200,600,200]

    for result in results:
                if len(result.boxes) > 0:
                    for box in result.boxes:
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        conf = box.conf[0].item()
                        cls = int(box.cls[0].item())
                        class_name = class_names[cls]
                        color = class_colors[class_name]

                        if(class_name == "quadruped"):
                            quadruped_xy = [x1, -y1, x2, -y2]
                            final = cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness=2)
                        elif(class_name =="sand"):
                            sand_xy = [x1, -y1, x2, -y2]
                            final = cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness=2)
                        elif(class_name =="stairs"):
                            stair_xy_temp = stair_xy
                            stair_xy = [x1, -y1, x2, -y2]
                            temp_area = abs((stair_xy_temp[0]-stair_xy_temp[2])*(stair_xy_temp[1]-stair_xy_temp[3]))
                            main_area = abs((stair_xy[0]-stair_xy[2])*(stair_xy[1]-stair_xy[3]))
                            area_diff = abs(main_area - temp_area)
                            if main_area>=temp_area or area_diff < 5000:    
                                stair_xy = stair_xy
                            else:
                                # stair_xy = stair_xy_temp
                                stair_xy = stair_xy
                        
                            final = cv2.rectangle(frame, (int(stair_xy[0]), int(-stair_xy[1])), (int(stair_xy[2]), int(-stair_xy[3])), color, thickness=2)
                            
                        elif(class_name =="stones"):
                            stones_xy = [x1, -y1, x2, -y2]
                            final = cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness=2)
                        
                        # Draw bounding box and display class
                        text = f"{class_name}: {confidence_format.format(conf)}"
                        cv2.putText(final, text, (int(x1) + 10, int(y1) + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    return frame, results, quadruped_xy, sand_xy, stair_xy, stones_xy
