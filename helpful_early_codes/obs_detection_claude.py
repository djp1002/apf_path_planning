path_to_image1 =  '/home/chitti/legged_ws/src/unitree_legged_sdk/example_py/test_images_depth/frame_191.jpg'
path_to_image2 =  '/home/chitti/legged_ws/src/unitree_legged_sdk/example_py/test_images_depth/frame_208.jpg'
path_to_image3 =  '/home/chitti/legged_ws/src/unitree_legged_sdk/example_py/test_images_depth/frame_191.jpg'
path_to_image4 =  '/home/chitti/legged_ws/src/unitree_legged_sdk/example_py/test_images_depth/frame_191.jpg'
path_to_image5 =  '/home/chitti/legged_ws/src/unitree_legged_sdk/example_py/test_images_depth/frame_208.jpg'




import cv2
import numpy as np

def detect_boxes(image):
    # Convert to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Define range for red color
    lower_red1 = np.array([0, 120, 70])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 120, 70])
    upper_red2 = np.array([180, 255, 255])
    
    # Create mask for red color
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = mask1 + mask2
    
    # Noise reduction
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter and draw bounding boxes
    min_area = 500  # Adjust this value based on the size of your boxes
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > min_area:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    return image

# Test the function on your images
# image_paths = ['path_to_image1.jpg', 'path_to_image2.jpg', 'path_to_image3.jpg', 'path_to_image4.jpg', 'path_to_image5.jpg']
image_paths = [path_to_image1,path_to_image2,path_to_image3]


for i, path in enumerate(image_paths, 1):
    image = cv2.imread(path)
    result = detect_boxes(image)
    cv2.imshow(f'Detected Boxes - Image {i}', result)
    cv2.waitKey(0)

cv2.destroyAllWindows()