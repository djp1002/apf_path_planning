import pyrealsense2 as rs
import numpy as np
import cv2

def detect_boxes(image):
    # image = cv2.applyColorMap(cv2.convertScaleAbs(gray_image, alpha=0.03), cv2.COLORMAP_JET)
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
    min_area = 1500  # Adjust this value based on the size of your boxes
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > min_area:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    return image


fourcc = cv2.VideoWriter_fourcc(*'DIVX')
writer_filtered = cv2.VideoWriter('depth_filtered_1.avi', fourcc, 30, (640, 480))
writer2 = cv2.VideoWriter('depth_w/o_filter_1.avi', fourcc, 30, (640, 480))
writer3 = cv2.VideoWriter('rgb_1.avi', fourcc, 30, (640, 480))

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()

# Get device product line for setting a supporting resolution
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))

found_rgb = False
for s in device.sensors:
    if s.get_info(rs.camera_info.name) == 'RGB Camera':
        found_rgb = True
        break
if not found_rgb:
    print("The demo requires Depth camera with Color sensor")
    exit(0)

config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
pipeline.start(config)

colorizer = rs.colorizer()
decimation = rs.decimation_filter()
spatial = rs.spatial_filter()
depth_to_disparity = rs.disparity_transform(True)
disparity_to_depth = rs.disparity_transform(False)
hole_filling = rs.hole_filling_filter()
temporal_filter = rs.temporal_filter()

align_to = rs.stream.color
align = rs.align(align_to)

# Configure spatial filter for long-range
spatial.set_option(rs.option.filter_magnitude, 5)
spatial.set_option(rs.option.filter_smooth_alpha, 0.25)
spatial.set_option(rs.option.filter_smooth_delta, 50)
spatial.set_option(rs.option.holes_fill, 5)

# hole_filling.set_option(rs.option.holes_fill,2)

# Configure temporal filter
temporal_filter.set_option(rs.option.filter_smooth_alpha, 0.5)
temporal_filter.set_option(rs.option.filter_smooth_delta, 100)
temporal_filter.set_option(rs.option.holes_fill, 6)

decimation.set_option(rs.option.filter_magnitude, 2)


try:
    while True:

        # Wait for a coherent pair of frames: depth and color
        non_aligned_frames = pipeline.wait_for_frames()
        frames = align.process(non_aligned_frames)
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()


        
        
        frame = depth_frame
        
        frame = decimation.process(frame)        
        frame = depth_to_disparity.process(frame)
        frame = spatial.process(frame)
        frame = temporal_filter.process(frame)
        frame = disparity_to_depth.process(frame)
        frame = hole_filling.process(frame)
        
        colorized_depth_filtered = np.asanyarray(colorizer.colorize(frame).get_data())
        colorized_depth_filtered = cv2.resize(colorized_depth_filtered, (640, 480), interpolation = cv2.INTER_LINEAR)

        if not depth_frame or not color_frame:
            continue

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        result = detect_boxes(colorized_depth_filtered)

        # # Show images
        # writer_filtered.write(colorized_depth_filtered)
        # writer2.write(depth_colormap)
        # writer3.write(color_image)
        # cv2.namedWindow('RealSense w/ filter', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense w/ filter', result)
        # cv2.imshow('RealSense w/o filter', depth_colormap )
        # cv2.imshow('RealSense rgb', color_image )
         
        if cv2.waitKey(1) == 27:
            break
            

finally:

    # Stop streaming
    writer_filtered.release()
    pipeline.stop()