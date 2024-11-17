import pyrealsense2 as rs
import numpy as np
import cv2

def initialize_realsense():
    pipeline = rs.pipeline()
    config = rs.config()

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

    return pipeline, config

def initialize_filters():
    align_to = rs.stream.color
    align = rs.align(align_to)
    
    colorizer = rs.colorizer()
    decimation = rs.decimation_filter()
    spatial = rs.spatial_filter()
    temporal = rs.temporal_filter()
    hole_filling = rs.hole_filling_filter()
    depth_to_disparity = rs.disparity_transform(True)
    disparity_to_depth = rs.disparity_transform(False)

    # Configure filters
    spatial.set_option(rs.option.filter_magnitude, 1)
    spatial.set_option(rs.option.filter_smooth_alpha, 0.5)
    spatial.set_option(rs.option.filter_smooth_delta, 50)
    spatial.set_option(rs.option.holes_fill, 1)

    temporal.set_option(rs.option.filter_smooth_alpha, 0.25)
    temporal.set_option(rs.option.filter_smooth_delta, 100)
    temporal.set_option(rs.option.holes_fill,3)

    decimation.set_option(rs.option.filter_magnitude, 2)

    return align, colorizer, decimation, spatial, temporal, hole_filling, depth_to_disparity, disparity_to_depth

def process_frames(pipeline, align, colorizer, decimation, spatial, temporal, hole_filling, depth_to_disparity, disparity_to_depth):
    non_aligned_frames = pipeline.wait_for_frames()
    frames = align.process(non_aligned_frames)
    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()

    if not depth_frame or not color_frame:
        return None, None
    
    frame = depth_frame
    frame = decimation.process(frame)        
    frame = depth_to_disparity.process(frame)
    frame = spatial.process(frame)
    frame = temporal.process(frame)
    frame = disparity_to_depth.process(frame)
    frame = hole_filling.process(frame)
    

    depth_norm_np = cv2.resize(np.asanyarray(frame.get_data()), (640, 480), interpolation=cv2.INTER_LINEAR)
    max_depth = float(depth_norm_np.max())
    # max_depth = 65535
    # max_depth = (65535 + 3*max_depth)/4
    # cv2.imshow('depth_norm_np', depth_norm_np)
    depth_norm_np = depth_norm_np/65535
    # cv2.imshow('depth_norm_np', depth_norm_np)
    # depth_norm_np = 1 - depth_norm_np
    depth_norm_np = (depth_norm_np)**0.5
    depth_norm_np[depth_norm_np>0] = 1/(1+((1/depth_norm_np[depth_norm_np>0])-1)**1)
    # depth_norm_np = 1/(1+((1/depth_norm_np)-1)**1)
    # depth_norm_np = 1 - depth_norm_np
    min_depth = float(depth_norm_np.min())
    max_depth = float(depth_norm_np.max())
    # print(min_depth,max_depth,np.percentile(depth_norm_np, 5),np.percentile(depth_norm_np, 95))
    # depth_norm_np = 1 - depth_norm_np
    # min_depth = 0
    # max_depth = 1
    # depth_norm_np[depth_norm_np>]
    image = depth_norm_np
    y, x = np.ogrid[:480, :640]
    # Calculate the distance from each point to the center (320, 240)
    distance = np.sqrt((x - 320)**2 + (y - 480)**2)
    # Create a mask for points inside the circle
    circle_mask = distance <= 200
    # Create a multiplier array, initialized with 1s
    image = image.astype(float)  # Convert to float to avoid overflow
    image[circle_mask] *= 1.01
    # Multiply the image by the multiplier
    # result = (image * multiplier).astype(np.uint8)
    # cv2.imshow('multiplied',result)
    # depth_norm_np = image





    depth_norm_np = cv2.normalize(depth_norm_np, None, 0, 255, cv2.NORM_MINMAX,dtype=cv2.CV_8U, mask=cv2.inRange(depth_norm_np,np.percentile(depth_norm_np, 5),np.percentile(depth_norm_np, 90)))
    depth_norm_np = np.clip(depth_norm_np,0,255)
    
    depth_norm_np = 255 - depth_norm_np
    # Apply colormap
    colorized_depth_np = cv2.applyColorMap(depth_norm_np.astype(np.uint8), cv2.COLORMAP_JET)
    color_image = np.asanyarray(color_frame.get_data())

    return colorized_depth_np, color_image

def detect_boxes(image):
    # Step 1: Convert to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # cv2.imshow('Step 1: HSV Conversion', hsv)
    
    # Step 2: Define range for red color
    lower_red1 = np.array([0, 120, 70])
    upper_red1 = np.array([10, 255, 255])
    # lower_red2 = np.array([70, 120, 70])
    # upper_red2 = np.array([180, 255, 255])
    
    # Step 3: Create mask for red color
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    # mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = mask1
    # cv2.imshow('Step 3: Color Mask', mask)
    
    # Step 4: Noise reduction
    kernel = np.ones((5,5), np.uint8)
    mask_opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    # cv2.imshow('Step 4a: After Opening', mask_opened)
    mask_closed = cv2.morphologyEx(mask_opened, cv2.MORPH_CLOSE, kernel)
    # cv2.imshow('Step 4b: After Closing', mask_closed)
    
    # Step 5: Find contours
    contours, _ = cv2.findContours(mask_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_image = image.copy()
    cv2.drawContours(contour_image, contours, -1, (0,255,0), 2)
    # cv2.imshow('Step 5: Contours', contour_image)
    
    # Step 6: Filter and draw bounding boxes
    min_area = 1500  # Adjust this value based on the size of your boxes
    max_area = 40000
    result_image = image.copy()
    raw_image = image.copy()
    rectangular_contours = []
    approx = []
    solidity = 0
    rectangles = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if max_area > area > min_area:
            epsilon = 0.05 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            if cv2.contourArea(cv2.convexHull(approx))==0:
                solidity=0
            else:
                solidity = float(area)/cv2.contourArea(cv2.convexHull(approx))
        # Check if the contour has 4 vertices (rectangular)
        if 7>len(approx) >2 and solidity>0.6:
            rectangular_contours.append(approx)
            x, y, w, h = cv2.boundingRect(approx)
            # aspect_ratio = float(w)/h
            cv2.rectangle(result_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
            rectangles.append([x, y, x+w, y+h,0])

    cv2.drawContours(result_image, rectangular_contours, -1, (0, 0, 255), 2)

    # Convert the list of lists to a set of tuples
    unique_set = set(map(tuple, rectangles))
    # Convert back to a list of lists
    rectangles = list(map(list, unique_set))
    # cv2.imshow('Step 6: Bounding Boxes', result_image)
    return result_image, raw_image, rectangles
