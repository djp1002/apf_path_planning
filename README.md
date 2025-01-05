```markdown
# Dyutkriti: Mobile Probe Potential Field Framework for UAV-Quadruped Navigation
![System Overview](assets/system_overview.png)
This repository contains the implementation of a multi-agent robotic framework that enables synergistic collaboration between a UAV and quadruped robot for autonomous industrial inspection tasks. The system features a novel Mobile Probe Potential Field (MPPF) algorithm for navigation and a comprehensive perception pipeline.

## Key Features

- **Mobile Probe Potential Field (MPPF) Algorithm**
  - Dynamic repulsive field generation proportional to obstacle geometries
  - Proactive local minima detection and avoidance
  - Virtual Obstacle Marker (VOM) placement strategy

- **Advanced Perception Pipeline**
  - Real-time RGB-based terrain classification using YOLOv11
  - Custom depth normalization with sigmoid transformation
  - Low-pass filtered obstacle tracking
  - Hybrid depth-RGB fusion architecture

- **Terrain-Adaptive Control**
  - Dynamic gait selection based on terrain type
  - Real-time path optimization
  - Velocity modulation for different surfaces

## System Requirements

### Hardware
- UAV equipped with:
  - RealSense D435i RGB-D camera
  - NVIDIA AGX Orin computing platform
  - Pixhawk 6C flight controller
- Unitree Go1 quadruped robot

### Software
- Ubuntu 20.04
- ROS Noetic
- Python 3.8+
- CUDA 11.4+ (for YOLOv11)
- OpenCV 4.5+

## Directory Structure

```
├── perception/

│   ├── depth_detection_refined.py      # Depth enhancement and obstacle detection

│   └── detect_quad_v11.py             # YOLOv11-based detection

├── planning/

│   ├── apf_vom_vector_minima_pract.py # MPPF implementation

│   └── filter_box_lowpass.py          # Obstacle tracking

├── control/

│   ├── go1_command.py                 # Quadruped control interface

│   └── main_combined.py               # Main system integration

└── utils/
    └── coordinate_transforms.py        # Frame transformation utilities
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/[username]/dyutkriti-mppf.git
cd dyutkriti-mppf
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up ROS workspace:
```bash
mkdir -p ~/catkin_ws/src
cd ~/catkin_ws/src
catkin_init_workspace
cd ..
catkin_make
```

## Usage

1. Launch the main system:
```bash
roslaunch dyutkriti_mppf main.launch
```

2. For standalone testing of MPPF:
```bash
python planning/apf_vom_vector_minima_pract.py
```

3. Run perception pipeline:
```bash
python perception/depth_detection_refined.py
```

```

This README provides a comprehensive overview of the project, installation instructions, usage guidelines, and proper attribution. You may want to customize the following elements:

1. Update the repository URL
2. Add specific version requirements for dependencies
3. Include any additional setup steps specific to your environment
4. Add troubleshooting guides if needed
5. Update the paper link once published
6. Modify the license type if necessary

Would you like me to make any adjustments to this draft?
