# Detection VLM in ROS

![License: BSD-3](https://img.shields.io/badge/License-BSD3-green.svg)
![ROS Version](https://img.shields.io/badge/ROS2-Humble-blue)
![ROS Version](https://img.shields.io/badge/ROS-Noetic-blue)


This package integrates two complementary vision-language model (VLM) modalities:
- Open-vocabulary object detection with 3D spatial grounding.
- Binary visual question-answering (Yes/No) with reasoning.

Both wrapped with ROS2/ROS nodes.

**Important Note:** Only ROS2 Humble and ROS Noetic are currently supported.
This instructions are for ROS2 Humble. For ROS Noetic check [this branch](https://github.com/ntnu-arl/detection_vlm/tree/noetic).

---

## Table of Contents

- [Setup](#setup)  
  - [General Requirements](#general-requirements)  
  - [Building](#building)  
  - [Python Virtual Environment](#python-virtual-environment)  
- [Usage](#usage) 
  - [Detection VLM](#detection-vlm)
  - [Q&A VLM](#q&a-vlm)
- [License](#license)  
- [Contact](#contact)  
---

## Setup

### General Requirements

These instructions assume that `ros-humble-desktop` is installed on **Ubuntu 22.04**. 

### Building

Build the repository:

```bash
mkdir -p vlm_ws/src
cd vlm_ws/src

git clone git@github.com:ntnu-arl/detection_vlm.git -b ros2_humble detection_vlm
rosdep install --from-paths . --ignore-src -r -y

cd ..
colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release
```

### Python Virtual Environment

It is highly recommended to set up a **Python virtual environment** to run ROS Python nodes:

```bash
cd vlm_ws/src/detection_vlm/detection_vlm_python
python3.8 -m venv --system-site-packages detection_vlm_env
source detection_vlm_env/bin/activate
pip install -U pip
pip install -r requirements.txt
```

---

## Usage

### Detection VLM

Object detection is performed using either an open-vocabulary object detector (YOLOe) or a VLM-based detector (GPT-4V via API call) initialized with a set of labels or a description of the objects to detect. These models operate on an image and produce 2D bounding boxes. In parallel, a downsampled voxel grid derived from a LiDAR point cloud and odometry estimates is maintained. Accordingly, LiDAR points are projected into the camera frame using the current pose estimate and the camera projection matrix. Valid points are clustered to identify those that fall within each 2D detection/mask. This produces aligned 2D detections and corresponding 3D bounding volumes.

The detection vlm can be run with:

```bash
ros2 launch detection_vlm_ros detection_vlm.launch.yaml
```

Input topics and necessary frame names (for TF querying) are set in [detection_vlm.launch.yaml](./detection_vlm_ros/launch/detection_vlm.launch.yaml).

Note that in this launch file you can set which config file to use. We provide two config file examples:
- YOLOe: [detection_yoloe.yaml](./detection_vlm_ros/config/detection_yoloe.yaml)
- OpenAI: [detection_vlm.yaml](./detection_vlm_ros/config/detection_vlm.yaml)

### Q&A VLM

For high-level semantic assessment, a VLM (GPT-4V via API call) processes the front-camera image together with a binary “Yes/No” question. For example, queries related to assessing safety or navigation-related properties of the scene (e.g., is the exit of this environment blocked). The model returns the binary answer, alongside a color-coded confidence overlay on the input image, and a brief explanation of its reasoning.

The detection vlm can be run with:

```bash
export OPENAI_API_KEY=<Your OpenAI API key>
ros2 launch detection_vlm_ros reasoning_vlm.launch.yaml
```

We provide a config file example here: [reasoning_vlm.yaml](./detection_vlm_ros/config/reasoning_vlm.yaml)

---

## License

Released under **BSD-3-Clause**.

---

## Contact

For questions or support, reach out via [GitHub Issues](https://github.com/ntnu-arl/detection_vlm/issues) or contact the authors directly:

- [Albert Gassol Puigjaner](mailto:albert.g.puigjaner@ntnu.no)  
- [Kostas Alexis](mailto:konstantinos.alexis@ntnu.no)