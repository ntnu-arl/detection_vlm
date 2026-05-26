# Detection VLM in ROS

![License: BSD-3](https://img.shields.io/badge/License-BSD3-green.svg)
![ROS Version](https://img.shields.io/badge/ROS2-Humble-blue)
![ROS Version](https://img.shields.io/badge/ROS-Noetic-blue)

This repository provides ROS wrappers for vision-language perception modules:

- Open-vocabulary 2D object detection with 3D LiDAR grounding.
- Semantic mapping, which accumulates detections into a persistent colored voxel map.
- Binary visual question-answering with confidence visualization.

The Noetic branch is documented here. For ROS 2 Humble, see the
[master branch](https://github.com/ntnu-arl/detection_vlm/tree/master).

---

## Table of Contents

- [Setup](#setup)
  - [General Requirements](#general-requirements)
  - [Building](#building)
  - [Python Virtual Environment](#python-virtual-environment)
- [Repository Layout](#repository-layout)
- [Configuration](#configuration)
- [Usage](#usage)
  - [Detection VLM](#detection-vlm)
  - [Semantic Map](#semantic-map)
  - [Q&A VLM](#qa-vlm)
- [License](#license)
- [Contact](#contact)

---

## Setup

### General Requirements

These instructions assume that `ros-noetic-desktop-full` is installed on
**Ubuntu 20.04**. Install general dependencies:

```bash
sudo apt install python3-rosdep python3-catkin-tools python3-vcstool
```

The ROS nodes also use common ROS packages such as `tf2_ros`, `tf2_sensor_msgs`,
`vision_msgs`, `visualization_msgs`, `dynamic_reconfigure`, and RViz. `rosdep`
should install the available ROS package dependencies.

### Building

Build the repository:

```bash
mkdir -p vlm_ws/src
cd vlm_ws
catkin init
catkin config -DCMAKE_BUILD_TYPE=Release

cd src
git clone git@github.com:ntnu-arl/detection_vlm.git -b noetic detection_vlm
rosdep install --from-paths . --ignore-src -r -y

cd ..
catkin build
source devel/setup.bash
```

### Python Virtual Environment

It is highly recommended to set up a Python virtual environment for the ROS
Python nodes:

```bash
cd vlm_ws/src/detection_vlm/detection_vlm_python
python3.8 -m venv --system-site-packages detection_vlm_env
source detection_vlm_env/bin/activate
pip install -U pip
pip install -r requirements.txt
```

The default launch files use this interpreter:

```bash
$(find detection_vlm_python)/detection_vlm_env/bin/python
```

If your environment lives elsewhere, pass `python_env:=/path/to/python` to the
launch file.

For OpenAI-backed VLM modes, export an API key before launching:

```bash
export OPENAI_API_KEY=<Your OpenAI API key>
```

YOLOE-backed modes do not require an OpenAI key. The first YOLOE run may download
model weights through Ultralytics if they are not already available locally.

---

## Repository Layout

- `detection_vlm_python`: model wrappers, OpenAI client, shared config helpers,
  and common output dataclasses.
- `detection_vlm_ros`: ROS nodes, launch files, RViz configs, prompts, and YAML
  examples.
- `detection_vlm_msgs`: ROS service definitions, currently `SetPrompt.srv`.

---

## Configuration

The launch files load YAML configs from `detection_vlm_ros/config` and then
override topics, frames, prompts, and selected options through launch arguments.

Provided examples:

- [detection_yoloe.yaml](./detection_vlm_ros/config/detection_yoloe.yaml):
  YOLOE detection with segmentation masks.
- [detection_vlm.yaml](./detection_vlm_ros/config/detection_vlm.yaml):
  OpenAI detection VLM.
- [semantic_mapping_yoloe.yaml](./detection_vlm_ros/config/semantic_mapping_yoloe.yaml):
  YOLOE-based semantic mapping.
- [reasoning_vlm.yaml](./detection_vlm_ros/config/reasoning_vlm.yaml):
  OpenAI binary reasoning VLM.

Common parameters:

- `vlm.type`: `yoloe` or `openai`.
- `prompt`: detection or reasoning query. This can also be changed at runtime
  with the `set_prompt` service.
- `worker.min_separation_s`: minimum time between processed images.
- `compressed_image`: subscribe to `sensor_msgs/CompressedImage` when `true`,
  otherwise `sensor_msgs/Image`.
- `target_frame`, `camera_frame`, `body_frame`: frame names used for TF lookup.
- `camera_intrinsics` and `distortion_coeffs`: optional calibration values. If a
  `CameraInfo` message is received, it is used instead.
- `classes_file`: optional class list for YOLOE models that support setting
  custom classes. Prompt-free YOLOE models with `pf` in the model name ignore it.

---

## Usage

### Detection VLM

The detection node runs 2D object detection on the input image and projects LiDAR
points into the camera frame to recover corresponding 3D object volumes. It can
use either YOLOE or an OpenAI VLM, depending on the selected config.

Launch:

```bash
roslaunch detection_vlm_ros detection_vlm.launch
```

Useful launch arguments:

```bash
roslaunch detection_vlm_ros detection_vlm.launch \
  config_path:=$(rospack find detection_vlm_ros)/config/detection_yoloe.yaml \
  input_image_topic:=/cam_front/image_raw/compressed \
  input_pointcloud_topic:=/mimosa_node/lidar/manager/points_full_res \
  input_camera_info_topic:=/camera/color/camera_info \
  target_frame:=mimosa_navigation \
  camera_frame:=camera0 \
  body_frame:=mimosa_body
```

Inputs:

- `/detection_vlm/input_image`
- `/detection_vlm/input_pointcloud`
- `/detection_vlm/input_camera_info`
- TF from the point cloud frame to `body_frame` and `target_frame`, and from
  `target_frame` to `camera_frame`.

Outputs:

- `/detection_vlm/detections_image`: annotated detection image.
- `/detection_vlm/accumulated_pointcloud`: accumulated point cloud in
  `target_frame`.
- `/detection_vlm/detected_bboxes_3d`: `vision_msgs/BoundingBox3DArray`.
- `/detection_vlm/visualization_3d_bboxes`: RViz markers.
- `/detection_vlm/debug_clusters_pointcloud`: optional colored cluster debug
  cloud when `publish_debug_clusters` is enabled.

Service:

- `/detection_vlm/set_prompt`: update the detection prompt at runtime.

Important config options:

- `association_method`: `dbscan` or `front_surface` for selecting which projected
  LiDAR points belong to a 2D detection.
- `use_masks_for_projection`: use segmentation masks when available instead of
  only 2D boxes.
- `voxel_size`, `eps_dbscan`, `min_points_per_cluster`: point cloud filtering and
  clustering behavior.
- `keep_boxes`: keep previous RViz boxes for debugging.

### Semantic Map

The semantic map node builds a persistent semantic voxel map from the same image,
point cloud, camera info, and TF inputs used by detection. Incoming point clouds
are accumulated as geometry in `target_frame`. Each processed image produces 2D
detections, projects active map points into the camera, associates points with
detections, and updates per-voxel semantic scores.

Launch:

```bash
roslaunch detection_vlm_ros semantic_map.launch
```

Useful launch arguments:

```bash
roslaunch detection_vlm_ros semantic_map.launch \
  config_path:=$(rospack find detection_vlm_ros)/config/semantic_mapping_yoloe.yaml \
  classes_file:=$(rospack find detection_vlm_ros)/config/classes.txt \
  input_image_topic:=/cam_front/image_raw/compressed \
  input_pointcloud_topic:=/mimosa_node/lidar/manager/points_full_res \
  input_camera_info_topic:=/camera/color/camera_info \
  target_frame:=mimosa_navigation \
  camera_frame:=camera0 \
  body_frame:=mimosa_body
```

Inputs:

- `/semantic_map/input_image`
- `/semantic_map/input_pointcloud`
- `/semantic_map/input_camera_info`
- TF from the point cloud frame to `body_frame` and `target_frame`, and from
  `target_frame` to `camera_frame`.

Outputs:

- `/semantic_map/detections_image`: annotated 2D detections used to update the
  map.
- `/semantic_map/semantic_map`: latched `sensor_msgs/PointCloud2` containing
  active semantic voxels colored by label.
- `/semantic_map/semantic_objects`: latched `visualization_msgs/MarkerArray`
  containing clustered semantic objects as boxes, connector lines, nodes, and
  labels.

Service:

- `/semantic_map/set_prompt`: update the detection prompt at runtime.

Important config options:

- `association_method`: `dbscan`, `front_surface`, or `none` for selecting which
  projected LiDAR points should update each detected label.
- `semantic_update_method`: `counter` for additive votes or `bayes` for Bayesian
  score updates.
- `semantic_publish_min_observations` and `semantic_publish_min_confidence`:
  thresholds before a voxel is published as semantic.
- `semantic_publish_period_s`: publish period for the semantic point cloud.
- `publish_semantic_objects` and `semantic_objects_period_s`: enable and rate
  limit semantic object marker publishing.
- `semantic_cluster_eps`, `semantic_cluster_min_points`,
  `semantic_object_min_voxels`: object clustering and filtering parameters.
- `active_window_enabled`: update only a local window around `body_frame`.
  `active_window_apply_to_updates` limits computation during map updates.
- `semantic_visualization_voxel_size` and `semantic_visualization_max_points`:
  downsample the published map for RViz performance.
- `max_active_update_points`: cap the number of map points projected into each
  image update.

There is also a site-specific launch file,
[semantic_map_nene.launch](./detection_vlm_ros/launch/semantic_map_nene.launch),
which adds static transforms and starts `rslidar_sdk`. Use
`semantic_map.launch` as the generic launch file unless you need that setup.

### Q&A VLM

The reasoning node sends the input image and a binary yes/no question to an
OpenAI VLM. It publishes a visualization image with the selected answer,
confidence, and explanation. The overlay is green/yellow/red depending on the
answer and confidence.

Launch:

```bash
export OPENAI_API_KEY=<Your OpenAI API key>
roslaunch detection_vlm_ros reasoning_vlm.launch
```

Inputs:

- `/reasoning_vlm/input_image`

Outputs:

- `/reasoning_vlm/output_image`: reasoning visualization.

Service:

- `/reasoning_vlm/set_prompt`: update the question at runtime.

Useful config options:

- `overlay`: enable or disable the confidence color overlay.
- `overlay_alpha`: overlay opacity.
- `footer_height`: text footer height.
- `runtime_logging_enabled`: write per-query runtime rows to CSV.

## License

Released under **BSD-3-Clause**.

---

## Contact

For questions or support, reach out via
[GitHub Issues](https://github.com/ntnu-arl/detection_vlm/issues) or contact the
authors directly:

- [Albert Gassol Puigjaner](mailto:albert.g.puigjaner@ntnu.no)
- [Kostas Alexis](mailto:konstantinos.alexis@ntnu.no)
