#!/usr/bin/env python3
# BSD 3-Clause License

# Copyright (c) 2025, NTNU Autonomous Robots Lab
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
"""Detection VLM ROS node."""

import struct
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional, Tuple

import cv2
import numpy as np
import open3d as o3d
import rclpy
import tf2_ros
import tf2_sensor_msgs.tf2_sensor_msgs as tf2sm
import yaml
from builtin_interfaces.msg import Duration as DurationMsg
from geometry_msgs.msg import Point, Pose, TransformStamped
from rcl_interfaces.msg import SetParametersResult
from rclpy.duration import Duration
from rclpy.node import Node
from scipy.spatial.transform import Rotation as R
from sensor_msgs.msg import CameraInfo, CompressedImage, Image, PointCloud2, PointField
from spark_config.config import Config, config_field
from std_msgs.msg import Header
from vision_msgs.msg import BoundingBox3D, BoundingBox3DArray
from visualization_msgs.msg import Marker, MarkerArray

import detection_vlm_python.models as models
import detection_vlm_ros
import detection_vlm_ros.point_cloud2 as pc2
from detection_vlm_msgs.srv import SetPrompt
from detection_vlm_python import BoundingBox
from detection_vlm_ros import ImageWorker, ImageWorkerConfig
from detection_vlm_ros.ros_conversions import Conversions


def to_sec(stamp):
    return stamp.sec + stamp.nanosec * 1e-9


@dataclass
class Translation(Config):
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0


@dataclass
class Quaternion(Config):
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    w: float = 1.0


@dataclass
class Lidar2BodyTransform(Config):
    translation: Translation = field(default_factory=Translation)
    quaternion: Quaternion = field(default_factory=Quaternion)


@dataclass
class CameraIntrinsics(Config):
    fx: float = None
    fy: float = None
    cx: float = None
    cy: float = None


@dataclass
class DistortionCoeffs(Config):
    k1: float = None
    k2: float = None
    p1: float = None
    p2: float = None


@dataclass
class DetectionVLMNodeConfig(Config):
    vlm: Any = config_field("detection_vlm_model", default="openai")
    prompt: str = ""
    worker: ImageWorkerConfig = field(default_factory=ImageWorkerConfig)
    verbose: bool = False
    target_frame: str = "world"
    camera_frame: str = "camera"
    body_frame: str = "body"
    lidar_to_body_transform: Lidar2BodyTransform = field(
        default_factory=Lidar2BodyTransform
    )
    min_point_r: float = 0.5
    max_point_r: float = 20.0
    use_tf_current_time: bool = False
    voxel_size: float = 0.05
    min_points_per_cluster: int = 30
    eps_dbscan: float = 0.2
    compressed_image: bool = False
    use_masks_for_projection: bool = True
    publish_debug_clusters: bool = False
    association_method: str = "dbscan"
    front_surface_cell_size_px: int = 20
    front_surface_depth_band: float = 0.20
    front_surface_depth_band_scale: float = 0.05
    front_surface_depth_percentile: float = 20.0
    front_surface_outlier_std_ratio: float = 1.5
    keep_boxes: bool = False
    classes_file: str = ""
    camera_intrinsics: CameraIntrinsics = field(default_factory=CameraIntrinsics)
    distortion_coeffs: DistortionCoeffs = field(default_factory=DistortionCoeffs)


class DetectionVLMNode(Node):
    """ROS node for detection VLM."""

    def __init__(self) -> None:
        """Initialize the Detection VLM ROS node."""
        super().__init__("detection_vlm_node")
        ros_config_params = (
            self.declare_parameter("config", "").get_parameter_value().string_value
        )
        # Load configuration
        config_path_param = (
            self.declare_parameter("config_path", "").get_parameter_value().string_value
        )
        config_path = Path(config_path_param).expanduser().absolute()
        if not config_path_param:
            self.config = DetectionVLMNodeConfig()
        elif not config_path.exists():
            self.get_logger().warn(f"config path '{config_path}' does not exist!")
            self.config = DetectionVLMNodeConfig()
        else:
            self.config = Config.load(DetectionVLMNodeConfig, config_path)
        overrides = yaml.safe_load(ros_config_params) if ros_config_params else {}
        if overrides:
            self.config.update(overrides)
        self.prompt = self.config.prompt
        self._initialized = False
        # Initialize VLM model
        self.vlm_model = self.config.vlm.create()
        self.confidence_threshold = getattr(
            self.vlm_model.config, "confidence_threshold", None
        )
        if self.confidence_threshold is not None:
            self.declare_parameter(
                "confidence_threshold", float(self.confidence_threshold)
            )
            self.confidence_threshold = (
                self.get_parameter("confidence_threshold")
                .get_parameter_value()
                .double_value
            )
            self.add_on_set_parameters_callback(self._on_parameter_update)
        self.association_method = self.config.association_method.strip().lower()
        if self.association_method not in {"dbscan", "front_surface"}:
            self.get_logger().warn(
                f"Unknown association_method={self.config.association_method!r}, falling back to 'dbscan'."
            )
            self.association_method = "dbscan"
        self.get_logger().info(f"Initializing with {self.config.show()}")

        vlm_model_name = getattr(self.config.vlm, "model", "")
        if "pf" not in vlm_model_name:
            classes_file = Path(self.config.classes_file)
            if classes_file.exists():
                with classes_file.open("r") as f:
                    class_names = [
                        line.strip() for line in f.readlines() if line.strip()
                    ]
                if not hasattr(self.vlm_model, "set_classes"):
                    success = False
                else:
                    success = self.vlm_model.set_classes(class_names)
                if not success:
                    self.get_logger().warn("Model does not support setting classes.")
                elif self.config.verbose:
                    self.get_logger().info(
                        f"Loaded {len(class_names)} classes from {classes_file}."
                    )

        # Set up image worker and ROS interfaces
        self.worker = ImageWorker(
            self,
            self.config.worker,
            "input_image",
            CompressedImage if self.config.compressed_image else Image,
            self._spin_once,
        )
        self.srv = self.create_service(SetPrompt, "set_prompt", self._handle_set_prompt)
        self.detections_image_pub = self.create_publisher(Image, "detections_image", 1)

        self.pcl_sub = self.create_subscription(
            PointCloud2, "input_pointcloud", self._pcl_callback, 1
        )
        self.pcl_pub = self.create_publisher(PointCloud2, "accumulated_pointcloud", 1)
        self.debug_clusters_pub = self.create_publisher(
            PointCloud2, "debug_clusters_pointcloud", 1
        )
        self.bbox3d_pub = self.create_publisher(
            BoundingBox3DArray, "detected_bboxes_3d", 1
        )
        self.vis_3dboxes_pub = self.create_publisher(
            MarkerArray, "visualization_3d_bboxes", 1
        )
        self.camera_info_sub = self.create_subscription(
            CameraInfo, "input_camera_info", self._camera_info_callback, 1
        )

        self.boxes_id = 0

        self.fx = self.config.camera_intrinsics.fx
        self.fy = self.config.camera_intrinsics.fy
        self.cx = self.config.camera_intrinsics.cx
        self.cy = self.config.camera_intrinsics.cy
        self.k1 = self.config.distortion_coeffs.k1
        self.k2 = self.config.distortion_coeffs.k2
        self.p1 = self.config.distortion_coeffs.p1
        self.p2 = self.config.distortion_coeffs.p2

        # Set up TF2 listener
        self.tf_buffer = tf2_ros.Buffer(cache_time=Duration(seconds=30.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.accumulated_cloud = o3d.geometry.PointCloud()

        # Generate N different random colors for visualization, N is number of classes
        names = list(getattr(self.vlm_model, "names", {}).values())
        colors = np.random.randint(0, 255, size=(len(names), 3))
        # Generate a dict mapping class names to colors
        self.class_colors = {
            names[i]: tuple(int(c) for c in colors[i]) for i in range(len(names))
        }

        model = getattr(self.vlm_model, "model", None)
        device = getattr(model, "device", "api")
        self.get_logger().info(f"Device: {device}")
        self.get_logger().info("Detection VLM node initialized.")

    def _camera_info_callback(self, msg: CameraInfo) -> None:
        """Callback to process incoming camera info messages.
        :param msg: Incoming CameraInfo message.
        """
        self.fx = msg.k[0]
        self.fy = msg.k[4]
        self.cx = msg.k[2]
        self.cy = msg.k[5]
        self.k1 = msg.d[0] if len(msg.d) > 0 else None
        self.k2 = msg.d[1] if len(msg.d) > 1 else None
        self.p1 = msg.d[2] if len(msg.d) > 2 else None
        self.p2 = msg.d[3] if len(msg.d) > 3 else None
        # Unsubscribe after receiving the first message
        self.destroy_subscription(self.camera_info_sub)
        self.get_logger().info(
            f"Camera intrinsics set: fx={self.fx}, fy={self.fy}, cx={self.cx}, cy={self.cy}"
        )

    def _on_parameter_update(self, params):
        for param in params:
            if param.name == "confidence_threshold":
                try:
                    self.confidence_threshold = float(param.value)
                except (TypeError, ValueError):
                    return SetParametersResult(
                        successful=False,
                        reason="confidence_threshold must be a number",
                    )
                self.get_logger().info(
                    f"Updated confidence threshold: {self.confidence_threshold}"
                )
        return SetParametersResult(successful=True)

    def _tf_time(self, stamp):
        if self.config.use_tf_current_time:
            return rclpy.time.Time()
        return rclpy.time.Time.from_msg(stamp)

    def _pcl_callback(self, msg: PointCloud2) -> None:
        """Callback to process incoming point cloud messages.
        :param msg: Incoming PointCloud2 message.
        """
        try:
            body2world = self.tf_buffer.lookup_transform(
                self.config.target_frame,
                self.config.body_frame,
                self._tf_time(msg.header.stamp),
                Duration(seconds=3.0),
            )
            lidar2body = TransformStamped()
            lidar2body.header.frame_id = self.config.body_frame
            lidar2body.header.stamp = self.get_clock().now().to_msg()
            lidar2body.child_frame_id = "lidar"
            lidar2body.transform.translation.x = (
                self.config.lidar_to_body_transform.translation.x
            )
            lidar2body.transform.translation.y = (
                self.config.lidar_to_body_transform.translation.y
            )
            lidar2body.transform.translation.z = (
                self.config.lidar_to_body_transform.translation.z
            )
            lidar2body.transform.rotation.x = (
                self.config.lidar_to_body_transform.quaternion.x
            )
            lidar2body.transform.rotation.y = (
                self.config.lidar_to_body_transform.quaternion.y
            )
            lidar2body.transform.rotation.z = (
                self.config.lidar_to_body_transform.quaternion.z
            )
            lidar2body.transform.rotation.w = (
                self.config.lidar_to_body_transform.quaternion.w
            )
            # compute inverse transform
            body_cloud = tf2sm.do_transform_cloud(msg, lidar2body)
            points_body = np.array(
                list(
                    pc2.read_points(
                        body_cloud, field_names=("x", "y", "z"), skip_nans=True
                    )
                )
            )
            # Get indices of points within min/max radius
            dists = np.linalg.norm(points_body, axis=1)
            valid_indices = np.where(
                (dists >= self.config.min_point_r) & (dists <= self.config.max_point_r)
            )[0]
            if valid_indices.size == 0:
                return

            transformed_cloud = tf2sm.do_transform_cloud(body_cloud, body2world)

            # Convert to numpy array
            points = np.array(
                list(
                    pc2.read_points(
                        transformed_cloud, field_names=("x", "y", "z"), skip_nans=True
                    )
                )
            )
            points = points[valid_indices]
            if points.size == 0:
                return

            # Convert to Open3D cloud and merge
            new_cloud = o3d.geometry.PointCloud()
            new_cloud.points = o3d.utility.Vector3dVector(points)

            # Combine with existing
            self.accumulated_cloud += new_cloud

            # Apply voxel downsampling in place (keeps only one downsampled cloud)
            self.accumulated_cloud = self.accumulated_cloud.voxel_down_sample(
                self.config.voxel_size
            )

        except (
            tf2_ros.LookupException,
            tf2_ros.ExtrapolationException,
            tf2_ros.ConnectivityException,
        ) as e:
            self.get_logger().warn(
                f"[PersistentVoxelAccumulator] TF lookup failed: {e}"
            )
        except Exception as e:
            self.get_logger().error(
                f"[PersistentVoxelAccumulator] Error processing pointcloud: {e}"
            )

        if len(self.accumulated_cloud.points) > 0:
            if self.config.verbose:
                self.get_logger().info(
                    f"Accumulated cloud has {len(self.accumulated_cloud.points)} points."
                )
            # Publish accumulated cloud
            pcl_msg = pc2.create_cloud_xyz32(
                Header(frame_id=self.config.target_frame, stamp=msg.header.stamp),
                np.asarray(self.accumulated_cloud.points),
            )
            self.pcl_pub.publish(pcl_msg)
            if self.config.verbose:
                self.get_logger().info("Published accumulated point cloud.")

    def _handle_set_prompt(self, req: SetPrompt, response):
        """Handle set prompt service call.
        :param req: Service request containing the new prompt.
        :return: Service response indicating success.
        """
        self.prompt = req.prompt
        self.get_logger().info(f"Prompt updated to: {self.prompt}")
        response.success = True
        return response

    def _spin_once(self, header: Header, image: np.ndarray) -> None:
        """Process a single image.
        :param header: ROS message header.
        :param image: Input image as a NumPy array.
        """
        if self.config.verbose:
            self.get_logger().info(f"Processing image at time {to_sec(header.stamp)}")
        if (
            self.k1 is not None
            and self.k2 is not None
            and self.p1 is not None
            and self.p2 is not None
        ):
            K = np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]])
            D = np.array([self.k1, self.k2, self.p1, self.p2])
            image = cv2.undistort(image, K, D)

        start_time = time.time()
        bboxes: List[BoundingBox] = self.vlm_model.detect(
            image, self.prompt, confidence_threshold=self.confidence_threshold
        )
        if len(bboxes) == 0:
            if self.config.verbose:
                self.get_logger().info("No objects detected.")
            return
        if self.config.verbose:
            self.get_logger().info(
                f"Detected {len(bboxes)} objects in {time.time() - start_time:.2f} seconds."
            )
        detection_image = image.copy()
        detection_image = cv2.cvtColor(detection_image, cv2.COLOR_RGB2BGR)
        for bbox in bboxes:
            if bbox.details in self.class_colors:
                random_color = self.class_colors[bbox.details]
            else:
                random_color = tuple(np.random.randint(0, 255, size=3).tolist())
            random_color_bgr = (random_color[2], random_color[1], random_color[0])
            cv2.rectangle(
                detection_image,
                (bbox.x0, bbox.y0),
                (bbox.x1, bbox.y1),
                random_color_bgr,
                2,
            )
            text = bbox.details
            if bbox.confidence is not None:
                text += f" {bbox.confidence:.2f}"
            cv2.putText(
                detection_image,
                text,
                (bbox.x0, bbox.y0 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                random_color_bgr,
                2,
            )
            if bbox.mask is not None:
                colored_mask = np.zeros_like(detection_image, dtype=np.uint8)
                colored_mask[bbox.mask] = random_color_bgr
                alpha = 0.5
                detection_image = cv2.addWeighted(
                    detection_image, 1.0, colored_mask, alpha, 0
                )

        ros_image = Conversions.to_sensor_image(detection_image)
        self.detections_image_pub.publish(ros_image)

        if len(self.accumulated_cloud.points) > 0 and self.fx is not None:
            self.process_3d_bboxes(header, bboxes, self.accumulated_cloud)

    @staticmethod
    def get_points_in_mask(
        points_3d: np.ndarray, proj_points_uv: np.ndarray, mask: np.ndarray
    ) -> np.ndarray:
        """Return 3D points whose projections lie inside the mask."""
        points_in_mask, _ = DetectionVLMNode.get_points_and_uv_in_mask(
            points_3d, proj_points_uv, mask
        )
        return points_in_mask

    @staticmethod
    def get_points_and_uv_in_mask(
        points_3d: np.ndarray, proj_points_uv: np.ndarray, mask: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return 3D points and projected UVs whose projections lie inside the mask."""
        H, W = mask.shape
        uv = np.round(proj_points_uv).astype(int)

        valid = (
            (uv[:, 0] >= 0)
            & (uv[:, 0] < W)
            & (uv[:, 1] >= 0)
            & (uv[:, 1] < H)
            & (points_3d[:, 2] > 0)
        )

        uv_valid = uv[valid]
        pts_valid = points_3d[valid]
        mask_values = mask[uv_valid[:, 1], uv_valid[:, 0]]
        inside_mask = mask_values.astype(bool)
        return pts_valid[inside_mask], uv_valid[inside_mask]

    @staticmethod
    def _camera_to_world(
        points_camera: np.ndarray, rot: np.ndarray, t: np.ndarray
    ) -> np.ndarray:
        """Transform Nx3 points from the camera frame into the target frame."""
        return (rot.T @ (points_camera - t.T).T).T

    def _get_detection_points(
        self,
        points_camera: np.ndarray,
        proj_points_uv: np.ndarray,
        bbox: BoundingBox,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Get projected map points associated with a 2D detection."""
        if self.config.use_masks_for_projection and bbox.mask is not None:
            return self.get_points_and_uv_in_mask(
                points_camera, proj_points_uv, bbox.mask
            )

        uv = np.round(proj_points_uv).astype(int)
        mask = (
            (uv[:, 0] >= bbox.x0)
            & (uv[:, 0] <= bbox.x1)
            & (uv[:, 1] >= bbox.y0)
            & (uv[:, 1] <= bbox.y1)
            & (points_camera[:, 2] > 0)
        )
        return points_camera[mask], uv[mask]

    def _select_dbscan_cluster(
        self,
        box_points_camera: np.ndarray,
        selected_color: Tuple[int, int, int],
        rot: np.ndarray,
        t: np.ndarray,
    ) -> Tuple[np.ndarray, List[Tuple[float, float, float, float]]]:
        """Select a DBSCAN cluster and prepare optional debug points."""
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(box_points_camera)
        labels = np.array(
            cloud.cluster_dbscan(
                eps=self.config.eps_dbscan,
                min_points=self.config.min_points_per_cluster,
                print_progress=False,
            )
        )
        valid_labels = labels[labels >= 0]
        if len(valid_labels) == 0:
            return np.empty((0, 3)), []

        clusters = [
            box_points_camera[labels == label]
            for label in np.unique(labels)
            if label != -1
        ]
        if not clusters:
            return np.empty((0, 3)), []

        largest_cluster = max(range(len(clusters)), key=lambda idx: len(clusters[idx]))
        closest_z_cluster = int(
            np.argmin([np.mean(cluster[:, 2]) for cluster in clusters])
        )
        if largest_cluster == closest_z_cluster:
            selected_idx = largest_cluster
        elif len(clusters[largest_cluster]) >= 20 * len(clusters[closest_z_cluster]):
            selected_idx = largest_cluster
        else:
            selected_idx = closest_z_cluster

        debug_points = []
        if self.config.publish_debug_clusters:
            for cluster_idx, cluster in enumerate(clusters):
                cluster_world = self._camera_to_world(cluster, rot, t)
                color = (
                    selected_color
                    if cluster_idx == selected_idx
                    else self._random_color_from_points(cluster_world)
                )
                debug_points.extend(
                    self._build_colored_point_entries(cluster_world, color)
                )

        return self._camera_to_world(clusters[selected_idx], rot, t), debug_points

    def _select_front_surface_points(
        self,
        box_points_camera: np.ndarray,
        box_uv: np.ndarray,
        selected_color: Tuple[int, int, int],
        rot: np.ndarray,
        t: np.ndarray,
    ) -> Tuple[np.ndarray, List[Tuple[float, float, float, float]]]:
        """Select a robust front surface from projected accumulated points."""
        if len(box_points_camera) == 0:
            return np.empty((0, 3)), []

        cell_size = max(1, int(self.config.front_surface_cell_size_px))
        abs_depth_band = max(1e-3, float(self.config.front_surface_depth_band))
        rel_depth_band = max(0.0, float(self.config.front_surface_depth_band_scale))
        depth_percentile = float(
            np.clip(self.config.front_surface_depth_percentile, 0.0, 100.0)
        )

        cell_indices = np.floor(box_uv / cell_size).astype(np.int32)
        _, inverse = np.unique(cell_indices, axis=0, return_inverse=True)

        keep_mask = np.zeros(len(box_points_camera), dtype=bool)
        for cell_idx in range(np.max(inverse) + 1):
            cell_mask = inverse == cell_idx
            cell_depths = box_points_camera[cell_mask, 2]
            if len(cell_depths) >= 3:
                front_depth = np.percentile(cell_depths, depth_percentile)
            else:
                front_depth = np.min(cell_depths)
            depth_band = max(abs_depth_band, rel_depth_band * front_depth)
            keep_mask[cell_mask] = cell_depths <= (front_depth + depth_band)

        selected_camera = box_points_camera[keep_mask]
        if len(selected_camera) < self.config.min_points_per_cluster:
            return np.empty((0, 3)), []

        if len(selected_camera) >= 6:
            selected_cloud = o3d.geometry.PointCloud()
            selected_cloud.points = o3d.utility.Vector3dVector(selected_camera)
            nb_neighbors = min(8, len(selected_camera) - 1)
            if nb_neighbors >= 3:
                selected_cloud, _ = selected_cloud.remove_statistical_outlier(
                    nb_neighbors=nb_neighbors,
                    std_ratio=max(
                        0.1, float(self.config.front_surface_outlier_std_ratio)
                    ),
                )
                selected_camera = np.asarray(selected_cloud.points)
                if len(selected_camera) < self.config.min_points_per_cluster:
                    return np.empty((0, 3)), []

        debug_points = []
        if self.config.publish_debug_clusters:
            selected_world = self._camera_to_world(selected_camera, rot, t)
            debug_points.extend(
                self._build_colored_point_entries(selected_world, selected_color)
            )
            rejected_camera = box_points_camera[~keep_mask]
            if len(rejected_camera) > 0:
                rejected_world = self._camera_to_world(rejected_camera, rot, t)
                debug_points.extend(
                    self._build_colored_point_entries(
                        rejected_world, self._random_color_from_points(rejected_world)
                    )
                )

        return self._camera_to_world(selected_camera, rot, t), debug_points

    def process_3d_bboxes(
        self,
        header: Header,
        bboxes: List[BoundingBox],
        cloud_map: o3d.geometry.PointCloud,
    ) -> None:
        """Project detections into the map, associate points, and publish 3D boxes."""
        try:
            points_world = np.asarray(cloud_map.points)
            if points_world.size == 0:
                return

            transform = self.tf_buffer.lookup_transform(
                self.config.camera_frame,
                self.config.target_frame,
                self._tf_time(header.stamp),
                Duration(seconds=3.0),
            )

            translation = np.array(
                [
                    transform.transform.translation.x,
                    transform.transform.translation.y,
                    transform.transform.translation.z,
                ]
            )
            quat = np.array(
                [
                    transform.transform.rotation.x,
                    transform.transform.rotation.y,
                    transform.transform.rotation.z,
                    transform.transform.rotation.w,
                ]
            )
            rot = R.from_quat(quat).as_matrix()
            t = translation.reshape(3, 1)

            points_camera = (rot @ points_world.T + t).T
            valid_depth = points_camera[:, 2] > 0
            points_camera = points_camera[valid_depth]
            if len(points_camera) == 0:
                return

            u = (points_camera[:, 0] * self.fx / points_camera[:, 2]) + self.cx
            v = (points_camera[:, 1] * self.fy / points_camera[:, 2]) + self.cy
            proj_points_uv = np.stack((u, v), axis=-1)

            bbs3d = BoundingBox3DArray()
            bbs3d.header = header
            all_labels = []
            all_confidences = []
            debug_cluster_points = []

            for bbox in bboxes:
                box_points_camera, box_uv = self._get_detection_points(
                    points_camera, proj_points_uv, bbox
                )
                if len(box_points_camera) < self.config.min_points_per_cluster:
                    continue

                selected_color = self.class_colors.get(
                    bbox.details,
                    tuple(np.random.randint(0, 255, size=3).tolist()),
                )

                if self.association_method == "front_surface":
                    selected_world, debug_points = self._select_front_surface_points(
                        box_points_camera, box_uv, selected_color, rot, t
                    )
                else:
                    selected_world, debug_points = self._select_dbscan_cluster(
                        box_points_camera, selected_color, rot, t
                    )

                if len(selected_world) < self.config.min_points_per_cluster:
                    continue
                debug_cluster_points.extend(debug_points)

                cluster_cloud = o3d.geometry.PointCloud()
                cluster_cloud.points = o3d.utility.Vector3dVector(selected_world)
                aabb = cluster_cloud.get_axis_aligned_bounding_box()

                center = aabb.get_center()
                extent = aabb.get_extent()

                bbox3d = BoundingBox3D()
                bbox3d.center.position.x = center[0]
                bbox3d.center.position.y = center[1]
                bbox3d.center.position.z = center[2]
                bbox3d.center.orientation.w = 1.0
                bbox3d.size.x = extent[0]
                bbox3d.size.y = extent[1]
                bbox3d.size.z = extent[2]
                bbs3d.boxes.append(bbox3d)
                all_labels.append(bbox.details)
                all_confidences.append(bbox.confidence)

            if len(bbs3d.boxes) > 0:
                self.bbox3d_pub.publish(bbs3d)
                self._pub_visualization_3dboxes(bbs3d, all_labels, all_confidences)
            if self.config.publish_debug_clusters and debug_cluster_points:
                self.debug_clusters_pub.publish(
                    self._create_colored_pointcloud_msg(header, debug_cluster_points)
                )

        except Exception as e:
            self.get_logger().error(f"Error in process_3d_bboxes: {e}")

    @staticmethod
    def _pack_rgb(color: Tuple[int, int, int]) -> float:
        """Pack RGB bytes into the float format expected by PointCloud2 rgb fields."""
        rgb_uint32 = (color[0] << 16) | (color[1] << 8) | color[2]
        return struct.unpack("f", struct.pack("I", rgb_uint32))[0]

    @staticmethod
    def _random_color_from_points(points: np.ndarray) -> Tuple[int, int, int]:
        """Generate a stable pseudo-random color for a cluster from its centroid."""
        centroid = np.mean(points, axis=0)
        seed = int(np.abs(np.round(np.sum(centroid) * 1000.0))) % (2**32)
        rng = np.random.default_rng(seed)
        return tuple(int(c) for c in rng.integers(0, 256, size=3))

    def _build_colored_point_entries(
        self, points: np.ndarray, color: Tuple[int, int, int]
    ) -> List[Tuple[float, float, float, float]]:
        """Convert XYZ points to PointCloud2 entries with a packed rgb field."""
        packed_rgb = self._pack_rgb(color)
        return [
            (float(point[0]), float(point[1]), float(point[2]), packed_rgb)
            for point in points
        ]

    def _create_colored_pointcloud_msg(
        self, header: Header, points: List[Tuple[float, float, float, float]]
    ) -> PointCloud2:
        """Create a colored PointCloud2 message in the target frame."""
        fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name="rgb", offset=12, datatype=PointField.FLOAT32, count=1),
        ]
        cloud_header = Header(frame_id=self.config.target_frame, stamp=header.stamp)
        return pc2.create_cloud(cloud_header, fields, points)

    def _pub_visualization_3dboxes(
        self,
        msg: BoundingBox3DArray,
        labels: List[str],
        confidences: List[Optional[float]],
    ) -> None:
        markers = MarkerArray()
        for i, box in enumerate(msg.boxes):
            # --- Wireframe marker ---
            m = Marker()
            m.header = msg.header
            m.header.frame_id = self.config.target_frame
            m.ns = "bbox3d_wireframe"
            m.id = i + self.boxes_id
            m.type = Marker.LINE_LIST
            m.action = Marker.ADD
            m.pose = box.center

            # Define 8 corners of the bounding box
            dx, dy, dz = box.size.x / 2.0, box.size.y / 2.0, box.size.z / 2.0
            corners = [
                (-dx, -dy, -dz),
                (dx, -dy, -dz),
                (dx, dy, -dz),
                (-dx, dy, -dz),
                (-dx, -dy, dz),
                (dx, -dy, dz),
                (dx, dy, dz),
                (-dx, dy, dz),
            ]

            # Define edges as pairs of corner indices
            edges = [
                (0, 1),
                (1, 2),
                (2, 3),
                (3, 0),  # bottom face
                (4, 5),
                (5, 6),
                (6, 7),
                (7, 4),  # top face
                (0, 4),
                (1, 5),
                (2, 6),
                (3, 7),  # vertical edges
            ]

            # Add edge points
            for start, end in edges:
                p1 = Point()
                p1.x, p1.y, p1.z = corners[start]
                p2 = Point()
                p2.x, p2.y, p2.z = corners[end]
                m.points.append(p1)
                m.points.append(p2)

            # Line color and thickness
            random_color = tuple(np.random.rand(3).tolist())
            if i < len(labels) and labels[i] in self.class_colors:
                random_color = tuple(c / 255.0 for c in self.class_colors[labels[i]])
            m.color.r = random_color[0]
            m.color.g = random_color[1]
            m.color.b = random_color[2]
            m.color.a = 1.0
            m.scale.x = 0.05  # line thickness

            m.lifetime = self._marker_lifetime()
            markers.markers.append(m)

            # --- Label marker ---
            label = Marker()
            label.header = msg.header
            label.header.frame_id = self.config.target_frame
            label.ns = "bbox3d_labels"
            label.id = 10000 + i + self.boxes_id
            label.type = Marker.TEXT_VIEW_FACING
            label.action = Marker.ADD

            # Position label slightly above the box
            label.pose = Pose()
            label.pose.position.x = box.center.position.x
            label.pose.position.y = box.center.position.y
            label.pose.position.z = box.center.position.z + (box.size.z / 2.0) + 0.2
            label.pose.orientation.w = 1.0

            label.scale.z = 1.0
            label.color.r = 1.0
            label.color.g = 1.0
            label.color.b = 1.0
            label.color.a = 1.0

            # Set label text (adjust depending on your message)
            label.text = labels[i] if i < len(labels) else "Object"
            if i < len(confidences) and confidences[i] is not None:
                label.text += f" {confidences[i]:.2f}"

            label.lifetime = self._marker_lifetime()
            markers.markers.append(label)

        if self.config.keep_boxes:
            self.boxes_id += len(msg.boxes)
        self.vis_3dboxes_pub.publish(markers)

    def _marker_lifetime(self) -> DurationMsg:
        lifetime = DurationMsg()
        if self.config.keep_boxes:
            return lifetime
        lifetime.sec = int(self.config.worker.min_separation_s)
        lifetime.nanosec = int((self.config.worker.min_separation_s % 1.0) * 1e9)
        return lifetime


def main(args=None) -> None:
    """Main function to run the Detection VLM ROS node."""
    rclpy.init(args=args)
    node = DetectionVLMNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
