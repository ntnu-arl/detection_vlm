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
"""Semantic map ROS 2 node."""

import csv
import hashlib
import os
import struct
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import open3d as o3d
import rclpy
import tf2_ros
import tf2_sensor_msgs.tf2_sensor_msgs as tf2sm
import yaml
from builtin_interfaces.msg import Duration as DurationMsg
from geometry_msgs.msg import Point, Pose, TransformStamped
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as R
from sensor_msgs.msg import CameraInfo, CompressedImage, Image, PointCloud2, PointField
from spark_config.config import Config, config_field
from std_msgs.msg import Header
from visualization_msgs.msg import Marker, MarkerArray

import detection_vlm_python.models as models
import detection_vlm_ros.point_cloud2 as pc2
from detection_vlm_msgs.srv import SetPrompt
from detection_vlm_python import BoundingBox
from detection_vlm_ros import ImageWorker, ImageWorkerConfig
from detection_vlm_ros.ros_conversions import Conversions

NO_SEMANTICS_COLOR = (200, 200, 200)


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
class SemanticMapNodeConfig(Config):
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
    voxel_size: float = 0.10
    min_points_per_cluster: int = 8
    eps_dbscan: float = 0.3
    compressed_image: bool = False
    classes_file: str = ""
    association_method: str = "dbscan"
    front_surface_cell_size_px: int = 20
    front_surface_depth_band: float = 0.20
    front_surface_depth_band_scale: float = 0.05
    front_surface_depth_percentile: float = 20.0
    front_surface_outlier_std_ratio: float = 1.5
    semantic_update_method: str = "counter"
    default_detection_confidence: float = 0.7
    semantic_publish_min_observations: int = 1
    semantic_publish_min_confidence: float = 0.0
    semantic_publish_period_s: float = 0.5
    publish_semantic_objects: bool = True
    semantic_objects_period_s: float = 2.0
    semantic_cluster_eps: float = 0.5
    semantic_cluster_min_points: int = 3
    semantic_object_min_voxels: int = 3
    semantic_object_min_confidence: float = 0.4
    semantic_object_node_height: float = 3.0
    semantic_object_node_radius: float = 0.18
    semantic_object_text_offset: float = 0.22
    show_confidence: bool = False
    active_window_enabled: bool = True
    active_window_radius_xy: float = 12.0
    active_window_min_z: float = -2.0
    active_window_max_z: float = 4.0
    active_window_apply_to_updates: bool = True
    active_window_apply_to_visualization: bool = False
    semantic_visualization_voxel_size: float = 0.75
    semantic_visualization_max_points: int = 50000
    semantic_object_visualization_voxel_size: float = 0.75
    semantic_object_max_dirty_labels_per_cycle: int = 4
    max_active_update_points: int = 30000
    runtime_logging_enabled: bool = False
    runtime_log_file: str = "/tmp/semantic_map_runtimes.csv"
    camera_intrinsics: CameraIntrinsics = field(default_factory=CameraIntrinsics)
    distortion_coeffs: DistortionCoeffs = field(default_factory=DistortionCoeffs)


@dataclass
class SemanticVoxel:
    scores: np.ndarray
    observations: int = 0


@dataclass
class SemanticObjectCluster:
    center: np.ndarray
    extent: np.ndarray
    confidence: float


class SemanticMapNode(Node):
    """ROS 2 node that builds a semantic voxel map from image detections."""

    NO_SEMANTICS_LABEL = "no_semantics"

    def __init__(self) -> None:
        super().__init__("semantic_map_node")
        ros_config_params = (
            self.declare_parameter("config", "").get_parameter_value().string_value
        )
        config_path_param = (
            self.declare_parameter("config_path", "").get_parameter_value().string_value
        )
        config_path = Path(config_path_param).expanduser().absolute()
        if not config_path_param:
            self.config = SemanticMapNodeConfig()
        elif not config_path.exists():
            self.get_logger().warn(f"config path '{config_path}' does not exist!")
            self.config = SemanticMapNodeConfig()
        else:
            self.config = Config.load(SemanticMapNodeConfig, config_path)

        overrides = yaml.safe_load(ros_config_params) if ros_config_params else {}
        if overrides:
            self.config.update(overrides)

        self.prompt = self.config.prompt
        self.vlm_model = self.config.vlm.create()
        self.confidence_threshold = getattr(
            self.vlm_model.config, "confidence_threshold", None
        )
        self.association_method = self.config.association_method.strip().lower()
        if self.association_method not in {"dbscan", "front_surface", "none"}:
            self.get_logger().warn(
                f"Unknown association_method={self.config.association_method!r}, falling back to 'none'."
            )
            self.association_method = "none"
        self.semantic_update_method = self.config.semantic_update_method.strip().lower()
        if self.semantic_update_method not in {"counter", "bayes"}:
            self.get_logger().warn(
                f"Unknown semantic_update_method={self.config.semantic_update_method!r}, falling back to 'counter'."
            )
            self.semantic_update_method = "counter"

        self._configure_classes()
        self.get_logger().info(f"Initializing with {self.config.show()}")

        self.runtime_log_lock = threading.Lock()
        self.runtime_log_file = None
        self.runtime_log_writer = None
        self._init_runtime_logger()

        self.worker = ImageWorker(
            self,
            self.config.worker,
            "input_image",
            CompressedImage if self.config.compressed_image else Image,
            self._spin_once,
        )

        latched_qos = QoSProfile(depth=1)
        latched_qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
        latched_qos.reliability = ReliabilityPolicy.RELIABLE

        self.srv = self.create_service(SetPrompt, "set_prompt", self._handle_set_prompt)
        self.detections_image_pub = self.create_publisher(Image, "detections_image", 1)
        self.semantic_map_pub = self.create_publisher(
            PointCloud2, "semantic_map", latched_qos
        )
        self.semantic_objects_pub = self.create_publisher(
            MarkerArray, "semantic_objects", latched_qos
        )
        self.pcl_sub = self.create_subscription(
            PointCloud2, "input_pointcloud", self._pcl_callback, 1
        )
        self.camera_info_sub = self.create_subscription(
            CameraInfo, "input_camera_info", self._camera_info_callback, 1
        )

        self.fx = self.config.camera_intrinsics.fx
        self.fy = self.config.camera_intrinsics.fy
        self.cx = self.config.camera_intrinsics.cx
        self.cy = self.config.camera_intrinsics.cy
        self.k1 = self.config.distortion_coeffs.k1
        self.k2 = self.config.distortion_coeffs.k2
        self.p1 = self.config.distortion_coeffs.p1
        self.p2 = self.config.distortion_coeffs.p2

        self.map_lock = threading.Lock()
        self.tf_buffer = tf2_ros.Buffer(cache_time=Duration(seconds=30.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.geometry_points = np.empty((0, 3), dtype=np.float32)
        self.geometry_key_to_index: Dict[Tuple[int, int, int], int] = {}
        self.semantic_voxels: Dict[Tuple[int, int, int], SemanticVoxel] = {}
        self.semantic_map_revision = 0
        self.semantic_map_published_revision = -1
        self.semantic_objects_revision = 0
        self.semantic_objects_published_revision = -1
        self.latest_output_stamp = self.get_clock().now().to_msg()

        self.label_names = list(getattr(self.vlm_model, "names", {}).values())
        if self.NO_SEMANTICS_LABEL not in self.label_names:
            self.label_names.append(self.NO_SEMANTICS_LABEL)
        self.label_to_index = {name: idx for idx, name in enumerate(self.label_names)}
        self.no_semantics_idx = self.label_to_index[self.NO_SEMANTICS_LABEL]
        self.class_colors = {
            name: self._color_for_label(name) for name in self.label_names
        }
        self.label_packed_colors = np.array(
            [self._pack_rgb(self.class_colors[name]) for name in self.label_names],
            dtype=np.float32,
        )
        self.semantic_cloud_fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name="rgb", offset=12, datatype=PointField.FLOAT32, count=1),
        ]

        self.semantic_map_timer = None
        if self.config.semantic_publish_period_s > 0.0:
            self.semantic_map_timer = self.create_timer(
                self.config.semantic_publish_period_s,
                self._semantic_map_timer_callback,
            )
        self.semantic_objects_timer = None
        if (
            self.config.publish_semantic_objects
            and self.config.semantic_objects_period_s > 0.0
        ):
            self.semantic_objects_timer = self.create_timer(
                self.config.semantic_objects_period_s,
                self._semantic_objects_timer_callback,
            )

        model = getattr(self.vlm_model, "model", None)
        device = getattr(model, "device", "api")
        self.get_logger().info(f"Device: {device}")
        self.get_logger().info("Semantic map node initialized.")

    def _configure_classes(self) -> None:
        vlm_model_name = getattr(self.config.vlm, "model", "")
        if "pf" in vlm_model_name:
            return
        classes_file = Path(self.config.classes_file)
        if not classes_file.exists():
            if self.config.classes_file:
                self.get_logger().warn(
                    f"Classes file {classes_file} does not exist. Using model defaults."
                )
            return
        with classes_file.open("r") as handle:
            class_names = [line.strip() for line in handle.readlines() if line.strip()]
        if not hasattr(self.vlm_model, "set_classes"):
            success = False
        else:
            success = self.vlm_model.set_classes(class_names)
        if not success:
            self.get_logger().warn("Model does not support setting classes.")
        else:
            self.get_logger().info(
                f"Loaded {len(class_names)} classes from {classes_file}."
            )

    def _init_runtime_logger(self) -> None:
        if not self.config.runtime_logging_enabled:
            return
        log_path = Path(os.path.expandvars(self.config.runtime_log_file)).expanduser()
        try:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            write_header = not log_path.exists() or log_path.stat().st_size == 0
            self.runtime_log_file = log_path.open("a", newline="", buffering=1)
            self.runtime_log_writer = csv.DictWriter(
                self.runtime_log_file,
                fieldnames=[
                    "wall_time_s",
                    "ros_time_s",
                    "module",
                    "duration_ms",
                    "detections",
                    "active_points",
                    "updated",
                    "semantic_objects",
                    "markers",
                    "notes",
                ],
            )
            if write_header:
                self.runtime_log_writer.writeheader()
            self.get_logger().info(f"Runtime logging enabled: {log_path}")
        except Exception as e:
            self.runtime_log_file = None
            self.runtime_log_writer = None
            self.get_logger().error(
                f"Failed to initialize runtime logger at {log_path}: {e}"
            )

    def _close_runtime_logger(self) -> None:
        with self.runtime_log_lock:
            if self.runtime_log_file is None:
                return
            try:
                self.runtime_log_file.close()
            except Exception as e:
                self.get_logger().warn(f"Failed to close runtime log file: {e}")
            finally:
                self.runtime_log_file = None
                self.runtime_log_writer = None

    def _log_runtime(
        self,
        module: str,
        duration_ms: float,
        header: Optional[Header] = None,
        detections: Optional[int] = None,
        active_points: Optional[int] = None,
        updated: Optional[bool] = None,
        semantic_objects: Optional[int] = None,
        markers: Optional[int] = None,
        notes: str = "",
    ) -> None:
        if self.runtime_log_writer is None:
            return
        ros_time_s = ""
        if header is not None and header.stamp is not None:
            ros_time_s = f"{to_sec(header.stamp):.9f}"
        row = {
            "wall_time_s": f"{time.time():.9f}",
            "ros_time_s": ros_time_s,
            "module": module,
            "duration_ms": f"{duration_ms:.3f}",
            "detections": "" if detections is None else detections,
            "active_points": "" if active_points is None else active_points,
            "updated": "" if updated is None else int(updated),
            "semantic_objects": "" if semantic_objects is None else semantic_objects,
            "markers": "" if markers is None else markers,
            "notes": notes,
        }
        with self.runtime_log_lock:
            if self.runtime_log_writer is None:
                return
            try:
                self.runtime_log_writer.writerow(row)
            except Exception as e:
                self.get_logger().warn(f"Failed to write runtime log row: {e}")

    def _camera_info_callback(self, msg: CameraInfo) -> None:
        self.fx = msg.k[0]
        self.fy = msg.k[4]
        self.cx = msg.k[2]
        self.cy = msg.k[5]
        self.k1 = msg.d[0] if len(msg.d) > 0 else None
        self.k2 = msg.d[1] if len(msg.d) > 1 else None
        self.p1 = msg.d[2] if len(msg.d) > 2 else None
        self.p2 = msg.d[3] if len(msg.d) > 3 else None
        self.destroy_subscription(self.camera_info_sub)
        self.get_logger().info(
            f"Camera intrinsics set: fx={self.fx}, fy={self.fy}, cx={self.cx}, cy={self.cy}"
        )

    def _handle_set_prompt(self, req: SetPrompt, response):
        self.prompt = req.prompt
        self.get_logger().info(f"Prompt updated to: {self.prompt}")
        response.success = True
        return response

    def _tf_time(self, stamp):
        if self.config.use_tf_current_time:
            return rclpy.time.Time()
        return rclpy.time.Time.from_msg(stamp)

    def _lidar_to_body_transform(self) -> TransformStamped:
        transform = TransformStamped()
        transform.header.frame_id = self.config.body_frame
        transform.header.stamp = self.get_clock().now().to_msg()
        transform.child_frame_id = "lidar"
        transform.transform.translation.x = (
            self.config.lidar_to_body_transform.translation.x
        )
        transform.transform.translation.y = (
            self.config.lidar_to_body_transform.translation.y
        )
        transform.transform.translation.z = (
            self.config.lidar_to_body_transform.translation.z
        )
        transform.transform.rotation.x = (
            self.config.lidar_to_body_transform.quaternion.x
        )
        transform.transform.rotation.y = (
            self.config.lidar_to_body_transform.quaternion.y
        )
        transform.transform.rotation.z = (
            self.config.lidar_to_body_transform.quaternion.z
        )
        transform.transform.rotation.w = (
            self.config.lidar_to_body_transform.quaternion.w
        )
        return transform

    def _pcl_callback(self, msg: PointCloud2) -> None:
        try:
            body2world = self.tf_buffer.lookup_transform(
                self.config.target_frame,
                self.config.body_frame,
                self._tf_time(msg.header.stamp),
                Duration(seconds=3.0),
            )
            body_cloud = tf2sm.do_transform_cloud(msg, self._lidar_to_body_transform())
            points_body = np.asarray(
                list(
                    pc2.read_points(
                        body_cloud, field_names=("x", "y", "z"), skip_nans=True
                    )
                ),
                dtype=np.float32,
            ).reshape(-1, 3)
            if points_body.size == 0:
                return
            dists = np.linalg.norm(points_body, axis=1)
            valid_indices = np.where(
                (dists >= self.config.min_point_r) & (dists <= self.config.max_point_r)
            )[0]
            if valid_indices.size == 0:
                return

            transformed_cloud = tf2sm.do_transform_cloud(body_cloud, body2world)
            points = np.asarray(
                list(
                    pc2.read_points(
                        transformed_cloud, field_names=("x", "y", "z"), skip_nans=True
                    )
                ),
                dtype=np.float32,
            ).reshape(-1, 3)
            points = points[valid_indices]
            if points.size == 0:
                return

            with self.map_lock:
                self._upsert_geometry_points_locked(points)
                self.latest_output_stamp = msg.header.stamp
        except (
            tf2_ros.LookupException,
            tf2_ros.ExtrapolationException,
            tf2_ros.ConnectivityException,
        ) as e:
            self.get_logger().warn(f"TF lookup failed: {e}")
        except Exception as e:
            self.get_logger().error(f"Error processing pointcloud: {e}")

    def _spin_once(self, header: Header, image: np.ndarray) -> None:
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
        detection_ms = (time.time() - start_time) * 1000.0
        self._log_runtime(
            "vlm_2d_object_detection",
            detection_ms,
            header,
            detections=len(bboxes),
        )
        if self.config.verbose:
            self.get_logger().info(
                f"Detected {len(bboxes)} objects in {detection_ms:.2f} ms."
            )

        self._publish_detections_image(image, bboxes)

        active_center = self._lookup_active_window_center(header.stamp)
        with self.map_lock:
            self.latest_output_stamp = header.stamp
        active_points = self._snapshot_active_geometry_points(active_center)
        if len(active_points) == 0:
            return

        cloud_snapshot = o3d.geometry.PointCloud()
        cloud_snapshot.points = o3d.utility.Vector3dVector(active_points)

        if len(bboxes) > 0 and self.fx is not None:
            semantic_update_start = time.time()
            updated = self._update_semantic_map(
                header, bboxes, cloud_snapshot, image.shape[:2]
            )
            semantic_update_ms = (time.time() - semantic_update_start) * 1000.0
            self._log_runtime(
                "semantic_map_update",
                semantic_update_ms,
                header,
                detections=len(bboxes),
                active_points=len(active_points),
                updated=updated,
            )

    def _publish_detections_image(
        self, image: np.ndarray, bboxes: List[BoundingBox]
    ) -> None:
        detection_image = cv2.cvtColor(image.copy(), cv2.COLOR_RGB2BGR)
        for bbox in bboxes:
            color = self.class_colors.get(
                bbox.details, self._color_for_label(bbox.details)
            )
            color_bgr = (color[2], color[1], color[0])
            cv2.rectangle(
                detection_image,
                (bbox.x0, bbox.y0),
                (bbox.x1, bbox.y1),
                color_bgr,
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
                color_bgr,
                2,
            )
            if bbox.mask is not None:
                colored_mask = np.zeros_like(detection_image, dtype=np.uint8)
                colored_mask[bbox.mask] = color_bgr
                detection_image = cv2.addWeighted(
                    detection_image, 1.0, colored_mask, 0.5, 0
                )
        self.detections_image_pub.publish(Conversions.to_sensor_image(detection_image))

    @staticmethod
    def get_points_and_uv_in_mask(
        points_3d: np.ndarray, proj_points_uv: np.ndarray, mask: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
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
        inside_mask = mask[uv_valid[:, 1], uv_valid[:, 0]].astype(bool)
        return pts_valid[inside_mask], uv_valid[inside_mask]

    @staticmethod
    def _camera_to_world(
        points_camera: np.ndarray, rot: np.ndarray, t: np.ndarray
    ) -> np.ndarray:
        return (rot.T @ (points_camera - t.T).T).T

    @classmethod
    def _color_for_label(cls, label: str) -> Tuple[int, int, int]:
        if label == cls.NO_SEMANTICS_LABEL:
            return NO_SEMANTICS_COLOR
        digest = hashlib.sha1(label.encode("utf-8")).digest()
        return (
            64 + digest[0] % 160,
            64 + digest[1] % 160,
            64 + digest[2] % 160,
        )

    def _upsert_geometry_points_locked(self, points_world: np.ndarray) -> None:
        if len(points_world) == 0:
            return
        voxel_indices = np.floor(points_world / self.config.voxel_size).astype(np.int32)
        unique_voxels, first_indices = np.unique(
            voxel_indices, axis=0, return_index=True
        )
        representative_points = points_world[first_indices].astype(
            np.float32, copy=False
        )
        new_geometry_points = []
        initial_scores = self._initial_scores()
        for voxel_key_arr, rep_point in zip(
            unique_voxels.tolist(), representative_points
        ):
            voxel_key = tuple(voxel_key_arr)
            geometry_idx = self.geometry_key_to_index.get(voxel_key)
            if geometry_idx is None:
                geometry_idx = len(self.geometry_points) + len(new_geometry_points)
                self.geometry_key_to_index[voxel_key] = geometry_idx
                new_geometry_points.append(rep_point)
            else:
                self.geometry_points[geometry_idx] = rep_point
            if voxel_key not in self.semantic_voxels:
                self.semantic_voxels[voxel_key] = SemanticVoxel(
                    scores=initial_scores.copy(), observations=1
                )
        if new_geometry_points:
            append_points = np.asarray(new_geometry_points, dtype=np.float32)
            if len(self.geometry_points) == 0:
                self.geometry_points = append_points
            else:
                self.geometry_points = np.vstack((self.geometry_points, append_points))
            self.semantic_map_revision += 1
            self.semantic_objects_revision += 1

    def _snapshot_active_geometry_points(self, active_center: np.ndarray) -> np.ndarray:
        with self.map_lock:
            points = self.geometry_points
            if len(points) == 0:
                return np.empty((0, 3), dtype=np.float32)
            if (
                self.config.active_window_enabled
                and self.config.active_window_apply_to_updates
                and active_center is not None
            ):
                points = points[self._active_window_mask(points, active_center)]
            if len(points) == 0:
                return np.empty((0, 3), dtype=np.float32)
            max_points = int(self.config.max_active_update_points)
            if max_points > 0 and len(points) > max_points:
                if active_center is not None:
                    dist2 = np.sum((points - active_center.reshape(1, 3)) ** 2, axis=1)
                    keep_idx = np.argpartition(dist2, max_points - 1)[:max_points]
                    points = points[keep_idx]
                else:
                    sample_idx = np.linspace(
                        0, len(points) - 1, max_points, dtype=np.int32
                    )
                    points = points[sample_idx]
            return points.copy()

    def _lookup_active_window_center(self, stamp) -> Optional[np.ndarray]:
        if not self.config.active_window_enabled:
            return None
        try:
            transform = self.tf_buffer.lookup_transform(
                self.config.target_frame,
                self.config.body_frame,
                self._tf_time(stamp),
                Duration(seconds=0.5),
            )
            return np.array(
                [
                    transform.transform.translation.x,
                    transform.transform.translation.y,
                    transform.transform.translation.z,
                ],
                dtype=np.float32,
            )
        except (
            tf2_ros.LookupException,
            tf2_ros.ExtrapolationException,
            tf2_ros.ConnectivityException,
        ) as e:
            self.get_logger().warn(f"Active-window TF lookup failed: {e}")
            return None

    def _active_window_mask(self, points: np.ndarray, center: np.ndarray) -> np.ndarray:
        if center is None or len(points) == 0 or not self.config.active_window_enabled:
            return np.ones(len(points), dtype=bool)
        dx = points[:, 0] - center[0]
        dy = points[:, 1] - center[1]
        dz = points[:, 2] - center[2]
        radius_mask = (dx * dx + dy * dy) <= (self.config.active_window_radius_xy**2)
        z_mask = (dz >= self.config.active_window_min_z) & (
            dz <= self.config.active_window_max_z
        )
        return radius_mask & z_mask

    def _current_output_header(self) -> Header:
        with self.map_lock:
            stamp = self.latest_output_stamp
        if stamp.sec == 0 and stamp.nanosec == 0:
            stamp = self.get_clock().now().to_msg()
        return Header(frame_id=self.config.target_frame, stamp=stamp)

    def _semantic_map_timer_callback(self) -> None:
        with self.map_lock:
            revision = self.semantic_map_revision
        if revision == self.semantic_map_published_revision:
            return
        self._publish_semantic_map(self._current_output_header())
        self.semantic_map_published_revision = revision

    def _semantic_objects_timer_callback(self) -> None:
        with self.map_lock:
            revision = self.semantic_objects_revision
        if revision == self.semantic_objects_published_revision:
            return
        self._publish_semantic_objects(self._current_output_header())
        self.semantic_objects_published_revision = revision

    def _get_detection_points(
        self,
        points_camera: np.ndarray,
        proj_points_uv: np.ndarray,
        bbox: BoundingBox,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if bbox.mask is not None:
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
        rot: np.ndarray,
        t: np.ndarray,
    ) -> np.ndarray:
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
            return np.empty((0, 3))
        clusters = [
            box_points_camera[labels == label]
            for label in np.unique(labels)
            if label != -1
        ]
        if not clusters:
            return np.empty((0, 3))
        selected_idx = max(range(len(clusters)), key=lambda idx: len(clusters[idx]))
        return self._camera_to_world(clusters[selected_idx], rot, t)

    def _select_front_surface_points(
        self,
        box_points_camera: np.ndarray,
        box_uv: np.ndarray,
        rot: np.ndarray,
        t: np.ndarray,
    ) -> np.ndarray:
        if len(box_points_camera) == 0:
            return np.empty((0, 3))
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
            return np.empty((0, 3))
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
        return self._camera_to_world(selected_camera, rot, t)

    def _associate_points(
        self,
        box_points_camera: np.ndarray,
        box_uv: np.ndarray,
        rot: np.ndarray,
        t: np.ndarray,
    ) -> np.ndarray:
        if self.association_method == "front_surface":
            return self._select_front_surface_points(box_points_camera, box_uv, rot, t)
        if self.association_method == "dbscan":
            return self._select_dbscan_cluster(box_points_camera, rot, t)
        return self._camera_to_world(box_points_camera, rot, t)

    def _update_semantic_map(
        self,
        header: Header,
        bboxes: List[BoundingBox],
        cloud_map: o3d.geometry.PointCloud,
        image_shape: Tuple[int, int],
    ) -> bool:
        try:
            points_world = np.asarray(cloud_map.points)
            if points_world.size == 0:
                return False

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
                return False

            u = (points_camera[:, 0] * self.fx / points_camera[:, 2]) + self.cx
            v = (points_camera[:, 1] * self.fy / points_camera[:, 2]) + self.cy
            image_h, image_w = image_shape
            in_image = (u >= 0) & (u < image_w) & (v >= 0) & (v < image_h)
            points_camera = points_camera[in_image]
            u = u[in_image]
            v = v[in_image]
            if len(points_camera) == 0:
                return False
            proj_points_uv = np.stack((u, v), axis=-1)

            updated = False
            for bbox in bboxes:
                label_idx = self.label_to_index.get(bbox.details)
                if label_idx is None:
                    self.get_logger().warn(f"Skipping unknown label {bbox.details!r}.")
                    continue
                box_points_camera, box_uv = self._get_detection_points(
                    points_camera, proj_points_uv, bbox
                )
                if len(box_points_camera) < self.config.min_points_per_cluster:
                    continue
                selected_world = self._associate_points(
                    box_points_camera, box_uv, rot, t
                )
                if len(selected_world) < self.config.min_points_per_cluster:
                    continue
                self._update_semantic_voxels(selected_world, label_idx, bbox.confidence)
                updated = True
            if updated:
                with self.map_lock:
                    self.semantic_map_revision += 1
                    self.semantic_objects_revision += 1
            return updated
        except Exception as e:
            self.get_logger().error(f"Error updating semantic map: {e}")
            return False

    def _update_semantic_voxels(
        self, points_world: np.ndarray, label_idx: int, confidence: float
    ) -> None:
        if len(points_world) == 0:
            return
        unique_voxels = np.unique(
            np.floor(points_world / self.config.voxel_size).astype(np.int32), axis=0
        )
        with self.map_lock:
            initial_scores = self._initial_scores()
            for voxel_key in map(tuple, unique_voxels.tolist()):
                voxel = self.semantic_voxels.get(voxel_key)
                if voxel is None:
                    voxel = SemanticVoxel(scores=initial_scores.copy())
                    self.semantic_voxels[voxel_key] = voxel
                voxel.observations += 1
                self._apply_semantic_update(voxel.scores, label_idx, confidence)

    def _initial_scores(self) -> np.ndarray:
        scores = np.full(len(self.label_names), 1e-3, dtype=np.float64)
        scores[self.no_semantics_idx] = 1.0
        if self.semantic_update_method == "bayes":
            scores /= np.sum(scores)
        return scores

    def _apply_semantic_update(
        self, scores: np.ndarray, label_idx: int, confidence: float
    ) -> None:
        detection_conf = confidence
        if detection_conf is None:
            detection_conf = self.config.default_detection_confidence
        detection_conf = float(np.clip(detection_conf, 1e-3, 0.999))
        if self.semantic_update_method == "bayes":
            off_prob = (1.0 - detection_conf) / max(1, len(scores) - 1)
            likelihood = np.full(len(scores), off_prob, dtype=np.float64)
            likelihood[label_idx] = detection_conf
            posterior = scores * likelihood
            denom = np.sum(posterior)
            if denom <= 0:
                posterior = self._initial_scores()
            else:
                posterior /= denom
            scores[:] = posterior
        else:
            if label_idx != self.no_semantics_idx:
                scores[self.no_semantics_idx] = max(
                    0.0, scores[self.no_semantics_idx] - detection_conf
                )
            scores[label_idx] += max(detection_conf, 0.1)

    def _voxel_center(self, voxel_key: Tuple[int, int, int]) -> np.ndarray:
        return (np.asarray(voxel_key, dtype=np.float64) + 0.5) * self.config.voxel_size

    def _voxel_probabilities(self, voxel: SemanticVoxel) -> np.ndarray:
        total = np.sum(voxel.scores)
        if total <= 0:
            return np.zeros_like(voxel.scores)
        if self.semantic_update_method == "bayes":
            return voxel.scores
        return voxel.scores / total

    @staticmethod
    def _pack_rgb(color: Tuple[int, int, int]) -> float:
        rgb_uint32 = (color[0] << 16) | (color[1] << 8) | color[2]
        return struct.unpack("f", struct.pack("I", rgb_uint32))[0]

    def _semantic_visual_state(
        self, voxel: SemanticVoxel
    ) -> Tuple[bool, Optional[int], float]:
        if voxel.observations < self.config.semantic_publish_min_observations:
            return False, None, 0.0
        probs = self._voxel_probabilities(voxel)
        if probs.size == 0:
            return False, None, 0.0
        label_idx = int(np.argmax(probs))
        confidence = float(probs[label_idx])
        if confidence < self.config.semantic_publish_min_confidence:
            return False, label_idx, confidence
        return True, label_idx, confidence

    def _snapshot_active_semantic_records(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        records = []
        label_indices = []
        confidences = []
        with self.map_lock:
            voxel_items = list(self.semantic_voxels.items())
        for voxel_key, voxel in voxel_items:
            active, label_idx, confidence = self._semantic_visual_state(voxel)
            if not active or label_idx is None:
                continue
            center = self._voxel_center(voxel_key)
            records.append(
                [
                    float(center[0]),
                    float(center[1]),
                    float(center[2]),
                    self.label_packed_colors[label_idx],
                ]
            )
            label_indices.append(label_idx)
            confidences.append(confidence)
        if not records:
            return (
                np.empty((0, 4), dtype=np.float32),
                np.empty((0,), dtype=np.int32),
                np.empty((0,), dtype=np.float32),
            )
        return (
            np.asarray(records, dtype=np.float32),
            np.asarray(label_indices, dtype=np.int32),
            np.asarray(confidences, dtype=np.float32),
        )

    def _downsample_visual_indices(
        self,
        points: np.ndarray,
        confidences: np.ndarray,
        voxel_size: float,
        max_points: int,
    ) -> np.ndarray:
        if len(points) == 0:
            return np.empty((0,), dtype=np.int32)
        indices = np.arange(len(points), dtype=np.int32)
        if voxel_size > self.config.voxel_size:
            voxel_indices = np.floor(points / voxel_size).astype(np.int32)
            _, inverse = np.unique(voxel_indices, axis=0, return_inverse=True)
            order = np.lexsort((-confidences, inverse))
            ordered_inverse = inverse[order]
            keep = np.ones(len(order), dtype=bool)
            keep[1:] = ordered_inverse[1:] != ordered_inverse[:-1]
            indices = indices[order[keep]]
        if max_points > 0 and len(indices) > max_points:
            sample_idx = np.linspace(0, len(indices) - 1, max_points, dtype=np.int32)
            indices = indices[sample_idx]
        return indices

    def _build_semantic_cloud_message(self, header: Header) -> Optional[PointCloud2]:
        records, _, confidences = self._snapshot_active_semantic_records()
        if len(records) == 0:
            return None
        if self.config.active_window_apply_to_visualization:
            active_center = self._lookup_active_window_center(header.stamp)
            if active_center is not None:
                mask = self._active_window_mask(records[:, :3], active_center)
                records = records[mask]
                confidences = confidences[mask]
        publish_idx = self._downsample_visual_indices(
            records[:, :3],
            confidences,
            self.config.semantic_visualization_voxel_size,
            self.config.semantic_visualization_max_points,
        )
        if len(publish_idx) == 0:
            return None
        publish_records = records[publish_idx]
        msg = PointCloud2()
        msg.header = Header(frame_id=self.config.target_frame, stamp=header.stamp)
        msg.height = 1
        msg.width = len(publish_records)
        msg.fields = self.semantic_cloud_fields
        msg.is_bigendian = False
        msg.point_step = 16
        msg.row_step = msg.point_step * msg.width
        msg.is_dense = True
        msg.data = publish_records.astype(np.float32, copy=False).tobytes()
        return msg

    def _publish_semantic_map(self, header: Header) -> None:
        cloud_msg = self._build_semantic_cloud_message(header)
        if cloud_msg is not None:
            self.semantic_map_pub.publish(cloud_msg)

    @staticmethod
    def _euclidean_cluster_indices(
        points: np.ndarray, tolerance: float, min_points: int
    ) -> List[np.ndarray]:
        if len(points) < min_points:
            return []
        tree = cKDTree(points)
        pairs = list(tree.query_pairs(tolerance))
        parent = np.arange(len(points), dtype=np.int32)

        def find(idx: int) -> int:
            while parent[idx] != idx:
                parent[idx] = parent[parent[idx]]
                idx = parent[idx]
            return idx

        def union(a: int, b: int) -> None:
            root_a = find(a)
            root_b = find(b)
            if root_a != root_b:
                parent[root_b] = root_a

        for a, b in pairs:
            union(a, b)

        groups: Dict[int, List[int]] = {}
        for idx in range(len(points)):
            root = find(idx)
            groups.setdefault(root, []).append(idx)

        return [
            np.asarray(indices, dtype=np.int32)
            for indices in groups.values()
            if len(indices) >= min_points
        ]

    def _build_object_clusters(self) -> Dict[int, List[SemanticObjectCluster]]:
        records, label_indices, confidences = self._snapshot_active_semantic_records()
        clusters_by_label: Dict[int, List[SemanticObjectCluster]] = {}
        for label_idx in sorted(set(label_indices.tolist())):
            if label_idx == self.no_semantics_idx:
                continue
            label_mask = label_indices == label_idx
            label_points = records[label_mask, :3]
            label_confidences = confidences[label_mask]
            if len(label_points) < self.config.semantic_object_min_voxels:
                clusters_by_label[label_idx] = []
                continue
            downsample_size = float(
                self.config.semantic_object_visualization_voxel_size
            )
            if downsample_size > self.config.voxel_size:
                keep_idx = self._downsample_visual_indices(
                    label_points,
                    label_confidences,
                    downsample_size,
                    0,
                )
                label_points = label_points[keep_idx]
                label_confidences = label_confidences[keep_idx]
            label_clusters = []
            cluster_indices = self._euclidean_cluster_indices(
                label_points,
                self.config.semantic_cluster_eps,
                self.config.semantic_cluster_min_points,
            )
            for indices in cluster_indices:
                cluster = label_points[indices]
                if len(cluster) < self.config.semantic_object_min_voxels:
                    continue
                confidence = float(np.mean(label_confidences[indices]))
                if confidence < self.config.semantic_object_min_confidence:
                    continue
                mins = np.min(cluster, axis=0)
                maxs = np.max(cluster, axis=0)
                label_clusters.append(
                    SemanticObjectCluster(
                        center=((mins + maxs) * 0.5).astype(np.float64),
                        extent=(maxs - mins).astype(np.float64),
                        confidence=confidence,
                    )
                )
            clusters_by_label[label_idx] = label_clusters
        return clusters_by_label

    def _publish_semantic_objects(self, header: Header) -> None:
        object_detection_start = time.time()
        object_cluster_cache = self._build_object_clusters()
        markers = MarkerArray()
        delete_marker = Marker()
        delete_marker.action = Marker.DELETEALL
        markers.markers.append(delete_marker)
        marker_id = 0

        for label_idx in sorted(object_cluster_cache.keys()):
            label_name = self.label_names[int(label_idx)]
            color = self.class_colors.get(label_name, (255, 255, 255))
            for cluster in object_cluster_cache[label_idx]:
                marker_id = self._append_bbox_markers(
                    markers,
                    marker_id,
                    header,
                    cluster.center,
                    cluster.extent,
                    label_name,
                    cluster.confidence,
                    color,
                )

        self.semantic_objects_pub.publish(markers)
        object_count = sum(len(clusters) for clusters in object_cluster_cache.values())
        self._log_runtime(
            "object_detection_3d",
            (time.time() - object_detection_start) * 1000.0,
            header,
            semantic_objects=object_count,
            markers=len(markers.markers),
        )

    def _append_bbox_markers(
        self,
        markers: MarkerArray,
        marker_id: int,
        header: Header,
        center: np.ndarray,
        extent: np.ndarray,
        label: str,
        confidence: float,
        color_rgb: Tuple[int, int, int],
    ) -> int:
        dx, dy, dz = extent[0] / 2.0, extent[1] / 2.0, extent[2] / 2.0
        corner_offsets = np.array(
            [
                (-dx, -dy, -dz),
                (dx, -dy, -dz),
                (dx, dy, -dz),
                (-dx, dy, -dz),
                (-dx, -dy, dz),
                (dx, -dy, dz),
                (dx, dy, dz),
                (-dx, dy, dz),
            ],
            dtype=np.float64,
        )
        corners_world = corner_offsets + center
        edges = [
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 0),
            (4, 5),
            (5, 6),
            (6, 7),
            (7, 4),
            (0, 4),
            (1, 5),
            (2, 6),
            (3, 7),
        ]
        node_center = np.array(
            [center[0], center[1], self.config.semantic_object_node_height],
            dtype=np.float64,
        )

        box_marker = Marker()
        box_marker.header = header
        box_marker.header.frame_id = self.config.target_frame
        box_marker.ns = "semantic_objects"
        box_marker.id = marker_id
        box_marker.type = Marker.LINE_LIST
        box_marker.action = Marker.ADD
        box_marker.pose = Pose()
        box_marker.pose.orientation.w = 1.0
        for start_idx, end_idx in edges:
            box_marker.points.append(self._make_point(corners_world[start_idx]))
            box_marker.points.append(self._make_point(corners_world[end_idx]))
        box_marker.scale.x = 0.5
        box_marker.color.r = color_rgb[0] / 255.0
        box_marker.color.g = color_rgb[1] / 255.0
        box_marker.color.b = color_rgb[2] / 255.0
        box_marker.color.a = 1.0
        box_marker.lifetime = DurationMsg()
        markers.markers.append(box_marker)

        connector = Marker()
        connector.header = header
        connector.header.frame_id = self.config.target_frame
        connector.ns = "semantic_object_connectors"
        connector.id = marker_id + 1
        connector.type = Marker.LINE_LIST
        connector.action = Marker.ADD
        connector.pose = Pose()
        connector.pose.orientation.w = 1.0
        for corner_idx in (4, 5, 6, 7):
            connector.points.append(self._make_point(corners_world[corner_idx]))
            connector.points.append(self._make_point(node_center))
        connector.scale.x = 0.5
        connector.color.r = color_rgb[0] / 255.0
        connector.color.g = color_rgb[1] / 255.0
        connector.color.b = color_rgb[2] / 255.0
        connector.color.a = 0.8
        connector.lifetime = DurationMsg()
        markers.markers.append(connector)

        sphere = Marker()
        sphere.header = header
        sphere.header.frame_id = self.config.target_frame
        sphere.ns = "semantic_object_nodes"
        sphere.id = marker_id + 2
        sphere.type = Marker.SPHERE
        sphere.action = Marker.ADD
        sphere.pose.position.x = node_center[0]
        sphere.pose.position.y = node_center[1]
        sphere.pose.position.z = node_center[2]
        sphere.pose.orientation.w = 1.0
        sphere.scale.x = self.config.semantic_object_node_radius * 2.0
        sphere.scale.y = self.config.semantic_object_node_radius * 2.0
        sphere.scale.z = self.config.semantic_object_node_radius * 2.0
        sphere.color.r = color_rgb[0] / 255.0
        sphere.color.g = color_rgb[1] / 255.0
        sphere.color.b = color_rgb[2] / 255.0
        sphere.color.a = 0.95
        sphere.lifetime = DurationMsg()
        markers.markers.append(sphere)

        text = Marker()
        text.header = header
        text.header.frame_id = self.config.target_frame
        text.ns = "semantic_object_labels"
        text.id = marker_id + 3
        text.type = Marker.TEXT_VIEW_FACING
        text.action = Marker.ADD
        text.pose.position.x = node_center[0]
        text.pose.position.y = node_center[1]
        text.pose.position.z = (
            node_center[2]
            + self.config.semantic_object_node_radius
            + self.config.semantic_object_text_offset
        )
        text.pose.orientation.w = 1.0
        text.scale.z = 2.5
        text.color.r = 1.0
        text.color.g = 1.0
        text.color.b = 1.0
        text.color.a = 1.0
        text.text = label
        if self.config.show_confidence:
            text.text += f" {confidence:.2f}"
        text.lifetime = DurationMsg()
        markers.markers.append(text)
        return marker_id + 4

    @staticmethod
    def _make_point(values: np.ndarray) -> Point:
        point = Point()
        point.x = float(values[0])
        point.y = float(values[1])
        point.z = float(values[2])
        return point

    def destroy_node(self) -> bool:
        self._close_runtime_logger()
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = SemanticMapNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
