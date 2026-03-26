#!/usr/bin/env python3
# BSD 3-Clause License
#
# Copyright (c) 2025, NTNU Autonomous Robots Lab
# All rights reserved.
#
"""Semantic map ROS node."""

import hashlib
import struct
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import cv2
import numpy as np
import open3d as o3d
import rospy
import sensor_msgs.point_cloud2 as pc2
import tf2_ros
import tf2_sensor_msgs.tf2_sensor_msgs as tf2sm
from geometry_msgs.msg import Point, Pose
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as R
from sensor_msgs.msg import CameraInfo, CompressedImage, Image, PointCloud2, PointField
from std_msgs.msg import Header
from visualization_msgs.msg import Marker, MarkerArray

import detection_vlm_python.models as models
import detection_vlm_ros
from detection_vlm_msgs.srv import SetPrompt, SetPromptResponse
from detection_vlm_python import BoundingBox
from detection_vlm_python.config import Config, config_field
from detection_vlm_ros import Conversions, ImageWorker, ImageWorkerConfig

NO_SEMANTICS_COLOR = (200, 200, 200)


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


class SemanticMapNode:
    """ROS node that builds a semantic voxel map from image detections."""

    NO_SEMANTICS_LABEL = "no_semantics"

    def __init__(self) -> None:
        self.config = detection_vlm_ros.load_from_ros(SemanticMapNodeConfig, ns="~")
        self.vlm_model = self.config.vlm.create()
        rospy.loginfo(f"[{rospy.get_name()}] Initializing with {self.config.show()}")

        self.worker = ImageWorker(
            self.config.worker,
            "input_image",
            CompressedImage if self.config.compressed_image else Image,
            self._spin_once,
        )
        self.prompt = self.config.prompt
        self.confidence_threshold = self.vlm_model.config.confidence_threshold
        self.association_method = self.config.association_method.strip().lower()
        if self.association_method not in {"dbscan", "front_surface", "none"}:
            rospy.logwarn(
                f"[{rospy.get_name()}] Unknown association_method={self.config.association_method!r}, falling back to 'none'."
            )
            self.association_method = "none"
        self.semantic_update_method = self.config.semantic_update_method.strip().lower()
        if self.semantic_update_method not in {"counter", "bayes"}:
            rospy.logwarn(
                f"[{rospy.get_name()}] Unknown semantic_update_method={self.config.semantic_update_method!r}, falling back to 'counter'."
            )
            self.semantic_update_method = "counter"

        if "pf" not in self.config.vlm.model:
            classes_file = Path(self.config.classes_file)
            if classes_file.exists():
                with classes_file.open("r") as f:
                    class_names = [
                        line.strip() for line in f.readlines() if line.strip()
                    ]
                success = self.vlm_model.set_classes(class_names)
                if not success:
                    rospy.logwarn(
                        f"[{rospy.get_name()}] Model does not support setting classes."
                    )
                else:
                    rospy.loginfo(
                        f"[{rospy.get_name()}] Loaded {len(class_names)} classes from {classes_file}."
                    )
            else:
                rospy.logwarn(
                    f"[{rospy.get_name()}] Classes file {classes_file} does not exist. Using default classes from the model."
                )

        self.srv = rospy.Service("set_prompt", SetPrompt, self._handle_set_prompt)
        self.detections_image_pub = rospy.Publisher(
            "detections_image", Image, queue_size=1
        )
        self.semantic_map_pub = rospy.Publisher(
            "semantic_map", PointCloud2, queue_size=1, latch=True
        )
        self.semantic_objects_pub = rospy.Publisher(
            "semantic_objects", MarkerArray, queue_size=1, latch=True
        )
        self.pcl_sub = rospy.Subscriber(
            "input_pointcloud", PointCloud2, self._pcl_callback, queue_size=1
        )
        self.camera_info_sub = rospy.Subscriber(
            "input_camera_info", CameraInfo, self._camera_info_callback, queue_size=1
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
        self.tf_buffer = tf2_ros.Buffer(rospy.Duration(30.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.geometry_points = np.empty((0, 3), dtype=np.float32)
        self.geometry_key_to_index: Dict[Tuple[int, int, int], int] = {}
        self.semantic_voxels: Dict[Tuple[int, int, int], SemanticVoxel] = {}
        self.semantic_map_revision = 0
        self.semantic_map_published_revision = -1
        self.semantic_objects_revision = 0
        self.semantic_objects_published_revision = -1
        self.latest_output_stamp = rospy.Time(0)

        self.label_names = list(self.vlm_model.names.values())
        if self.NO_SEMANTICS_LABEL not in self.label_names:
            self.label_names.append(self.NO_SEMANTICS_LABEL)
        self.label_to_index = {name: idx for idx, name in enumerate(self.label_names)}
        self.no_semantics_idx = self.label_to_index[self.NO_SEMANTICS_LABEL]
        self.class_colors = {
            name: self._color_for_label(name) for name in self.label_names
        }
        self.label_colors = np.array(
            [self.class_colors[name] for name in self.label_names], dtype=np.uint8
        )
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
        self.semantic_cloud_records = np.empty((0, 4), dtype=np.float32)
        self.semantic_cloud_label_indices = np.empty((0,), dtype=np.int32)
        self.semantic_cloud_confidences = np.empty((0,), dtype=np.float32)
        self.semantic_cloud_active = np.empty((0,), dtype=bool)
        self.semantic_cloud_key_to_index: Dict[Tuple[int, int, int], int] = {}
        self.label_voxel_keys: Dict[int, Set[Tuple[int, int, int]]] = {
            idx: set() for idx in range(len(self.label_names))
        }
        self.object_cluster_cache: Dict[int, List[SemanticObjectCluster]] = {}
        self.object_dirty_labels: Set[int] = set()

        self.semantic_map_timer = None
        if self.config.semantic_publish_period_s > 0.0:
            self.semantic_map_timer = rospy.Timer(
                rospy.Duration(self.config.semantic_publish_period_s),
                self._semantic_map_timer_callback,
            )
        self.semantic_objects_timer = None
        if (
            self.config.publish_semantic_objects
            and self.config.semantic_objects_period_s > 0.0
        ):
            self.semantic_objects_timer = rospy.Timer(
                rospy.Duration(self.config.semantic_objects_period_s),
                self._semantic_objects_timer_callback,
            )

        rospy.loginfo(f"[{rospy.get_name()}] Device: {self.vlm_model.model.device}")
        rospy.loginfo(f"[{rospy.get_name()}] finished initializing!")

    def _camera_info_callback(self, msg: CameraInfo) -> None:
        self.fx = msg.K[0]
        self.fy = msg.K[4]
        self.cx = msg.K[2]
        self.cy = msg.K[5]
        self.k1 = msg.D[0] if len(msg.D) > 0 else None
        self.k2 = msg.D[1] if len(msg.D) > 1 else None
        self.p1 = msg.D[2] if len(msg.D) > 2 else None
        self.p2 = msg.D[3] if len(msg.D) > 3 else None
        self.camera_info_sub.unregister()
        rospy.loginfo(
            f"[{rospy.get_name()}] Camera intrinsics set: fx={self.fx}, fy={self.fy}, cx={self.cx}, cy={self.cy}"
        )

    def _handle_set_prompt(self, req: SetPrompt) -> SetPromptResponse:
        self.prompt = req.prompt
        rospy.loginfo(f"[{rospy.get_name()}] Prompt updated to: {self.prompt}")
        return SetPromptResponse(success=True)

    def _pcl_callback(self, msg: PointCloud2) -> None:
        try:
            transform_2_body = self.tf_buffer.lookup_transform(
                self.config.body_frame,
                msg.header.frame_id,
                msg.header.stamp
                if not self.config.use_tf_current_time
                else rospy.Time(0),
                rospy.Duration(3.0),
            )
            transformed_body_cloud = tf2sm.do_transform_cloud(msg, transform_2_body)
            point_bodys = np.array(
                list(
                    pc2.read_points(
                        transformed_body_cloud,
                        field_names=("x", "y", "z"),
                        skip_nans=True,
                    )
                )
            )
            if point_bodys.size == 0:
                return
            dists = np.linalg.norm(point_bodys, axis=1)
            valid_indices = np.where(
                (dists >= self.config.min_point_r) & (dists <= self.config.max_point_r)
            )[0]
            if valid_indices.size == 0:
                return

            transform = self.tf_buffer.lookup_transform(
                self.config.target_frame,
                msg.header.frame_id,
                msg.header.stamp
                if not self.config.use_tf_current_time
                else rospy.Time(0),
                rospy.Duration(3.0),
            )
            transformed_cloud = tf2sm.do_transform_cloud(msg, transform)
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

            with self.map_lock:
                self._upsert_geometry_points_locked(points)
                self.latest_output_stamp = msg.header.stamp
        except (
            tf2_ros.LookupException,
            tf2_ros.ExtrapolationException,
            tf2_ros.ConnectivityException,
        ) as e:
            rospy.logwarn_throttle(2.0, f"[{rospy.get_name()}] TF lookup failed: {e}")
        except Exception as e:
            rospy.logerr(f"[{rospy.get_name()}] Error processing pointcloud: {e}")

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
        if self.config.verbose:
            rospy.loginfo(
                f"[{rospy.get_name()}] Detected {len(bboxes)} objects in {time.time() - start_time:.2f} seconds."
            )

        detection_image = image.copy()
        detection_image = cv2.cvtColor(detection_image, cv2.COLOR_RGB2BGR)
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

        active_center = self._lookup_active_window_center(header.stamp)
        with self.map_lock:
            self.latest_output_stamp = header.stamp
        active_points = self._snapshot_active_geometry_points(active_center)
        if len(active_points) == 0:
            return

        cloud_snapshot = o3d.geometry.PointCloud()
        cloud_snapshot.points = o3d.utility.Vector3dVector(active_points)

        if len(bboxes) > 0 and self.fx is not None:
            self._update_semantic_map(header, bboxes, cloud_snapshot, image.shape[:2])

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

    def _append_new_semantic_visual_voxels_locked(
        self,
        voxel_keys: List[Tuple[int, int, int]],
        voxels: List[SemanticVoxel],
    ) -> None:
        if not voxel_keys:
            return
        centers = np.array(
            [self._voxel_center(key) for key in voxel_keys], dtype=np.float32
        )
        label_indices = np.full(len(voxel_keys), self.no_semantics_idx, dtype=np.int32)
        confidences = np.zeros(len(voxel_keys), dtype=np.float32)
        active = np.zeros(len(voxel_keys), dtype=bool)
        rgb = np.empty(len(voxel_keys), dtype=np.float32)
        for idx, voxel in enumerate(voxels):
            is_active, label_idx, confidence = self._semantic_visual_state(voxel)
            target_label = label_idx if label_idx is not None else self.no_semantics_idx
            label_indices[idx] = target_label
            confidences[idx] = confidence
            active[idx] = is_active
            rgb[idx] = self.label_packed_colors[target_label]
            if is_active and target_label != self.no_semantics_idx:
                self.label_voxel_keys[target_label].add(voxel_keys[idx])
                self.object_dirty_labels.add(target_label)
        new_records = np.column_stack((centers, rgb)).astype(np.float32, copy=False)
        start_idx = len(self.semantic_cloud_records)
        if start_idx == 0:
            self.semantic_cloud_records = new_records
            self.semantic_cloud_label_indices = label_indices
            self.semantic_cloud_confidences = confidences
            self.semantic_cloud_active = active
        else:
            self.semantic_cloud_records = np.vstack(
                (self.semantic_cloud_records, new_records)
            )
            self.semantic_cloud_label_indices = np.concatenate(
                (self.semantic_cloud_label_indices, label_indices)
            )
            self.semantic_cloud_confidences = np.concatenate(
                (self.semantic_cloud_confidences, confidences)
            )
            self.semantic_cloud_active = np.concatenate(
                (self.semantic_cloud_active, active)
            )
        for offset, voxel_key in enumerate(voxel_keys):
            self.semantic_cloud_key_to_index[voxel_key] = start_idx + offset

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
        new_geometry_keys: List[Tuple[int, int, int]] = []
        new_geometry_points: List[np.ndarray] = []
        new_semantic_keys: List[Tuple[int, int, int]] = []
        new_semantic_voxels: List[SemanticVoxel] = []
        initial_scores = self._initial_scores()
        for voxel_key_arr, rep_point in zip(
            unique_voxels.tolist(), representative_points
        ):
            voxel_key = tuple(voxel_key_arr)
            geometry_idx = self.geometry_key_to_index.get(voxel_key)
            if geometry_idx is None:
                geometry_idx = len(self.geometry_points) + len(new_geometry_keys)
                self.geometry_key_to_index[voxel_key] = geometry_idx
                new_geometry_keys.append(voxel_key)
                new_geometry_points.append(rep_point)
            else:
                self.geometry_points[geometry_idx] = rep_point
            if voxel_key not in self.semantic_voxels:
                voxel = SemanticVoxel(scores=initial_scores.copy(), observations=1)
                self.semantic_voxels[voxel_key] = voxel
                new_semantic_keys.append(voxel_key)
                new_semantic_voxels.append(voxel)
        if new_geometry_points:
            append_points = np.asarray(new_geometry_points, dtype=np.float32)
            if len(self.geometry_points) == 0:
                self.geometry_points = append_points
            else:
                self.geometry_points = np.vstack((self.geometry_points, append_points))
        if new_semantic_voxels:
            self._append_new_semantic_visual_voxels_locked(
                new_semantic_keys, new_semantic_voxels
            )
            self.semantic_map_revision += 1

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
                active_mask = self._active_window_mask(points, active_center)
                points = points[active_mask]
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

    def _lookup_active_window_center(self, stamp: rospy.Time) -> np.ndarray:
        if not self.config.active_window_enabled:
            return None
        try:
            transform = self.tf_buffer.lookup_transform(
                self.config.target_frame,
                self.config.body_frame,
                stamp if not self.config.use_tf_current_time else rospy.Time(0),
                rospy.Duration(0.5),
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
            rospy.logwarn_throttle(
                2.0, f"[{rospy.get_name()}] Active-window TF lookup failed: {e}"
            )
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
        if stamp == rospy.Time(0):
            stamp = rospy.Time.now()
        return Header(frame_id=self.config.target_frame, stamp=stamp)

    def _semantic_map_timer_callback(self, _event) -> None:
        with self.map_lock:
            revision = self.semantic_map_revision
        if revision == self.semantic_map_published_revision:
            return
        self._publish_semantic_map(self._current_output_header())
        self.semantic_map_published_revision = revision

    def _semantic_objects_timer_callback(self, _event) -> None:
        with self.map_lock:
            revision = self.semantic_objects_revision
        if revision == self.semantic_objects_published_revision:
            return
        self._publish_semantic_objects(self._current_output_header())
        self.semantic_objects_published_revision = revision

    def _get_mask_points(
        self,
        points_camera: np.ndarray,
        proj_points_uv: np.ndarray,
        bbox: BoundingBox,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if bbox.mask is None:
            return np.empty((0, 3)), np.empty((0, 2), dtype=np.int32)
        return self.get_points_and_uv_in_mask(points_camera, proj_points_uv, bbox.mask)

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
        elif self.association_method == "dbscan":
            return self._select_dbscan_cluster(box_points_camera, rot, t)
        else:
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
                header.stamp if not self.config.use_tf_current_time else rospy.Time(0),
                rospy.Duration(3.0),
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
                    rospy.logwarn_throttle(
                        5.0,
                        f"[{rospy.get_name()}] Skipping unknown label {bbox.details!r}.",
                    )
                    continue
                box_points_camera, box_uv = self._get_mask_points(
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
                self.semantic_map_revision += 1
                self.semantic_objects_revision += 1
            return updated
        except Exception as e:
            rospy.logerr(f"[{rospy.get_name()}] Error updating semantic map: {e}")
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
                prev_state = None
                if voxel is None:
                    voxel = SemanticVoxel(scores=initial_scores.copy())
                    self.semantic_voxels[voxel_key] = voxel
                else:
                    prev_state = self._semantic_visual_state(voxel)
                voxel.observations += 1
                self._apply_semantic_update(voxel.scores, label_idx, confidence)
                self._upsert_semantic_visual_voxel_locked(voxel_key, voxel, prev_state)

    def _initial_scores(self) -> np.ndarray:
        scores = np.full(len(self.label_names), 1e-3, dtype=np.float64)
        scores[self.no_semantics_idx] = 1.0
        if self.semantic_update_method == "bayes":
            scores /= np.sum(scores)
            return scores
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

    def _upsert_semantic_visual_voxel_locked(
        self,
        voxel_key: Tuple[int, int, int],
        voxel: SemanticVoxel,
        prev_state: Optional[Tuple[bool, Optional[int], float]],
    ) -> None:
        prev_active = False
        prev_label_idx = None
        if prev_state is not None:
            prev_active, prev_label_idx, _ = prev_state
        active, label_idx, confidence = self._semantic_visual_state(voxel)
        record_idx = self.semantic_cloud_key_to_index.get(voxel_key)
        if record_idx is None:
            record_idx = len(self.semantic_cloud_records)
            center = self._voxel_center(voxel_key).astype(np.float32)
            rgb = self.label_packed_colors[
                label_idx if label_idx is not None else self.no_semantics_idx
            ]
            new_record = np.array(
                [[center[0], center[1], center[2], rgb]], dtype=np.float32
            )
            self.semantic_cloud_records = np.vstack(
                (self.semantic_cloud_records, new_record)
            )
            self.semantic_cloud_label_indices = np.append(
                self.semantic_cloud_label_indices,
                label_idx if label_idx is not None else self.no_semantics_idx,
            )
            self.semantic_cloud_confidences = np.append(
                self.semantic_cloud_confidences, confidence
            ).astype(np.float32, copy=False)
            self.semantic_cloud_active = np.append(self.semantic_cloud_active, active)
            self.semantic_cloud_key_to_index[voxel_key] = record_idx
        else:
            if prev_active and prev_label_idx is not None:
                self.label_voxel_keys[prev_label_idx].discard(voxel_key)
            self.semantic_cloud_label_indices[record_idx] = (
                label_idx if label_idx is not None else self.no_semantics_idx
            )
            self.semantic_cloud_confidences[record_idx] = confidence
            self.semantic_cloud_active[record_idx] = active
            target_label = label_idx if label_idx is not None else self.no_semantics_idx
            self.semantic_cloud_records[record_idx, 3] = self.label_packed_colors[
                target_label
            ]

        if active and label_idx is not None:
            self.label_voxel_keys[label_idx].add(voxel_key)
        if prev_label_idx is not None:
            self.object_dirty_labels.add(prev_label_idx)
        if label_idx is not None:
            self.object_dirty_labels.add(label_idx)
        self.semantic_objects_revision += 1

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
        with self.map_lock:
            active_mask = self.semantic_cloud_active.copy()
            active_records = self.semantic_cloud_records[active_mask].copy()
            active_confidences = self.semantic_cloud_confidences[active_mask].copy()
        if len(active_records) == 0:
            return None
        if len(active_records) != len(active_confidences):
            rospy.logwarn_throttle(
                2.0,
                f"[{rospy.get_name()}] Semantic cloud snapshot mismatch: {len(active_records)} records vs {len(active_confidences)} confidences.",
            )
            size = min(len(active_records), len(active_confidences))
            active_records = active_records[:size]
            active_confidences = active_confidences[:size]
            if size == 0:
                return None
        publish_idx = self._downsample_visual_indices(
            active_records[:, :3],
            active_confidences,
            self.config.semantic_visualization_voxel_size,
            self.config.semantic_visualization_max_points,
        )
        if len(publish_idx) == 0:
            return None
        publish_records = active_records[publish_idx]
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

    def _rebuild_dirty_object_clusters(self) -> None:
        with self.map_lock:
            dirty_labels_all = sorted(self.object_dirty_labels)
            max_dirty = max(
                1, int(self.config.semantic_object_max_dirty_labels_per_cycle)
            )
            dirty_labels = dirty_labels_all[:max_dirty]
            self.object_dirty_labels = set(dirty_labels_all[max_dirty:])
            label_snapshots = {}
            for label_idx in dirty_labels:
                if label_idx == self.no_semantics_idx:
                    continue
                voxel_keys = list(self.label_voxel_keys.get(label_idx, set()))
                if not voxel_keys:
                    label_snapshots[label_idx] = (
                        np.empty((0, 3), dtype=np.float32),
                        np.empty((0,), dtype=np.float32),
                    )
                    continue
                indices = np.array(
                    [self.semantic_cloud_key_to_index[key] for key in voxel_keys],
                    dtype=np.int32,
                )
                active_mask = self.semantic_cloud_active[indices]
                indices = indices[active_mask]
                if len(indices) == 0:
                    label_snapshots[label_idx] = (
                        np.empty((0, 3), dtype=np.float32),
                        np.empty((0,), dtype=np.float32),
                    )
                    continue
                label_points = self.semantic_cloud_records[indices, :3].copy()
                label_confidences = self.semantic_cloud_confidences[indices].copy()
                label_snapshots[label_idx] = (
                    label_points,
                    label_confidences,
                )

        rebuilt = {}
        for label_idx, (label_points, label_confidences) in label_snapshots.items():
            clusters: List[SemanticObjectCluster] = []
            if len(label_points) >= self.config.semantic_object_min_voxels:
                cluster_indices = self._euclidean_cluster_indices(
                    label_points,
                    self.config.semantic_cluster_eps,
                    self.config.semantic_cluster_min_points,
                )
                for indices in cluster_indices:
                    cluster = label_points[indices]
                    if len(cluster) < self.config.semantic_object_min_voxels:
                        continue
                    mins = np.min(cluster, axis=0)
                    maxs = np.max(cluster, axis=0)
                    clusters.append(
                        SemanticObjectCluster(
                            center=((mins + maxs) * 0.5).astype(np.float64),
                            extent=(maxs - mins).astype(np.float64),
                            confidence=float(np.mean(label_confidences[indices])),
                        )
                    )
            rebuilt[label_idx] = clusters

        with self.map_lock:
            for label_idx, clusters in rebuilt.items():
                self.object_cluster_cache[label_idx] = clusters

    def _publish_semantic_map(self, header: Header) -> None:
        cloud_msg = self._build_semantic_cloud_message(header)
        if cloud_msg is None:
            return
        self.semantic_map_pub.publish(cloud_msg)

    def _publish_semantic_objects(self, header: Header) -> None:
        self._rebuild_dirty_object_clusters()
        with self.map_lock:
            object_cluster_cache = {
                label_idx: list(clusters)
                for label_idx, clusters in self.object_cluster_cache.items()
            }
        markers = MarkerArray()
        delete_marker = Marker()
        delete_marker.action = Marker.DELETEALL
        markers.markers.append(delete_marker)
        marker_id = 0

        for label_idx in sorted(object_cluster_cache.keys()):
            if label_idx == self.no_semantics_idx:
                continue
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
            box_marker.points.append(Point(*corners_world[start_idx]))
            box_marker.points.append(Point(*corners_world[end_idx]))
        box_marker.scale.x = 0.2
        box_marker.color.r = color_rgb[0] / 255.0
        box_marker.color.g = color_rgb[1] / 255.0
        box_marker.color.b = color_rgb[2] / 255.0
        box_marker.color.a = 1.0
        box_marker.lifetime = rospy.Duration(0)
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
            connector.points.append(Point(*corners_world[corner_idx]))
            connector.points.append(Point(*node_center))
        connector.scale.x = 0.2
        connector.color.r = color_rgb[0] / 255.0
        connector.color.g = color_rgb[1] / 255.0
        connector.color.b = color_rgb[2] / 255.0
        connector.color.a = 0.8
        connector.lifetime = rospy.Duration(0)
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
        sphere.lifetime = rospy.Duration(0)
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
        text.scale.z = 1.5
        text.color.r = 1.0
        text.color.g = 1.0
        text.color.b = 1.0
        text.color.a = 1.0
        text.text = f"{label}"
        if self.config.show_confidence:
            text.text += f" {confidence:.2f}"
        text.lifetime = rospy.Duration(0)
        markers.markers.append(text)
        return marker_id + 4

    def spin(self) -> None:
        rospy.spin()


def main():
    rospy.init_node("semantic_map_node")
    node = SemanticMapNode()
    node.spin()


if __name__ == "__main__":
    main()
