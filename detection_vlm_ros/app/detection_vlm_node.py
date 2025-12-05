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

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List

import cv2
import numpy as np
import open3d as o3d
import rospy
import sensor_msgs.point_cloud2 as pc2
import tf2_ros
import tf2_sensor_msgs.tf2_sensor_msgs as tf2sm
from dynamic_reconfigure.server import Server
from geometry_msgs.msg import Point, Pose
from scipy.spatial.transform import Rotation as R
from sensor_msgs.msg import CameraInfo, CompressedImage, Image, PointCloud2
from std_msgs.msg import Header
from vision_msgs.msg import BoundingBox3D, BoundingBox3DArray
from visualization_msgs.msg import Marker, MarkerArray

import detection_vlm_python.models as models
import detection_vlm_ros
from detection_vlm_msgs.srv import SetPrompt, SetPromptResponse
from detection_vlm_python import BoundingBox
from detection_vlm_python.config import Config, config_field
from detection_vlm_ros import ImageWorker, ImageWorkerConfig
from detection_vlm_ros.cfg import DetectionVLMConfig
from detection_vlm_ros.ros_conversions import Conversions


@dataclass
class CameraIntrinsics(Config):
    fx: float = None
    fy: float = None
    cx: float = None
    cy: float = None


@dataclass
class DetectionVLMNodeConfig(Config):
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
    voxel_size: float = 0.05
    min_points_per_cluster: int = 30
    eps_dbscan: float = 0.2
    compressed_image: bool = False
    use_masks_for_projection: bool = True
    classes_file: str = ""
    camera_intrinsics: CameraIntrinsics = field(default_factory=CameraIntrinsics)


class DetectionVLMNode:
    """ROS node for detection VLM."""

    def __init__(self) -> None:
        """Initialize the Detection VLM ROS node."""
        self.config = detection_vlm_ros.load_from_ros(DetectionVLMNodeConfig, ns="~")
        self.vlm_model = self.config.vlm.create()
        rospy.loginfo(f"[{rospy.get_name()}] Initializing with {self.config.show()}")
        self.worker = ImageWorker(
            self.config.worker,
            "input_image",
            CompressedImage if self.config.compressed_image else Image,
            self._spin_once,
        )
        self.prompt = self.config.prompt
        self._initialized = False
        self.confidence_threshold = self.vlm_model.config.confidence_threshold
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
                elif self.config.verbose:
                    rospy.loginfo(
                        f"[{rospy.get_name()}] Loaded {len(class_names)} classes from {classes_file}."
                    )

        self.srv = rospy.Service("set_prompt", SetPrompt, self._handle_set_prompt)
        self.detections_image_pub = rospy.Publisher(
            "detections_image", Image, queue_size=1
        )
        self.pcl_sub = rospy.Subscriber(
            "input_pointcloud", PointCloud2, self._pcl_callback, queue_size=1
        )
        self.pcl_pub = rospy.Publisher(
            "accumulated_pointcloud", PointCloud2, queue_size=1
        )
        self.bbox3d_pub = rospy.Publisher(
            "detected_bboxes_3d", BoundingBox3DArray, queue_size=1
        )
        self.vis_3dboxes_pub = rospy.Publisher(
            "visualization_3d_bboxes", MarkerArray, queue_size=1
        )
        self.camera_info_sub = rospy.Subscriber(
            "input_camera_info", CameraInfo, self._camera_info_callback, queue_size=1
        )

        self.fx = self.config.camera_intrinsics.fx
        self.fy = self.config.camera_intrinsics.fy
        self.cx = self.config.camera_intrinsics.cx
        self.cy = self.config.camera_intrinsics.cy

        self.tf_buffer = tf2_ros.Buffer(rospy.Duration(30.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.accumulated_cloud = o3d.geometry.PointCloud()

        # Generate N different random colors for visualization, N is number of classes
        names = list(self.vlm_model.names.values())
        colors = np.random.randint(0, 255, size=(len(names), 3))
        # Generate a dict mapping class names to colors
        self.class_colors = {
            names[i]: tuple(int(c) for c in colors[i]) for i in range(len(names))
        }

        self.dynamic_reconf_server = Server(
            DetectionVLMConfig, self._dynamic_reconf_callback
        )

        rospy.loginfo(f"[{rospy.get_name()}] Device: {self.vlm_model.model.device}")
        rospy.loginfo(f"[{rospy.get_name()}] finished initializing!")

    def _camera_info_callback(self, msg: CameraInfo) -> None:
        """Callback to process incoming camera info messages.
        :param msg: Incoming CameraInfo message.
        """
        self.fx = msg.K[0]
        self.fy = msg.K[4]
        self.cx = msg.K[2]
        self.cy = msg.K[5]
        # Unsubscribe after receiving the first message
        self.camera_info_sub.unregister()
        rospy.loginfo(
            f"[{rospy.get_name()}] Camera intrinsics set: fx={self.fx}, fy={self.fy}, cx={self.cx}, cy={self.cy}"
        )

    def _dynamic_reconf_callback(self, config, level):
        if not self._initialized:
            self._initialized = True
            return config
        self.confidence_threshold = config.confidence_threshold
        rospy.loginfo(
            f"[{rospy.get_name()}] Updated confidence threshold: {self.confidence_threshold}"
        )
        return config

    def _pcl_callback(self, msg: PointCloud2) -> None:
        """Callback to process incoming point cloud messages.
        :param msg: Incoming PointCloud2 message.
        """
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
            # Get indices of points within min/max radius
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

            # Convert to numpy array
            points = np.array(
                list(
                    pc2.read_points(
                        transformed_cloud, field_names=("x", "y", "z"), skip_nans=True
                    )
                )
            )
            # Keep only valid points
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
            rospy.logwarn_throttle(
                2.0, f"[PersistentVoxelAccumulator] TF lookup failed: {e}"
            )
        except Exception as e:
            rospy.logerr(
                f"[PersistentVoxelAccumulator] Error processing pointcloud: {e}"
            )

        if len(self.accumulated_cloud.points) > 0:
            if self.config.verbose:
                rospy.loginfo(
                    f"[{rospy.get_name()}] Accumulated cloud has {len(self.accumulated_cloud.points)} points."
                )
            # Publish accumulated cloud
            pcl_msg = pc2.create_cloud_xyz32(
                Header(frame_id=self.config.target_frame, stamp=msg.header.stamp),
                np.asarray(self.accumulated_cloud.points),
            )
            self.pcl_pub.publish(pcl_msg)
            rospy.loginfo(f"[{rospy.get_name()}] Published accumulated point cloud.")

    def _handle_set_prompt(self, req: SetPrompt) -> SetPromptResponse:
        """Handle set prompt service call.
        :param req: Service request containing the new prompt.
        :return: Service response indicating success.
        """
        self.prompt = req.prompt
        rospy.loginfo(f"[{rospy.get_name()}] Prompt updated to: {self.prompt}")
        return SetPromptResponse(success=True)

    def _spin_once(self, header: Header, image: np.ndarray) -> None:
        """Process a single image.
        :param header: ROS message header.
        :param image: Input image as a NumPy array.
        """
        if self.config.verbose:
            rospy.loginfo(
                f"[{rospy.get_name()}] Processing image at time {header.stamp.to_sec()}"
            )
        start_time = time.time()
        bboxes: List[BoundingBox] = self.vlm_model.detect(
            image, self.prompt, confidence_threshold=self.confidence_threshold
        )
        if len(bboxes) == 0:
            if self.config.verbose:
                rospy.loginfo(f"[{rospy.get_name()}] No objects detected.")
            return
        if self.config.verbose:
            rospy.loginfo(
                f"[{rospy.get_name()}] Detected {len(bboxes)} objects in {time.time() - start_time:.2f} seconds."
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
        """
        :param points_3d: Nx3 numpy array
        :param proj_points_uv: Nx2 numpy array (pixel [u, v] coords)
        :param mask: HxW numpy array (True/False or 1/0)
        :return: Mx3 numpy array of 3D points inside the mask
        """

        H, W = mask.shape

        # round pixel coords and convert to int
        uv = np.round(proj_points_uv).astype(int)

        # keep points inside image bounds
        valid = (
            (uv[:, 0] >= 0)
            & (uv[:, 0] < W)
            & (uv[:, 1] >= 0)
            & (uv[:, 1] < H)
            & (points_3d[:, 2] > 0)
        )

        uv_valid = uv[valid]
        pts_valid = points_3d[valid]

        # check mask for those valid uv indices
        mask_values = mask[uv_valid[:, 1], uv_valid[:, 0]]  # mask[y,x]

        inside_mask = mask_values.astype(bool)

        return pts_valid[inside_mask]

    def process_3d_bboxes(
        self,
        header: Header,
        bboxes: List[BoundingBox],
        cloud_map: o3d.geometry.PointCloud,
    ) -> None:
        """
        For each 2D detection, extract 3D points inside the bounding box,
        cluster them with Open3D DBSCAN, pick the closest cluster,
        and compute a 3D bounding box.
        :param header: ROS message header.
        :param bboxes: List of 2D bounding boxes detected in the image.
        :param cloud_map: Point cloud in the target frame.
        """
        try:
            # --- Convert PointCloud2 to numpy array ---
            points = np.asarray(cloud_map.points)
            if points.size == 0:
                return

            # Get transform from world -> camera
            transform = self.tf_buffer.lookup_transform(
                self.config.camera_frame,
                self.config.target_frame,
                header.stamp if not self.config.use_tf_current_time else rospy.Time(0),
                rospy.Duration(3.0),
            )

            # Extract pose
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

            # Convert world points → camera points
            points = (rot @ points.T + t).T

            bbs3d = BoundingBox3DArray()
            bbs3d.header = header
            all_labels = []
            all_confidences = []
            for i, bbox in enumerate(bboxes):
                # Project 3D points → 2D image plane
                u = (points[:, 0] * self.fx / points[:, 2]) + self.cx
                v = (points[:, 1] * self.fy / points[:, 2]) + self.cy
                if self.config.use_masks_for_projection and bbox.mask is not None:
                    box_points = self.get_points_in_mask(
                        points, np.stack((u, v), axis=-1), bbox.mask
                    )
                    if len(box_points) < self.config.min_points_per_cluster:
                        continue
                else:
                    mask = (
                        (u >= bbox.x0)
                        & (u <= bbox.x1)
                        & (v >= bbox.y0)
                        & (v <= bbox.y1)
                        & (points[:, 2] > 0)
                    )
                    box_points = points[mask]
                if len(box_points) < self.config.min_points_per_cluster:
                    continue

                # --- Convert to Open3D cloud ---
                cloud = o3d.geometry.PointCloud()
                cloud.points = o3d.utility.Vector3dVector(box_points)

                # --- Cluster with Open3D DBSCAN ---
                labels = np.array(
                    cloud.cluster_dbscan(
                        eps=self.config.eps_dbscan,
                        min_points=self.config.min_points_per_cluster,
                        print_progress=False,
                    )
                )
                valid_labels = labels[labels >= 0]
                if len(valid_labels) == 0:
                    continue

                clusters = [
                    box_points[labels == label]
                    for label in np.unique(labels)
                    if label != -1
                ]
                if not clusters:
                    continue

                # --- Pick closest cluster (smallest mean Z) ---
                closest_cluster = min(clusters, key=lambda c: np.mean(c[:, 2]))

                # --- Project to world frame ---
                closest_cluster = (rot.T @ (closest_cluster - t.T).T).T

                # --- Compute axis-aligned bounding box ---
                cluster_cloud = o3d.geometry.PointCloud()
                cluster_cloud.points = o3d.utility.Vector3dVector(closest_cluster)
                aabb = cluster_cloud.get_axis_aligned_bounding_box()

                # --- Get center ---
                center = aabb.get_center()
                extent = aabb.get_extent()  # [x, y, z] size in meters

                bbox3d = BoundingBox3D()
                bbox3d.center.position.x = center[0]
                bbox3d.center.position.y = center[1]
                bbox3d.center.position.z = center[2]
                bbox3d.center.orientation.w = 1.0  # No rotation
                bbox3d.size.x = extent[0]
                bbox3d.size.y = extent[1]
                bbox3d.size.z = extent[2]
                bbs3d.boxes.append(bbox3d)
                all_labels.append(bbox.details)
                if bbox.confidence is not None:
                    all_confidences.append(bbox.confidence)

            if len(bbs3d.boxes) > 0:
                self.bbox3d_pub.publish(bbs3d)
                self._pub_visualization_3dboxes(bbs3d, all_labels, all_confidences)

        except Exception as e:
            rospy.logerr(f"[{rospy.get_name()}] Error in process_3d_bboxes: {e}")

    def _pub_visualization_3dboxes(
        self, msg: BoundingBox3DArray, labels: List[str], confidences: List[float]
    ) -> None:
        markers = MarkerArray()
        for i, box in enumerate(msg.boxes):
            # --- Wireframe marker ---
            m = Marker()
            m.header = msg.header
            m.header.frame_id = self.config.target_frame
            m.ns = "bbox3d_wireframe"
            m.id = i
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
                p1 = Point(*corners[start])
                p2 = Point(*corners[end])
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

            m.lifetime = rospy.Duration(self.config.worker.min_separation_s)
            markers.markers.append(m)

            # --- Label marker ---
            label = Marker()
            label.header = msg.header
            label.header.frame_id = self.config.target_frame
            label.ns = "bbox3d_labels"
            label.id = 10000 + i
            label.type = Marker.TEXT_VIEW_FACING
            label.action = Marker.ADD

            # Position label slightly above the box
            label.pose = Pose()
            label.pose.position.x = box.center.position.x
            label.pose.position.y = box.center.position.y
            label.pose.position.z = box.center.position.z + (box.size.z / 2.0) + 0.2
            label.pose.orientation.w = 1.0

            label.scale.z = 0.5
            label.color.r = 1  # random_color[0]
            label.color.g = 1  # random_color[1]
            label.color.b = 1  # random_color[2]
            label.color.a = 1.0

            # Set label text (adjust depending on your message)
            label.text = labels[i] if i < len(labels) else "Object"
            if i < len(confidences):
                label.text += f" {confidences[i]:.2f}"

            label.lifetime = rospy.Duration(self.config.worker.min_separation_s)
            markers.markers.append(label)

        self.vis_3dboxes_pub.publish(markers)

    def spin(self) -> None:
        """Spin the ROS node."""
        rospy.spin()


def main():
    """Main function to start the Detection VLM ROS node."""
    rospy.init_node("detection_vlm_node")
    node = DetectionVLMNode()
    node.spin()


if __name__ == "__main__":
    main()
