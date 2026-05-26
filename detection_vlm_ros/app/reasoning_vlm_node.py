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
"""Reasoning VLM ROS node."""

import csv
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import rclpy
import yaml
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage, Image
from spark_config.config import Config, config_field
from std_msgs.msg import Header

import detection_vlm_python.models as models
import detection_vlm_ros
from detection_vlm_msgs.srv import SetPrompt
from detection_vlm_python import ReasoningOutput
from detection_vlm_ros import ImageWorker, ImageWorkerConfig
from detection_vlm_ros.ros_conversions import Conversions


def to_sec(stamp):
    return stamp.sec + stamp.nanosec * 1e-9


@dataclass
class ReasoningVLMNodeConfig(Config):
    """Configuration for Reasoning VLM Node."""

    vlm: any = config_field("reason_vlm_model", default="openai")
    prompt: str = ""
    worker: ImageWorkerConfig = field(default_factory=ImageWorkerConfig)
    verbose: bool = False
    overlay_alpha: float = 0.3
    footer_height: int = 80
    overlay: bool = True
    compressed_image: bool = False
    runtime_logging_enabled: bool = False
    runtime_log_file: str = "/tmp/reasoning_vlm_runtimes.csv"


class ReasoningVLMNode(Node):
    """ROS Node for Reasoning VLM."""

    def __init__(self) -> None:
        """Initialize Reasoning VLM Node."""
        super().__init__("reasoning_vlm_node")
        ros_config_params = (
            self.declare_parameter("config", "").get_parameter_value().string_value
        )

        # Load configuration
        config_path_param = (
            self.declare_parameter("config_path", "").get_parameter_value().string_value
        )
        config_path = Path(config_path_param).expanduser().absolute()
        if not config_path_param:
            self.config = ReasoningVLMNodeConfig()
        elif not config_path.exists():
            self.get_logger().warn(f"config path '{config_path}' does not exist!")
            self.config = ReasoningVLMNodeConfig()
        else:
            self.config = Config.load(ReasoningVLMNodeConfig, config_path)
        overrides = yaml.safe_load(ros_config_params) if ros_config_params else {}
        if overrides:
            self.config.update(overrides)
        self.prompt = self.config.prompt

        # Initialize VLM model
        self.vlm_model = self.config.vlm.create()
        self.get_logger().info(f"Initializing with {self.config.show()}")

        self.runtime_log_lock = threading.Lock()
        self.runtime_log_file = None
        self.runtime_log_writer = None
        self._init_runtime_logger()

        # Initialize ROS components
        self.worker = ImageWorker(
            self,
            self.config.worker,
            "input_image",
            CompressedImage if self.config.compressed_image else Image,
            self._spin_once,
        )
        self.image_pub = self.create_publisher(Image, "reasoning_image", 1)
        self.srv = self.create_service(SetPrompt, "set_prompt", self._handle_set_prompt)
        self.get_logger().info("Reasoning VLM finished initializing!")

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
                    "select",
                    "probability",
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
        select: str = "",
        probability: Optional[float] = None,
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
            "select": select,
            "probability": "" if probability is None else f"{probability:.6f}",
        }
        with self.runtime_log_lock:
            if self.runtime_log_writer is None:
                return
            try:
                self.runtime_log_writer.writerow(row)
            except Exception as e:
                self.get_logger().warn(f"Failed to write runtime log row: {e}")

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
        """Process incoming image message."""
        if self.config.verbose:
            self.get_logger().info(f"Processing image at time {to_sec(header.stamp)}")

        start_time = time.time()
        reasoning_output: ReasoningOutput = self.vlm_model.reason(image, self.prompt)
        vlm_runtime_ms = (time.time() - start_time) * 1000.0
        self._log_runtime(
            "reasoning_vlm",
            vlm_runtime_ms,
            header,
            select=str(reasoning_output.select),
            probability=reasoning_output.probability,
        )

        # Generate color based on probability (red→green)
        prob = np.clip(reasoning_output.probability, 0.0, 1.0)
        color = (0, int(255 * prob), int(255 * (1 - prob)))  # BGR (red→green)

        output_image = cv2.cvtColor(image.copy(), cv2.COLOR_RGB2BGR)
        overlay = np.full_like(output_image, color, dtype=np.uint8)
        if self.config.overlay:
            heatmap = cv2.addWeighted(
                output_image,
                1 - self.config.overlay_alpha,
                overlay,
                self.config.overlay_alpha,
                0,
            )
        else:
            heatmap = output_image

        # Prepare text
        prob_text = f"Probability: {prob:.2f}"
        expl_text = reasoning_output.explanation or "No explanation provided"

        # Font settings
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.6
        thickness = 1
        line_spacing = 8  # extra space between lines
        max_text_width = output_image.shape[1] - 20

        # --- Dynamic text wrapping ---
        words = expl_text.split(" ")
        lines, current_line = [], ""
        for word in words:
            test_line = current_line + word + " "
            (w, _) = cv2.getTextSize(test_line, font, scale, thickness)[0]
            if w < max_text_width:
                current_line = test_line
            else:
                lines.append(current_line.strip())
                current_line = word + " "
        if current_line:
            lines.append(current_line.strip())

        # --- Dynamically size footer ---
        line_height = (
            cv2.getTextSize("Test", font, scale, thickness)[0][1] + line_spacing
        )
        footer_height = int(
            30 + len(lines) * line_height + 20
        )  # base padding + lines + margin
        footer = (
            np.ones((footer_height, output_image.shape[1], 3), dtype=np.uint8) * 255
        )
        combined = np.vstack((heatmap, footer))

        # --- Draw text ---
        y0 = output_image.shape[0] + 25
        cv2.putText(combined, prob_text, (10, y0), font, 0.7, (0, 0, 0), 2, cv2.LINE_AA)

        y = y0 + 30
        for line in lines:
            cv2.putText(
                combined, line, (10, y), font, scale, (0, 0, 0), thickness, cv2.LINE_AA
            )
            y += line_height

        # Publish resulting image
        output_msg = Conversions.to_sensor_image(combined)
        self.image_pub.publish(output_msg)

        if self.config.verbose:
            self.get_logger().info(
                f"Published result with prob={prob:.2f}; VLM runtime={vlm_runtime_ms / 1000.0:.2f}s"
            )

    def destroy_node(self) -> bool:
        self._close_runtime_logger()
        return super().destroy_node()


def main(args=None) -> None:
    """Main function to run the Detection VLM ROS node."""
    rclpy.init(args=args)
    node = ReasoningVLMNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
