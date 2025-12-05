#!/usr/bin/env python3
"""Reasoning VLM ROS node."""

import time
from dataclasses import dataclass, field
from pathlib import Path

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


@dataclass
class ReasoningVLMNodeConfig(Config):
    """Configuration for Reasoning VLM Node."""

    vlm: any = config_field("reason_vlm_model", default="openai")
    prompt: str = ""
    worker: ImageWorkerConfig = field(default_factory=ImageWorkerConfig)
    verbose: bool = False
    overlay_alpha: float = 0.3
    footer_height: int = 80
    compressed_image: bool = False


class ReasoningVLMNode(Node):
    """ROS Node for Reasoning VLM."""

    def __init__(self) -> None:
        """Initialize Reasoning VLM Node."""
        super().__init__("reasoning_vlm_node")
        ros_config_params = (
            self.declare_parameter("config", "").get_parameter_value().string_value
        )

        # Load configuration
        config_path = (
            self.declare_parameter("config_path", "").get_parameter_value().string_value
        )
        config_path = Path(config_path).expanduser().absolute()
        if not config_path.exists() and config_path != "":
            self.get_logger().warn(f"config path '{config_path}' does not exist!")
            self.config = ReasoningVLMNodeConfig()
        else:
            self.config = Config.load(ReasoningVLMNodeConfig, config_path)
        self.config.update(yaml.safe_load(ros_config_params))
        self.prompt = self.config.prompt

        # Initialize VLM model
        self.vlm_model = self.config.vlm.create()
        self.get_logger().info(f"Initializing with {self.config.show()}")

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
            self.get_logger().info(f"Processing image at time {header.stamp.to_sec()}")

        start_time = time.time()
        reasoning_output: ReasoningOutput = self.vlm_model.reason(image, self.prompt)

        # Generate color based on probability (red→green)
        prob = np.clip(reasoning_output.probability, 0.0, 1.0)
        color = (0, int(255 * prob), int(255 * (1 - prob)))  # BGR (red→green)

        output_image = cv2.cvtColor(image.copy(), cv2.COLOR_RGB2BGR)
        overlay = np.full_like(output_image, color, dtype=np.uint8)
        heatmap = cv2.addWeighted(
            output_image,
            1 - self.config.overlay_alpha,
            overlay,
            self.config.overlay_alpha,
            0,
        )

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
                f"Published result with prob={prob:.2f} in {time.time() - start_time:.2f}s"
            )


def main(args=None) -> None:
    """Main function to run the Detection VLM ROS node."""
    rclpy.init(args=args)
    node = ReasoningVLMNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
