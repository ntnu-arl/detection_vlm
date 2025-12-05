"""Set of VLM models for object detection in images."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import numpy as np
import torch
from spark_config import Config, register_config

from detection_vlm_python import BoundingBox, ReasoningOutput
from detection_vlm_python.openai_client import OpenAIClient, OpenAIClientConfig

_ = torch.cuda.is_available()


class OpenAIVLM:
    """Base class for OpenAI Vision-Language Models."""

    def __init__(self, config) -> None:
        """Initialize OpenAI VLM model with given configuration."""
        self.config = config
        system_prompt_path = Path(config.system_prompt_path)
        if system_prompt_path.exists() and system_prompt_path.is_file():
            with open(system_prompt_path, "r") as f:
                self.config.client_config.system_prompt = f.read()
        self.client = OpenAIClient(self.config.client_config)


class OpenAIDetectionVLM(OpenAIVLM):
    """OpenAI Vision-Language Model for object detection in images."""

    def __init__(self, config) -> None:
        """Initialize OpenAI VLM model with given configuration."""
        super().__init__(config)

    @classmethod
    def construct(cls, **kwargs):
        """Load model from configuration dictionary."""
        config = OpenAIDetectionVLMConfig()
        config.update(kwargs)
        return cls(config)

    def detect(self, image: np.ndarray, prompt: str) -> List[BoundingBox]:
        """Detect objects in the image based on the given prompt.

        :param image: Input image as a NumPy array.
        :param prompt: Text prompt specifying what to detect.
        :return: List of BoundingBox instances representing detected objects.
        """
        result, success = self.client.inference(image, prompt)
        if not success:
            return []

        bboxes = [BoundingBox(**box) for box in result]
        return bboxes


class OpenAIReasonVLM(OpenAIVLM):
    """OpenAI Vision-Language Model for reasoning about images."""

    def __init__(self, config) -> None:
        """Initialize OpenAI VLM model with given configuration."""
        super().__init__(config)

    @classmethod
    def construct(cls, **kwargs):
        """Load model from configuration dictionary."""
        config = OpenAIReasonVLMConfig()
        config.update(kwargs)
        return cls(config)

    def reason(self, image: np.ndarray, prompt: str) -> ReasoningOutput:
        """Generate reasoning about the image based on the given prompt.

        :param image: Input image as a NumPy array.
        :param prompt: Text prompt specifying the reasoning task.
        :return: ReasoningOutput instance containing the reasoning results.
        """
        result, success = self.client.inference(image, prompt)
        if not success:
            return ReasoningOutput(
                select=False, probability=0.0, explanation="Reasoning failed."
            )

        return ReasoningOutput(**result)


class YOLOEDetection:
    """YOLOE Detection model."""

    def __init__(self, config) -> None:
        """Initialize YOLOE Detection model with given configuration."""
        from ultralytics import YOLOE

        self.config = config
        self.model = YOLOE(self.config.model)
        if self.config.cuda:
            self.model.to("cuda")

    @classmethod
    def construct(cls, **kwargs):
        """Load model from configuration dictionary."""
        config = YOLOEDetectionConfig()
        config.update(kwargs)
        return cls(config)

    def detect(self, image: np.ndarray, prompt: str) -> List[BoundingBox]:
        """Detect objects in the image based on the given prompt.

        :param image: Input image as a NumPy array.
        :param prompt: Text prompt specifying what to detect (not used in YOLOE).
        :return: List of BoundingBox instances representing detected objects.
        """
        results = self.model.predict(
            image,
            device="cuda" if self.config.cuda else "cpu",
            conf=self.config.confidence_threshold,
            imgsz=self.config.output_size,
            verbose=self.config.verbose,
        )

        bboxes = []
        for box, label in zip(results[0].boxes.xyxy, results[0].boxes.cls):
            x_min, y_min, x_max, y_max = box.cpu().numpy()
            class_id = int(label.cpu().numpy())
            bboxes.append(
                BoundingBox(
                    x0=int(x_min),
                    y0=int(y_min),
                    x1=int(x_max),
                    y1=int(y_max),
                    details=str(self.model.names[class_id]),
                )
            )
        return bboxes


@register_config("detection_vlm_model", name="openai", constructor=OpenAIDetectionVLM)
@dataclass
class OpenAIDetectionVLMConfig(Config):
    """Configuration for OpenAI VLM model."""

    client_config: OpenAIClientConfig = field(default_factory=OpenAIClientConfig)
    system_prompt_path: str = ""

    @classmethod
    def load(cls, filepath):
        """Load config from file."""
        return Config.load(cls, filepath)


@register_config("reason_vlm_model", name="openai", constructor=OpenAIReasonVLM)
@dataclass
class OpenAIReasonVLMConfig(Config):
    """Configuration for OpenAI VLM reasoning model."""

    client_config: OpenAIClientConfig = field(default_factory=OpenAIClientConfig)
    system_prompt_path: str = ""

    @classmethod
    def load(cls, filepath):
        """Load config from file."""
        return Config.load(cls, filepath)


@register_config("detection_vlm_model", name="yoloe", constructor=YOLOEDetection)
@dataclass
class YOLOEDetectionConfig(Config):
    """YOLOE Detection model configuration."""

    model: str = "yoloe-11l-seg-pf.pt"
    confidence_threshold: float = 0.25
    output_size: int = 736
    verbose: bool = False
    cuda: bool = True

    @classmethod
    def load(cls, filepath):
        """Load config from file."""
        return Config.load(cls, filepath)
