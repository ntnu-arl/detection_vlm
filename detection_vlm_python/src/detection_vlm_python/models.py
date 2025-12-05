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
"""Set of VLM models for object detection in images."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import numpy as np
import ultralytics.utils.ops

from detection_vlm_python import BoundingBox, ReasoningOutput
from detection_vlm_python.config import Config, register_config
from detection_vlm_python.openai_client import OpenAIClient, OpenAIClientConfig


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

    def detect(
        self, image: np.ndarray, prompt: str, confidence_threshold: float = None
    ) -> List[BoundingBox]:
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
        self.names = self.model.names

    @classmethod
    def construct(cls, **kwargs):
        """Load model from configuration dictionary."""
        config = YOLOEDetectionConfig()
        config.update(kwargs)
        return cls(config)

    def set_classes(self, class_names: List[str]) -> bool:
        if "pf" in self.config.model:
            return False  # PF models do not support setting classes
        self.model.set_classes(class_names, self.model.get_text_pe(class_names))
        self.names = {i: name for i, name in enumerate(class_names)}
        return True

    def detect(
        self, image: np.ndarray, prompt: str, confidence_threshold: float = None
    ) -> List[BoundingBox]:
        """Detect objects in the image based on the given prompt.

        :param image: Input image as a NumPy array.
        :param prompt: Text prompt specifying what to detect (not used in YOLOE).
        :return: List of BoundingBox instances representing detected objects.
        """
        results = self.model.predict(
            image,
            device="cuda" if self.config.cuda else "cpu",
            conf=confidence_threshold,
            imgsz=self.config.output_size,
            verbose=self.config.verbose,
        )

        bboxes = []
        if len(results[0].boxes.xyxy) == 0:
            return bboxes
        masks = (
            ultralytics.utils.ops.scale_image(
                results[0].masks.data.permute(1, 2, 0).cpu().numpy(), image.shape
            ).transpose(2, 0, 1)
            > 0.5
        )
        for box, label, conf, mask in zip(
            results[0].boxes.xyxy, results[0].boxes.cls, results[0].boxes.conf, masks
        ):
            x_min, y_min, x_max, y_max = box.cpu().numpy()
            class_id = int(label.cpu().numpy())
            bboxes.append(
                BoundingBox(
                    x0=int(x_min),
                    y0=int(y_min),
                    x1=int(x_max),
                    y1=int(y_max),
                    details=str(self.model.names[class_id]),
                    confidence=float(conf.cpu().numpy()),
                    mask=mask,
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
