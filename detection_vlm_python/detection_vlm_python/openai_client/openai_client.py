import base64
import json
import os
import re
from io import BytesIO
from typing import Dict, Tuple, Union

import cv2
import numpy as np
from openai import OpenAI
from PIL import Image

from detection_vlm_python.openai_client.config import OpenAIClientConfig


class OpenAIClient:
    """Client for OpenAI API to generate navigation or vision-based prompts."""

    def __init__(self, config: OpenAIClientConfig) -> None:
        """Initialize OpenAI client with model, max tokens, and system prompt."""
        self.config = config
        self.api_key = os.getenv("OPENAI_API_KEY", "")
        self.client = OpenAI(api_key=self.api_key)

    def _encode_image(self, image: Union[str, np.ndarray]) -> str:
        """
        Convert an image (path or np.ndarray) to base64 string for API input.
        :param image: Either a file path (str) or an image array (np.ndarray, RGB or BGR).
        :return: Base64-encoded string representation of the image (JPEG format).
        """
        if isinstance(image, str):
            # If a file path is passed
            with open(image, "rb") as f:
                return base64.b64encode(f.read()).decode("utf-8")

        elif isinstance(image, np.ndarray):
            # Handle both OpenCV BGR and normal RGB images
            if image.ndim == 3 and image.shape[2] == 3:
                # Convert BGR (OpenCV default) → RGB for correct colors
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                image_rgb = image

            # Encode to JPEG in memory
            pil_img = Image.fromarray(image_rgb.astype(np.uint8))
            buffer = BytesIO()
            pil_img.save(buffer, format="JPEG")
            encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
            return encoded

        else:
            raise TypeError("Input must be a file path or a NumPy image array.")

    def _parse_json_from_model_output(self, raw_text: str):
        """
        Extracts and parses JSON content from a model response that may contain Markdown code fences.
        Works even if the model wraps the JSON in ```json ... ``` or ``` ... ```.
        """
        if not raw_text:
            return None

        # Remove Markdown code fences (```json ... ``` or ``` ... ```)
        cleaned = re.sub(
            r"^```(?:json)?\s*|\s*```$", "", raw_text.strip(), flags=re.DOTALL
        )

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as e:
            # Sometimes the model returns multiple objects or text + json — try to extract the first JSON block
            match = re.search(r"\{.*\}|\[.*\]", raw_text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(0))
                except json.JSONDecodeError:
                    pass
            raise ValueError(
                f"Failed to parse JSON from model output: {e}\nRaw text:\n{raw_text}"
            )

    def inference(
        self,
        image: Union[str, np.ndarray],
        prompt: str = "Detect the described object in the image and return bounding boxes.",
        log: bool = False,
    ) -> Tuple[Dict, bool]:
        """
        Use OpenAI’s VLM.
        :param image: Image input as file path or np.ndarray.
        :param prompt: Text prompt.
        :param log: Whether to print debug logs.
        :return: A tuple (result_dict, success_flag)
        """
        try:
            base64_image = self._encode_image(image)

            messages = [
                {"role": "system", "content": self.config.system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": prompt},
                        {
                            "type": "input_image",
                            "image_url": f"data:image/jpeg;base64,{base64_image}",
                        },
                    ],
                },
            ]

            response = self.client.responses.create(
                model=self.config.model,
                input=messages,
            )

            raw_text = response.output_text.strip()

            if log:
                print(f"[OpenAIClient] Prompt: {prompt}")
                print(f"[OpenAIClient] Response: {raw_text}")

            # Try parsing structured JSON-like output
            try:
                parsed = self._parse_json_from_model_output(raw_text)
                return parsed, True
            except Exception as e:
                # If model didn't return valid JSON, handle gracefully
                print(
                    f"[OpenAIClient] Warning: Failed to parse JSON from model output: {e}"
                )
                return {"error": "Failed to parse JSON"}, False

        except Exception as e:
            print(f"[OpenAIClient] Error during object detection: {e}")
            return {"error": str(e)}, False
