from dataclasses import dataclass

from detection_vlm_python.config import Config


@dataclass
class OpenAIClientConfig(Config):
    """Configuration for the OpenAI client."""

    model: str = "gpt-4o"
    system_prompt: str = "You are a vision model that detects objects."

    @classmethod
    def load(cls, filepath):
        """Load config from file."""
        return Config.load(cls, filepath)
