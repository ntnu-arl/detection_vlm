from dataclasses import dataclass

from spark_config import Config


@dataclass
class OpenAIClientConfig(Config):
    """Configuration for the OpenAI client."""

    model: str = "gpt-4o"
    system_prompt: str = "You are a vision model that detects objects."

    @classmethod
    def load(cls, filepath):
        """Load config from file."""
        return Config.load(cls, filepath)
