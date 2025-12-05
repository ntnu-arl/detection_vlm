"""Module handling detection with vision-language models."""

import pathlib

from detection_vlm_python.misc import Logger
from detection_vlm_python.output import BoundingBox, ReasoningOutput


def root_path():
    """Get root path of package."""
    return pathlib.Path(__file__).absolute().parent
