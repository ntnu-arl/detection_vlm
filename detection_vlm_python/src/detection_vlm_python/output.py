from dataclasses import dataclass

import numpy as np


@dataclass
class BoundingBox:
    x0: int
    y0: int
    x1: int
    y1: int
    details: str
    confidence: float = None
    mask: np.ndarray = None


@dataclass
class ReasoningOutput:
    select: bool
    probability: float
    explanation: str
