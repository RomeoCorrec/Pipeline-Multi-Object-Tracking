from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class Detection:
    bbox: list[float]          # [x1, y1, x2, y2]
    conf: float
    class_id: int
    embedding: Optional[np.ndarray] = None
