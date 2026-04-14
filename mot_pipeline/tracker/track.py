from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import Optional
import numpy as np


class TrackState(Enum):
    New = 1
    Active = 2
    Lost = 3
    Removed = 4


@dataclass
class Track:
    track_id: int
    state: TrackState
    bbox: list[float]          # [x1, y1, x2, y2]
    class_id: int = 0
    age: int = 0
    hits: int = 0
    time_since_update: int = 0
    embedding: Optional[np.ndarray] = None
    mean: Optional[np.ndarray] = None        # Kalman state (8D)
    covariance: Optional[np.ndarray] = None  # Kalman covariance (8x8)
