from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np
from ..types import Detection


class BaseDetector(ABC):
    @abstractmethod
    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Detect objects in a BGR frame. Returns list of Detection."""
