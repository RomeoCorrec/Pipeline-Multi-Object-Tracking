from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np


class BaseEmbedder(ABC):
    @abstractmethod
    def embed(self, crops: list[np.ndarray]) -> np.ndarray:
        """
        Args:
            crops: list of BGR images (variable sizes)
        Returns:
            np.ndarray shape [N, D], L2-normalized float32
        """
