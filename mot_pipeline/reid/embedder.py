from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
import cv2
from .base import BaseEmbedder


class MobileNetV2Embedder(BaseEmbedder):
    """
    MobileNetV2 feature extractor -> 512D L2-normalized embedding.
    Swap for OSNet/ResNet50 by subclassing BaseEmbedder.
    """

    def __init__(self, config: dict) -> None:
        device_str = config.get("device", "cpu")
        if device_str == "auto":
            device_str = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device_str)
        dim = config.get("dim", 512)

        backbone = models.mobilenet_v2(
            weights=models.MobileNet_V2_Weights.DEFAULT
        )
        extractor = nn.Sequential(
            backbone.features,
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )
        projector = nn.Linear(1280, dim)

        self.model = nn.Sequential(extractor, projector)
        self.model.eval().to(self.device)

        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    @torch.no_grad()
    def embed(self, crops: list[np.ndarray]) -> np.ndarray:
        if not crops:
            return np.empty((0, self.model[-1].out_features), dtype=np.float32)

        tensors = [
            self.transform(cv2.cvtColor(c, cv2.COLOR_BGR2RGB))
            for c in crops
        ]
        batch = torch.stack(tensors).to(self.device)
        embeddings = self.model(batch).cpu().numpy().astype(np.float32)

        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms > 0, norms, 1.0)
        return embeddings / norms
