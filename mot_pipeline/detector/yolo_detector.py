from __future__ import annotations
import numpy as np
from ultralytics import YOLO
from .base import BaseDetector
from ..types import Detection


class YOLOv8Detector(BaseDetector):
    def __init__(self, config: dict) -> None:
        self.model = YOLO(config.get("model_path", "yolov8n.pt"))
        self.conf_thresh = config.get("conf_thresh", 0.25)
        self.iou_thresh = config.get("iou_thresh", 0.45)
        self.classes = config.get("classes", [0])
        device = config.get("device", "cpu")
        self.model.to(device)

    def detect(self, frame: np.ndarray) -> list[Detection]:
        results = self.model.predict(
            frame,
            conf=self.conf_thresh,
            iou=self.iou_thresh,
            classes=self.classes,
            verbose=False,
        )[0]

        if results.boxes is None:
            return []

        detections: list[Detection] = []
        boxes = results.boxes
        for i in range(len(boxes)):
            bbox = boxes.xyxy[i].cpu().numpy().tolist()
            conf = float(boxes.conf[i].cpu().numpy())
            class_id = int(boxes.cls[i].cpu().numpy())
            detections.append(Detection(bbox=bbox, conf=conf, class_id=class_id))
        return detections
