from __future__ import annotations
import time
from collections import deque
import cv2
import numpy as np
import yaml
from .detector.base import BaseDetector
from .detector.yolo_detector import YOLOv8Detector
from .tracker.bytetrack import ByteTrack
from .reid.base import BaseEmbedder
from .reid.embedder import MobileNetV2Embedder
from .counter import VirtualLineCounter
from .visualizer import Visualizer


def load_config(path: str = "config/default.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _resolve_device(cfg: str) -> str:
    if cfg == "auto":
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    return cfg


class MOTPipeline:
    def __init__(
        self,
        config: dict,
        detector: BaseDetector | None = None,
        tracker: ByteTrack | None = None,
        embedder: BaseEmbedder | None = None,
        counter: VirtualLineCounter | None = None,
    ) -> None:
        device = _resolve_device(config.get("device", "auto"))

        self.detector = detector or YOLOv8Detector(
            {**config.get("detector", {}), "device": device}
        )
        self.tracker = tracker or ByteTrack(config.get("tracker", {}))
        self.embedder = embedder or MobileNetV2Embedder(
            {**config.get("embedder", {}), "device": device}
        )
        self.counter = counter
        self.visualizer = Visualizer()
        self.config = config
        self._fps_buf: deque[float] = deque(maxlen=30)

    def run(self, source: int | str) -> None:
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open source: {source!r}")

        pipe_cfg = self.config.get("pipeline", {})
        display = pipe_cfg.get("display", True)
        writer = self._make_writer(cap, pipe_cfg.get("output_video"))

        try:
            while True:
                t0 = time.perf_counter()
                ret, frame = cap.read()
                if not ret:
                    break

                detections = self.detector.detect(frame)
                self._attach_embeddings(frame, detections)
                tracks = self.tracker.update(detections)

                if self.counter is not None:
                    self.counter.update(tracks)

                dt = time.perf_counter() - t0
                self._fps_buf.append(1.0 / dt if dt > 0 else 0.0)
                fps = sum(self._fps_buf) / len(self._fps_buf)

                annotated = self.visualizer.draw(frame, tracks, fps, self.counter)

                if writer:
                    writer.write(annotated)
                if display:
                    cv2.imshow("MOT Pipeline", annotated)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
        finally:
            cap.release()
            if writer:
                writer.release()
            cv2.destroyAllWindows()
            if self.counter:
                csv_path = pipe_cfg.get("counter_csv", "results/crossings.csv")
                self.counter.export_csv(csv_path)

    def _attach_embeddings(self, frame: np.ndarray, detections) -> None:
        high = [d for d in detections if d.conf >= self.tracker.conf_high]
        if not high:
            return
        crops = []
        valid_dets = []
        for d in high:
            x1, y1, x2, y2 = (int(v) for v in d.bbox)
            crop = frame[max(0, y1):y2, max(0, x1):x2]
            if crop.size > 0:
                crops.append(crop)
                valid_dets.append(d)
        if crops:
            embeddings = self.embedder.embed(crops)
            for det, emb in zip(valid_dets, embeddings):
                det.embedding = emb

    @staticmethod
    def _make_writer(cap, output_path: str | None):
        if not output_path:
            return None
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        return cv2.VideoWriter(output_path, fourcc, fps, (w, h))
