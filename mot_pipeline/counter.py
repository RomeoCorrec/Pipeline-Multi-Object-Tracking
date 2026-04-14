from __future__ import annotations
import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from .tracker.track import Track


@dataclass
class LineCount:
    in_count: int = 0
    out_count: int = 0


class VirtualLineCounter:
    """
    Detects when a track's centroid crosses a configurable virtual line.
    Uses cross-product sign change for direction detection.
    """

    def __init__(
        self,
        config_path: str,
        class_names: dict[int, str] | None = None,
    ) -> None:
        with open(config_path) as f:
            cfg = json.load(f)
        self.p1 = np.array([cfg["x1"], cfg["y1"]], dtype=float)
        self.p2 = np.array([cfg["x2"], cfg["y2"]], dtype=float)
        self.class_names: dict[int, str] = class_names or {}

        self._prev: dict[int, np.ndarray] = {}
        self.counts: dict[str, LineCount] = {}
        self._crossings: list[dict] = []

    def update(self, tracks: list[Track]) -> None:
        live_ids = {t.track_id for t in tracks}
        self._prev = {k: v for k, v in self._prev.items() if k in live_ids}

        for t in tracks:
            centroid = np.array([
                (t.bbox[0] + t.bbox[2]) / 2,
                (t.bbox[1] + t.bbox[3]) / 2,
            ])
            if t.track_id in self._prev:
                prev = self._prev[t.track_id]
                s_prev = self._cross(prev)
                s_curr = self._cross(centroid)
                if s_prev * s_curr < 0:
                    direction = "in" if s_curr < 0 else "out"
                    cls = self.class_names.get(t.class_id, "object")
                    self.counts.setdefault(cls, LineCount())
                    if direction == "in":
                        self.counts[cls].in_count += 1
                    else:
                        self.counts[cls].out_count += 1
                    self._crossings.append({
                        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                        "track_id": t.track_id,
                        "class_name": cls,
                        "direction": direction,
                    })
            self._prev[t.track_id] = centroid

    def export_csv(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(
                f, fieldnames=["timestamp", "track_id", "class_name", "direction"]
            )
            w.writeheader()
            w.writerows(self._crossings)

    def _cross(self, p: np.ndarray) -> float:
        d = self.p2 - self.p1
        return float(d[0] * (p[1] - self.p1[1]) - d[1] * (p[0] - self.p1[0]))
