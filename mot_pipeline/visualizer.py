from __future__ import annotations
from collections import deque
import cv2
import numpy as np
from .tracker.track import Track

# 20-color HSV palette, deterministic
_PALETTE: list[tuple[int, int, int]] = [
    tuple(int(c) for c in cv2.cvtColor(  # type: ignore[misc]
        np.array([[[int(h * 180 / 20), 200, 230]]], dtype=np.uint8),
        cv2.COLOR_HSV2BGR,
    )[0][0])
    for h in range(20)
]


class Visualizer:
    def __init__(self, trail_length: int = 30) -> None:
        self.trail_length = trail_length
        self._trails: dict[int, deque[tuple[int, int]]] = {}

    def draw(
        self,
        frame: np.ndarray,
        tracks: list[Track],
        fps: float = 0.0,
        counter=None,
    ) -> np.ndarray:
        output = frame.copy()
        live_ids = {t.track_id for t in tracks}
        self._trails = {k: v for k, v in self._trails.items() if k in live_ids}

        for t in tracks:
            color = _PALETTE[t.track_id % len(_PALETTE)]
            x1, y1, x2, y2 = (int(v) for v in t.bbox)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            # Trail
            if t.track_id not in self._trails:
                self._trails[t.track_id] = deque(maxlen=self.trail_length)
            self._trails[t.track_id].append((cx, cy))
            pts = list(self._trails[t.track_id])
            for i in range(1, len(pts)):
                alpha = i / len(pts)
                cv2.line(output, pts[i - 1], pts[i], color, max(1, int(2 * alpha)))

            # Bounding box
            cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
            label = f"ID:{t.track_id}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(output, (x1, y1 - th - 4), (x1 + tw, y1), color, -1)
            cv2.putText(
                output, label, (x1, y1 - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1,
            )

        # HUD
        cv2.putText(output, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(output, f"Tracks: {len(tracks)}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Virtual line + counts
        if counter is not None:
            cv2.line(
                output,
                tuple(map(int, counter.p1)),
                tuple(map(int, counter.p2)),
                (0, 0, 255), 2,
            )
            y_off = 90
            for cls_name, cnt in counter.counts.items():
                cv2.putText(
                    output,
                    f"{cls_name}  In:{cnt.in_count}  Out:{cnt.out_count}",
                    (10, y_off),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2,
                )
                y_off += 25

        return output
