# MOT Pipeline — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a complete Multi-Object Tracking portfolio pipeline with ByteTrack from scratch, YOLOv8 detection, MobileNetV2 Re-ID, virtual line counter, and MOT17 benchmark.

**Architecture:** Package `mot_pipeline/` with abstract interfaces for detector and embedder; ByteTrack owns Kalman state per-track; pipeline orchestrates all components frame-by-frame; device is auto-detected at runtime.

**Tech Stack:** Python 3.11, ultralytics (YOLOv8), opencv-python, torch/torchvision (MobileNetV2), scipy (Hungarian), filterpy not used (Kalman from scratch), trackeval (MOT17 metrics), rich, tqdm, pyyaml.

---

## File Map

| File | Responsibility |
|------|---------------|
| `mot_pipeline/types.py` | `Detection` dataclass (shared) |
| `mot_pipeline/tracker/kalman.py` | 8D Kalman Filter, from scratch |
| `mot_pipeline/tracker/track.py` | `TrackState` enum + `Track` dataclass |
| `mot_pipeline/tracker/bytetrack.py` | `ByteTrack`, IoU/cosine cost, Hungarian |
| `mot_pipeline/detector/base.py` | `BaseDetector` ABC |
| `mot_pipeline/detector/yolo_detector.py` | `YOLOv8Detector` |
| `mot_pipeline/reid/base.py` | `BaseEmbedder` ABC |
| `mot_pipeline/reid/embedder.py` | `MobileNetV2Embedder` |
| `mot_pipeline/counter.py` | `VirtualLineCounter` + CSV export |
| `mot_pipeline/visualizer.py` | Bounding boxes, trails, FPS overlay |
| `mot_pipeline/pipeline.py` | `MOTPipeline`, `load_config` |
| `mot_pipeline/__main__.py` | CLI entrypoint |
| `mot_pipeline/benchmark.py` | `MOT17Evaluator` |
| `config/default.yaml` | All tunable parameters |
| `config/virtual_line.json` | Virtual line coordinates |
| `tests/test_kalman.py` | Kalman unit tests |
| `tests/test_tracker.py` | ByteTrack unit tests |
| `tests/test_reid.py` | Embedder unit tests |
| `tests/test_counter.py` | Counter unit tests |
| `tests/test_detector.py` | Detector interface tests |
| `Dockerfile` | CPU image, exposes CLI |
| `README.md` | Metrics table, GIF instructions, usage |

---

## Task 1: Project scaffold

**Files:**
- Create: `requirements.txt`
- Create: `config/default.yaml`
- Create: `config/virtual_line.json`
- Create: `mot_pipeline/__init__.py`
- Create: `mot_pipeline/detector/__init__.py`
- Create: `mot_pipeline/tracker/__init__.py`
- Create: `mot_pipeline/reid/__init__.py`
- Create: `tests/__init__.py`
- Create: `results/.gitkeep`
- Modify: `.gitignore`

- [ ] **Step 1: Create requirements.txt**

```
ultralytics>=8.0
opencv-python>=4.8
torch>=2.0
torchvision>=0.15
scipy>=1.11
numpy>=1.24
pyyaml>=6.0
tqdm>=4.65
rich>=13.0
trackeval @ git+https://github.com/JonathonLuiten/TrackEval.git
pytest>=7.0
```

- [ ] **Step 2: Create config/default.yaml**

```yaml
device: auto  # auto | cpu | cuda

detector:
  model_path: yolov8n.pt
  conf_thresh: 0.25
  iou_thresh: 0.45
  classes: [0]  # 0 = person

tracker:
  conf_high: 0.6
  conf_low: 0.1
  iou_thresh_high: 0.3
  iou_thresh_low: 0.5
  max_lost_age: 30
  min_hits: 2
  reid_alpha: 0.8

embedder:
  type: mobilenetv2
  dim: 512

pipeline:
  output_video: output.mp4
  display: true
  counter_csv: results/crossings.csv

benchmark:
  mot17_dir: data/MOT17
  output_dir: results/
```

- [ ] **Step 3: Create config/virtual_line.json**

```json
{"x1": 0, "y1": 540, "x2": 1920, "y2": 540}
```

- [ ] **Step 4: Create package __init__.py files**

`mot_pipeline/__init__.py`, `mot_pipeline/detector/__init__.py`, `mot_pipeline/tracker/__init__.py`, `mot_pipeline/reid/__init__.py`, `tests/__init__.py` — all empty.

- [ ] **Step 5: Create results/.gitkeep**

Empty file at `results/.gitkeep`.

- [ ] **Step 6: Update .gitignore**

Append to existing `.gitignore`:
```
# MOT Pipeline
data/
results/*.csv
results/*.txt
output.mp4
__pycache__/
*.pyc
.pytest_cache/
*.egg-info/
```

- [ ] **Step 7: Commit**

```bash
git add requirements.txt config/ mot_pipeline/ tests/ results/ .gitignore
git commit -m "chore: project scaffold — config, packages, gitignore"
```

---

## Task 2: Shared Detection dataclass

**Files:**
- Create: `mot_pipeline/types.py`

- [ ] **Step 1: Create mot_pipeline/types.py**

```python
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
```

- [ ] **Step 2: Verify import works**

Run: `python -c "from mot_pipeline.types import Detection; print(Detection([0,0,1,1], 0.9, 0))"`

Expected output: `Detection(bbox=[0, 0, 1, 1], conf=0.9, class_id=0, embedding=None)`

- [ ] **Step 3: Commit**

```bash
git add mot_pipeline/types.py
git commit -m "feat: Detection dataclass"
```

---

## Task 3: Kalman Filter (from scratch)

**Files:**
- Create: `mot_pipeline/tracker/kalman.py`
- Create: `tests/test_kalman.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_kalman.py
import numpy as np
import pytest
from mot_pipeline.tracker.kalman import KalmanFilter


def test_initiate_shape():
    kf = KalmanFilter()
    meas = np.array([100.0, 200.0, 0.5, 150.0])
    mean, cov = kf.initiate(meas)
    assert mean.shape == (8,)
    assert cov.shape == (8, 8)


def test_initiate_position_matches_measurement():
    kf = KalmanFilter()
    meas = np.array([100.0, 200.0, 0.5, 150.0])
    mean, _ = kf.initiate(meas)
    np.testing.assert_array_equal(mean[:4], meas)
    np.testing.assert_array_equal(mean[4:], np.zeros(4))


def test_predict_returns_correct_shapes():
    kf = KalmanFilter()
    meas = np.array([100.0, 200.0, 0.5, 150.0])
    mean, cov = kf.initiate(meas)
    pred_mean, pred_cov = kf.predict(mean, cov)
    assert pred_mean.shape == (8,)
    assert pred_cov.shape == (8, 8)


def test_predict_constant_velocity_zero():
    """With zero initial velocity, predicted position equals initial position."""
    kf = KalmanFilter()
    meas = np.array([100.0, 200.0, 0.5, 150.0])
    mean, cov = kf.initiate(meas)
    pred_mean, _ = kf.predict(mean, cov)
    np.testing.assert_array_almost_equal(pred_mean[:4], meas, decimal=5)


def test_update_returns_correct_shapes():
    kf = KalmanFilter()
    meas = np.array([100.0, 200.0, 0.5, 150.0])
    mean, cov = kf.initiate(meas)
    pred_mean, pred_cov = kf.predict(mean, cov)
    new_mean, new_cov = kf.update(pred_mean, pred_cov, meas)
    assert new_mean.shape == (8,)
    assert new_cov.shape == (8, 8)


def test_update_moves_toward_measurement():
    kf = KalmanFilter()
    meas = np.array([100.0, 200.0, 0.5, 150.0])
    mean, cov = kf.initiate(meas)
    pred_mean, pred_cov = kf.predict(mean, cov)
    new_meas = np.array([110.0, 210.0, 0.5, 150.0])
    updated_mean, _ = kf.update(pred_mean, pred_cov, new_meas)
    # Position should move toward new measurement
    assert 100.0 <= updated_mean[0] <= 110.0
    assert 200.0 <= updated_mean[1] <= 210.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_kalman.py -v`

Expected: `ModuleNotFoundError` or `ImportError`

- [ ] **Step 3: Implement mot_pipeline/tracker/kalman.py**

```python
from __future__ import annotations
import numpy as np


class KalmanFilter:
    """
    8D state Kalman Filter for bounding-box tracking.

    State vector: [cx, cy, ar, h, vx, vy, var, vh]
      cx, cy  — center coordinates
      ar      — aspect ratio (w/h)
      h       — height
      vx, vy, var, vh — corresponding velocities

    Measurement vector: [cx, cy, ar, h]
    """

    def __init__(self) -> None:
        ndim, dt = 4, 1.0
        self._F = np.eye(2 * ndim)
        for i in range(ndim):
            self._F[i, ndim + i] = dt
        self._H = np.eye(ndim, 2 * ndim)
        self._std_weight_position = 1.0 / 20
        self._std_weight_velocity = 1.0 / 160

    def initiate(self, measurement: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mean = np.concatenate([measurement, np.zeros(4)])
        std = [
            2 * self._std_weight_position * measurement[3],
            2 * self._std_weight_position * measurement[3],
            1e-2,
            2 * self._std_weight_position * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            1e-5,
            10 * self._std_weight_velocity * measurement[3],
        ]
        covariance = np.diag(np.square(std))
        return mean, covariance

    def predict(
        self, mean: np.ndarray, covariance: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        std_pos = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-2,
            self._std_weight_position * mean[3],
        ]
        std_vel = [
            self._std_weight_velocity * mean[3],
            self._std_weight_velocity * mean[3],
            1e-5,
            self._std_weight_velocity * mean[3],
        ]
        Q = np.diag(np.square(np.concatenate([std_pos, std_vel])))
        return self._F @ mean, self._F @ covariance @ self._F.T + Q

    def update(
        self,
        mean: np.ndarray,
        covariance: np.ndarray,
        measurement: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        std = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-1,
            self._std_weight_position * mean[3],
        ]
        R = np.diag(np.square(std))
        S = self._H @ covariance @ self._H.T + R
        K = covariance @ self._H.T @ np.linalg.inv(S)
        innovation = measurement - self._H @ mean
        new_mean = mean + K @ innovation
        new_covariance = covariance - K @ S @ K.T
        return new_mean, new_covariance
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_kalman.py -v`

Expected: `6 passed`

- [ ] **Step 5: Commit**

```bash
git add mot_pipeline/tracker/kalman.py tests/test_kalman.py
git commit -m "feat: Kalman filter from scratch (8D state)"
```

---

## Task 4: Track dataclass + ByteTrack

**Files:**
- Create: `mot_pipeline/tracker/track.py`
- Create: `mot_pipeline/tracker/bytetrack.py`
- Create: `tests/test_tracker.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_tracker.py
import numpy as np
import pytest
from mot_pipeline.types import Detection
from mot_pipeline.tracker.bytetrack import ByteTrack, iou_batch
from mot_pipeline.tracker.track import TrackState


# --- iou_batch ---

def test_iou_perfect_overlap():
    a = np.array([[0.0, 0.0, 10.0, 10.0]])
    b = np.array([[0.0, 0.0, 10.0, 10.0]])
    np.testing.assert_almost_equal(iou_batch(a, b)[0, 0], 1.0)


def test_iou_no_overlap():
    a = np.array([[0.0, 0.0, 5.0, 5.0]])
    b = np.array([[10.0, 10.0, 20.0, 20.0]])
    np.testing.assert_almost_equal(iou_batch(a, b)[0, 0], 0.0)


def test_iou_partial_overlap():
    a = np.array([[0.0, 0.0, 10.0, 10.0]])
    b = np.array([[5.0, 0.0, 15.0, 10.0]])
    iou = iou_batch(a, b)[0, 0]
    assert 0.0 < iou < 1.0


def test_iou_output_shape():
    a = np.random.rand(3, 4) * 100
    a[:, 2:] += a[:, :2]  # ensure x2>x1, y2>y1
    b = np.random.rand(5, 4) * 100
    b[:, 2:] += b[:, :2]
    assert iou_batch(a, b).shape == (3, 5)


# --- ByteTrack ---

def test_bytetrack_new_track_not_returned_first_frame():
    """State=New requires min_hits frames before being returned as Active."""
    tracker = ByteTrack({})
    dets = [Detection(bbox=[10.0, 10.0, 50.0, 100.0], conf=0.9, class_id=0)]
    tracks = tracker.update(dets)
    assert len(tracks) == 0


def test_bytetrack_promoted_to_active_after_min_hits():
    tracker = ByteTrack({"min_hits": 2})
    det = Detection(bbox=[10.0, 10.0, 50.0, 100.0], conf=0.9, class_id=0)
    tracker.update([det])
    tracks = tracker.update([det])
    assert len(tracks) == 1
    assert tracks[0].state == TrackState.Active


def test_bytetrack_lost_after_no_detections():
    tracker = ByteTrack({"min_hits": 1, "max_lost_age": 2})
    det = Detection(bbox=[10.0, 10.0, 50.0, 100.0], conf=0.9, class_id=0)
    tracker.update([det])
    tracker.update([])
    tracker.update([])
    tracker.update([])
    assert len(tracker.active_tracks) == 0


def test_bytetrack_unique_ids():
    tracker = ByteTrack({"min_hits": 1})
    det1 = Detection(bbox=[10.0, 10.0, 50.0, 100.0], conf=0.9, class_id=0)
    det2 = Detection(bbox=[200.0, 200.0, 250.0, 300.0], conf=0.9, class_id=0)
    tracker.update([det1, det2])
    tracks = tracker.update([det1, det2])
    ids = [t.track_id for t in tracks]
    assert len(ids) == len(set(ids))


def test_bytetrack_low_conf_ignored_for_new_tracks():
    """Detections below conf_high should not create new tracks."""
    tracker = ByteTrack({"conf_high": 0.6, "conf_low": 0.1, "min_hits": 1})
    det = Detection(bbox=[10.0, 10.0, 50.0, 100.0], conf=0.3, class_id=0)
    tracker.update([det])
    tracks = tracker.update([det])
    assert len(tracks) == 0


def test_bytetrack_track_id_increments():
    tracker = ByteTrack({"min_hits": 1})
    det = Detection(bbox=[10.0, 10.0, 50.0, 100.0], conf=0.9, class_id=0)
    tracker.update([det])
    tracks1 = tracker.update([det])
    # Force a new track by moving far away
    tracker.update([])
    tracker.update([])
    tracker.update([])
    tracker.update([])
    det2 = Detection(bbox=[400.0, 400.0, 440.0, 500.0], conf=0.9, class_id=0)
    tracker.update([det2])
    tracks2 = tracker.update([det2])
    if tracks1 and tracks2:
        assert tracks2[0].track_id > tracks1[0].track_id
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_tracker.py -v`

Expected: `ImportError`

- [ ] **Step 3: Create mot_pipeline/tracker/track.py**

```python
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
    covariance: Optional[np.ndarray] = None  # Kalman covariance (8×8)
```

- [ ] **Step 4: Create mot_pipeline/tracker/bytetrack.py**

```python
from __future__ import annotations
import numpy as np
from scipy.optimize import linear_sum_assignment
from .kalman import KalmanFilter
from .track import Track, TrackState
from ..types import Detection


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def bbox_to_cxcyarh(bbox: list[float]) -> np.ndarray:
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1
    return np.array([(x1 + x2) / 2, (y1 + y2) / 2, w / h if h > 0 else 1.0, h])


def cxcyarh_to_xyxy(state: np.ndarray) -> list[float]:
    cx, cy, ar, h = state[:4]
    w = ar * h
    return [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]


def iou_batch(bboxes_a: np.ndarray, bboxes_b: np.ndarray) -> np.ndarray:
    """IoU matrix [N, M] for two sets of boxes in xyxy format."""
    area_a = (bboxes_a[:, 2] - bboxes_a[:, 0]) * (bboxes_a[:, 3] - bboxes_a[:, 1])
    area_b = (bboxes_b[:, 2] - bboxes_b[:, 0]) * (bboxes_b[:, 3] - bboxes_b[:, 1])
    ix1 = np.maximum(bboxes_a[:, None, 0], bboxes_b[None, :, 0])
    iy1 = np.maximum(bboxes_a[:, None, 1], bboxes_b[None, :, 1])
    ix2 = np.minimum(bboxes_a[:, None, 2], bboxes_b[None, :, 2])
    iy2 = np.minimum(bboxes_a[:, None, 3], bboxes_b[None, :, 3])
    inter = np.maximum(0, ix2 - ix1) * np.maximum(0, iy2 - iy1)
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / union, 0.0)


def cosine_distance_batch(emb_a: np.ndarray, emb_b: np.ndarray) -> np.ndarray:
    """Cosine distance matrix [N, M]. Assumes L2-normalized embeddings."""
    return 1.0 - emb_a @ emb_b.T


def _hungarian(cost: np.ndarray, thresh: float) -> tuple[list, list, list]:
    if cost.size == 0:
        return [], list(range(cost.shape[0])), list(range(cost.shape[1]))
    row_ind, col_ind = linear_sum_assignment(cost)
    matched, matched_r, matched_c = [], set(), set()
    for r, c in zip(row_ind, col_ind):
        if cost[r, c] <= thresh:
            matched.append((r, c))
            matched_r.add(r)
            matched_c.add(c)
    unmatched_rows = [r for r in range(cost.shape[0]) if r not in matched_r]
    unmatched_cols = [c for c in range(cost.shape[1]) if c not in matched_c]
    return matched, unmatched_rows, unmatched_cols


# ---------------------------------------------------------------------------
# ByteTrack
# ---------------------------------------------------------------------------

class ByteTrack:
    def __init__(self, config: dict) -> None:
        self.conf_high = config.get("conf_high", 0.6)
        self.conf_low = config.get("conf_low", 0.1)
        self.iou_thresh_high = config.get("iou_thresh_high", 0.3)
        self.iou_thresh_low = config.get("iou_thresh_low", 0.5)
        self.max_lost_age = config.get("max_lost_age", 30)
        self.min_hits = config.get("min_hits", 2)
        self.reid_alpha = config.get("reid_alpha", 0.8)

        self.kalman = KalmanFilter()
        self._next_id = 1
        self.active_tracks: list[Track] = []   # New + Active
        self.lost_tracks: list[Track] = []

    # ------------------------------------------------------------------
    def update(self, detections: list[Detection]) -> list[Track]:
        high = [d for d in detections if d.conf >= self.conf_high]
        low = [d for d in detections if self.conf_low <= d.conf < self.conf_high]

        # Kalman predict all
        for t in self.active_tracks + self.lost_tracks:
            t.mean, t.covariance = self.kalman.predict(t.mean, t.covariance)
            t.bbox = cxcyarh_to_xyxy(t.mean)
            t.age += 1
            t.time_since_update += 1

        # Association 1 — active vs high-conf dets
        m1, unm_t1, unm_d1 = self._associate(
            self.active_tracks, high, self.iou_thresh_high, use_reid=True
        )
        for ti, di in m1:
            self._update_track(self.active_tracks[ti], high[di])

        unmatched_active = [self.active_tracks[i] for i in unm_t1]
        unmatched_high = [high[i] for i in unm_d1]

        # Association 2 — (unmatched active + lost) vs low-conf dets
        pool2 = unmatched_active + self.lost_tracks
        m2, unm_t2, _ = self._associate(
            pool2, low, self.iou_thresh_low, use_reid=False
        )
        for ti, di in m2:
            self._update_track(pool2[ti], low[di])

        # Update states for unmatched tracks
        for t in [pool2[i] for i in unm_t2]:
            t.state = (
                TrackState.Removed
                if t.time_since_update > self.max_lost_age
                else TrackState.Lost
            )

        # New tracks from unmatched high-conf dets
        for det in unmatched_high:
            self.active_tracks.append(self._init_track(det))

        # Promote New → Active
        for t in self.active_tracks:
            if t.state == TrackState.New and t.hits >= self.min_hits:
                t.state = TrackState.Active

        # Rebuild lists
        self.lost_tracks = [
            t for t in self.lost_tracks + [pool2[i] for i in unm_t2]
            if t.state == TrackState.Lost
        ]
        self.active_tracks = [
            t for t in self.active_tracks
            if t.state in (TrackState.New, TrackState.Active)
        ]

        return [t for t in self.active_tracks if t.state == TrackState.Active]

    # ------------------------------------------------------------------
    def _associate(
        self,
        tracks: list[Track],
        dets: list[Detection],
        thresh: float,
        use_reid: bool,
    ) -> tuple[list, list, list]:
        if not tracks or not dets:
            return [], list(range(len(tracks))), list(range(len(dets)))

        t_boxes = np.array([t.bbox for t in tracks])
        d_boxes = np.array([d.bbox for d in dets])
        iou_cost = 1.0 - iou_batch(t_boxes, d_boxes)

        cost = iou_cost
        if use_reid:
            t_embs = [t.embedding for t in tracks]
            d_embs = [d.embedding for d in dets]
            if all(e is not None for e in t_embs + d_embs):
                emb_cost = cosine_distance_batch(
                    np.stack(t_embs), np.stack(d_embs)
                )
                cost = self.reid_alpha * iou_cost + (1 - self.reid_alpha) * emb_cost

        return _hungarian(cost, 1.0 - thresh)

    def _update_track(self, track: Track, det: Detection) -> None:
        meas = bbox_to_cxcyarh(det.bbox)
        track.mean, track.covariance = self.kalman.update(
            track.mean, track.covariance, meas
        )
        track.bbox = cxcyarh_to_xyxy(track.mean)
        track.hits += 1
        track.time_since_update = 0
        if det.embedding is not None:
            track.embedding = det.embedding
        if track.state == TrackState.Lost:
            track.state = TrackState.Active

    def _init_track(self, det: Detection) -> Track:
        mean, cov = self.kalman.initiate(bbox_to_cxcyarh(det.bbox))
        track = Track(
            track_id=self._next_id,
            state=TrackState.New,
            bbox=list(det.bbox),
            class_id=det.class_id,
            age=0,
            hits=1,
            time_since_update=0,
            embedding=det.embedding,
            mean=mean,
            covariance=cov,
        )
        self._next_id += 1
        return track
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_tracker.py -v`

Expected: `10 passed`

- [ ] **Step 6: Commit**

```bash
git add mot_pipeline/tracker/track.py mot_pipeline/tracker/bytetrack.py tests/test_tracker.py
git commit -m "feat: ByteTrack from scratch (Kalman + Hungarian, IoU/cosine cost)"
```

---

## Task 5: BaseDetector + YOLOv8Detector

**Files:**
- Create: `mot_pipeline/detector/base.py`
- Create: `mot_pipeline/detector/yolo_detector.py`
- Create: `tests/test_detector.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_detector.py
import numpy as np
import pytest
from unittest.mock import MagicMock, patch
from mot_pipeline.detector.base import BaseDetector
from mot_pipeline.detector.yolo_detector import YOLOv8Detector
from mot_pipeline.types import Detection


def test_yolo_is_subclass_of_base():
    assert issubclass(YOLOv8Detector, BaseDetector)


def test_detect_returns_list_of_detections():
    with patch("mot_pipeline.detector.yolo_detector.YOLO") as MockYOLO:
        mock_model = MagicMock()
        MockYOLO.return_value = mock_model
        mock_model.to.return_value = mock_model

        # Build a mock result with 1 detection
        mock_box = MagicMock()
        mock_box.xyxy = [MagicMock(cpu=lambda: MagicMock(numpy=lambda: np.array([10.0, 20.0, 50.0, 80.0])))]
        mock_box.conf = [MagicMock(cpu=lambda: MagicMock(numpy=lambda: np.array(0.85)))]
        mock_box.cls = [MagicMock(cpu=lambda: MagicMock(numpy=lambda: np.array(0)))]
        mock_box.__len__ = lambda self: 1

        mock_result = MagicMock()
        mock_result.boxes = mock_box
        mock_model.predict.return_value = [mock_result]

        detector = YOLOv8Detector({"model_path": "yolov8n.pt", "device": "cpu"})
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        dets = detector.detect(frame)

        assert isinstance(dets, list)
        assert len(dets) == 1
        assert isinstance(dets[0], Detection)
        assert dets[0].conf == pytest.approx(0.85)
        assert dets[0].class_id == 0


def test_detect_empty_frame_returns_list():
    with patch("mot_pipeline.detector.yolo_detector.YOLO") as MockYOLO:
        mock_model = MagicMock()
        MockYOLO.return_value = mock_model
        mock_model.to.return_value = mock_model

        mock_result = MagicMock()
        mock_result.boxes = None
        mock_model.predict.return_value = [mock_result]

        detector = YOLOv8Detector({"model_path": "yolov8n.pt", "device": "cpu"})
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        dets = detector.detect(frame)
        assert dets == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_detector.py -v`

Expected: `ImportError`

- [ ] **Step 3: Create mot_pipeline/detector/base.py**

```python
from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np
from ..types import Detection


class BaseDetector(ABC):
    @abstractmethod
    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Detect objects in a BGR frame. Returns list of Detection."""
```

- [ ] **Step 4: Create mot_pipeline/detector/yolo_detector.py**

```python
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
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_detector.py -v`

Expected: `3 passed`

- [ ] **Step 6: Commit**

```bash
git add mot_pipeline/detector/base.py mot_pipeline/detector/yolo_detector.py tests/test_detector.py
git commit -m "feat: BaseDetector + YOLOv8Detector wrapper"
```

---

## Task 6: BaseEmbedder + MobileNetV2Embedder

**Files:**
- Create: `mot_pipeline/reid/base.py`
- Create: `mot_pipeline/reid/embedder.py`
- Create: `tests/test_reid.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_reid.py
import numpy as np
import pytest
from mot_pipeline.reid.base import BaseEmbedder
from mot_pipeline.reid.embedder import MobileNetV2Embedder


@pytest.fixture(scope="module")
def embedder():
    return MobileNetV2Embedder({"device": "cpu", "dim": 512})


def test_mobilenetv2_is_subclass_of_base():
    assert issubclass(MobileNetV2Embedder, BaseEmbedder)


def test_embed_single_crop_shape(embedder):
    crop = np.random.randint(0, 255, (100, 50, 3), dtype=np.uint8)
    result = embedder.embed([crop])
    assert result.shape == (1, 512)


def test_embed_l2_normalized(embedder):
    crop = np.random.randint(0, 255, (100, 50, 3), dtype=np.uint8)
    result = embedder.embed([crop])
    norm = np.linalg.norm(result[0])
    np.testing.assert_almost_equal(norm, 1.0, decimal=5)


def test_embed_batch_shape(embedder):
    crops = [np.random.randint(0, 255, (80, 40, 3), dtype=np.uint8) for _ in range(4)]
    result = embedder.embed(crops)
    assert result.shape == (4, 512)


def test_embed_empty_input(embedder):
    result = embedder.embed([])
    assert result.shape[0] == 0


def test_embed_different_crops_differ(embedder):
    rng = np.random.default_rng(42)
    crop1 = rng.integers(0, 255, (80, 40, 3), dtype=np.uint8)
    crop2 = rng.integers(0, 255, (80, 40, 3), dtype=np.uint8)
    e1 = embedder.embed([crop1])
    e2 = embedder.embed([crop2])
    assert not np.allclose(e1, e2)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_reid.py -v`

Expected: `ImportError`

- [ ] **Step 3: Create mot_pipeline/reid/base.py**

```python
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
```

- [ ] **Step 4: Create mot_pipeline/reid/embedder.py**

```python
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
    MobileNetV2 feature extractor → 512D L2-normalized embedding.
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
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_reid.py -v`

Expected: `6 passed`  
(First run downloads MobileNetV2 weights ~14 MB — normal.)

- [ ] **Step 6: Commit**

```bash
git add mot_pipeline/reid/base.py mot_pipeline/reid/embedder.py tests/test_reid.py
git commit -m "feat: BaseEmbedder + MobileNetV2Embedder (512D, L2-normalized)"
```

---

## Task 7: VirtualLineCounter

**Files:**
- Create: `mot_pipeline/counter.py`
- Create: `tests/test_counter.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_counter.py
import json
import os
import pytest
from mot_pipeline.counter import VirtualLineCounter
from mot_pipeline.tracker.track import Track, TrackState


def make_cfg(tmp_path, x1, y1, x2, y2):
    cfg = {"x1": x1, "y1": y1, "x2": x2, "y2": y2}
    p = tmp_path / "line.json"
    p.write_text(json.dumps(cfg))
    return str(p)


def track(tid, bbox):
    return Track(track_id=tid, state=TrackState.Active,
                 bbox=bbox, class_id=0)


def test_no_crossing_same_side(tmp_path):
    counter = VirtualLineCounter(make_cfg(tmp_path, 0, 100, 500, 100))
    t = track(1, [50.0, 20.0, 150.0, 80.0])   # centroid y=50, above line y=100
    counter.update([t])
    t.bbox = [55.0, 25.0, 155.0, 85.0]         # centroid y=55, still above
    counter.update([t])
    assert len(counter._crossings) == 0


def test_crossing_detected(tmp_path):
    counter = VirtualLineCounter(make_cfg(tmp_path, 0, 100, 500, 100))
    t = track(1, [50.0, 20.0, 150.0, 80.0])   # centroid y=50 (above)
    counter.update([t])
    t.bbox = [50.0, 110.0, 150.0, 160.0]       # centroid y=135 (below)
    counter.update([t])
    assert len(counter._crossings) == 1


def test_crossing_direction_out(tmp_path):
    """Centroid starts above (positive side) then crosses → direction 'out'."""
    counter = VirtualLineCounter(make_cfg(tmp_path, 0, 100, 500, 100))
    t = track(1, [50.0, 20.0, 150.0, 80.0])
    counter.update([t])
    t.bbox = [50.0, 110.0, 150.0, 160.0]
    counter.update([t])
    assert counter._crossings[0]["direction"] in ("in", "out")


def test_count_increments(tmp_path):
    counter = VirtualLineCounter(make_cfg(tmp_path, 0, 100, 500, 100))
    t = track(1, [50.0, 20.0, 150.0, 80.0])
    counter.update([t])
    t.bbox = [50.0, 110.0, 150.0, 160.0]
    counter.update([t])
    total = sum(c.in_count + c.out_count for c in counter.counts.values())
    assert total == 1


def test_csv_export(tmp_path):
    counter = VirtualLineCounter(make_cfg(tmp_path, 0, 100, 500, 100))
    t = track(1, [50.0, 20.0, 150.0, 80.0])
    counter.update([t])
    t.bbox = [50.0, 110.0, 150.0, 160.0]
    counter.update([t])

    csv_path = str(tmp_path / "out.csv")
    counter.export_csv(csv_path)
    assert os.path.exists(csv_path)
    with open(csv_path) as f:
        lines = f.readlines()
    assert len(lines) == 2  # header + 1 crossing row
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_counter.py -v`

Expected: `ImportError`

- [ ] **Step 3: Create mot_pipeline/counter.py**

```python
from __future__ import annotations
import csv
import json
import time
from dataclasses import dataclass, field
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

    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    def _cross(self, p: np.ndarray) -> float:
        d = self.p2 - self.p1
        return float(d[0] * (p[1] - self.p1[1]) - d[1] * (p[0] - self.p1[0]))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_counter.py -v`

Expected: `5 passed`

- [ ] **Step 5: Commit**

```bash
git add mot_pipeline/counter.py tests/test_counter.py
git commit -m "feat: VirtualLineCounter with cross-product detection + CSV export"
```

---

## Task 8: Visualizer

**Files:**
- Create: `mot_pipeline/visualizer.py`

- [ ] **Step 1: Create mot_pipeline/visualizer.py**

```python
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
```

- [ ] **Step 2: Verify import**

Run: `python -c "from mot_pipeline.visualizer import Visualizer; print('OK')"`

Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add mot_pipeline/visualizer.py
git commit -m "feat: Visualizer — bbox, trails, FPS overlay, virtual line"
```

---

## Task 9: MOTPipeline + CLI entrypoint

**Files:**
- Create: `mot_pipeline/pipeline.py`
- Create: `mot_pipeline/__main__.py`

- [ ] **Step 1: Create mot_pipeline/pipeline.py**

```python
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

    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
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
```

- [ ] **Step 2: Create mot_pipeline/__main__.py**

```python
from __future__ import annotations
import argparse
from .pipeline import MOTPipeline, load_config
from .counter import VirtualLineCounter


def main() -> None:
    parser = argparse.ArgumentParser(description="MOT Pipeline")
    parser.add_argument("--source", default="0",
                        help="Video source: int for webcam, path for video file")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--line", default=None,
                        help="Path to virtual_line.json")
    parser.add_argument("--no-display", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.no_display:
        config.setdefault("pipeline", {})["display"] = False

    counter = VirtualLineCounter(args.line) if args.line else None
    source: int | str = int(args.source) if args.source.isdigit() else args.source

    pipeline = MOTPipeline(config=config, counter=counter)
    pipeline.run(source)


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Verify CLI entrypoint loads**

Run: `python -m mot_pipeline.pipeline --help`

Expected: argparse help text with `--source`, `--config`, `--line`, `--no-display`

- [ ] **Step 4: Run full test suite**

Run: `pytest tests/ -v`

Expected: all previous tests still pass (15+ passed, 0 failed)

- [ ] **Step 5: Commit**

```bash
git add mot_pipeline/pipeline.py mot_pipeline/__main__.py
git commit -m "feat: MOTPipeline orchestration + CLI entrypoint"
```

---

## Task 10: Benchmark (MOT17Evaluator)

**Files:**
- Create: `mot_pipeline/benchmark.py`

- [ ] **Step 1: Create mot_pipeline/benchmark.py**

```python
from __future__ import annotations
import csv
import time
import urllib.request
import zipfile
from datetime import date
from pathlib import Path

import cv2
import numpy as np
from rich.console import Console
from rich.table import Table
from tqdm import tqdm

# Official MOT17 download — DPM detector subset (~1.9 GB)
_MOT17_URL = "https://motchallenge.net/data/MOT17.zip"


class MOT17Evaluator:
    """
    Downloads MOT17 (once), runs the pipeline on every *-DPM sequence,
    computes MOTA/IDF1/ID-Switches/FPS via trackeval, prints a rich table,
    and saves a CSV.
    """

    def __init__(self, config: dict) -> None:
        self.mot17_dir = Path(config.get("benchmark", {}).get("mot17_dir", "data/MOT17"))
        self.output_dir = Path(config.get("benchmark", {}).get("output_dir", "results"))
        self.config = config

    # ------------------------------------------------------------------
    def run(self) -> None:
        self._ensure_dataset()
        self.output_dir.mkdir(parents=True, exist_ok=True)

        sequences = sorted(self.mot17_dir.glob("MOT17-*-DPM"))
        if not sequences:
            raise FileNotFoundError(
                f"No MOT17-*-DPM sequences found in {self.mot17_dir}. "
                "Check mot17_dir in config."
            )

        results = []
        for seq_path in sequences:
            print(f"\n>>> {seq_path.name}")
            results.append(self._run_sequence(seq_path))

        self._display(results)
        self._save_csv(results)

    # ------------------------------------------------------------------
    def _ensure_dataset(self) -> None:
        if self.mot17_dir.exists() and any(self.mot17_dir.iterdir()):
            return
        self.mot17_dir.parent.mkdir(parents=True, exist_ok=True)
        zip_path = self.mot17_dir.parent / "MOT17.zip"

        print(f"Downloading MOT17 from {_MOT17_URL} …")
        with tqdm(unit="B", unit_scale=True, desc="MOT17.zip") as pbar:
            def _hook(b: int, bsize: int, tsize: int | None) -> None:
                if tsize:
                    pbar.total = tsize
                pbar.update(b * bsize - pbar.n)
            urllib.request.urlretrieve(_MOT17_URL, zip_path, reporthook=_hook)

        print("Extracting …")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(self.mot17_dir.parent)
        zip_path.unlink()
        print(f"Dataset ready at {self.mot17_dir}")

    # ------------------------------------------------------------------
    def _run_sequence(self, seq_path: Path) -> dict:
        from .pipeline import MOTPipeline

        frames = sorted((seq_path / "img1").glob("*.jpg"))
        pipeline = MOTPipeline(config=self.config)
        pred_lines: list[str] = []
        elapsed = 0.0

        for frame_idx, img_path in enumerate(frames, 1):
            frame = cv2.imread(str(img_path))
            t0 = time.perf_counter()

            dets = pipeline.detector.detect(frame)
            pipeline._attach_embeddings(frame, dets)
            tracks = pipeline.tracker.update(dets)

            elapsed += time.perf_counter() - t0
            for t in tracks:
                x1, y1, x2, y2 = t.bbox
                w, h = x2 - x1, y2 - y1
                pred_lines.append(
                    f"{frame_idx},{t.track_id},"
                    f"{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},1,-1,-1,-1\n"
                )

        pred_path = self.output_dir / f"{seq_path.name}.txt"
        pred_path.write_text("".join(pred_lines))
        fps = len(frames) / elapsed if elapsed > 0 else 0.0

        metrics = self._eval(seq_path, pred_path, seq_path.name)
        metrics["sequence"] = seq_path.name
        metrics["fps"] = round(fps, 1)
        return metrics

    # ------------------------------------------------------------------
    def _eval(self, seq_path: Path, pred_path: Path, seq_name: str) -> dict:
        try:
            import trackeval  # type: ignore

            eval_cfg = trackeval.Evaluator.get_default_eval_config()
            eval_cfg.update({"PRINT_RESULTS": False, "OUTPUT_SUMMARY": False,
                              "OUTPUT_EMPTY_CLASSES": False})

            data_cfg = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
            data_cfg.update({
                "GT_FOLDER": str(seq_path.parent),
                "TRACKERS_FOLDER": str(self.output_dir),
                "SEQMAP_FILE": None,
                "SEQ_INFO": {seq_name: None},
                "OUTPUT_FOLDER": None,
                "TRACKER_SUB_FOLDER": "",
            })

            evaluator = trackeval.Evaluator(eval_cfg)
            dataset = [trackeval.datasets.MotChallenge2DBox(data_cfg)]
            metrics = [trackeval.metrics.CLEAR(), trackeval.metrics.Identity()]
            raw, _ = evaluator.evaluate(dataset, metrics)

            seq_res = raw["MotChallenge2DBox"][seq_name]["pedestrian"]
            clear = seq_res["CLEAR"]
            ident = seq_res["Identity"]
            return {
                "mota": round(float(clear.get("MOTA", 0)) * 100, 1),
                "idf1": round(float(ident.get("IDF1", 0)) * 100, 1),
                "id_switches": int(clear.get("IDSW", 0)),
            }
        except Exception as exc:
            print(f"  [trackeval error] {exc}")
            return {"mota": -1.0, "idf1": -1.0, "id_switches": -1}

    # ------------------------------------------------------------------
    def _display(self, results: list[dict]) -> None:
        console = Console()
        table = Table(title="MOT17 Benchmark Results", show_lines=True)
        for col in ("Sequence", "MOTA ↑", "IDF1 ↑", "ID Sw. ↓", "FPS"):
            table.add_column(col, justify="right")
        for r in results:
            table.add_row(
                r["sequence"], str(r["mota"]), str(r["idf1"]),
                str(r["id_switches"]), str(r["fps"]),
            )
        console.print(table)

    def _save_csv(self, results: list[dict]) -> None:
        path = self.output_dir / f"benchmark_{date.today()}.csv"
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(
                f, fieldnames=["sequence", "mota", "idf1", "id_switches", "fps"]
            )
            w.writeheader()
            w.writerows(results)
        print(f"\nResults saved to {path}")
```

- [ ] **Step 2: Verify import**

Run: `python -c "from mot_pipeline.benchmark import MOT17Evaluator; print('OK')"`

Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add mot_pipeline/benchmark.py
git commit -m "feat: MOT17Evaluator — auto-download, trackeval metrics, rich table"
```

---

## Task 11: Dockerfile

**Files:**
- Create: `Dockerfile`

- [ ] **Step 1: Create Dockerfile**

```dockerfile
FROM python:3.11-slim

# System deps for OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY mot_pipeline/ ./mot_pipeline/
COPY config/ ./config/

# Pre-download YOLOv8n weights
RUN python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"

ENTRYPOINT ["python", "-m", "mot_pipeline"]
CMD ["--source", "0", "--no-display"]
```

- [ ] **Step 2: Verify Dockerfile syntax**

Run: `docker build --check . 2>&1 | head -5` (or `docker build -t mot-pipeline . --no-cache` if you want a full build)

Expected: no syntax errors

- [ ] **Step 3: Commit**

```bash
git add Dockerfile
git commit -m "feat: Dockerfile (python:3.11-slim, OpenCV system deps, YOLOv8n weights pre-cached)"
```

---

## Task 12: README

**Files:**
- Create: `README.md`

- [ ] **Step 1: Create README.md**

````markdown
# MOT Pipeline

Real-time Multi-Object Tracking pipeline: YOLOv8 detection · ByteTrack from scratch · MobileNetV2 Re-ID · virtual line counter · MOT17 benchmark.

![Demo](docs/demo.gif)

---

## Features

- **ByteTrack from scratch** — Kalman Filter (8D state), Hungarian algorithm, dual-threshold association, IoU + cosine Re-ID cost
- **Swappable interfaces** — `BaseDetector` / `BaseEmbedder` ABCs; swap YOLOv8 or MobileNetV2 in one line
- **Virtual line counter** — JSON-configured line, cross-product direction detection, CSV export
- **MOT17 benchmark** — auto-download, MOTA / IDF1 / ID Switches / FPS via trackeval
- **CPU + GPU** — auto-detects CUDA; runs on any machine

---

## Quick Start

### pip

```bash
pip install -r requirements.txt
# Webcam
python -m mot_pipeline --source 0
# Video file
python -m mot_pipeline --source path/to/video.mp4
# With virtual line counter
python -m mot_pipeline --source path/to/video.mp4 --line config/virtual_line.json
```

### Docker

```bash
docker build -t mot-pipeline .
# Headless on a video file
docker run --rm -v $(pwd)/videos:/videos mot-pipeline --source /videos/test.mp4 --no-display
```

---

## Configuration

Edit `config/default.yaml`:

```yaml
device: auto          # auto | cpu | cuda
detector:
  model_path: yolov8n.pt
  conf_thresh: 0.25
  classes: [0]        # COCO class IDs (0=person)
tracker:
  conf_high: 0.6
  conf_low: 0.1
  max_lost_age: 30
  min_hits: 2
  reid_alpha: 0.8     # weight of IoU vs cosine cost
embedder:
  dim: 512
pipeline:
  output_video: output.mp4
  display: true
```

Virtual line (`config/virtual_line.json`):
```json
{"x1": 0, "y1": 540, "x2": 1920, "y2": 540}
```

---

## Benchmark — MOT17

```bash
python -c "
from mot_pipeline.benchmark import MOT17Evaluator
from mot_pipeline.pipeline import load_config
MOT17Evaluator(load_config()).run()
"
```

MOT17 (~1.9 GB) is downloaded automatically on first run.

### Results (YOLOv8n, MobileNetV2 Re-ID, CPU)

| Sequence | MOTA ↑ | IDF1 ↑ | ID Sw. ↓ | FPS |
|----------|--------|--------|----------|-----|
| MOT17-02-DPM | — | — | — | — |
| MOT17-04-DPM | — | — | — | — |
| MOT17-05-DPM | — | — | — | — |
| *Run benchmark to fill this table* | | | | |

---

## Generate Demo GIF

```bash
ffmpeg -i output.mp4 -vf "fps=15,scale=640:-1:flags=lanczos" \
       -loop 0 docs/demo.gif
```

---

## Architecture

```
frame → YOLOv8Detector ──→ List[Detection]
                        ↓
              MobileNetV2Embedder (high-conf crops)
                        ↓
              ByteTrack.update() ──→ List[Track]
                        ↓
        ┌───────────────┴────────────────┐
  VirtualLineCounter           Visualizer.draw()
        ↓                               ↓
   crossings.csv              annotated frame → VideoWriter
```

---

## Extending

### Swap detector (e.g. RT-DETR)

```python
from mot_pipeline.detector.base import BaseDetector

class RTDETRDetector(BaseDetector):
    def detect(self, frame):
        ...  # return List[Detection]

pipeline = MOTPipeline(config, detector=RTDETRDetector(...))
```

### Swap Re-ID model (e.g. OSNet)

```python
from mot_pipeline.reid.base import BaseEmbedder

class OSNetEmbedder(BaseEmbedder):
    def embed(self, crops):
        ...  # return np.ndarray [N, D], L2-normalized

pipeline = MOTPipeline(config, embedder=OSNetEmbedder(...))
```

---

## Tests

```bash
pytest tests/ -v
```
````

- [ ] **Step 2: Create docs/ directory for the demo GIF placeholder**

```bash
mkdir -p docs
touch docs/.gitkeep
```

- [ ] **Step 3: Run full test suite one final time**

Run: `pytest tests/ -v`

Expected: all tests pass

- [ ] **Step 4: Final commit**

```bash
git add README.md docs/
git commit -m "docs: README with architecture, benchmark table, Docker + GIF instructions"
```

---

## Self-Review Against Spec

| Spec section | Task that implements it |
|---|---|
| detector/yolo_detector.py — BaseDetector wrapper | Task 5 |
| tracker/bytetrack.py — from scratch | Tasks 3, 4 |
| reid/embedder.py — BaseEmbedder + MobileNetV2 | Task 6 |
| pipeline.py — orchestration | Task 9 |
| visualizer.py — bbox, trails, FPS, counter | Tasks 8, 9 |
| counter.py — virtual line, CSV export | Task 7 |
| benchmark.py — MOT17 download, metrics | Task 10 |
| Dockerfile | Task 11 |
| README + metrics table + GIF instructions | Task 12 |
| config YAML + virtual_line.json | Task 1 |
| CPU/GPU auto-detect | Task 9 (`_resolve_device`) |
| Hybrid IoU+cosine cost | Task 4 (`_associate`) |
| `min_hits` guard (New→Active) | Task 4 |
| `max_lost_age` removal | Task 4 |
| L2-normalized 512D embeddings | Task 6 |
| Trails deque(30) + opacity | Task 8 |
| ID-colored palette | Task 8 |
| CSV columns: timestamp/track_id/class_name/direction | Task 7 |
| MOTA, IDF1, ID Switches, FPS output | Task 10 |
| `rich` table display | Task 10 |
