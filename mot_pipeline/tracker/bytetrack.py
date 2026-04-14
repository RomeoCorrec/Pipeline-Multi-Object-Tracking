from __future__ import annotations
import numpy as np
from scipy.optimize import linear_sum_assignment
from .kalman import KalmanFilter
from .track import Track, TrackState
from ..types import Detection


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
        self.active_tracks: list[Track] = []
        self.lost_tracks: list[Track] = []

    def update(self, detections: list[Detection]) -> list[Track]:
        high = [d for d in detections if d.conf >= self.conf_high]
        low = [d for d in detections if self.conf_low <= d.conf < self.conf_high]

        for t in self.active_tracks + self.lost_tracks:
            t.mean, t.covariance = self.kalman.predict(t.mean, t.covariance)
            t.bbox = cxcyarh_to_xyxy(t.mean)
            t.age += 1
            t.time_since_update += 1

        m1, unm_t1, unm_d1 = self._associate(
            self.active_tracks, high, self.iou_thresh_high, use_reid=True
        )
        for ti, di in m1:
            self._update_track(self.active_tracks[ti], high[di])

        unmatched_active = [self.active_tracks[i] for i in unm_t1]
        unmatched_high = [high[i] for i in unm_d1]

        pool2 = unmatched_active + self.lost_tracks
        m2, unm_t2, _ = self._associate(
            pool2, low, self.iou_thresh_low, use_reid=False
        )
        for ti, di in m2:
            self._update_track(pool2[ti], low[di])

        for t in [pool2[i] for i in unm_t2]:
            t.state = (
                TrackState.Removed
                if t.time_since_update > self.max_lost_age
                else TrackState.Lost
            )

        for det in unmatched_high:
            self.active_tracks.append(self._init_track(det))

        for t in self.active_tracks:
            if t.state == TrackState.New and t.hits >= self.min_hits:
                t.state = TrackState.Active

        self.lost_tracks = [
            t for t in self.lost_tracks + [pool2[i] for i in unm_t2]
            if t.state == TrackState.Lost
        ]
        self.active_tracks = [
            t for t in self.active_tracks
            if t.state in (TrackState.New, TrackState.Active)
        ]

        return [t for t in self.active_tracks if t.state == TrackState.Active]

    def _associate(self, tracks, dets, thresh, use_reid):
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
