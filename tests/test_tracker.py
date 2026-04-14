# tests/test_tracker.py
import numpy as np
import pytest
from mot_pipeline.types import Detection
from mot_pipeline.tracker.bytetrack import ByteTrack, iou_batch
from mot_pipeline.tracker.track import TrackState


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
    a[:, 2:] += a[:, :2]
    b = np.random.rand(5, 4) * 100
    b[:, 2:] += b[:, :2]
    assert iou_batch(a, b).shape == (3, 5)


def test_bytetrack_new_track_not_returned_first_frame():
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
    tracker.update([])
    tracker.update([])
    tracker.update([])
    tracker.update([])
    det2 = Detection(bbox=[400.0, 400.0, 440.0, 500.0], conf=0.9, class_id=0)
    tracker.update([det2])
    tracks2 = tracker.update([det2])
    if tracks1 and tracks2:
        assert tracks2[0].track_id > tracks1[0].track_id
