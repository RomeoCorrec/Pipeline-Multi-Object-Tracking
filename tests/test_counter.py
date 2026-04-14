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


def test_crossing_direction_is_valid(tmp_path):
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
