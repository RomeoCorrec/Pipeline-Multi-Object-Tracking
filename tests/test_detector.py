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
