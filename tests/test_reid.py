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
