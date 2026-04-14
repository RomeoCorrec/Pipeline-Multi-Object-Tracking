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
    assert 100.0 <= updated_mean[0] <= 110.0
    assert 200.0 <= updated_mean[1] <= 210.0
