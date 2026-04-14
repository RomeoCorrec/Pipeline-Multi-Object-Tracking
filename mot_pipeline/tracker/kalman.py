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
