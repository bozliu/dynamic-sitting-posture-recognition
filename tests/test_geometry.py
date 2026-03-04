from __future__ import annotations

import numpy as np

from posture_recognition.features.geometry import (
    KP,
    angle_between_vectors_deg,
    estimate_side_sign,
    joint_angle_deg,
)


def test_angle_between_vectors_deg() -> None:
    assert angle_between_vectors_deg(np.array([1.0, 0.0]), np.array([0.0, 1.0])) == 90.0


def test_joint_angle_deg() -> None:
    a = np.array([0.0, 0.0])
    b = np.array([1.0, 0.0])
    c = np.array([2.0, 0.0])
    assert joint_angle_deg(a, b, c) == 180.0


def test_estimate_side_sign_from_nose_and_ears() -> None:
    keypoints = np.zeros((17, 2), dtype=np.float32)
    scores = np.zeros((17,), dtype=np.float32)

    keypoints[KP["nose"]] = np.array([100.0, 10.0], dtype=np.float32)
    keypoints[KP["left_ear"]] = np.array([80.0, 12.0], dtype=np.float32)
    keypoints[KP["right_ear"]] = np.array([82.0, 9.0], dtype=np.float32)
    scores[KP["nose"]] = 0.9
    scores[KP["left_ear"]] = 0.9
    scores[KP["right_ear"]] = 0.9

    sign = estimate_side_sign(keypoints, scores, 0.3)
    assert sign == 1.0
