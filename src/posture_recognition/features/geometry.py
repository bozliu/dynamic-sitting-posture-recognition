from __future__ import annotations

import math
from typing import Iterable

import numpy as np


COCO_KEYPOINT_NAMES = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]

KP = {name: idx for idx, name in enumerate(COCO_KEYPOINT_NAMES)}


def point_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def midpoint(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (a + b) / 2.0


def angle_between_vectors_deg(v1: np.ndarray, v2: np.ndarray) -> float | None:
    n1 = float(np.linalg.norm(v1))
    n2 = float(np.linalg.norm(v2))
    if n1 == 0.0 or n2 == 0.0:
        return None
    cos_theta = float(np.dot(v1, v2) / (n1 * n2))
    cos_theta = max(-1.0, min(1.0, cos_theta))
    return float(math.degrees(math.acos(cos_theta)))


def joint_angle_deg(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float | None:
    v1 = a - b
    v2 = c - b
    return angle_between_vectors_deg(v1, v2)


def safe_mean(values: Iterable[float | None]) -> float | None:
    clean = [v for v in values if v is not None]
    if not clean:
        return None
    return float(sum(clean) / len(clean))


def keypoint_available(scores: np.ndarray, index: int, threshold: float) -> bool:
    return bool(index < len(scores) and scores[index] >= threshold)


def estimate_side_sign(keypoints: np.ndarray, scores: np.ndarray, threshold: float) -> float | None:
    nose_idx = KP["nose"]
    left_ear_idx = KP["left_ear"]
    right_ear_idx = KP["right_ear"]
    left_eye_idx = KP["left_eye"]
    right_eye_idx = KP["right_eye"]

    if keypoint_available(scores, nose_idx, threshold):
        nose_x = float(keypoints[nose_idx][0])

        ear_points = []
        if keypoint_available(scores, left_ear_idx, threshold):
            ear_points.append(float(keypoints[left_ear_idx][0]))
        if keypoint_available(scores, right_ear_idx, threshold):
            ear_points.append(float(keypoints[right_ear_idx][0]))

        if ear_points:
            ear_mid_x = float(sum(ear_points) / len(ear_points))
            return 1.0 if nose_x >= ear_mid_x else -1.0

        eye_points = []
        if keypoint_available(scores, left_eye_idx, threshold):
            eye_points.append(float(keypoints[left_eye_idx][0]))
        if keypoint_available(scores, right_eye_idx, threshold):
            eye_points.append(float(keypoints[right_eye_idx][0]))

        if eye_points:
            eye_mid_x = float(sum(eye_points) / len(eye_points))
            return 1.0 if nose_x >= eye_mid_x else -1.0

    return None
