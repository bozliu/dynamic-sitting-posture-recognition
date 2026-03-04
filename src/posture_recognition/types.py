from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


@dataclass
class PoseInstance:
    box_xyxy: np.ndarray
    score: float
    label: int
    keypoints_xy: np.ndarray
    keypoint_scores: np.ndarray


@dataclass
class FeatureVector:
    torso_tilt_deg: float | None
    forward_displacement_norm: float | None
    left_knee_angle_deg: float | None
    right_knee_angle_deg: float | None
    shoulder_width: float | None
    torso_length: float | None
    wrist_distance_norm: float | None
    wrists_to_torso_norm: float | None
    side_sign: float | None
    left_hip_ankle_norm: float | None
    right_hip_ankle_norm: float | None


@dataclass
class PostureDecision:
    posture_label: str
    kneeling: bool
    hands_folded: bool
    confidence_person: float
    confidence_keypoints: float
    confidence_decision: float


@dataclass
class InferenceResult:
    posture_label: str
    kneeling: bool
    hands_folded: bool
    angles: dict[str, float | None]
    confidence: dict[str, float]
    device: str
    timing_ms: dict[str, float]
    warnings: list[str]
    status_color: str | None = None
    result_age_ms: float | None = None
    stream_fps: float | None = None
    mode: str | None = None
    stream_stats: dict[str, float | int | None] | None = None
    frame_index: int | None = None
    timestamp_sec: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
