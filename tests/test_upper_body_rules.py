from __future__ import annotations

from posture_recognition.rules.classifier import RuleConfig, classify
from posture_recognition.types import FeatureVector


def make_features(**overrides: float | None) -> FeatureVector:
    base = FeatureVector(
        torso_tilt_deg=8.0,
        forward_displacement_norm=0.0,
        left_knee_angle_deg=None,
        right_knee_angle_deg=None,
        shoulder_width=100.0,
        torso_length=120.0,
        wrist_distance_norm=0.9,
        wrists_to_torso_norm=1.3,
        side_sign=1.0,
        left_hip_ankle_norm=None,
        right_hip_ankle_norm=None,
    )
    for key, value in overrides.items():
        setattr(base, key, value)
    return base


def test_upper_body_classifies_straight() -> None:
    decision = classify(make_features(), person_score=0.9, keypoint_confidence=0.9, rules=RuleConfig(), pose_mode="upper_body")
    assert decision.posture_label == "straight"


def test_upper_body_classifies_hunchback() -> None:
    decision = classify(
        make_features(forward_displacement_norm=0.28),
        person_score=0.9,
        keypoint_confidence=0.9,
        rules=RuleConfig(),
        pose_mode="upper_body",
    )
    assert decision.posture_label == "hunchback"


def test_upper_body_classifies_reclined() -> None:
    decision = classify(
        make_features(forward_displacement_norm=-0.30),
        person_score=0.9,
        keypoint_confidence=0.9,
        rules=RuleConfig(),
        pose_mode="upper_body",
    )
    assert decision.posture_label == "reclined"
