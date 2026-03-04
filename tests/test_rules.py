from __future__ import annotations

from posture_recognition.rules.classifier import RuleConfig, classify
from posture_recognition.types import FeatureVector


def make_features(**overrides: float | None) -> FeatureVector:
    base = FeatureVector(
        torso_tilt_deg=10.0,
        forward_displacement_norm=0.0,
        left_knee_angle_deg=150.0,
        right_knee_angle_deg=150.0,
        shoulder_width=100.0,
        torso_length=120.0,
        wrist_distance_norm=0.8,
        wrists_to_torso_norm=1.5,
        side_sign=1.0,
        left_hip_ankle_norm=1.5,
        right_hip_ankle_norm=1.5,
    )
    for key, value in overrides.items():
        setattr(base, key, value)
    return base


def test_classify_straight() -> None:
    rules = RuleConfig()
    decision = classify(make_features(), person_score=0.9, keypoint_confidence=0.9, rules=rules)
    assert decision.posture_label == "straight"


def test_classify_hunchback() -> None:
    rules = RuleConfig()
    decision = classify(
        make_features(forward_displacement_norm=0.30),
        person_score=0.9,
        keypoint_confidence=0.9,
        rules=rules,
    )
    assert decision.posture_label == "hunchback"


def test_classify_reclined() -> None:
    rules = RuleConfig()
    decision = classify(
        make_features(forward_displacement_norm=-0.30),
        person_score=0.9,
        keypoint_confidence=0.9,
        rules=rules,
    )
    assert decision.posture_label == "reclined"


def test_kneeling_detected() -> None:
    rules = RuleConfig()
    decision = classify(
        make_features(left_knee_angle_deg=85.0, left_hip_ankle_norm=0.8),
        person_score=0.9,
        keypoint_confidence=0.9,
        rules=rules,
    )
    assert decision.kneeling is True


def test_folded_hands_detected() -> None:
    rules = RuleConfig()
    decision = classify(
        make_features(wrist_distance_norm=0.3, wrists_to_torso_norm=0.8),
        person_score=0.9,
        keypoint_confidence=0.9,
        rules=rules,
    )
    assert decision.hands_folded is True
