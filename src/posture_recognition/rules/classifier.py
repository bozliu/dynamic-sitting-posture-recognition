from __future__ import annotations

from dataclasses import dataclass

from posture_recognition.types import FeatureVector, PostureDecision


@dataclass
class RuleConfig:
    hunch_forward_min: float = 0.16
    recline_forward_min: float = 0.16
    straight_forward_max: float = 0.10
    straight_tilt_max_deg: float = 18.0
    kneeling_knee_angle_max_deg: float = 115.0
    kneeling_hip_ankle_max_norm: float = 0.90
    hands_wrist_distance_max_norm: float = 0.55
    hands_wrist_to_torso_max_norm: float = 1.10
    min_keypoint_confidence: float = 0.25


def classify(
    features: FeatureVector,
    person_score: float,
    keypoint_confidence: float,
    rules: RuleConfig,
    pose_mode: str = "full_body",
) -> PostureDecision:
    posture_label = "unknown"

    if (
        features.forward_displacement_norm is not None
        and features.torso_tilt_deg is not None
        and keypoint_confidence >= rules.min_keypoint_confidence
    ):
        forward = features.forward_displacement_norm
        torso_tilt = features.torso_tilt_deg

        if forward >= rules.hunch_forward_min:
            posture_label = "hunchback"
        elif forward <= -rules.recline_forward_min:
            posture_label = "reclined"
        elif abs(forward) <= rules.straight_forward_max and torso_tilt <= rules.straight_tilt_max_deg:
            posture_label = "straight"

    kneeling = False if pose_mode == "upper_body" else _detect_kneeling(features, keypoint_confidence, rules)
    hands_folded = _detect_hands_folded(features, keypoint_confidence, rules)
    confidence_decision = max(0.0, min(1.0, 0.55 * person_score + 0.45 * keypoint_confidence))

    return PostureDecision(
        posture_label=posture_label,
        kneeling=kneeling,
        hands_folded=hands_folded,
        confidence_person=float(person_score),
        confidence_keypoints=float(keypoint_confidence),
        confidence_decision=float(confidence_decision),
    )


def _detect_kneeling(features: FeatureVector, keypoint_confidence: float, rules: RuleConfig) -> bool:
    if keypoint_confidence < rules.min_keypoint_confidence:
        return False

    conditions = []
    if features.left_knee_angle_deg is not None:
        conditions.append(
            features.left_knee_angle_deg <= rules.kneeling_knee_angle_max_deg
            and (
                features.left_hip_ankle_norm is None
                or features.left_hip_ankle_norm <= rules.kneeling_hip_ankle_max_norm
            )
        )
    if features.right_knee_angle_deg is not None:
        conditions.append(
            features.right_knee_angle_deg <= rules.kneeling_knee_angle_max_deg
            and (
                features.right_hip_ankle_norm is None
                or features.right_hip_ankle_norm <= rules.kneeling_hip_ankle_max_norm
            )
        )

    return any(conditions)


def _detect_hands_folded(features: FeatureVector, keypoint_confidence: float, rules: RuleConfig) -> bool:
    if keypoint_confidence < rules.min_keypoint_confidence:
        return False
    if features.wrist_distance_norm is None or features.wrists_to_torso_norm is None:
        return False

    return (
        features.wrist_distance_norm <= rules.hands_wrist_distance_max_norm
        and features.wrists_to_torso_norm <= rules.hands_wrist_to_torso_max_norm
    )
