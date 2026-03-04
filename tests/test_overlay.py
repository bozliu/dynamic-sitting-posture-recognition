from __future__ import annotations

import numpy as np

from posture_recognition.render.overlay import _compute_scales, draw_overlay
from posture_recognition.types import InferenceResult


def _result(posture_label: str) -> InferenceResult:
    return InferenceResult(
        posture_label=posture_label,
        kneeling=False,
        hands_folded=False,
        angles={"torso_tilt_deg": 0.0, "left_knee_deg": 160.0, "right_knee_deg": 160.0},
        confidence={"person": 0.9, "keypoints": 0.8, "decision": 0.85},
        device="cpu",
        timing_ms={"pose_infer": 15.0, "total": 20.0},
        warnings=[],
    )


def test_overlay_green_for_straight_posture() -> None:
    image = np.zeros((720, 1280, 3), dtype=np.uint8)
    overlay = draw_overlay(
        image,
        instance=None,
        result=_result("straight"),
        features=None,
        keypoint_threshold=0.3,
        alert_blink=False,
    )
    pixel = overlay[12, 1260]  # top-right point, avoids status text
    assert int(pixel[1]) > int(pixel[2])  # green channel dominates red


def test_overlay_red_for_incorrect_posture() -> None:
    image = np.zeros((720, 1280, 3), dtype=np.uint8)
    overlay = draw_overlay(
        image,
        instance=None,
        result=_result("hunchback"),
        features=None,
        keypoint_threshold=0.3,
        alert_blink=False,
    )
    pixel = overlay[12, 1260]
    assert int(pixel[2]) > int(pixel[1])  # red channel dominates green


def test_overlay_font_scale_has_minimum_readable_size() -> None:
    title_scale, text_scale, thick, line_h = _compute_scales(1920, 1080, 1.0)
    assert title_scale >= 1.2
    assert text_scale >= 0.8
    assert thick >= 1
    assert line_h >= 24
