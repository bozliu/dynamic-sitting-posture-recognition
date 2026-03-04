from __future__ import annotations

import time

import cv2
import numpy as np

from posture_recognition.features.geometry import KP
from posture_recognition.types import FeatureVector, InferenceResult, PoseInstance


SKELETON = [
    (KP["nose"], KP["left_eye"]),
    (KP["nose"], KP["right_eye"]),
    (KP["left_eye"], KP["left_ear"]),
    (KP["right_eye"], KP["right_ear"]),
    (KP["left_shoulder"], KP["right_shoulder"]),
    (KP["left_shoulder"], KP["left_elbow"]),
    (KP["left_elbow"], KP["left_wrist"]),
    (KP["right_shoulder"], KP["right_elbow"]),
    (KP["right_elbow"], KP["right_wrist"]),
    (KP["left_shoulder"], KP["left_hip"]),
    (KP["right_shoulder"], KP["right_hip"]),
    (KP["left_hip"], KP["right_hip"]),
    (KP["left_hip"], KP["left_knee"]),
    (KP["left_knee"], KP["left_ankle"]),
    (KP["right_hip"], KP["right_knee"]),
    (KP["right_knee"], KP["right_ankle"]),
]

GREEN = (50, 190, 50)
RED = (40, 50, 230)
WHITE = (245, 245, 245)
BLACK = (20, 20, 20)
SKELETON_COLOR = (255, 210, 70)
KEYPOINT_COLOR = (0, 140, 255)


def draw_overlay(
    image_bgr: np.ndarray,
    instance: PoseInstance | None,
    result: InferenceResult,
    features: FeatureVector | None,
    keypoint_threshold: float,
    alert_blink: bool = True,
    font_scale_base: float = 1.0,
    stream_fps: float | None = None,
) -> np.ndarray:
    canvas = image_bgr.copy()
    height, width = canvas.shape[:2]

    title_scale, text_scale, thick, line_h = _compute_scales(width, height, font_scale_base)

    posture_ok = result.posture_label == "straight"
    theme = GREEN if posture_ok else RED
    result.status_color = "green" if posture_ok else "red"

    border_thickness = max(2, int(4 * (thick / 2)))
    cv2.rectangle(canvas, (2, 2), (width - 3, height - 3), theme, border_thickness)

    if instance is not None:
        x1, y1, x2, y2 = [int(v) for v in instance.box_xyxy]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), theme, max(2, thick))

        for a, b in SKELETON:
            if (
                instance.keypoint_scores[a] >= keypoint_threshold
                and instance.keypoint_scores[b] >= keypoint_threshold
            ):
                p1 = tuple(int(v) for v in instance.keypoints_xy[a])
                p2 = tuple(int(v) for v in instance.keypoints_xy[b])
                cv2.line(canvas, p1, p2, SKELETON_COLOR, max(2, thick))

        radius = max(4, int(4 * text_scale))
        for idx, point in enumerate(instance.keypoints_xy):
            if instance.keypoint_scores[idx] >= keypoint_threshold:
                cv2.circle(canvas, tuple(int(v) for v in point), radius, KEYPOINT_COLOR, -1)

    top_bar_h = max(64, int(height * 0.12))
    top_overlay = canvas.copy()
    cv2.rectangle(top_overlay, (0, 0), (width, top_bar_h), theme, -1)
    alpha = 0.85 if posture_ok else 0.78
    if (not posture_ok) and alert_blink and int(time.time() * 2) % 2 == 0:
        alpha = 0.55
    cv2.addWeighted(top_overlay, alpha, canvas, 1.0 - alpha, 0, canvas)

    status_text = "POSTURE: STRAIGHT (CORRECT)" if posture_ok else "POSTURE NOT CORRECT - SIT UPRIGHT"
    _draw_text(
        canvas,
        status_text,
        (18, int(top_bar_h * 0.70)),
        title_scale,
        WHITE,
        max(2, thick + 1),
    )

    info_lines = [
        f"posture: {result.posture_label}",
        f"mode: {result.mode or 'n/a'}",
        f"kneeling: {result.kneeling}",
        f"hands_folded: {result.hands_folded}",
        f"device: {result.device}",
        f"infer(ms): {result.timing_ms['pose_infer']:.1f}",
        f"total(ms): {result.timing_ms['total']:.1f}",
    ]

    if result.result_age_ms is not None:
        info_lines.append(f"result_age(ms): {result.result_age_ms:.1f}")
    if stream_fps is not None:
        info_lines.append(f"display_fps: {stream_fps:.2f}")
    if result.stream_stats is not None:
        dropped = result.stream_stats.get("dropped_frames")
        infer_n = result.stream_stats.get("infer_every_n")
        if dropped is not None:
            info_lines.append(f"dropped_frames: {dropped}")
        if infer_n is not None:
            info_lines.append(f"infer_every_n: {infer_n}")
    if result.warnings:
        info_lines.append(f"status: {result.warnings[0]}")

    if features is not None:
        info_lines.append(f"torso_tilt_deg: {_fmt(features.torso_tilt_deg)}")
        info_lines.append(f"forward_norm: {_fmt(features.forward_displacement_norm)}")
        info_lines.append(f"left_knee_deg: {_fmt(features.left_knee_angle_deg)}")
        info_lines.append(f"right_knee_deg: {_fmt(features.right_knee_angle_deg)}")

    card_x = 16
    card_y = top_bar_h + 14
    card_w = max(260, int(width * 0.42))
    card_h = line_h * len(info_lines) + 22

    _draw_card(canvas, card_x, card_y, card_w, card_h)
    text_y = card_y + line_h
    for line in info_lines:
        _draw_text(canvas, line, (card_x + 10, text_y), text_scale, WHITE, max(1, thick))
        text_y += line_h

    return canvas


def _compute_scales(width: int, height: int, font_scale_base: float) -> tuple[float, float, int, int]:
    ref = min(width, height) / 720.0
    text_scale = max(0.80, 0.85 * ref * font_scale_base)
    title_scale = max(1.20, 1.40 * ref * font_scale_base)
    thick = max(1, int(round(1.6 * ref)))
    line_h = max(24, int(28 * ref * font_scale_base))
    return title_scale, text_scale, thick, line_h


def _draw_text(
    image: np.ndarray,
    text: str,
    origin: tuple[int, int],
    scale: float,
    color: tuple[int, int, int],
    thickness: int,
) -> None:
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, BLACK, thickness + 2, cv2.LINE_AA)
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def _draw_card(image: np.ndarray, x: int, y: int, w: int, h: int) -> None:
    overlay = image.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), (18, 18, 18), -1)
    cv2.addWeighted(overlay, 0.72, image, 0.28, 0, image)
    cv2.rectangle(image, (x, y), (x + w, y + h), (140, 140, 140), 1)


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"
