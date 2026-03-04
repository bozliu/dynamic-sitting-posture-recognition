from __future__ import annotations

import numpy as np

from posture_recognition.pipeline import PosturePipeline
from posture_recognition.types import PoseInstance


def _dummy_person_with_knees() -> PoseInstance:
    keypoints_xy = np.zeros((17, 2), dtype=np.float32)
    keypoint_scores = np.ones((17,), dtype=np.float32)
    # Build a shape that would normally trigger kneeling in full-body mode.
    keypoints_xy[5] = np.array([100.0, 120.0])  # left_shoulder
    keypoints_xy[6] = np.array([120.0, 120.0])  # right_shoulder
    keypoints_xy[11] = np.array([100.0, 170.0])  # left_hip
    keypoints_xy[12] = np.array([122.0, 170.0])  # right_hip
    keypoints_xy[13] = np.array([95.0, 185.0])  # left_knee
    keypoints_xy[15] = np.array([90.0, 190.0])  # left_ankle
    keypoints_xy[0] = np.array([110.0, 100.0])  # nose
    return PoseInstance(
        box_xyxy=np.array([60.0, 60.0, 180.0, 230.0], dtype=np.float32),
        score=0.95,
        label=1,
        keypoints_xy=keypoints_xy,
        keypoint_scores=keypoint_scores,
    )


class DummyBackend:
    def infer(self, image_bgr):
        _ = image_bgr
        return [_dummy_person_with_knees()], {"pose_infer_ms": 1.0, "device": "cpu", "warnings": []}


def test_upper_body_mode_forces_kneeling_false(monkeypatch) -> None:
    monkeypatch.setattr(PosturePipeline, "_build_realtime_backend", staticmethod(lambda **kwargs: DummyBackend()))
    pipeline = PosturePipeline(backend="realtime", pose_mode="upper_body")
    result, _, _ = pipeline.predict(np.zeros((240, 320, 3), dtype=np.uint8))
    assert result.mode == "upper_body"
    assert result.kneeling is False
    assert any("kneeling disabled in upper-body mode" in w for w in result.warnings)
