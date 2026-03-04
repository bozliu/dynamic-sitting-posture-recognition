from __future__ import annotations

import numpy as np

from posture_recognition.pose.openpose_torch import OpenPoseTorchBackend, OpenPoseTorchBackendConfig
from posture_recognition.types import PoseInstance


def _dummy_instance() -> PoseInstance:
    return PoseInstance(
        box_xyxy=np.array([10.0, 10.0, 80.0, 120.0], dtype=np.float32),
        score=0.9,
        label=1,
        keypoints_xy=np.zeros((17, 2), dtype=np.float32),
        keypoint_scores=np.ones((17,), dtype=np.float32),
    )


def test_openpose_backend_falls_back_when_load_fails(monkeypatch) -> None:
    backend = OpenPoseTorchBackend(OpenPoseTorchBackendConfig(device="cpu"))

    class DummyFallback:
        def infer(self, image_bgr):
            _ = image_bgr
            return [_dummy_instance()], {"pose_infer_ms": 8.0, "device": "cpu", "warnings": ["fallback backend used"]}

    backend.fallback_backend = DummyFallback()
    monkeypatch.setattr(backend, "_load_model", lambda: (_ for _ in ()).throw(RuntimeError("boom")))

    instances, meta = backend.infer(np.zeros((64, 64, 3), dtype=np.uint8))
    assert len(instances) == 1
    assert meta["device"] == "cpu"
    assert any("fallback" in str(w) for w in meta["warnings"])


def test_openpose_18_to_coco17_mapping_shape() -> None:
    backend = OpenPoseTorchBackend(OpenPoseTorchBackendConfig(device="cpu"))
    person_kps = np.random.rand(18, 3).astype(np.float32)
    xy, score = backend._to_coco17(person_kps)
    assert xy is not None and score is not None
    assert xy.shape == (17, 2)
    assert score.shape == (17,)
