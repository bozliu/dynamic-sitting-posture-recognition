from __future__ import annotations

import numpy as np

from posture_recognition.pipeline import PosturePipeline
from posture_recognition.pose.torchvision_pose import TorchvisionPoseBackend


class DummyRealtimeBackend:
    def infer(self, image_bgr: np.ndarray):
        _ = image_bgr
        return [], {"pose_infer_ms": 1.0, "device": "mps", "warnings": []}


def test_pipeline_uses_requested_accurate_backend() -> None:
    pipeline = PosturePipeline(backend="accurate")
    assert pipeline.backend_name == "accurate"
    assert isinstance(pipeline.pose_backend, TorchvisionPoseBackend)


def test_pipeline_uses_requested_realtime_backend(monkeypatch) -> None:
    monkeypatch.setattr(
        PosturePipeline,
        "_build_realtime_backend",
        staticmethod(lambda **kwargs: DummyRealtimeBackend()),
    )
    pipeline = PosturePipeline(backend="realtime")
    assert pipeline.backend_name == "realtime"
    assert isinstance(pipeline.pose_backend, DummyRealtimeBackend)


def test_pipeline_uses_requested_openpose_backend(monkeypatch) -> None:
    class DummyOpenPoseBackend:
        def infer(self, image_bgr: np.ndarray):
            _ = image_bgr
            return [], {"pose_infer_ms": 1.0, "device": "cpu", "warnings": []}

    monkeypatch.setattr(
        PosturePipeline,
        "_build_openpose_backend",
        staticmethod(lambda **kwargs: DummyOpenPoseBackend()),
    )
    pipeline = PosturePipeline(backend="openpose_torch")
    assert pipeline.backend_name == "openpose_torch"
    assert isinstance(pipeline.pose_backend, DummyOpenPoseBackend)
