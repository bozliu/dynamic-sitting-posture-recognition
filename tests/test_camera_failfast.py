from __future__ import annotations

import pytest

import posture_recognition.cli as cli


class DummyClosedCapture:
    def __init__(self, *args, **kwargs) -> None:
        _ = args, kwargs

    def isOpened(self) -> bool:
        return False

    def release(self) -> None:
        return None

    def read(self):
        return False, None


def test_open_camera_failfast_when_unavailable(monkeypatch) -> None:
    clock = {"t": 0.0}

    def fake_perf_counter() -> float:
        clock["t"] += 0.2
        return clock["t"]

    monkeypatch.setattr(cli.cv2, "VideoCapture", lambda *args, **kwargs: DummyClosedCapture())
    monkeypatch.setattr(cli.time, "perf_counter", fake_perf_counter)
    monkeypatch.setattr(cli.time, "sleep", lambda *_: None)

    with pytest.raises(RuntimeError, match="Unable to open webcam id=0"):
        cli._open_camera_with_failfast(
            camera_id=0,
            camera_backend="auto",
            camera_open_timeout_sec=0.6,
            first_frame_timeout_sec=0.6,
            capture_width=640,
            capture_height=360,
        )


def test_camera_backend_attempts_prefers_avfoundation_on_macos(monkeypatch) -> None:
    monkeypatch.setattr(cli.sys, "platform", "darwin")
    attempts = cli._camera_backend_attempts("auto")
    assert attempts[0][1] == "avfoundation"
