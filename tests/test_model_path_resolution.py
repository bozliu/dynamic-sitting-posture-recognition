from __future__ import annotations

from pathlib import Path

from posture_recognition.pipeline import PosturePipeline


def test_resolve_model_path_prefers_env_dir(monkeypatch, tmp_path: Path) -> None:
    models_dir = tmp_path / "models"
    target = models_dir / "realtime" / "yolo11n-pose.pt"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b"demo")

    monkeypatch.setenv("POSTUREMIRROR_MODELS_DIR", str(models_dir))
    resolved = PosturePipeline._resolve_model_path(None, "realtime/yolo11n-pose.pt")
    assert resolved == str(target)
