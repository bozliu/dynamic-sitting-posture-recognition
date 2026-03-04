from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from typer.testing import CliRunner

import posture_recognition.cli as cli
from posture_recognition.types import InferenceResult


class DummyPipeline:
    keypoint_threshold = 0.3

    def predict_image_file(self, _input: Path):
        image = np.zeros((64, 64, 3), dtype=np.uint8)
        return _dummy_result(), None, None, image

    def predict(self, _frame, frame_index=None, timestamp_sec=None):
        _ = frame_index, timestamp_sec
        return _dummy_result(), None, None


runner = CliRunner()


def _dummy_result() -> InferenceResult:
    return InferenceResult(
        posture_label="straight",
        kneeling=False,
        hands_folded=False,
        angles={"torso_tilt_deg": 5.0, "left_knee_deg": 160.0, "right_knee_deg": 162.0},
        confidence={"person": 0.9, "keypoints": 0.8, "decision": 0.85},
        device="cpu",
        timing_ms={"pose_infer": 10.0, "total": 12.0},
        warnings=[],
    )


def test_image_command_writes_outputs(tmp_path, monkeypatch) -> None:
    input_path = tmp_path / "input.jpg"
    cv2.imwrite(str(input_path), np.zeros((32, 32, 3), dtype=np.uint8))

    output_image = tmp_path / "overlay.jpg"
    output_json = tmp_path / "result.json"

    monkeypatch.setattr(cli, "_make_pipeline", lambda *args, **kwargs: DummyPipeline())

    result = runner.invoke(
        cli.app,
        [
            "image",
            "--input",
            str(input_path),
            "--output-image",
            str(output_image),
            "--output-json",
            str(output_json),
        ],
    )

    assert result.exit_code == 0
    assert output_image.exists()
    assert output_json.exists()


def test_evaluate_command_writes_report(tmp_path, monkeypatch) -> None:
    report = tmp_path / "eval.json"
    images_dir = tmp_path / "images"
    images_dir.mkdir()

    monkeypatch.setattr(cli, "_make_pipeline", lambda *args, **kwargs: DummyPipeline())
    monkeypatch.setattr(
        cli,
        "run_evaluation",
        lambda pipeline, images_dir: {
            "num_images": 1,
            "posture": {"accuracy": 1.0, "macro_f1": 1.0},
            "kneeling": {"accuracy": 1.0},
            "hands_folded": {"accuracy": 1.0},
            "rows": [],
        },
    )

    result = runner.invoke(
        cli.app,
        [
            "evaluate",
            "--images-dir",
            str(images_dir),
            "--report",
            str(report),
        ],
    )

    assert result.exit_code == 0
    assert report.exists()
