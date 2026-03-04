from __future__ import annotations

import json
import sys
import threading
import time
from collections import deque
from dataclasses import replace
from pathlib import Path
from typing import Optional

import cv2
import typer
import yaml
from rich import print
from tqdm import tqdm

from posture_recognition.experiments import run_benchmark, run_calibration, run_evaluation
from posture_recognition.pipeline import PosturePipeline, write_json, write_jsonl
from posture_recognition.render.overlay import draw_overlay
from posture_recognition.runtime.webcam_runtime import (
    WebcamRuntimeController,
    WebcamRuntimeOptions,
    camera_backend_attempts,
    open_camera_with_failfast,
    waiting_result,
)
from posture_recognition.types import InferenceResult


app = typer.Typer(help="PyTorch-based sitting posture recognition toolkit")


def _make_pipeline(
    config: Path,
    calibration: Optional[Path],
    device: str,
    backend: Optional[str] = None,
    pose_mode: Optional[str] = None,
    infer_max_dim: Optional[int] = None,
    infer_every_n: Optional[int] = None,
) -> PosturePipeline:
    return PosturePipeline(
        config_path=config,
        calibration_path=calibration,
        device=device,
        backend=backend,
        pose_mode=pose_mode,
        infer_max_dim=infer_max_dim,
        infer_every_n=infer_every_n,
    )


@app.command()
def image(
    input: Path = typer.Option(..., exists=True, readable=True, help="Input image path"),
    output_image: Path = typer.Option(..., help="Path to save annotated image"),
    output_json: Path = typer.Option(..., help="Path to save JSON result"),
    config: Path = typer.Option(Path("configs/default.yaml"), help="Config YAML"),
    calibration: Path = typer.Option(Path("configs/calibration.yaml"), help="Calibration YAML"),
    device: str = typer.Option("auto", help="auto|mps|cpu"),
) -> None:
    pipeline = _make_pipeline(config, calibration, device)
    ui_alert_blink = bool(getattr(pipeline, "ui_alert_blink", True))
    ui_font_scale_base = float(getattr(pipeline, "ui_font_scale_base", 1.0))
    result, features, instance, image_bgr = pipeline.predict_image_file(input)
    overlay = draw_overlay(
        image_bgr,
        instance,
        result,
        features,
        pipeline.keypoint_threshold,
        alert_blink=ui_alert_blink,
        font_scale_base=ui_font_scale_base,
    )

    output_image.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_image), overlay)
    write_json(output_json, result.to_dict())

    print(f"[green]Saved image:[/green] {output_image}")
    print(f"[green]Saved json:[/green] {output_json}")
    print(json.dumps(result.to_dict(), indent=2))


@app.command()
def video(
    input: Path = typer.Option(..., exists=True, readable=True, help="Input video path"),
    output_video: Path = typer.Option(..., help="Output annotated video path"),
    output_jsonl: Path = typer.Option(..., help="Output JSONL predictions"),
    config: Path = typer.Option(Path("configs/default.yaml"), help="Config YAML"),
    calibration: Path = typer.Option(Path("configs/calibration.yaml"), help="Calibration YAML"),
    device: str = typer.Option("auto", help="auto|mps|cpu"),
) -> None:
    pipeline = _make_pipeline(config, calibration, device)
    ui_alert_blink = bool(getattr(pipeline, "ui_alert_blink", True))
    ui_font_scale_base = float(getattr(pipeline, "ui_font_scale_base", 1.0))

    cap = cv2.VideoCapture(str(input))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {input}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    output_video.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_video),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    rows = []
    for frame_idx in tqdm(range(frame_count if frame_count > 0 else 0), desc="Processing video", disable=frame_count <= 0):
        ret, frame = cap.read()
        if not ret:
            break

        timestamp = frame_idx / fps if fps > 0 else None
        result, features, instance = pipeline.predict(frame, frame_index=frame_idx, timestamp_sec=timestamp)
        overlay = draw_overlay(
            frame,
            instance,
            result,
            features,
            pipeline.keypoint_threshold,
            alert_blink=ui_alert_blink,
            font_scale_base=ui_font_scale_base,
        )
        writer.write(overlay)
        rows.append(result.to_dict())

    if frame_count <= 0:
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            timestamp = frame_idx / fps if fps > 0 else None
            result, features, instance = pipeline.predict(frame, frame_index=frame_idx, timestamp_sec=timestamp)
            overlay = draw_overlay(
                frame,
                instance,
                result,
                features,
                pipeline.keypoint_threshold,
                alert_blink=ui_alert_blink,
                font_scale_base=ui_font_scale_base,
            )
            writer.write(overlay)
            rows.append(result.to_dict())
            frame_idx += 1

    cap.release()
    writer.release()

    write_jsonl(output_jsonl, rows)
    print(f"[green]Saved video:[/green] {output_video}")
    print(f"[green]Saved jsonl:[/green] {output_jsonl}")


@app.command()
def webcam(
    camera_id: int = typer.Option(0, help="Webcam device id"),
    display: bool = typer.Option(True, "--display/--no-display", help="Display annotated stream"),
    display_backend: Optional[str] = typer.Option(None, help="auto|opencv|tk"),
    output_jsonl: Optional[Path] = typer.Option(None, help="Optional JSONL path"),
    max_frames: Optional[int] = typer.Option(None, help="Optional frame limit"),
    backend: str = typer.Option("realtime", help="realtime|accurate|openpose_torch"),
    pose_mode: Optional[str] = typer.Option(None, help="upper_body|full_body"),
    camera_backend: Optional[str] = typer.Option(None, help="avfoundation|auto"),
    camera_open_timeout_sec: Optional[float] = typer.Option(None, min=0.5, help="Camera open timeout in seconds"),
    first_frame_timeout_sec: Optional[float] = typer.Option(None, min=0.5, help="First-frame timeout in seconds"),
    read_fail_max: Optional[int] = typer.Option(None, min=1, help="Max consecutive read failures before fail-fast"),
    capture_width: int = typer.Option(960, min=0, help="Webcam capture width; 0 keeps driver default"),
    capture_height: int = typer.Option(540, min=0, help="Webcam capture height; 0 keeps driver default"),
    infer_max_dim: Optional[int] = typer.Option(None, min=128, help="Override max inference image dimension"),
    infer_every_n: Optional[int] = typer.Option(None, min=1, help="Run inference every N frames"),
    alert_blink: bool = typer.Option(True, "--alert-blink/--no-alert-blink", help="Blink red alert for incorrect posture"),
    config: Path = typer.Option(Path("configs/default.yaml"), help="Config YAML"),
    calibration: Path = typer.Option(Path("configs/calibration.yaml"), help="Calibration YAML"),
    device: str = typer.Option("auto", help="auto|mps|cpu"),
) -> None:
    pipeline = _make_pipeline(
        config=config,
        calibration=calibration,
        device=device,
        backend=backend,
        pose_mode=pose_mode,
        infer_max_dim=infer_max_dim,
        infer_every_n=infer_every_n,
    )
    ui_alert_blink = bool(getattr(pipeline, "ui_alert_blink", True))
    ui_font_scale_base = float(getattr(pipeline, "ui_font_scale_base", 1.0))
    ui_cfg = pipeline.config.get("ui", {})
    display_backend = str(display_backend or ui_cfg.get("display_backend", "auto")).lower().strip()
    camera_cfg = pipeline.config.get("inference", {}).get("camera", {})
    camera_backend = str(camera_backend or camera_cfg.get("backend", "auto"))
    camera_open_timeout_sec = float(camera_open_timeout_sec or camera_cfg.get("open_timeout_sec", 2.0))
    first_frame_timeout_sec = float(first_frame_timeout_sec or camera_cfg.get("first_frame_timeout_sec", 2.0))
    read_fail_max = int(read_fail_max or camera_cfg.get("read_fail_max", 120))

    options = WebcamRuntimeOptions(
        camera_id=camera_id,
        max_frames=max_frames,
        camera_backend=camera_backend,
        camera_open_timeout_sec=camera_open_timeout_sec,
        first_frame_timeout_sec=first_frame_timeout_sec,
        read_fail_max=read_fail_max,
        capture_width=capture_width,
        capture_height=capture_height,
        alert_blink=alert_blink,
        ui_alert_blink=ui_alert_blink,
        ui_font_scale_base=ui_font_scale_base,
        display_backend=display_backend,
        device_name=device,
        pose_mode=str(getattr(pipeline, "pose_mode", pose_mode or "upper_body")),
    )
    runtime = WebcamRuntimeController(pipeline, options)
    try:
        runtime.start()
    except RuntimeError as exc:
        print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1)

    window_name = "posture-recognition"
    active_display_backend = "none"
    try:
        if display:
            want_tk = display_backend == "tk" or (display_backend == "auto" and sys.platform == "darwin")
            tk_ready = False
            tk_state: dict[str, object] = {}
            if want_tk:
                try:
                    import tkinter as tk
                    from PIL import Image, ImageTk

                    tk_state = {"tk": tk, "Image": Image, "ImageTk": ImageTk}
                    tk_ready = True
                except Exception:
                    tk_ready = False

            if tk_ready:
                active_display_backend = "tk"
                tk = tk_state["tk"]  # type: ignore[assignment]
                Image = tk_state["Image"]  # type: ignore[assignment]
                ImageTk = tk_state["ImageTk"]  # type: ignore[assignment]
                root = tk.Tk()
                root.title(window_name)
                panel = tk.Label(root)
                panel.pack(fill="both", expand=True)
                closed = {"value": False}

                def on_close() -> None:
                    if closed["value"]:
                        return
                    closed["value"] = True
                    runtime.stop()
                    runtime.close()
                    try:
                        root.destroy()
                    except Exception:
                        pass

                root.protocol("WM_DELETE_WINDOW", on_close)
                root.bind("<Escape>", lambda _evt: on_close())
                root.bind("<KeyPress-q>", lambda _evt: on_close())

                def tick() -> None:
                    if closed["value"]:
                        return
                    overlay, _result = runtime.build_overlay()
                    if overlay is not None:
                        rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
                        photo = ImageTk.PhotoImage(Image.fromarray(rgb))
                        panel.configure(image=photo)
                        panel.image = photo
                    if not runtime.is_running():
                        on_close()
                        return
                    root.after(10, tick)

                root.after(0, tick)
                root.mainloop()
            else:
                active_display_backend = "opencv"
                cv2.namedWindow(window_name, cv2.WINDOW_NORMAL | int(getattr(cv2, "WINDOW_GUI_NORMAL", 0)))
                while runtime.is_running():
                    try:
                        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                            runtime.stop()
                            break
                    except cv2.error:
                        runtime.stop()
                        break

                    overlay, _result = runtime.build_overlay()
                    if overlay is None:
                        time.sleep(0.003)
                        continue

                    cv2.imshow(window_name, overlay)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord("q"), 27):
                        runtime.stop()
                        break
        else:
            while runtime.is_running():
                time.sleep(0.01)
    finally:
        runtime.close()
        if display and active_display_backend == "opencv":
            try:
                cv2.destroyWindow(window_name)
            except cv2.error:
                cv2.destroyAllWindows()

    if output_jsonl is not None:
        write_jsonl(output_jsonl, runtime.rows_payload())
        print(f"[green]Saved jsonl:[/green] {output_jsonl}")


def _waiting_result(
    device: str,
    mode: str = "upper_body",
    warnings: Optional[list[str]] = None,
    stream_stats: Optional[dict[str, float | int | None]] = None,
) -> InferenceResult:
    return waiting_result(
        device=device,
        mode=mode,
        warnings=warnings,
        stream_stats=stream_stats,
    )


def _open_camera_with_failfast(
    camera_id: int,
    camera_backend: str,
    camera_open_timeout_sec: float,
    first_frame_timeout_sec: float,
    capture_width: int,
    capture_height: int,
) -> tuple[cv2.VideoCapture, Optional[object], str]:
    return open_camera_with_failfast(
        camera_id=camera_id,
        camera_backend=camera_backend,
        camera_open_timeout_sec=camera_open_timeout_sec,
        first_frame_timeout_sec=first_frame_timeout_sec,
        capture_width=capture_width,
        capture_height=capture_height,
    )


def _camera_backend_attempts(camera_backend: str) -> list[tuple[int, str]]:
    return camera_backend_attempts(camera_backend)


@app.command()
def evaluate(
    images_dir: Path = typer.Option(Path("sample_images"), exists=True, file_okay=False, help="Evaluation image directory"),
    report: Path = typer.Option(Path("artifacts/eval.json"), help="Evaluation report output path"),
    config: Path = typer.Option(Path("configs/default.yaml"), help="Config YAML"),
    calibration: Path = typer.Option(Path("configs/calibration.yaml"), help="Calibration YAML"),
    device: str = typer.Option("auto", help="auto|mps|cpu"),
) -> None:
    pipeline = _make_pipeline(config, calibration, device)
    eval_report = run_evaluation(pipeline, images_dir)
    write_json(report, eval_report)
    print(f"[green]Saved evaluation report:[/green] {report}")
    print(json.dumps({k: v for k, v in eval_report.items() if k != "rows"}, indent=2))


@app.command()
def benchmark(
    images_dir: Path = typer.Option(Path("sample_images"), exists=True, file_okay=False, help="Image directory used as frame source"),
    num_frames: int = typer.Option(300, min=1, help="Number of frames to benchmark"),
    report: Path = typer.Option(Path("artifacts/benchmark.json"), help="Benchmark report output path"),
    config: Path = typer.Option(Path("configs/default.yaml"), help="Config YAML"),
    calibration: Path = typer.Option(Path("configs/calibration.yaml"), help="Calibration YAML"),
    device: str = typer.Option("auto", help="auto|mps|cpu"),
) -> None:
    pipeline = _make_pipeline(config, calibration, device)
    report_data = run_benchmark(pipeline, images_dir, num_frames=num_frames)
    write_json(report, report_data)
    print(f"[green]Saved benchmark report:[/green] {report}")
    print(json.dumps({k: v for k, v in report_data.items() if k != "rows"}, indent=2))


@app.command()
def calibrate(
    images_dir: Path = typer.Option(Path("sample_images"), exists=True, file_okay=False, help="Labeled images directory"),
    write_config: Path = typer.Option(Path("configs/calibration.yaml"), help="Calibration YAML output"),
    report: Path = typer.Option(Path("artifacts/calibration_report.json"), help="Calibration report output path"),
    config: Path = typer.Option(Path("configs/default.yaml"), help="Config YAML"),
    calibration: Path = typer.Option(Path("configs/calibration.yaml"), help="Calibration YAML"),
    device: str = typer.Option("auto", help="auto|mps|cpu"),
) -> None:
    pipeline = _make_pipeline(config, calibration, device)
    calibration_report = run_calibration(pipeline, images_dir)

    write_json(report, calibration_report)
    if "recommended_rules" in calibration_report:
        payload = {"rules": calibration_report["recommended_rules"]}
        write_config.parent.mkdir(parents=True, exist_ok=True)
        write_config.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
        print(f"[green]Saved calibration config:[/green] {write_config}")

    print(f"[green]Saved calibration report:[/green] {report}")
    print(json.dumps({k: v for k, v in calibration_report.items() if k != "folds"}, indent=2))


if __name__ == "__main__":
    app()
