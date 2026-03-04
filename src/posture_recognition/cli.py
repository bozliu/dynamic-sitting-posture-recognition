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
from posture_recognition.realtime import LatestFrameBuffer
from posture_recognition.render.overlay import draw_overlay
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

    startup_stage = {"value": "camera_opening"}
    cap, first_frame, backend_name = _open_camera_with_failfast(
        camera_id=camera_id,
        camera_backend=camera_backend,
        camera_open_timeout_sec=camera_open_timeout_sec,
        first_frame_timeout_sec=first_frame_timeout_sec,
        capture_width=capture_width,
        capture_height=capture_height,
    )
    startup_stage["value"] = "model_warming"
    cap_ref = {"cap": cap}
    cap_lock = threading.Lock()
    camera_backend_ref = {"name": backend_name}

    frame_buffer = LatestFrameBuffer()
    if first_frame is not None:
        frame_buffer.push(first_frame, frame_index=0, captured_at=time.perf_counter())
    stop_event = threading.Event()
    rows: list[dict] = []
    rows_lock = threading.Lock()
    latest_prediction_lock = threading.Lock()
    stream_stats_lock = threading.Lock()
    runtime_state_lock = threading.Lock()
    dropped_frames = {"count": 0}
    runtime_state: dict[str, str | None] = {"warning": None}
    latest_prediction = {
        "result": None,
        "features": None,
        "instance": None,
        "inferred_at": 0.0,
        "source_frame_index": -1,
    }

    def capture_worker() -> None:
        frame_idx = 1 if first_frame is not None else 0
        consecutive_read_failures = 0
        reconnect_attempts = 0
        next_reconnect_ts = 0.0
        try:
            while not stop_event.is_set():
                with cap_lock:
                    active_cap = cap_ref["cap"]
                    ret, frame = active_cap.read()

                if not ret:
                    consecutive_read_failures += 1
                    if consecutive_read_failures >= read_fail_max and time.perf_counter() >= next_reconnect_ts:
                        startup_stage["value"] = "camera_recovering"
                        reconnect_attempts += 1
                        try:
                            new_cap, recovered_frame, recovered_backend = _open_camera_with_failfast(
                                camera_id=camera_id,
                                camera_backend=camera_backend,
                                camera_open_timeout_sec=max(1.0, camera_open_timeout_sec),
                                first_frame_timeout_sec=max(1.0, first_frame_timeout_sec),
                                capture_width=capture_width,
                                capture_height=capture_height,
                            )
                            with cap_lock:
                                old_cap = cap_ref["cap"]
                                cap_ref["cap"] = new_cap
                            old_cap.release()
                            camera_backend_ref["name"] = recovered_backend
                            if recovered_frame is not None:
                                captured_at = time.perf_counter()
                                frame_buffer.push(recovered_frame, frame_idx, captured_at)
                                frame_idx += 1
                            consecutive_read_failures = 0
                            next_reconnect_ts = 0.0
                            startup_stage["value"] = "live"
                            with runtime_state_lock:
                                runtime_state["warning"] = (
                                    f"camera stream recovered after reconnect attempt #{reconnect_attempts}"
                                )
                        except Exception as exc:
                            next_reconnect_ts = time.perf_counter() + 1.0
                            with runtime_state_lock:
                                runtime_state["warning"] = f"camera reconnect failed (attempt #{reconnect_attempts}): {exc}"
                    time.sleep(0.01)
                    continue

                if consecutive_read_failures >= read_fail_max:
                    with runtime_state_lock:
                        runtime_state["warning"] = "camera stream recovered"
                consecutive_read_failures = 0
                captured_at = time.perf_counter()
                frame_buffer.push(frame, frame_idx, captured_at)
                frame_idx += 1
                if max_frames is not None and frame_idx >= max_frames:
                    break
        except Exception as exc:
            startup_stage["value"] = "capture_exception"
            with runtime_state_lock:
                runtime_state["warning"] = f"capture worker exception: {exc}"
        finally:
            frame_buffer.mark_done()

    def infer_worker() -> None:
        last_processed = -1
        infer_n = max(1, pipeline.infer_every_n)

        while True:
            packet = frame_buffer.snapshot(copy_frame=True)
            if packet.frame_index <= last_processed:
                if packet.done:
                    break
                time.sleep(0.003)
                continue

            skipped = packet.frame_index - last_processed - 1
            if skipped > 0:
                with stream_stats_lock:
                    dropped_frames["count"] += skipped

            last_processed = packet.frame_index
            if packet.frame is None:
                if packet.done:
                    break
                continue
            if packet.frame_index % infer_n != 0:
                if packet.done and packet.frame_index == last_processed:
                    continue
                continue

            try:
                result, features, instance = pipeline.predict(packet.frame, frame_index=packet.frame_index)
            except Exception as exc:
                startup_stage["value"] = "inference_recovering"
                with runtime_state_lock:
                    runtime_state["warning"] = f"inference error, retrying: {exc}"
                time.sleep(0.01)
                continue

            inferred_at = time.perf_counter()
            result.status_color = "green" if result.posture_label == "straight" else "red"
            result.result_age_ms = max(0.0, (inferred_at - packet.captured_at) * 1000.0)
            result.mode = getattr(pipeline, "pose_mode", pose_mode)
            with runtime_state_lock:
                runtime_warning = runtime_state.get("warning")
            if runtime_warning:
                result.warnings = list(result.warnings) + [runtime_warning]
            with stream_stats_lock:
                dropped_count = int(dropped_frames["count"])
            result.stream_stats = {
                "display_fps": None,
                "dropped_frames": dropped_count,
                "infer_every_n": infer_n,
                "camera_backend": camera_backend_ref["name"],
            }
            startup_stage["value"] = "live"

            with latest_prediction_lock:
                latest_prediction["result"] = result
                latest_prediction["features"] = features
                latest_prediction["instance"] = instance
                latest_prediction["inferred_at"] = inferred_at
                latest_prediction["source_frame_index"] = packet.frame_index

            with rows_lock:
                rows.append(result.to_dict())

    capture_thread = threading.Thread(target=capture_worker, daemon=True)
    infer_thread = threading.Thread(target=infer_worker, daemon=True)
    capture_thread.start()
    infer_thread.start()

    fps_window: deque[float] = deque(maxlen=30)
    previous_display_ts: float | None = None
    window_name = "posture-recognition"
    active_display_backend = "none"

    def compute_stream_fps(now: float) -> float | None:
        nonlocal previous_display_ts
        if previous_display_ts is not None:
            delta = now - previous_display_ts
            if delta > 0:
                fps_window.append(1.0 / delta)
        previous_display_ts = now
        return (sum(fps_window) / len(fps_window)) if fps_window else None

    def build_overlay(packet, stream_fps: float | None, now: float):
        with latest_prediction_lock:
            result = latest_prediction["result"]
            features = latest_prediction["features"]
            instance = latest_prediction["instance"]
            inferred_at = latest_prediction["inferred_at"]

        with runtime_state_lock:
            runtime_warning = runtime_state.get("warning")

        if result is None:
            with stream_stats_lock:
                dropped_count = int(dropped_frames["count"])
            waiting_warnings = [f"startup_stage={startup_stage['value']}"]
            if runtime_warning:
                waiting_warnings.append(runtime_warning)
            waiting = _waiting_result(
                device=str(getattr(pipeline.pose_backend, "device", device)),
                mode=getattr(pipeline, "pose_mode", pose_mode),
                warnings=waiting_warnings,
                stream_stats={
                    "display_fps": stream_fps,
                    "dropped_frames": dropped_count,
                    "infer_every_n": max(1, pipeline.infer_every_n),
                    "camera_backend": camera_backend_ref["name"],
                },
            )
            waiting.stream_fps = stream_fps
            return draw_overlay(
                packet.frame,
                None,
                waiting,
                None,
                pipeline.keypoint_threshold,
                alert_blink=alert_blink and ui_alert_blink,
                font_scale_base=ui_font_scale_base,
                stream_fps=stream_fps,
            )

        display_result = replace(result)
        display_result.result_age_ms = max(0.0, (now - float(inferred_at)) * 1000.0)
        display_result.stream_fps = stream_fps
        stream_stats = dict(display_result.stream_stats or {})
        stream_stats["display_fps"] = stream_fps
        with stream_stats_lock:
            stream_stats["dropped_frames"] = int(dropped_frames["count"])
        display_result.stream_stats = stream_stats
        if runtime_warning:
            display_result.warnings = list(display_result.warnings) + [runtime_warning]
        return draw_overlay(
            packet.frame,
            instance,
            display_result,
            features,
            pipeline.keypoint_threshold,
            alert_blink=alert_blink and ui_alert_blink,
            font_scale_base=ui_font_scale_base,
            stream_fps=stream_fps,
        )

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
            except Exception as exc:
                with runtime_state_lock:
                    runtime_state["warning"] = f"tk display unavailable, fallback to opencv: {exc}"

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
                stop_event.set()
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
                packet = frame_buffer.snapshot(copy_frame=True)
                if packet.frame is not None:
                    now = time.perf_counter()
                    stream_fps = compute_stream_fps(now)
                    overlay = build_overlay(packet, stream_fps, now)
                    rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
                    photo = ImageTk.PhotoImage(Image.fromarray(rgb))
                    panel.configure(image=photo)
                    panel.image = photo
                if not (capture_thread.is_alive() or infer_thread.is_alive()):
                    on_close()
                    return
                root.after(10, tick)

            root.after(0, tick)
            root.mainloop()
        else:
            active_display_backend = "opencv"
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL | int(getattr(cv2, "WINDOW_GUI_NORMAL", 0)))
            while capture_thread.is_alive() or infer_thread.is_alive():
                try:
                    if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                        stop_event.set()
                        break
                except cv2.error:
                    stop_event.set()
                    break

                packet = frame_buffer.snapshot(copy_frame=True)
                if packet.frame is None:
                    time.sleep(0.003)
                    continue

                now = time.perf_counter()
                stream_fps = compute_stream_fps(now)
                overlay = build_overlay(packet, stream_fps, now)

                cv2.imshow(window_name, overlay)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or key == 27:
                    stop_event.set()
                    break
                try:
                    if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                        stop_event.set()
                        break
                except cv2.error:
                    stop_event.set()
                    break

    capture_thread.join()
    infer_thread.join()

    with cap_lock:
        cap_ref["cap"].release()
    if display and active_display_backend == "opencv":
        try:
            cv2.destroyWindow(window_name)
        except cv2.error:
            cv2.destroyAllWindows()

    if output_jsonl is not None:
        with rows_lock:
            rows_payload = list(rows)
        if not rows_payload:
            packet = frame_buffer.snapshot(copy_frame=False)
            if packet.frame_index >= 0:
                with stream_stats_lock:
                    dropped_count = int(dropped_frames["count"])
                with runtime_state_lock:
                    runtime_warning = runtime_state.get("warning")
                waiting_warnings = [f"startup_stage={startup_stage['value']}"]
                if runtime_warning:
                    waiting_warnings.append(runtime_warning)
                rows_payload = [
                    _waiting_result(
                        device=str(getattr(pipeline.pose_backend, "device", device)),
                        mode=getattr(pipeline, "pose_mode", pose_mode),
                        warnings=waiting_warnings,
                        stream_stats={
                            "display_fps": None,
                            "dropped_frames": dropped_count,
                            "infer_every_n": max(1, pipeline.infer_every_n),
                            "camera_backend": camera_backend_ref["name"],
                        },
                    ).to_dict()
                ]
        write_jsonl(output_jsonl, rows_payload)
        print(f"[green]Saved jsonl:[/green] {output_jsonl}")


def _waiting_result(
    device: str,
    mode: str = "upper_body",
    warnings: Optional[list[str]] = None,
    stream_stats: Optional[dict[str, float | int | None]] = None,
) -> InferenceResult:
    return InferenceResult(
        posture_label="unknown",
        kneeling=False,
        hands_folded=False,
        angles={"torso_tilt_deg": None, "left_knee_deg": None, "right_knee_deg": None},
        confidence={"person": 0.0, "keypoints": 0.0, "decision": 0.0},
        device=device,
        timing_ms={"pose_infer": 0.0, "total": 0.0},
        warnings=warnings or ["awaiting first inference"],
        status_color="red",
        result_age_ms=None,
        stream_fps=None,
        mode=mode,
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
    attempts = _camera_backend_attempts(camera_backend)
    errors: list[str] = []
    global_deadline = time.perf_counter() + camera_open_timeout_sec

    for backend_id, backend_name in attempts:
        while time.perf_counter() < global_deadline:
            cap = cv2.VideoCapture(camera_id, backend_id)
            if not cap.isOpened():
                cap.release()
                time.sleep(0.05)
                continue

            if capture_width > 0:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(capture_width))
            if capture_height > 0:
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(capture_height))
            if hasattr(cv2, "CAP_PROP_BUFFERSIZE"):
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            remaining_budget = max(0.05, global_deadline - time.perf_counter())
            first_deadline = time.perf_counter() + min(first_frame_timeout_sec, remaining_budget)
            while time.perf_counter() < first_deadline:
                ret, frame = cap.read()
                if ret and frame is not None:
                    return cap, frame, backend_name
                time.sleep(0.01)

            cap.release()
            errors.append(
                f"{backend_name}: opened but failed to get first frame within {first_frame_timeout_sec:.1f}s"
            )
            break
        else:
            errors.append(f"{backend_name}: unable to open within remaining startup budget")

    details = "; ".join(errors) if errors else "unknown camera backend error"
    raise RuntimeError(
        f"Unable to open webcam id={camera_id}. {details}. "
        "On macOS, run from Terminal.app and grant Camera permission. "
        "Codex app process cannot access the camera due missing camera entitlement."
    )


def _camera_backend_attempts(camera_backend: str) -> list[tuple[int, str]]:
    backend = camera_backend.lower().strip()
    avfoundation = int(getattr(cv2, "CAP_AVFOUNDATION", cv2.CAP_ANY))

    if backend == "avfoundation":
        return [(avfoundation, "avfoundation")]

    if backend == "auto":
        if sys.platform == "darwin":
            return [(avfoundation, "avfoundation"), (int(cv2.CAP_ANY), "any")]
        return [(int(cv2.CAP_ANY), "any")]

    return [(int(cv2.CAP_ANY), "any")]


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
