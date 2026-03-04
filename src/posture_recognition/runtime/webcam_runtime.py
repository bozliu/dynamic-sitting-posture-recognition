from __future__ import annotations

import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Optional

import cv2

from posture_recognition.pipeline import PosturePipeline
from posture_recognition.realtime import LatestFrameBuffer
from posture_recognition.render.overlay import draw_overlay
from posture_recognition.types import InferenceResult


@dataclass
class WebcamRuntimeOptions:
    camera_id: int = 0
    max_frames: int | None = None
    camera_backend: str = "auto"
    camera_open_timeout_sec: float = 2.0
    first_frame_timeout_sec: float = 2.0
    read_fail_max: int = 120
    capture_width: int = 960
    capture_height: int = 540
    alert_blink: bool = True
    ui_alert_blink: bool = True
    ui_font_scale_base: float = 1.0
    display_backend: str = "auto"
    device_name: str = "auto"
    pose_mode: str = "upper_body"


class WebcamRuntimeController:
    """Shared webcam runtime used by CLI and desktop app."""

    def __init__(self, pipeline: PosturePipeline, options: WebcamRuntimeOptions) -> None:
        self.pipeline = pipeline
        self.options = options

        self.startup_stage = {"value": "init"}
        self.frame_buffer = LatestFrameBuffer()
        self.stop_event = threading.Event()

        self.rows: list[dict] = []
        self.rows_lock = threading.Lock()
        self.latest_prediction_lock = threading.Lock()
        self.stream_stats_lock = threading.Lock()
        self.runtime_state_lock = threading.Lock()

        self.dropped_frames = {"count": 0}
        self.runtime_state: dict[str, str | None] = {"warning": None}
        self.latest_prediction = {
            "result": None,
            "features": None,
            "instance": None,
            "inferred_at": 0.0,
            "source_frame_index": -1,
        }

        self.cap_ref: dict[str, cv2.VideoCapture | None] = {"cap": None}
        self.cap_lock = threading.Lock()
        self.camera_backend_ref = {"name": "unknown"}

        self.capture_thread: threading.Thread | None = None
        self.infer_thread: threading.Thread | None = None

        self.fps_window: deque[float] = deque(maxlen=30)
        self.previous_display_ts: float | None = None

    def start(self) -> None:
        if self.capture_thread is not None or self.infer_thread is not None:
            return

        self.startup_stage["value"] = "camera_opening"
        cap, first_frame, backend_name = open_camera_with_failfast(
            camera_id=self.options.camera_id,
            camera_backend=self.options.camera_backend,
            camera_open_timeout_sec=self.options.camera_open_timeout_sec,
            first_frame_timeout_sec=self.options.first_frame_timeout_sec,
            capture_width=self.options.capture_width,
            capture_height=self.options.capture_height,
        )
        self.cap_ref["cap"] = cap
        self.camera_backend_ref["name"] = backend_name
        self.startup_stage["value"] = "model_warming"

        if first_frame is not None:
            self.frame_buffer.push(first_frame, frame_index=0, captured_at=time.perf_counter())

        self.capture_thread = threading.Thread(target=self._capture_worker, daemon=True)
        self.infer_thread = threading.Thread(target=self._infer_worker, daemon=True)
        self.capture_thread.start()
        self.infer_thread.start()

    def stop(self) -> None:
        self.stop_event.set()

    def close(self) -> None:
        self.stop_event.set()
        if self.capture_thread is not None:
            self.capture_thread.join(timeout=3.0)
        if self.infer_thread is not None:
            self.infer_thread.join(timeout=3.0)

        with self.cap_lock:
            cap = self.cap_ref.get("cap")
            if cap is not None:
                cap.release()
                self.cap_ref["cap"] = None

    def is_running(self) -> bool:
        capture_alive = self.capture_thread.is_alive() if self.capture_thread is not None else False
        infer_alive = self.infer_thread.is_alive() if self.infer_thread is not None else False
        return capture_alive or infer_alive

    def compute_stream_fps(self) -> float | None:
        now = time.perf_counter()
        if self.previous_display_ts is not None:
            delta = now - self.previous_display_ts
            if delta > 0:
                self.fps_window.append(1.0 / delta)
        self.previous_display_ts = now
        return (sum(self.fps_window) / len(self.fps_window)) if self.fps_window else None

    def build_overlay(self) -> tuple[object | None, InferenceResult | None]:
        packet = self.frame_buffer.snapshot(copy_frame=True)
        if packet.frame is None:
            return None, None

        stream_fps = self.compute_stream_fps()
        now = time.perf_counter()

        with self.latest_prediction_lock:
            result = self.latest_prediction["result"]
            features = self.latest_prediction["features"]
            instance = self.latest_prediction["instance"]
            inferred_at = self.latest_prediction["inferred_at"]

        with self.runtime_state_lock:
            runtime_warning = self.runtime_state.get("warning")

        if result is None:
            with self.stream_stats_lock:
                dropped_count = int(self.dropped_frames["count"])
            waiting_warnings = [f"startup_stage={self.startup_stage['value']}"]
            if runtime_warning:
                waiting_warnings.append(runtime_warning)
            waiting = waiting_result(
                device=str(getattr(self.pipeline.pose_backend, "device", self.options.device_name)),
                mode=getattr(self.pipeline, "pose_mode", self.options.pose_mode),
                warnings=waiting_warnings,
                stream_stats={
                    "display_fps": stream_fps,
                    "dropped_frames": dropped_count,
                    "infer_every_n": max(1, self.pipeline.infer_every_n),
                    "camera_backend": self.camera_backend_ref["name"],
                },
            )
            waiting.stream_fps = stream_fps
            overlay = draw_overlay(
                packet.frame,
                None,
                waiting,
                None,
                self.pipeline.keypoint_threshold,
                alert_blink=self.options.alert_blink and self.options.ui_alert_blink,
                font_scale_base=self.options.ui_font_scale_base,
                stream_fps=stream_fps,
            )
            return overlay, waiting

        display_result = replace(result)
        display_result.result_age_ms = max(0.0, (now - float(inferred_at)) * 1000.0)
        display_result.stream_fps = stream_fps
        stream_stats = dict(display_result.stream_stats or {})
        stream_stats["display_fps"] = stream_fps
        with self.stream_stats_lock:
            stream_stats["dropped_frames"] = int(self.dropped_frames["count"])
        display_result.stream_stats = stream_stats
        if runtime_warning:
            display_result.warnings = list(display_result.warnings) + [runtime_warning]

        overlay = draw_overlay(
            packet.frame,
            instance,
            display_result,
            features,
            self.pipeline.keypoint_threshold,
            alert_blink=self.options.alert_blink and self.options.ui_alert_blink,
            font_scale_base=self.options.ui_font_scale_base,
            stream_fps=stream_fps,
        )
        return overlay, display_result

    def rows_payload(self) -> list[dict]:
        with self.rows_lock:
            payload = list(self.rows)
        if payload:
            return payload

        packet = self.frame_buffer.snapshot(copy_frame=False)
        if packet.frame_index < 0:
            return payload

        with self.stream_stats_lock:
            dropped_count = int(self.dropped_frames["count"])
        with self.runtime_state_lock:
            runtime_warning = self.runtime_state.get("warning")

        waiting_warnings = [f"startup_stage={self.startup_stage['value']}"]
        if runtime_warning:
            waiting_warnings.append(runtime_warning)

        return [
            waiting_result(
                device=str(getattr(self.pipeline.pose_backend, "device", self.options.device_name)),
                mode=getattr(self.pipeline, "pose_mode", self.options.pose_mode),
                warnings=waiting_warnings,
                stream_stats={
                    "display_fps": None,
                    "dropped_frames": dropped_count,
                    "infer_every_n": max(1, self.pipeline.infer_every_n),
                    "camera_backend": self.camera_backend_ref["name"],
                },
            ).to_dict()
        ]

    def _capture_worker(self) -> None:
        frame_idx = 1 if self.frame_buffer.snapshot(copy_frame=False).frame_index >= 0 else 0
        consecutive_read_failures = 0
        reconnect_attempts = 0
        next_reconnect_ts = 0.0

        try:
            while not self.stop_event.is_set():
                with self.cap_lock:
                    active_cap = self.cap_ref.get("cap")
                    if active_cap is None:
                        break
                    ret, frame = active_cap.read()

                if not ret:
                    consecutive_read_failures += 1
                    if (
                        consecutive_read_failures >= self.options.read_fail_max
                        and time.perf_counter() >= next_reconnect_ts
                    ):
                        self.startup_stage["value"] = "camera_recovering"
                        reconnect_attempts += 1
                        try:
                            new_cap, recovered_frame, recovered_backend = open_camera_with_failfast(
                                camera_id=self.options.camera_id,
                                camera_backend=self.options.camera_backend,
                                camera_open_timeout_sec=max(1.0, self.options.camera_open_timeout_sec),
                                first_frame_timeout_sec=max(1.0, self.options.first_frame_timeout_sec),
                                capture_width=self.options.capture_width,
                                capture_height=self.options.capture_height,
                            )
                            with self.cap_lock:
                                old_cap = self.cap_ref.get("cap")
                                self.cap_ref["cap"] = new_cap
                            if old_cap is not None:
                                old_cap.release()
                            self.camera_backend_ref["name"] = recovered_backend
                            if recovered_frame is not None:
                                captured_at = time.perf_counter()
                                self.frame_buffer.push(recovered_frame, frame_idx, captured_at)
                                frame_idx += 1
                            consecutive_read_failures = 0
                            next_reconnect_ts = 0.0
                            self.startup_stage["value"] = "live"
                            with self.runtime_state_lock:
                                self.runtime_state["warning"] = (
                                    f"camera stream recovered after reconnect attempt #{reconnect_attempts}"
                                )
                        except Exception as exc:
                            next_reconnect_ts = time.perf_counter() + 1.0
                            with self.runtime_state_lock:
                                self.runtime_state["warning"] = (
                                    f"camera reconnect failed (attempt #{reconnect_attempts}): {exc}"
                                )
                    time.sleep(0.01)
                    continue

                if consecutive_read_failures >= self.options.read_fail_max:
                    with self.runtime_state_lock:
                        self.runtime_state["warning"] = "camera stream recovered"
                consecutive_read_failures = 0

                captured_at = time.perf_counter()
                self.frame_buffer.push(frame, frame_idx, captured_at)
                frame_idx += 1

                if self.options.max_frames is not None and frame_idx >= self.options.max_frames:
                    break
        except Exception as exc:
            self.startup_stage["value"] = "capture_exception"
            with self.runtime_state_lock:
                self.runtime_state["warning"] = f"capture worker exception: {exc}"
        finally:
            self.frame_buffer.mark_done()

    def _infer_worker(self) -> None:
        last_processed = -1
        infer_n = max(1, self.pipeline.infer_every_n)

        while True:
            packet = self.frame_buffer.snapshot(copy_frame=True)
            if packet.frame_index <= last_processed:
                if packet.done:
                    break
                time.sleep(0.003)
                continue

            skipped = packet.frame_index - last_processed - 1
            if skipped > 0:
                with self.stream_stats_lock:
                    self.dropped_frames["count"] += skipped

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
                result, features, instance = self.pipeline.predict(packet.frame, frame_index=packet.frame_index)
            except Exception as exc:
                self.startup_stage["value"] = "inference_recovering"
                with self.runtime_state_lock:
                    self.runtime_state["warning"] = f"inference error, retrying: {exc}"
                time.sleep(0.01)
                continue

            inferred_at = time.perf_counter()
            result.status_color = "green" if result.posture_label == "straight" else "red"
            result.result_age_ms = max(0.0, (inferred_at - packet.captured_at) * 1000.0)
            result.mode = getattr(self.pipeline, "pose_mode", self.options.pose_mode)

            with self.runtime_state_lock:
                runtime_warning = self.runtime_state.get("warning")
            if runtime_warning:
                result.warnings = list(result.warnings) + [runtime_warning]

            with self.stream_stats_lock:
                dropped_count = int(self.dropped_frames["count"])
            result.stream_stats = {
                "display_fps": None,
                "dropped_frames": dropped_count,
                "infer_every_n": infer_n,
                "camera_backend": self.camera_backend_ref["name"],
            }
            self.startup_stage["value"] = "live"

            with self.latest_prediction_lock:
                self.latest_prediction["result"] = result
                self.latest_prediction["features"] = features
                self.latest_prediction["instance"] = instance
                self.latest_prediction["inferred_at"] = inferred_at
                self.latest_prediction["source_frame_index"] = packet.frame_index

            with self.rows_lock:
                self.rows.append(result.to_dict())


def waiting_result(
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


def open_camera_with_failfast(
    camera_id: int,
    camera_backend: str,
    camera_open_timeout_sec: float,
    first_frame_timeout_sec: float,
    capture_width: int,
    capture_height: int,
) -> tuple[cv2.VideoCapture, Optional[object], str]:
    attempts = camera_backend_attempts(camera_backend)
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
        "Codex app process cannot access the camera due to missing camera entitlement."
    )


def camera_backend_attempts(camera_backend: str) -> list[tuple[int, str]]:
    backend = camera_backend.lower().strip()
    avfoundation = int(getattr(cv2, "CAP_AVFOUNDATION", cv2.CAP_ANY))

    if backend == "avfoundation":
        return [(avfoundation, "avfoundation")]

    if backend == "auto":
        if sys.platform == "darwin":
            return [(avfoundation, "avfoundation"), (int(cv2.CAP_ANY), "any")]
        return [(int(cv2.CAP_ANY), "any")]

    return [(int(cv2.CAP_ANY), "any")]
