from __future__ import annotations

import argparse
import threading
from pathlib import Path
from typing import Optional

import cv2

from posture_recognition.pipeline import PosturePipeline
from posture_recognition.runtime.webcam_runtime import WebcamRuntimeController, WebcamRuntimeOptions


def _build_pipeline(args: argparse.Namespace) -> PosturePipeline:
    return PosturePipeline(
        config_path=Path(args.config),
        calibration_path=Path(args.calibration),
        device=args.device,
        backend=args.backend,
        pose_mode=args.pose_mode,
        infer_max_dim=args.infer_max_dim,
        infer_every_n=args.infer_every_n,
    )


class PostureMirrorApp:
    def __init__(self, args: argparse.Namespace) -> None:
        import tkinter as tk
        from PIL import Image, ImageTk

        self.tk = tk
        self.Image = Image
        self.ImageTk = ImageTk
        self.args = args

        self.root = tk.Tk()
        self.root.title("PostureMirror")
        self.root.geometry("1180x760")

        self.runtime_lock = threading.Lock()
        self.runtime: WebcamRuntimeController | None = None

        self._build_layout()
        self.root.protocol("WM_DELETE_WINDOW", self.on_quit)
        self.root.bind("<Escape>", lambda _evt: self.on_quit())
        self.root.bind("<KeyPress-q>", lambda _evt: self.on_quit())

        self._tick()
        if self.args.autostart:
            self.on_start()

    def _build_layout(self) -> None:
        tk = self.tk

        controls = tk.Frame(self.root)
        controls.pack(fill="x", padx=10, pady=10)

        self.start_button = tk.Button(controls, text="Start", width=10, command=self.on_start)
        self.start_button.pack(side="left", padx=(0, 8))

        self.stop_button = tk.Button(controls, text="Stop", width=10, command=self.on_stop, state="disabled")
        self.stop_button.pack(side="left", padx=(0, 8))

        self.quit_button = tk.Button(controls, text="Quit", width=10, command=self.on_quit)
        self.quit_button.pack(side="left", padx=(0, 8))

        self.state_label = tk.Label(controls, text="State: idle", anchor="w")
        self.state_label.pack(side="left", padx=(12, 0))

        self.metrics_label = tk.Label(controls, text="", anchor="w")
        self.metrics_label.pack(side="right")

        self.video_panel = tk.Label(self.root, bg="#101010")
        self.video_panel.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        self.info_label = tk.Label(
            self.root,
            text="POSTURE: UNKNOWN",
            font=("Helvetica", 18, "bold"),
            anchor="w",
            bg="#1f1f1f",
            fg="#ffffff",
            padx=12,
            pady=8,
        )
        self.info_label.pack(fill="x", padx=10, pady=(0, 10))

    def on_start(self) -> None:
        with self.runtime_lock:
            if self.runtime is not None and self.runtime.is_running():
                return

            pipeline = _build_pipeline(self.args)
            ui_cfg = pipeline.config.get("ui", {})
            camera_cfg = pipeline.config.get("inference", {}).get("camera", {})
            options = WebcamRuntimeOptions(
                camera_id=self.args.camera_id,
                max_frames=self.args.max_frames,
                camera_backend=self.args.camera_backend or str(camera_cfg.get("backend", "avfoundation")),
                camera_open_timeout_sec=float(
                    self.args.camera_open_timeout_sec or camera_cfg.get("open_timeout_sec", 2.0)
                ),
                first_frame_timeout_sec=float(
                    self.args.first_frame_timeout_sec or camera_cfg.get("first_frame_timeout_sec", 2.0)
                ),
                read_fail_max=int(self.args.read_fail_max or camera_cfg.get("read_fail_max", 120)),
                capture_width=self.args.capture_width,
                capture_height=self.args.capture_height,
                alert_blink=self.args.alert_blink,
                ui_alert_blink=bool(ui_cfg.get("alert_blink", True)),
                ui_font_scale_base=float(ui_cfg.get("font_scale_base", 1.0)),
                display_backend="tk",
                device_name=self.args.device,
                pose_mode=pipeline.pose_mode,
            )

            runtime = WebcamRuntimeController(pipeline, options)
            try:
                runtime.start()
            except RuntimeError as exc:
                self.state_label.config(text=f"State: startup failed ({exc})")
                self.info_label.config(text="POSTURE: CAMERA ERROR", bg="#c73737")
                return
            self.runtime = runtime

        self.start_button.config(state="disabled")
        self.stop_button.config(state="normal")
        self.state_label.config(text="State: running")

    def on_stop(self) -> None:
        with self.runtime_lock:
            runtime = self.runtime
            self.runtime = None

        if runtime is not None:
            runtime.stop()
            runtime.close()

        self.start_button.config(state="normal")
        self.stop_button.config(state="disabled")
        self.state_label.config(text="State: stopped")

    def on_quit(self) -> None:
        self.on_stop()
        try:
            self.root.quit()
        finally:
            self.root.destroy()

    def _tick(self) -> None:
        runtime: Optional[WebcamRuntimeController]
        with self.runtime_lock:
            runtime = self.runtime

        if runtime is not None:
            overlay, result = runtime.build_overlay()
            if overlay is not None:
                rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
                photo = self.ImageTk.PhotoImage(self.Image.fromarray(rgb))
                self.video_panel.configure(image=photo)
                self.video_panel.image = photo

            if result is not None:
                posture = str(result.posture_label).upper()
                color = "#1f9d3a" if result.posture_label == "straight" else "#c73737"
                self.info_label.config(text=f"POSTURE: {posture}", bg=color)
                fps = result.stream_fps if result.stream_fps is not None else 0.0
                age = result.result_age_ms if result.result_age_ms is not None else 0.0
                self.metrics_label.config(
                    text=(
                        f"Device={result.device}  "
                        f"FPS={fps:.2f}  "
                        f"Age(ms)={age:.1f}  "
                        f"Backend={self.args.backend}  "
                        f"Mode={result.mode or 'n/a'}"
                    )
                )

            if not runtime.is_running():
                self.on_stop()

        self.root.after(15, self._tick)

    def run(self) -> None:
        self.root.mainloop()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PostureMirror macOS desktop app")
    parser.add_argument("--camera-id", type=int, default=0)
    parser.add_argument("--backend", choices=["realtime", "accurate", "openpose_torch"], default="realtime")
    parser.add_argument("--pose-mode", choices=["upper_body", "full_body"], default="upper_body")
    parser.add_argument("--camera-backend", choices=["avfoundation", "auto"], default="avfoundation")
    parser.add_argument("--camera-open-timeout-sec", type=float, default=None)
    parser.add_argument("--first-frame-timeout-sec", type=float, default=None)
    parser.add_argument("--read-fail-max", type=int, default=None)
    parser.add_argument("--capture-width", type=int, default=960)
    parser.add_argument("--capture-height", type=int, default=540)
    parser.add_argument("--infer-max-dim", type=int, default=512)
    parser.add_argument("--infer-every-n", type=int, default=2)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--calibration", default="configs/calibration.yaml")
    parser.add_argument("--device", choices=["auto", "mps", "cpu"], default="auto")
    parser.add_argument("--alert-blink", action="store_true", default=True)
    parser.add_argument("--no-alert-blink", dest="alert_blink", action="store_false")
    parser.add_argument("--autostart", action="store_true", default=True)
    parser.add_argument("--no-autostart", dest="autostart", action="store_false")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    app = PostureMirrorApp(args)
    app.run()


if __name__ == "__main__":
    main()
