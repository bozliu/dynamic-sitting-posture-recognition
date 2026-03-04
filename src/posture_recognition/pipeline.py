from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

from posture_recognition.features.geometry import (
    KP,
    angle_between_vectors_deg,
    estimate_side_sign,
    joint_angle_deg,
    keypoint_available,
    midpoint,
    point_distance,
)
from posture_recognition.pose.torchvision_pose import PoseBackendConfig, TorchvisionPoseBackend
from posture_recognition.rules.classifier import RuleConfig, classify
from posture_recognition.types import FeatureVector, InferenceResult, PoseInstance


DEFAULT_CONFIG_PATH = Path("configs/default.yaml")
DEFAULT_CALIBRATION_PATH = Path("configs/calibration.yaml")


class PosturePipeline:
    def __init__(
        self,
        config_path: str | Path = DEFAULT_CONFIG_PATH,
        calibration_path: str | Path | None = DEFAULT_CALIBRATION_PATH,
        device: str = "auto",
        backend: str | None = None,
        pose_mode: str | None = None,
        infer_max_dim: int | None = None,
        infer_every_n: int | None = None,
    ) -> None:
        self.config = self._load_config(Path(config_path), Path(calibration_path) if calibration_path else None)

        inference_cfg = self.config["inference"]
        realtime_cfg = inference_cfg.get("realtime", {})
        openpose_cfg = inference_cfg.get("openpose_torch", {})
        selected_backend = str(backend or inference_cfg.get("backend", "realtime")).lower().strip()
        selected_device = device if device != "auto" else str(inference_cfg.get("device", "auto"))
        person_score_threshold = float(inference_cfg["person_score_threshold"])
        keypoint_score_threshold = float(inference_cfg["keypoint_score_threshold"])
        selected_pose_mode = str(pose_mode or inference_cfg.get("pose_mode", "full_body")).lower().strip()

        self.backend_name = selected_backend
        self.startup_warnings: list[str] = []
        self.infer_every_n = max(1, int(infer_every_n or realtime_cfg.get("infer_every_n", 1)))
        self.ui_font_scale_base = float(self.config.get("ui", {}).get("font_scale_base", 1.0))
        self.ui_alert_blink = bool(self.config.get("ui", {}).get("alert_blink", True))
        self.pose_mode = selected_pose_mode if selected_pose_mode in {"upper_body", "full_body"} else "full_body"
        if self.pose_mode != selected_pose_mode:
            self.startup_warnings.append(
                f"Unsupported pose_mode={selected_pose_mode}, fallback to full_body"
            )

        if selected_backend == "realtime":
            realtime_max_dim = int(infer_max_dim or realtime_cfg.get("max_dim", 640))
            try:
                self.pose_backend = self._build_realtime_backend(
                    person_score_threshold=person_score_threshold,
                    keypoint_score_threshold=keypoint_score_threshold,
                    device=selected_device,
                    max_image_dim=realtime_max_dim,
                    model_name=str(realtime_cfg.get("model_name", "yolo11n-pose.pt")),
                )
            except Exception as exc:
                self.startup_warnings.append(
                    f"Realtime backend unavailable, fallback to accurate backend: {exc}"
                )
                self.backend_name = "accurate"
                accurate_max_dim = int(infer_max_dim or inference_cfg.get("max_image_dim", 960))
                self.pose_backend = self._build_accurate_backend(
                    person_score_threshold=person_score_threshold,
                    keypoint_score_threshold=keypoint_score_threshold,
                    device=selected_device,
                    max_image_dim=accurate_max_dim,
                )
        elif selected_backend == "openpose_torch":
            openpose_max_dim = int(infer_max_dim or openpose_cfg.get("max_dim", 512))
            try:
                self.pose_backend = self._build_openpose_backend(
                    person_score_threshold=person_score_threshold,
                    keypoint_score_threshold=keypoint_score_threshold,
                    device=selected_device,
                    max_image_dim=openpose_max_dim,
                    model_name=str(openpose_cfg.get("model_name", "openpose_mobilenetv2")),
                )
            except Exception as exc:
                self.startup_warnings.append(
                    f"openpose_torch backend unavailable, fallback to realtime backend: {exc}"
                )
                self.backend_name = "realtime"
                realtime_max_dim = int(infer_max_dim or realtime_cfg.get("max_dim", 640))
                self.pose_backend = self._build_realtime_backend(
                    person_score_threshold=person_score_threshold,
                    keypoint_score_threshold=keypoint_score_threshold,
                    device=selected_device,
                    max_image_dim=realtime_max_dim,
                    model_name=str(realtime_cfg.get("model_name", "yolo11n-pose.pt")),
                )
        elif selected_backend == "accurate":
            accurate_max_dim = int(infer_max_dim or inference_cfg.get("max_image_dim", 960))
            self.pose_backend = self._build_accurate_backend(
                person_score_threshold=person_score_threshold,
                keypoint_score_threshold=keypoint_score_threshold,
                device=selected_device,
                max_image_dim=accurate_max_dim,
            )
        else:
            self.startup_warnings.append(
                f"Unsupported backend={selected_backend}, fallback to realtime backend"
            )
            self.backend_name = "realtime"
            realtime_max_dim = int(infer_max_dim or realtime_cfg.get("max_dim", 640))
            self.pose_backend = self._build_realtime_backend(
                person_score_threshold=person_score_threshold,
                keypoint_score_threshold=keypoint_score_threshold,
                device=selected_device,
                max_image_dim=realtime_max_dim,
                model_name=str(realtime_cfg.get("model_name", "yolo11n-pose.pt")),
            )

        self.keypoint_threshold = keypoint_score_threshold
        self.rules = RuleConfig(**self.config["rules"])

    @staticmethod
    def _build_accurate_backend(
        person_score_threshold: float,
        keypoint_score_threshold: float,
        device: str,
        max_image_dim: int,
    ) -> TorchvisionPoseBackend:
        pose_cfg = PoseBackendConfig(
            person_score_threshold=person_score_threshold,
            keypoint_score_threshold=keypoint_score_threshold,
            device=device,
            max_image_dim=max_image_dim,
        )
        return TorchvisionPoseBackend(pose_cfg)

    @staticmethod
    def _build_realtime_backend(
        person_score_threshold: float,
        keypoint_score_threshold: float,
        device: str,
        max_image_dim: int,
        model_name: str,
    ):
        from posture_recognition.pose.ultralytics_pose import (
            UltralyticsPoseBackend,
            UltralyticsPoseBackendConfig,
        )

        pose_cfg = UltralyticsPoseBackendConfig(
            person_score_threshold=person_score_threshold,
            keypoint_score_threshold=keypoint_score_threshold,
            device=device,
            model_name=model_name,
            max_image_dim=max_image_dim,
        )
        return UltralyticsPoseBackend(pose_cfg)

    @staticmethod
    def _build_openpose_backend(
        person_score_threshold: float,
        keypoint_score_threshold: float,
        device: str,
        max_image_dim: int,
        model_name: str,
    ):
        from posture_recognition.pose.openpose_torch import (
            OpenPoseTorchBackend,
            OpenPoseTorchBackendConfig,
        )

        pose_cfg = OpenPoseTorchBackendConfig(
            person_score_threshold=person_score_threshold,
            keypoint_score_threshold=keypoint_score_threshold,
            device=device,
            model_name=model_name,
            max_image_dim=max_image_dim,
        )
        return OpenPoseTorchBackend(pose_cfg)

    @staticmethod
    def _load_config(config_path: Path, calibration_path: Path | None) -> dict[str, Any]:
        with config_path.open("r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        if calibration_path and calibration_path.exists():
            with calibration_path.open("r", encoding="utf-8") as f:
                calibration = yaml.safe_load(f) or {}
            config = _deep_merge(config, calibration)

        return config

    def predict(
        self,
        image_bgr: np.ndarray,
        frame_index: int | None = None,
        timestamp_sec: float | None = None,
    ) -> tuple[InferenceResult, FeatureVector | None, PoseInstance | None]:
        total_start = time.perf_counter()
        instances, pose_meta = self.pose_backend.infer(image_bgr)

        warnings = list(self.startup_warnings) + list(pose_meta.get("warnings", []))
        selected = self._select_primary_instance(instances)

        if selected is None:
            total_ms = (time.perf_counter() - total_start) * 1000.0
            if self.pose_mode == "upper_body":
                warnings.append("kneeling disabled in upper-body mode")
            result = InferenceResult(
                posture_label="unknown",
                kneeling=False,
                hands_folded=False,
                angles={"torso_tilt_deg": None, "left_knee_deg": None, "right_knee_deg": None},
                confidence={"person": 0.0, "keypoints": 0.0, "decision": 0.0},
                device=str(pose_meta.get("device", "cpu")),
                timing_ms={"pose_infer": float(pose_meta.get("pose_infer_ms", 0.0)), "total": float(total_ms)},
                warnings=warnings + ["no person detected"],
                status_color="red",
                mode=self.pose_mode,
                frame_index=frame_index,
                timestamp_sec=timestamp_sec,
            )
            return result, None, None

        features = self.extract_features(selected)
        keypoint_conf = self._keypoint_confidence(selected)
        decision = classify(features, selected.score, keypoint_conf, self.rules, pose_mode=self.pose_mode)
        total_ms = (time.perf_counter() - total_start) * 1000.0
        if self.pose_mode == "upper_body":
            warnings.append("kneeling disabled in upper-body mode")

        result = InferenceResult(
            posture_label=decision.posture_label,
            kneeling=decision.kneeling,
            hands_folded=decision.hands_folded,
            angles={
                "torso_tilt_deg": features.torso_tilt_deg,
                "left_knee_deg": features.left_knee_angle_deg,
                "right_knee_deg": features.right_knee_angle_deg,
            },
            confidence={
                "person": decision.confidence_person,
                "keypoints": decision.confidence_keypoints,
                "decision": decision.confidence_decision,
            },
            device=str(pose_meta.get("device", "cpu")),
            timing_ms={"pose_infer": float(pose_meta.get("pose_infer_ms", 0.0)), "total": float(total_ms)},
            warnings=warnings,
            status_color="green" if decision.posture_label == "straight" else "red",
            mode=self.pose_mode,
            frame_index=frame_index,
            timestamp_sec=timestamp_sec,
        )
        return result, features, selected

    def extract_features(self, instance: PoseInstance) -> FeatureVector:
        kps = instance.keypoints_xy
        scores = instance.keypoint_scores
        threshold = self.keypoint_threshold

        left_shoulder = _point_if(kps, scores, KP["left_shoulder"], threshold)
        right_shoulder = _point_if(kps, scores, KP["right_shoulder"], threshold)
        left_hip = _point_if(kps, scores, KP["left_hip"], threshold)
        right_hip = _point_if(kps, scores, KP["right_hip"], threshold)
        left_knee = _point_if(kps, scores, KP["left_knee"], threshold)
        right_knee = _point_if(kps, scores, KP["right_knee"], threshold)
        left_ankle = _point_if(kps, scores, KP["left_ankle"], threshold)
        right_ankle = _point_if(kps, scores, KP["right_ankle"], threshold)
        left_wrist = _point_if(kps, scores, KP["left_wrist"], threshold)
        right_wrist = _point_if(kps, scores, KP["right_wrist"], threshold)

        head_points = []
        for idx in [KP["nose"], KP["left_ear"], KP["right_ear"], KP["left_eye"], KP["right_eye"]]:
            if keypoint_available(scores, idx, threshold):
                head_points.append(kps[idx])

        head_point = np.mean(np.stack(head_points, axis=0), axis=0) if head_points else None
        shoulder_mid = midpoint(left_shoulder, right_shoulder) if left_shoulder is not None and right_shoulder is not None else None
        hip_mid = midpoint(left_hip, right_hip) if left_hip is not None and right_hip is not None else None
        shoulder_width = point_distance(left_shoulder, right_shoulder) if left_shoulder is not None and right_shoulder is not None else None

        torso_tilt_deg = None
        torso_length = None
        if shoulder_mid is not None and hip_mid is not None:
            torso_vec = shoulder_mid - hip_mid
            torso_length = float(np.linalg.norm(torso_vec))
            torso_tilt_deg = angle_between_vectors_deg(torso_vec, np.array([0.0, -1.0], dtype=np.float32))
        elif shoulder_mid is not None and head_point is not None:
            neck_vec = head_point - shoulder_mid
            neck_len = float(np.linalg.norm(neck_vec))
            torso_tilt_deg = angle_between_vectors_deg(neck_vec, np.array([0.0, -1.0], dtype=np.float32))
            if shoulder_width is not None and shoulder_width > 1e-6:
                torso_length = shoulder_width
            elif neck_len > 1e-6:
                torso_length = neck_len

        side_sign = estimate_side_sign(kps, scores, threshold)

        forward_norm = None
        reference_point = hip_mid if hip_mid is not None else shoulder_mid
        if (
            head_point is not None
            and reference_point is not None
            and torso_length is not None
            and torso_length > 1e-6
            and side_sign is not None
        ):
            forward_norm = float(side_sign * (head_point[0] - reference_point[0]) / torso_length)

        left_knee_angle = None
        if left_hip is not None and left_knee is not None and left_ankle is not None:
            left_knee_angle = joint_angle_deg(left_hip, left_knee, left_ankle)

        right_knee_angle = None
        if right_hip is not None and right_knee is not None and right_ankle is not None:
            right_knee_angle = joint_angle_deg(right_hip, right_knee, right_ankle)

        left_hip_ankle_norm = None
        if left_hip is not None and left_ankle is not None and torso_length and torso_length > 1e-6:
            left_hip_ankle_norm = point_distance(left_hip, left_ankle) / torso_length

        right_hip_ankle_norm = None
        if right_hip is not None and right_ankle is not None and torso_length and torso_length > 1e-6:
            right_hip_ankle_norm = point_distance(right_hip, right_ankle) / torso_length

        wrist_distance_norm = None
        wrists_to_torso_norm = None
        torso_center = None
        if shoulder_mid is not None and hip_mid is not None:
            torso_center = midpoint(shoulder_mid, hip_mid)
        elif shoulder_mid is not None:
            torso_center = shoulder_mid

        if (
            left_wrist is not None
            and right_wrist is not None
            and shoulder_width is not None
            and shoulder_width > 1e-6
            and torso_length is not None
            and torso_length > 1e-6
            and torso_center is not None
        ):
            wrist_distance_norm = point_distance(left_wrist, right_wrist) / shoulder_width
            wrists_to_torso_norm = (
                point_distance(left_wrist, torso_center) + point_distance(right_wrist, torso_center)
            ) / (2.0 * torso_length)

        return FeatureVector(
            torso_tilt_deg=torso_tilt_deg,
            forward_displacement_norm=forward_norm,
            left_knee_angle_deg=left_knee_angle,
            right_knee_angle_deg=right_knee_angle,
            shoulder_width=shoulder_width,
            torso_length=torso_length,
            wrist_distance_norm=wrist_distance_norm,
            wrists_to_torso_norm=wrists_to_torso_norm,
            side_sign=side_sign,
            left_hip_ankle_norm=left_hip_ankle_norm,
            right_hip_ankle_norm=right_hip_ankle_norm,
        )

    def predict_image_file(
        self,
        input_path: str | Path,
    ) -> tuple[InferenceResult, FeatureVector | None, PoseInstance | None, np.ndarray]:
        image_bgr = cv2.imread(str(input_path))
        if image_bgr is None:
            raise FileNotFoundError(f"Unable to read image: {input_path}")
        result, features, instance = self.predict(image_bgr)
        return result, features, instance, image_bgr

    @staticmethod
    def _select_primary_instance(instances: list[PoseInstance]) -> PoseInstance | None:
        if not instances:
            return None

        def score_fn(inst: PoseInstance) -> float:
            x1, y1, x2, y2 = inst.box_xyxy
            area = max(1.0, float((x2 - x1) * (y2 - y1)))
            return float(inst.score) * area

        return max(instances, key=score_fn)

    @staticmethod
    def _keypoint_confidence(instance: PoseInstance) -> float:
        return float(np.mean(instance.keypoint_scores))


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_jsonl(path: str | Path, records: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")


def _point_if(kps: np.ndarray, scores: np.ndarray, idx: int, threshold: float) -> np.ndarray | None:
    if idx >= len(kps):
        return None
    if scores[idx] < threshold:
        return None
    return kps[idx]


def _deep_merge(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    output = dict(base)
    for key, value in update.items():
        if key in output and isinstance(output[key], dict) and isinstance(value, dict):
            output[key] = _deep_merge(output[key], value)
        else:
            output[key] = value
    return output
