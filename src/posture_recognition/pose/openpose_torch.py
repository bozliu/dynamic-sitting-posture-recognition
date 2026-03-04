from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from posture_recognition.pose.ultralytics_pose import (
    UltralyticsPoseBackend,
    UltralyticsPoseBackendConfig,
)
from posture_recognition.types import PoseInstance


@dataclass
class OpenPoseTorchBackendConfig:
    person_score_threshold: float = 0.40
    keypoint_score_threshold: float = 0.25
    device: str = "auto"
    max_image_dim: int = 512
    model_name: str = "openpose_mobilenetv2"
    repo_or_dir: str | None = None
    checkpoint_path: str | None = None


class OpenPoseTorchBackend:
    """
    Optional PyTorch OpenPose backend.

    This backend attempts to load a lightweight OpenPose Mobilenet model from torch.hub.
    If loading/inference is unavailable in the runtime environment, it automatically
    falls back to the realtime Ultralytics backend.
    """

    def __init__(self, config: OpenPoseTorchBackendConfig | None = None) -> None:
        self.config = config or OpenPoseTorchBackendConfig()
        self.device = self._resolve_device(self.config.device)
        self.warnings: list[str] = []

        fallback_cfg = UltralyticsPoseBackendConfig(
            person_score_threshold=self.config.person_score_threshold,
            keypoint_score_threshold=self.config.keypoint_score_threshold,
            device=self.device,
            max_image_dim=self.config.max_image_dim,
        )
        self.fallback_backend = UltralyticsPoseBackend(fallback_cfg)

        self.model: Any | None = None
        self._predictor_name: str | None = None

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device == "cpu":
            return "cpu"
        if device == "mps":
            return "mps"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _load_model(self) -> None:
        if self.model is not None:
            return
        repo_or_dir = self.config.repo_or_dir or "Daniil-Osokin/lightweight-human-pose-estimation.pytorch"
        source = "local" if self.config.repo_or_dir and Path(self.config.repo_or_dir).exists() else "github"
        try:
            self.model = torch.hub.load(
                repo_or_dir,
                self.config.model_name,
                pretrained=True,
                trust_repo=True,
                source=source,
            )
            if hasattr(self.model, "to"):
                self.model = self.model.to(self.device)
            if hasattr(self.model, "eval"):
                self.model.eval()

            checkpoint_path = Path(self.config.checkpoint_path) if self.config.checkpoint_path else None
            if checkpoint_path is not None:
                if checkpoint_path.exists() and hasattr(self.model, "load_state_dict"):
                    state_dict = torch.load(str(checkpoint_path), map_location="cpu")
                    if isinstance(state_dict, dict) and "state_dict" in state_dict:
                        state_dict = state_dict["state_dict"]
                    self.model.load_state_dict(state_dict, strict=False)
                else:
                    self.warnings.append(f"Configured openpose checkpoint_path not found: {checkpoint_path}")
        except Exception as exc:  # pragma: no cover - depends on internet/runtime
            raise RuntimeError(f"Failed to load PyTorch OpenPose model: {exc}") from exc

        # Support a few common high-level inference entry points.
        if hasattr(self.model, "predict_keypoints"):
            self._predictor_name = "predict_keypoints"
        elif hasattr(self.model, "predict"):
            self._predictor_name = "predict"
        elif callable(self.model):
            self._predictor_name = "__call__"
        else:
            raise RuntimeError("Loaded OpenPose model has no supported inference API.")

    def infer(self, image_bgr: np.ndarray) -> tuple[list[PoseInstance], dict[str, object]]:
        if image_bgr is None:
            return [], {"pose_infer_ms": 0.0, "device": self.device, "warnings": ["empty image"]}

        start = time.perf_counter()
        try:
            self._load_model()
            raw_pred = self._run_predict(image_bgr)
            instances = self._decode_predictions(raw_pred, image_bgr.shape[1], image_bgr.shape[0])
            infer_ms = (time.perf_counter() - start) * 1000.0

            if not instances:
                self.warnings.append("openpose_torch produced no usable instances; fallback to realtime backend")
                fallback_instances, fallback_meta = self.fallback_backend.infer(image_bgr)
                fallback_warnings = list(fallback_meta.get("warnings", []))
                return fallback_instances, {
                    "pose_infer_ms": float(fallback_meta.get("pose_infer_ms", infer_ms)),
                    "device": str(fallback_meta.get("device", self.device)),
                    "warnings": list(dict.fromkeys(self.warnings + fallback_warnings)),
                }

            return instances, {"pose_infer_ms": infer_ms, "device": self.device, "warnings": list(self.warnings)}
        except Exception as exc:  # pragma: no cover - hardware/runtime dependent
            self.warnings.append(f"openpose_torch unavailable, fallback to realtime backend: {exc}")
            fallback_instances, fallback_meta = self.fallback_backend.infer(image_bgr)
            fallback_warnings = list(fallback_meta.get("warnings", []))
            return fallback_instances, {
                "pose_infer_ms": float(fallback_meta.get("pose_infer_ms", 0.0)),
                "device": str(fallback_meta.get("device", self.device)),
                "warnings": list(dict.fromkeys(self.warnings + fallback_warnings)),
            }

    def _run_predict(self, image_bgr: np.ndarray) -> Any:
        assert self.model is not None
        assert self._predictor_name is not None
        if self._predictor_name == "__call__":
            return self.model(image_bgr)
        fn = getattr(self.model, self._predictor_name)
        return fn(image_bgr)

    def _decode_predictions(self, raw_pred: Any, width: int, height: int) -> list[PoseInstance]:
        """
        Accepts a few generic prediction formats:
        - ndarray/list with shape [N, K, 2 or 3]
        - dict with key 'keypoints'
        """
        keypoints_batch = self._extract_keypoints_batch(raw_pred)
        if keypoints_batch is None:
            return []

        instances: list[PoseInstance] = []
        for person in keypoints_batch:
            keypoints_xy, keypoint_scores = self._to_coco17(person)
            if keypoints_xy is None or keypoint_scores is None:
                continue

            valid = keypoint_scores >= self.config.keypoint_score_threshold
            if int(np.sum(valid)) < 4:
                continue

            xs = keypoints_xy[valid, 0]
            ys = keypoints_xy[valid, 1]
            if xs.size == 0 or ys.size == 0:
                continue

            x1 = float(np.clip(np.min(xs), 0, max(0, width - 1)))
            y1 = float(np.clip(np.min(ys), 0, max(0, height - 1)))
            x2 = float(np.clip(np.max(xs), 0, max(0, width - 1)))
            y2 = float(np.clip(np.max(ys), 0, max(0, height - 1)))
            score = float(np.mean(keypoint_scores[valid])) if np.any(valid) else 0.0
            if score < self.config.person_score_threshold:
                continue

            instances.append(
                PoseInstance(
                    box_xyxy=np.array([x1, y1, x2, y2], dtype=np.float32),
                    score=score,
                    label=1,
                    keypoints_xy=keypoints_xy.astype(np.float32),
                    keypoint_scores=keypoint_scores.astype(np.float32),
                )
            )
        return instances

    @staticmethod
    def _extract_keypoints_batch(raw_pred: Any) -> np.ndarray | None:
        if raw_pred is None:
            return None
        if isinstance(raw_pred, dict):
            if "keypoints" in raw_pred:
                arr = np.asarray(raw_pred["keypoints"])
                return arr if arr.ndim == 3 else None
            return None
        arr = np.asarray(raw_pred)
        if arr.ndim == 3:
            return arr
        if arr.ndim == 2:
            return np.expand_dims(arr, axis=0)
        return None

    @staticmethod
    def _to_coco17(person_kps: np.ndarray) -> tuple[np.ndarray | None, np.ndarray | None]:
        """
        Converts person keypoints to COCO-17 order expected by the pipeline.
        Supports K = 17 (COCO), 18 (OpenPose body-18), 25 (OpenPose body-25).
        """
        if person_kps.ndim != 2 or person_kps.shape[1] < 2:
            return None, None

        k = person_kps.shape[0]
        if person_kps.shape[1] >= 3:
            xy = person_kps[:, :2]
            score = np.clip(person_kps[:, 2], 0.0, 1.0)
        else:
            xy = person_kps[:, :2]
            score = np.ones((k,), dtype=np.float32)

        if k == 17:
            return xy.astype(np.float32), score.astype(np.float32)

        if k == 18:
            # OpenPose body-18 -> COCO-17 (drop neck idx=1)
            mapping = [0, 15, 14, 17, 16, 5, 2, 6, 3, 7, 4, 11, 8, 12, 9, 13, 10]
            return xy[mapping].astype(np.float32), score[mapping].astype(np.float32)

        if k >= 25:
            # OpenPose body-25 -> COCO-17
            mapping = [0, 16, 15, 18, 17, 5, 2, 6, 3, 7, 4, 12, 9, 13, 10, 14, 11]
            return xy[mapping].astype(np.float32), score[mapping].astype(np.float32)

        return None, None
