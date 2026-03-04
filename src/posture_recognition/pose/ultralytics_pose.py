from __future__ import annotations

import time
from dataclasses import dataclass

import cv2
import numpy as np
import torch

from posture_recognition.types import PoseInstance


@dataclass
class UltralyticsPoseBackendConfig:
    person_score_threshold: float = 0.4
    keypoint_score_threshold: float = 0.25
    device: str = "auto"
    model_name: str = "yolo11n-pose.pt"
    max_image_dim: int = 640


class UltralyticsPoseBackend:
    def __init__(self, config: UltralyticsPoseBackendConfig | None = None) -> None:
        self.config = config or UltralyticsPoseBackendConfig()
        self.model = None
        self.warnings: list[str] = []
        self.device = self._resolve_device(self.config.device)

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
        try:
            from ultralytics import YOLO
        except Exception as exc:  # pragma: no cover - depends on environment package install
            raise RuntimeError(
                "Ultralytics is not installed. Install it with `pip install ultralytics` "
                "or use `--backend accurate`."
            ) from exc

        self.model = YOLO(self.config.model_name)

    @staticmethod
    def _normalize_keypoints(
        keypoints_xy: np.ndarray,
        keypoint_scores: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        if keypoints_xy.shape[0] == 17 and keypoint_scores.shape[0] == 17:
            return keypoints_xy.astype(np.float32), keypoint_scores.astype(np.float32)

        padded_xy = np.zeros((17, 2), dtype=np.float32)
        padded_scores = np.zeros((17,), dtype=np.float32)

        count = min(17, keypoints_xy.shape[0], keypoint_scores.shape[0])
        if count > 0:
            padded_xy[:count] = keypoints_xy[:count].astype(np.float32)
            padded_scores[:count] = keypoint_scores[:count].astype(np.float32)
        return padded_xy, padded_scores

    def _run_once(self, image_bgr: np.ndarray) -> tuple[list[PoseInstance], float]:
        self._load_model()

        height, width = image_bgr.shape[:2]
        scale = 1.0
        working_image = image_bgr
        max_dim = int(self.config.max_image_dim)
        if max_dim > 0 and max(height, width) > max_dim:
            scale = max_dim / float(max(height, width))
            new_w = max(1, int(round(width * scale)))
            new_h = max(1, int(round(height * scale)))
            working_image = cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)

        start = time.perf_counter()
        results = self.model.predict(
            source=working_image,
            conf=float(self.config.person_score_threshold),
            imgsz=int(max_dim) if max_dim > 0 else max(working_image.shape[:2]),
            device=self.device,
            verbose=False,
            classes=[0],  # person
        )
        infer_ms = (time.perf_counter() - start) * 1000.0

        if not results:
            return [], infer_ms

        result = results[0]
        if result.boxes is None or result.keypoints is None:
            return [], infer_ms

        boxes_xyxy = result.boxes.xyxy.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy() if result.boxes.cls is not None else np.zeros_like(scores)
        kpts_xy = result.keypoints.xy.cpu().numpy()
        kpts_conf = result.keypoints.conf
        if kpts_conf is None:
            kpts_scores = np.ones((kpts_xy.shape[0], kpts_xy.shape[1]), dtype=np.float32)
        else:
            kpts_scores = np.clip(kpts_conf.cpu().numpy(), 0.0, 1.0)

        instances: list[PoseInstance] = []
        for i in range(len(scores)):
            score = float(scores[i])
            cls = int(classes[i]) if i < len(classes) else 0
            if cls != 0 or score < self.config.person_score_threshold:
                continue

            box_xyxy = boxes_xyxy[i].astype(np.float32)
            keypoints_xy, keypoint_scores = self._normalize_keypoints(kpts_xy[i], kpts_scores[i])
            if scale != 1.0:
                box_xyxy = box_xyxy / scale
                keypoints_xy = keypoints_xy / scale

            instances.append(
                PoseInstance(
                    box_xyxy=box_xyxy,
                    score=score,
                    label=1,
                    keypoints_xy=keypoints_xy,
                    keypoint_scores=keypoint_scores,
                )
            )

        return instances, infer_ms

    def infer(self, image_bgr: np.ndarray) -> tuple[list[PoseInstance], dict[str, object]]:
        if image_bgr is None:
            return [], {"pose_infer_ms": 0.0, "device": self.device, "warnings": ["empty image"]}

        try:
            instances, infer_ms = self._run_once(image_bgr)
            return instances, {"pose_infer_ms": infer_ms, "device": self.device, "warnings": list(self.warnings)}
        except Exception as exc:  # pragma: no cover - hardware dependent
            if self.device == "mps":
                self.warnings.append(f"MPS inference failed, fallback to CPU: {exc}")
                self.device = "cpu"
                instances, infer_ms = self._run_once(image_bgr)
                return instances, {"pose_infer_ms": infer_ms, "device": self.device, "warnings": list(self.warnings)}
            raise
