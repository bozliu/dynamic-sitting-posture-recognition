from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision

from posture_recognition.types import PoseInstance


@dataclass
class PoseBackendConfig:
    person_score_threshold: float = 0.5
    keypoint_score_threshold: float = 0.3
    device: str = "auto"
    max_image_dim: int = 960
    weights_path: str | None = None


class TorchvisionPoseBackend:
    def __init__(self, config: PoseBackendConfig | None = None) -> None:
        self.config = config or PoseBackendConfig()
        self.model = None
        self.warnings: list[str] = []
        self.device = self._resolve_device(self.config.device)
        if self.config.device == "auto" and torch.backends.mps.is_available():
            self.warnings.append(
                "Auto device selected CPU for Keypoint R-CNN stability. Use --device mps to force MPS."
            )

    @staticmethod
    def _resolve_device(device: str) -> torch.device:
        if device == "cpu":
            return torch.device("cpu")
        if device == "mps":
            return torch.device("mps")
        # Keypoint R-CNN has partial MPS support and can cause large startup latency.
        # Keep auto mode stable/reproducible by defaulting to CPU.
        return torch.device("cpu")

    def _load_model(self) -> None:
        if self.model is not None:
            return
        weights_path = Path(self.config.weights_path) if self.config.weights_path else None
        if weights_path is not None and weights_path.exists():
            model = torchvision.models.detection.keypointrcnn_resnet50_fpn(weights=None)
            state_dict = torch.load(str(weights_path), map_location="cpu")
            model.load_state_dict(state_dict)
        else:
            if weights_path is not None and not weights_path.exists():
                self.warnings.append(f"Configured accurate weights_path not found: {weights_path}")
            weights = torchvision.models.detection.KeypointRCNN_ResNet50_FPN_Weights.DEFAULT
            model = torchvision.models.detection.keypointrcnn_resnet50_fpn(weights=weights)
        model.eval()
        self.model = model.to(self.device)

        if self.device.type == "mps":  # pragma: no cover - hardware dependent
            # Probe once on a tiny tensor to avoid paying a long failure cost on real input.
            try:
                with torch.no_grad():
                    _ = self.model([torch.zeros((3, 64, 64), dtype=torch.float32, device=self.device)])
            except Exception as exc:
                self.warnings.append(f"MPS unsupported for Keypoint R-CNN, using CPU: {exc}")
                self.device = torch.device("cpu")
                self.model = self.model.to(self.device)

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

        image_rgb = cv2.cvtColor(working_image, cv2.COLOR_BGR2RGB)
        image_tensor = (
            torch.from_numpy(image_rgb)
            .permute(2, 0, 1)
            .to(dtype=torch.float32, device=self.device)
            / 255.0
        )

        start = time.perf_counter()
        with torch.no_grad():
            output = self.model([image_tensor])[0]
        infer_ms = (time.perf_counter() - start) * 1000.0

        boxes = output["boxes"].detach().cpu().numpy()
        scores = output["scores"].detach().cpu().numpy()
        labels = output["labels"].detach().cpu().numpy()
        keypoints = output["keypoints"].detach().cpu().numpy()
        keypoint_scores_raw = output.get("keypoints_scores")
        keypoint_scores = (
            np.clip(keypoint_scores_raw.detach().cpu().numpy(), 0.0, 1.0)
            if keypoint_scores_raw is not None
            else np.clip(keypoints[:, :, 2] / 2.0, 0.0, 1.0)
        )

        instances: list[PoseInstance] = []
        for i in range(len(scores)):
            score = float(scores[i])
            label = int(labels[i])
            if label != 1 or score < self.config.person_score_threshold:
                continue
            box_xyxy = boxes[i].astype(np.float32)
            keypoints_xy = keypoints[i, :, :2].astype(np.float32)
            if scale != 1.0:
                box_xyxy = box_xyxy / scale
                keypoints_xy = keypoints_xy / scale
            instances.append(
                PoseInstance(
                    box_xyxy=box_xyxy,
                    score=score,
                    label=label,
                    keypoints_xy=keypoints_xy,
                    keypoint_scores=keypoint_scores[i].astype(np.float32),
                )
            )

        return instances, infer_ms

    def infer(self, image_bgr: np.ndarray) -> tuple[list[PoseInstance], dict[str, object]]:
        if image_bgr is None:
            return [], {"pose_infer_ms": 0.0, "device": str(self.device), "warnings": ["empty image"]}

        try:
            instances, infer_ms = self._run_once(image_bgr)
            return instances, {"pose_infer_ms": infer_ms, "device": str(self.device), "warnings": list(self.warnings)}
        except Exception as exc:  # pragma: no cover - hardware dependent
            if self.device.type == "mps":
                self.warnings.append(f"MPS inference failed, fallback to CPU: {exc}")
                self.device = torch.device("cpu")
                if self.model is not None:
                    self.model = self.model.to(self.device)
                instances, infer_ms = self._run_once(image_bgr)
                return instances, {
                    "pose_infer_ms": infer_ms,
                    "device": str(self.device),
                    "warnings": list(self.warnings),
                }
            raise
