#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

import torch
import torchvision


def _copy_if_exists(src: Path, dst: Path) -> Path | None:
    if not src.exists():
        return None
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return dst


def prefetch_realtime_models(output_dir: Path) -> dict[str, str | None]:
    output = output_dir / "realtime"
    output.mkdir(parents=True, exist_ok=True)

    try:
        from ultralytics import YOLO

        model = YOLO("yolo11n-pose.pt")
        ckpt_path = Path(str(getattr(model, "ckpt_path", "yolo11n-pose.pt")))
        if ckpt_path.exists():
            copied = _copy_if_exists(ckpt_path, output / "yolo11n-pose.pt")
            return {
                "status": "ok" if copied else "missing",
                "model": str(copied) if copied else None,
            }
        return {"status": "missing", "model": None}
    except Exception as exc:
        return {"status": f"error: {exc}", "model": None}


def prefetch_accurate_models(output_dir: Path) -> dict[str, str | None]:
    output = output_dir / "accurate"
    output.mkdir(parents=True, exist_ok=True)

    model_path = output / "keypointrcnn_resnet50_fpn_coco.pth"
    try:
        weights = torchvision.models.detection.KeypointRCNN_ResNet50_FPN_Weights.DEFAULT
        state_dict = weights.get_state_dict(progress=True)
        torch.save(state_dict, model_path)
        return {"status": "ok", "model": str(model_path)}
    except Exception as exc:
        return {"status": f"error: {exc}", "model": None}


def prefetch_openpose_models(output_dir: Path) -> dict[str, str | None]:
    output = output_dir / "openpose_torch"
    output.mkdir(parents=True, exist_ok=True)

    repo_dir = output / "lightweight-human-pose-estimation.pytorch"
    repo_url = "https://github.com/Daniil-Osokin/lightweight-human-pose-estimation.pytorch.git"

    try:
        if not repo_dir.exists():
            subprocess.run(
                ["git", "clone", "--depth", "1", repo_url, str(repo_dir)],
                check=True,
                capture_output=True,
                text=True,
            )

        # Best-effort warm cache for torch.hub local load.
        try:
            torch.hub.load(str(repo_dir), "openpose_mobilenetv2", pretrained=True, source="local")
            status = "ok"
        except Exception as exc:
            status = f"partial: {exc}"

        return {"status": status, "repo_or_dir": str(repo_dir)}
    except Exception as exc:
        return {"status": f"error: {exc}", "repo_or_dir": None}


def main() -> None:
    parser = argparse.ArgumentParser(description="Prefetch posture models for app bundling")
    parser.add_argument(
        "--output-dir",
        default="packaging/models",
        help="Output directory for bundled models",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "realtime": prefetch_realtime_models(output_dir),
        "accurate": prefetch_accurate_models(output_dir),
        "openpose_torch": prefetch_openpose_models(output_dir),
    }

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(json.dumps({"models_dir": str(output_dir), "manifest": manifest}, indent=2))


if __name__ == "__main__":
    main()
