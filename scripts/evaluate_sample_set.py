#!/usr/bin/env python3
from __future__ import annotations

import argparse

from posture_recognition.experiments import run_evaluation
from posture_recognition.pipeline import PosturePipeline, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate posture pipeline on sample images")
    parser.add_argument("--images-dir", default="sample_images")
    parser.add_argument("--report", default="artifacts/eval.json")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--calibration", default="configs/calibration.yaml")
    parser.add_argument("--device", default="auto", choices=["auto", "mps", "cpu"])
    args = parser.parse_args()

    pipeline = PosturePipeline(config_path=args.config, calibration_path=args.calibration, device=args.device)
    report = run_evaluation(pipeline, args.images_dir)
    write_json(args.report, report)
    print(f"Saved evaluation report: {args.report}")


if __name__ == "__main__":
    main()
