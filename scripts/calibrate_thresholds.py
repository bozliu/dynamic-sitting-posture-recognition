#!/usr/bin/env python3
from __future__ import annotations

import argparse

import yaml

from posture_recognition.experiments import run_calibration
from posture_recognition.pipeline import PosturePipeline, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate posture thresholds using sample images")
    parser.add_argument("--images-dir", default="sample_images")
    parser.add_argument("--write-config", default="configs/calibration.yaml")
    parser.add_argument("--report", default="artifacts/calibration_report.json")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--calibration", default="configs/calibration.yaml")
    parser.add_argument("--device", default="auto", choices=["auto", "mps", "cpu"])
    args = parser.parse_args()

    pipeline = PosturePipeline(config_path=args.config, calibration_path=args.calibration, device=args.device)
    report = run_calibration(pipeline, args.images_dir)
    write_json(args.report, report)

    if "recommended_rules" in report:
        with open(args.write_config, "w", encoding="utf-8") as f:
            yaml.safe_dump({"rules": report["recommended_rules"]}, f, sort_keys=False)
        print(f"Saved calibration config: {args.write_config}")

    print(f"Saved calibration report: {args.report}")


if __name__ == "__main__":
    main()
