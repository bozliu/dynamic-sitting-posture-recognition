#!/usr/bin/env python3
from __future__ import annotations

import argparse

from posture_recognition.experiments import run_benchmark
from posture_recognition.pipeline import PosturePipeline, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark posture runtime")
    parser.add_argument("--images-dir", default="sample_images")
    parser.add_argument("--num-frames", type=int, default=300)
    parser.add_argument("--report", default="artifacts/benchmark.json")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--calibration", default="configs/calibration.yaml")
    parser.add_argument("--device", default="auto", choices=["auto", "mps", "cpu"])
    args = parser.parse_args()

    pipeline = PosturePipeline(config_path=args.config, calibration_path=args.calibration, device=args.device)
    report = run_benchmark(pipeline, args.images_dir, num_frames=args.num_frames)
    write_json(args.report, report)
    print(f"Saved benchmark report: {args.report}")


if __name__ == "__main__":
    main()
