from __future__ import annotations

import itertools
import statistics
from dataclasses import asdict
from pathlib import Path
from typing import Any

import cv2

from posture_recognition.pipeline import PosturePipeline
from posture_recognition.rules.classifier import RuleConfig, classify


def parse_filename_labels(path: Path) -> dict[str, Any]:
    stem = path.stem.lower()

    posture_label = None
    if stem.startswith("hunchback"):
        posture_label = "hunchback"
    elif stem.startswith("recline"):
        posture_label = "reclined"
    elif stem.startswith("straight"):
        posture_label = "straight"

    return {
        "posture_label": posture_label,
        "kneeling": "kneeling" in stem,
        "hands_folded": "_hf" in stem,
    }


def list_images(images_dir: str | Path) -> list[Path]:
    images_dir = Path(images_dir)
    images = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"):
        images.extend(images_dir.glob(ext))
    return sorted(images)


def run_evaluation(pipeline: PosturePipeline, images_dir: str | Path) -> dict[str, Any]:
    rows = []
    for image_path in list_images(images_dir):
        labels = parse_filename_labels(image_path)
        image = cv2.imread(str(image_path))
        result, _, _ = pipeline.predict(image)
        rows.append(
            {
                "image": str(image_path.name),
                "gt": labels,
                "pred": {
                    "posture_label": result.posture_label,
                    "kneeling": result.kneeling,
                    "hands_folded": result.hands_folded,
                },
                "timing_ms": result.timing_ms,
                "confidence": result.confidence,
            }
        )

    posture_rows = [r for r in rows if r["gt"]["posture_label"] is not None]
    posture_classes = ["straight", "hunchback", "reclined"]
    posture_metrics = _multiclass_metrics(posture_rows, posture_classes, "posture_label")
    kneeling_metrics = _binary_metrics(rows, "kneeling")
    hands_metrics = _binary_metrics(rows, "hands_folded")

    avg_total_ms = statistics.mean([r["timing_ms"]["total"] for r in rows]) if rows else 0.0

    return {
        "num_images": len(rows),
        "posture": posture_metrics,
        "kneeling": kneeling_metrics,
        "hands_folded": hands_metrics,
        "avg_total_latency_ms": avg_total_ms,
        "rows": rows,
    }


def run_calibration(pipeline: PosturePipeline, images_dir: str | Path) -> dict[str, Any]:
    samples = []
    for image_path in list_images(images_dir):
        labels = parse_filename_labels(image_path)
        if labels["posture_label"] is None:
            continue
        image = cv2.imread(str(image_path))
        result, features, selected = pipeline.predict(image)
        if features is None or selected is None:
            continue
        samples.append(
            {
                "image": image_path.name,
                "gt_posture": labels["posture_label"],
                "features": features,
                "person_score": selected.score,
                "keypoint_conf": result.confidence["keypoints"],
            }
        )

    if len(samples) < 3:
        return {
            "message": "not enough labeled posture samples for calibration",
            "recommended_rules": asdict(pipeline.rules),
            "loo_accuracy": None,
            "folds": [],
        }

    hunch_values = [0.10, 0.14, 0.18, 0.22, 0.26, 0.30]
    recline_values = [0.10, 0.14, 0.18, 0.22, 0.26, 0.30]
    straight_values = [0.06, 0.10, 0.14, 0.18]
    tilt_values = [12.0, 16.0, 20.0, 24.0]

    candidate_rules = []
    for h, r, s, t in itertools.product(hunch_values, recline_values, straight_values, tilt_values):
        candidate_rules.append(
            RuleConfig(
                hunch_forward_min=h,
                recline_forward_min=r,
                straight_forward_max=s,
                straight_tilt_max_deg=t,
                kneeling_knee_angle_max_deg=pipeline.rules.kneeling_knee_angle_max_deg,
                kneeling_hip_ankle_max_norm=pipeline.rules.kneeling_hip_ankle_max_norm,
                hands_wrist_distance_max_norm=pipeline.rules.hands_wrist_distance_max_norm,
                hands_wrist_to_torso_max_norm=pipeline.rules.hands_wrist_to_torso_max_norm,
                min_keypoint_confidence=pipeline.rules.min_keypoint_confidence,
            )
        )

    folds = []
    selected_candidates = []

    for i in range(len(samples)):
        train = [samples[j] for j in range(len(samples)) if j != i]
        test = samples[i]

        best = None
        best_train_acc = -1.0
        for candidate in candidate_rules:
            correct = 0
            for row in train:
                pred = classify(row["features"], row["person_score"], row["keypoint_conf"], candidate)
                correct += int(pred.posture_label == row["gt_posture"])
            acc = correct / len(train)
            if acc > best_train_acc:
                best_train_acc = acc
                best = candidate

        assert best is not None
        selected_candidates.append(best)
        test_pred = classify(test["features"], test["person_score"], test["keypoint_conf"], best)
        folds.append(
            {
                "test_image": test["image"],
                "gt": test["gt_posture"],
                "pred": test_pred.posture_label,
                "train_accuracy": best_train_acc,
                "selected_rules": asdict(best),
            }
        )

    loo_accuracy = sum(int(f["gt"] == f["pred"]) for f in folds) / len(folds)

    recommended_rules = {
        "hunch_forward_min": statistics.median([c.hunch_forward_min for c in selected_candidates]),
        "recline_forward_min": statistics.median([c.recline_forward_min for c in selected_candidates]),
        "straight_forward_max": statistics.median([c.straight_forward_max for c in selected_candidates]),
        "straight_tilt_max_deg": statistics.median([c.straight_tilt_max_deg for c in selected_candidates]),
        "kneeling_knee_angle_max_deg": pipeline.rules.kneeling_knee_angle_max_deg,
        "kneeling_hip_ankle_max_norm": pipeline.rules.kneeling_hip_ankle_max_norm,
        "hands_wrist_distance_max_norm": pipeline.rules.hands_wrist_distance_max_norm,
        "hands_wrist_to_torso_max_norm": pipeline.rules.hands_wrist_to_torso_max_norm,
        "min_keypoint_confidence": pipeline.rules.min_keypoint_confidence,
    }

    return {
        "num_samples": len(samples),
        "loo_accuracy": loo_accuracy,
        "folds": folds,
        "recommended_rules": recommended_rules,
    }


def run_benchmark(pipeline: PosturePipeline, images_dir: str | Path, num_frames: int = 300) -> dict[str, Any]:
    images = list_images(images_dir)
    if not images:
        raise FileNotFoundError(f"No images found in {images_dir}")

    cached = []
    for img_path in images:
        image = cv2.imread(str(img_path))
        if image is not None:
            cached.append((img_path.name, image))

    if not cached:
        raise RuntimeError("Failed to read benchmark images")

    total_ms = []
    infer_ms = []
    rows = []

    for i in range(num_frames):
        name, frame = cached[i % len(cached)]
        result, _, _ = pipeline.predict(frame, frame_index=i)
        total_ms.append(result.timing_ms["total"])
        infer_ms.append(result.timing_ms["pose_infer"])
        rows.append({"frame": i, "source": name, "timing_ms": result.timing_ms, "device": result.device})

    total_median = statistics.median(total_ms)
    total_p95 = _percentile(total_ms, 95)
    infer_median = statistics.median(infer_ms)
    infer_p95 = _percentile(infer_ms, 95)

    return {
        "num_frames": num_frames,
        "device": rows[-1]["device"] if rows else "unknown",
        "total_latency_ms": {
            "mean": statistics.mean(total_ms),
            "median": total_median,
            "p95": total_p95,
            "fps_from_median": 1000.0 / total_median if total_median > 0 else 0.0,
        },
        "pose_infer_latency_ms": {
            "mean": statistics.mean(infer_ms),
            "median": infer_median,
            "p95": infer_p95,
            "fps_from_median": 1000.0 / infer_median if infer_median > 0 else 0.0,
        },
        "rows": rows,
    }


def _multiclass_metrics(rows: list[dict[str, Any]], classes: list[str], key: str) -> dict[str, Any]:
    if not rows:
        return {"num_samples": 0, "accuracy": 0.0, "macro_f1": 0.0, "per_class": {}, "confusion": {}}

    confusion = {gt: {pred: 0 for pred in classes + ["unknown"]} for gt in classes}
    correct = 0
    for row in rows:
        gt = row["gt"][key]
        pred = row["pred"][key]
        if gt in confusion:
            confusion[gt][pred if pred in confusion[gt] else "unknown"] += 1
        if gt == pred:
            correct += 1

    per_class = {}
    f1_values = []
    for cls in classes:
        tp = sum(1 for r in rows if r["gt"][key] == cls and r["pred"][key] == cls)
        fp = sum(1 for r in rows if r["gt"][key] != cls and r["pred"][key] == cls)
        fn = sum(1 for r in rows if r["gt"][key] == cls and r["pred"][key] != cls)

        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if precision + recall > 0 else 0.0
        per_class[cls] = {"precision": precision, "recall": recall, "f1": f1, "support": tp + fn}
        f1_values.append(f1)

    return {
        "num_samples": len(rows),
        "accuracy": correct / len(rows),
        "macro_f1": sum(f1_values) / len(f1_values),
        "per_class": per_class,
        "confusion": confusion,
    }


def _binary_metrics(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    if not rows:
        return {"num_samples": 0, "accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}

    tp = sum(1 for r in rows if r["gt"][key] and r["pred"][key])
    tn = sum(1 for r in rows if not r["gt"][key] and not r["pred"][key])
    fp = sum(1 for r in rows if not r["gt"][key] and r["pred"][key])
    fn = sum(1 for r in rows if r["gt"][key] and not r["pred"][key])

    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if precision + recall > 0 else 0.0
    accuracy = (tp + tn) / len(rows)

    return {
        "num_samples": len(rows),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def _percentile(values: list[float], q: int) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    idx = int(round((q / 100.0) * (len(sorted_values) - 1)))
    return sorted_values[idx]
