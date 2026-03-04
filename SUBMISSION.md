# Submission Summary (PyTorch-Only, `dl` Environment)

## Scope Delivered

- PyTorch-only posture recognition pipeline.
- Realtime webcam mode with stability improvements.
- Upper-body-only posture inference mode for practical webcam framing.
- Optional PyTorch OpenPose backend (`openpose_torch`) with automatic fallback.
- Strong red/green feedback visualization.

## Why Upper-Body-Only Matters

In normal laptop camera placement, lower body is often not visible.
This submission makes posture classification dependable in that real-world condition by using upper-body keypoint geometry as the primary signal.

## What Was Implemented

- `pose_mode=upper_body` default for webcam.
- Upper-body decision features and rule path.
- `kneeling` contract preserved for compatibility:
  - upper-body mode emits `kneeling=false` and warning.
- Latest-frame-first async loop.
- Runtime camera reconnect logic.
- macOS-friendly close behavior (`X`, `q`, `Esc`).

## Reproducibility Commands

```bash
conda env create -f environment.yml
conda activate dl
pip install -e '.[dev]'
conda run -n dl pytest -q
```

```bash
conda run -n dl posture-recognition webcam \
  --backend realtime \
  --pose-mode upper_body \
  --display-backend tk \
  --camera-id 0 \
  --infer-max-dim 512 \
  --infer-every-n 2
```

## Privacy/Security Cleanup

- Removed personal identifier fields from project metadata.
- Removed non-essential personal references from docs, keeping only the explicitly requested published Medium article link.
- Added ignore rules for `.env`, logs, outputs, caches, and local data folders.
- Added `.env.example` with placeholder only.

## Acceptance Snapshot

- Tests pass in `dl` environment.
- Image demo, benchmark, and webcam command paths are documented and runnable.


## Published Article

- https://bozliu.medium.com/dynamic-siting-posture-recognition-and-correction-68ae418fbc77


## License and Commercial Rights

- Repository license: PolyForm Noncommercial 1.0.0 (`LICENSE`).
- Commercial rights are reserved by the project owner.
- Commercial usage requires a separate written commercial license; request via GitHub issue (`Commercial License Request`).
