# Dynamic Sitting Posture Recognition (PyTorch-Only)

This project provides a real-time, single-camera posture feedback system inspired by the Alexander Technique.
It is built to help desk workers rebuild posture awareness and reduce long-term lower-back-risk behavior.

## Published Article

This implementation is the PyTorch rebuild and submission-quality engineering version of the published work:

- https://bozliu.medium.com/dynamic-siting-posture-recognition-and-correction-68ae418fbc77


## What Problem This Solves

Most laptop/desktop webcams only capture the upper body, not knees or ankles.
Traditional full-body posture rules fail in this common setup.

## Key Improvement: Upper-Body-Only Posture Inference

### Why This Is Important

- In real office use, camera framing usually includes only head, shoulders, torso, and hands.
- If the algorithm depends on leg keypoints, predictions become unstable or unavailable.
- A useful posture assistant must still work reliably with partial-body views.

### What Was Implemented

- Added `pose_mode=upper_body` as the default for webcam inference.
- Reworked decision features to rely on visible upper-body cues:
  - torso tilt,
  - forward displacement of head/torso,
  - normalized hand-to-hand / hand-to-torso geometry.
- Kept API compatibility:
  - `kneeling` remains in output schema,
  - in upper-body mode it is fixed to `false` with explicit warnings.
- Integrated mature PyTorch pose backends:
  - default low-latency `realtime` backend,
  - optional `openpose_torch` backend,
  - fallback strategy when optional backend is unavailable.

### Uniqueness of This Submission

- Works in the real constraint of office webcams (upper-body-only view), not just ideal full-body setups.
- Provides strong, immediate feedback UX:
  - `straight` = green,
  - all other posture states = red,
  - large readable status overlays.
- Prioritizes usability and stability:
  - latest-frame-first pipeline,
  - camera fail-fast on startup,
  - runtime auto-reconnect for temporary camera read failures,
  - macOS-friendly window behavior.
- Fully PyTorch-only and reproducible under a single `dl` conda environment.

## Project Structure

- CLI entry: `posture-recognition`
- Core pipeline: `src/posture_recognition/pipeline.py`
- Realtime webcam loop: `src/posture_recognition/cli.py`
- Rule engine: `src/posture_recognition/rules/classifier.py`
- Overlay rendering: `src/posture_recognition/render/overlay.py`

## Reproducible Setup

### 1) Create Environment

Use either file (same environment):

```bash
conda env create -f environment.yml
# or
conda env update -n dl -f environment.dl.yml --prune
```

Then install package:

```bash
conda activate dl
pip install -e '.[dev]'
```

### 2) Run Tests

```bash
conda run -n dl pytest -q
```

### 3) Minimal Demo (Image)

```bash
conda run -n dl posture-recognition image \
  --input sample_images/straight_hf_flip.jpg \
  --output-image artifacts/demo_upperbody_overlay.jpg \
  --output-json artifacts/demo_upperbody_result.json
```

### 4) Realtime Webcam Demo (Recommended on macOS)

```bash
conda run -n dl posture-recognition webcam \
  --backend realtime \
  --pose-mode upper_body \
  --display-backend tk \
  --camera-id 0 \
  --read-fail-max 120 \
  --capture-width 960 \
  --capture-height 540 \
  --infer-max-dim 512 \
  --infer-every-n 2 \
  --alert-blink
```

Close methods:
- click window `X`
- press `q`
- press `Esc`

## Optional Backend: PyTorch OpenPose

```bash
conda run -n dl posture-recognition webcam \
  --backend openpose_torch \
  --pose-mode upper_body \
  --display-backend tk \
  --camera-id 0
```

If OpenPose backend is unavailable in your runtime, the pipeline falls back to `realtime` and records warnings.

## Model Weights and Large Files

- Model weights are not committed to git history.
- First run may download model weights automatically via backend libraries.
- Downloaded weights are cached locally by your Python runtime (for example in user cache directories).
- Repository policy keeps generated outputs, logs, checkpoints, and local data out of version control.

## Output Contract

Each frame/image result contains:

- `posture_label`: `straight | hunchback | reclined | unknown`
- `kneeling`: bool
- `hands_folded`: bool
- `angles`: `{torso_tilt_deg, left_knee_deg, right_knee_deg}`
- `confidence`
- `timing_ms`
- `status_color`: `green | red`
- `mode`: `upper_body | full_body`
- `stream_stats`
- `warnings`

## Reproducibility & Safety Hygiene

This repository includes:

- conda environment files: `environment.yml`, `environment.dl.yml`
- lock-style pip list: `requirements.lock.txt`
- smoke/integration tests under `tests/`
- CI workflow: `.github/workflows/ci.yml`
- project license: `LICENSE`
- strict ignore policy for outputs/cache/secrets in `.gitignore`
- `.env.example` template with no real credentials


## License & Commercial Use

- This repository is licensed for **non-commercial use only** under PolyForm Noncommercial 1.0.0.
- Commercial use requires a separate written commercial license from the IP owner.
- For commercial licensing, open a GitHub issue in this repository with title `Commercial License Request`.

## Security and Privacy Notes

- No personal tokens/keys are stored in tracked files.
- Do not commit `.env`, raw data dumps, logs, or generated artifacts.
- Keep any local machine paths, account IDs, and private credentials out of commits.

## Current Limitations

- Designed for single primary person in frame.
- Performance may degrade in heavy occlusion or very low light.
- Upper-body mode intentionally disables kneeling inference.
