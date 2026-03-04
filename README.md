# PostureMirror (PyTorch-Only)

Canonical repository: [bozliu/PostureMirror](https://github.com/bozliu/PostureMirror)  
Previous URL (`dynamic-sitting-posture-recognition`) is kept by GitHub redirect.

This project delivers a practical posture-feedback system inspired by the Alexander Technique, focused on real office webcam conditions.

## Published Article

This codebase is the PyTorch rebuild and engineering release of:  
[Dynamic Siting Posture Recognition and Correction](https://bozliu.medium.com/dynamic-siting-posture-recognition-and-correction-68ae418fbc77)

## Why This Matters

Lower-back pain is common in office workers, but many posture systems assume full-body visibility. In real laptop usage, webcams often capture only the upper body.

## What We Improved (v2.0.0)

- Added **upper-body-first posture inference** (`pose_mode=upper_body`) for realistic webcam framing.
- Kept strong real-time feedback policy:
  - `straight` -> green
  - all other posture labels -> red
- Added **desktop app distribution** for Apple Silicon:
  - `PostureMirror.app`
  - `PostureMirror-v2.0.0-macos-arm64.dmg`
- Preserved full CLI toolkit for image/video/evaluate/benchmark.
- Kept PyTorch-only stack with optional `openpose_torch` backend.

## Uniqueness

- Designed for the real constraint that webcams miss legs.
- Decision logic works from head/shoulder/torso/hand geometry.
- Provides strong visual posture alerts and latency stats in live mode.
- Supports both power users (CLI) and end users (macOS app + DMG).

## Figures from Published Work

### Figure 5. Successful detection of standing posture
![Figure 5](docs/figures/figure5_successful_standing_detection.jpeg)

### Figure 6. Successful detection of sitting posture
![Figure 6](docs/figures/figure6_successful_sitting_detection.jpeg)

### Figure 7. Failure examples of pose estimation
![Figure 7](docs/figures/figure7_pose_estimation_failure_examples.png)

### Figure 8. Input/output from dynamic imaging system
![Figure 8](docs/figures/figure8_dynamic_imaging_input_output.png)

## Quickstart (CLI)

### 1) Environment

```bash
conda env update -n dl -f environment.dl.yml --prune
conda activate dl
pip install -e '.[dev]'
```

### 2) Tests

```bash
conda run -n dl pytest -q
```

### 3) Image Demo

```bash
conda run -n dl posture-recognition image \
  --input sample_images/straight_hf_flip.jpg \
  --output-image artifacts/demo_overlay.jpg \
  --output-json artifacts/demo_result.json
```

### 4) Realtime Webcam (recommended)

```bash
conda run -n dl posture-recognition webcam \
  --backend realtime \
  --pose-mode upper_body \
  --display-backend tk \
  --camera-id 0 \
  --capture-width 960 \
  --capture-height 540 \
  --infer-max-dim 512 \
  --infer-every-n 2
```

Close with window `X`, `Esc`, or `q`.

## Desktop App (Apple Silicon)

### Download DMG

- Release page: [PostureMirror Releases](https://github.com/bozliu/PostureMirror/releases)
- Artifact name: `PostureMirror-v2.0.0-macos-arm64.dmg`
- Checksum file: `PostureMirror-v2.0.0-macos-arm64.dmg.sha256`

### Unsigned App First Launch

v2.0.0 is an unsigned public beta.

- Open the DMG and drag `PostureMirror.app` into `Applications`.
- First launch may be blocked by Gatekeeper.
- Use Finder -> right click app -> `Open` -> confirm `Open`.

### Desktop App Scope vs CLI Scope

- Desktop app (`PostureMirror.app` / `posturemirror-app`): webcam realtime only.
- CLI (`posture-recognition`): webcam + image + video + evaluate + benchmark + calibrate.

## Run Desktop App from Python Environment

```bash
conda run -n dl posturemirror-app \
  --backend realtime \
  --pose-mode upper_body \
  --camera-id 0
```

## Optional Backend: PyTorch OpenPose

```bash
conda run -n dl posture-recognition webcam \
  --backend openpose_torch \
  --pose-mode upper_body \
  --display-backend tk \
  --camera-id 0
```

If OpenPose backend fails to initialize, runtime falls back to realtime backend with warnings.

## Build DMG Locally

```bash
bash scripts/build_macos_app.sh
bash scripts/build_dmg.sh v2.0.0
```

## Output Contract (no schema break)

- `posture_label`: `straight | hunchback | reclined | unknown`
- `kneeling`: bool
- `hands_folded`: bool
- `angles`
- `confidence`
- `timing_ms`
- `status_color`
- `mode`
- `stream_stats`
- `warnings`

## License & Commercial Use

- License: **PolyForm Noncommercial 1.0.0** (non-commercial only).
- Commercial rights are reserved by the IP owner.
- Commercial licensing requests: open a GitHub issue in this repository (`Commercial License Request`).

## Security and Privacy

- No secrets/tokens in tracked files.
- Local outputs and caches are git-ignored.
- Personal machine paths are removed from documentation and code comments.
