# Submission Summary (v2.0.2)

## Canonical Repository

- [bozliu/PostureMirror](https://github.com/bozliu/PostureMirror)
- Previous repo URL (`dynamic-sitting-posture-recognition`) is expected to redirect.

## Scope Delivered

- PyTorch-only posture recognition pipeline.
- Upper-body-first posture decision path for realistic laptop webcam framing.
- Stable realtime webcam runtime with latest-frame-first processing.
- Optional `openpose_torch` backend with fallback to realtime backend.
- Desktop distribution for Apple Silicon:
  - `PostureMirror.app`
  - `PostureMirror-v2.0.2-macos-arm64.dmg`

## Why Upper-Body-Only Is Important

In practical office usage, webcams usually miss legs. Full-body-only logic becomes unreliable. This release defaults to `upper_body` mode and classifies posture from head/shoulder/torso/hand geometry, so feedback remains usable in real deployment.

## Contract Compatibility

No JSON schema break for inference output fields. Existing keys remain available.

## Reproducibility Commands

```bash
conda env update -n dl -f environment.dl.yml --prune
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
  --capture-width 960 \
  --capture-height 540 \
  --infer-max-dim 512 \
  --infer-every-n 2
```

## DMG Build / Release Artifacts

```bash
bash scripts/build_macos_app.sh
bash scripts/build_dmg.sh v2.0.2
```

Expected files:
- `dist/PostureMirror-v2.0.2-macos-arm64.dmg`
- `dist/PostureMirror-v2.0.2-macos-arm64.dmg.sha256`

## Published Article

- [Dynamic Siting Posture Recognition and Correction](https://bozliu.medium.com/dynamic-siting-posture-recognition-and-correction-68ae418fbc77)

## License and Commercial Rights

- License: PolyForm Noncommercial 1.0.0 (`LICENSE`).
- Commercial rights reserved by project owner.
- Commercial usage requires separate written license (`COMMERCIAL-LICENSE.md`).

## Zero-Residual Policy

- Official install/update must use `scripts/install_macos_app.sh` (no backup folders under `/Applications`).
- Official cleanup command: `scripts/cleanup_app_residuals.sh`.
- Official uninstall command: `scripts/uninstall_macos_app.sh` (optional `--purge-user-data`).
