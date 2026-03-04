#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

SKIP_PREFETCH=0
if [[ "${1:-}" == "--skip-prefetch" ]]; then
  SKIP_PREFETCH=1
fi

if [[ "$SKIP_PREFETCH" -eq 0 ]]; then
  python scripts/prefetch_models.py --output-dir packaging/models
fi

python -m pip install --upgrade pip
python -m pip install --upgrade pyinstaller

export POSTUREMIRROR_MODELS_DIR="$ROOT_DIR/packaging/models"

rm -rf build dist
pyinstaller --noconfirm --clean packaging/macos/PostureMirror.spec

if [[ ! -d "dist/PostureMirror.app" ]]; then
  echo "PostureMirror.app was not produced" >&2
  exit 1
fi

echo "Built app: $ROOT_DIR/dist/PostureMirror.app"
