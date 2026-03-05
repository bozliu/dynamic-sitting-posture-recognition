#!/usr/bin/env bash
set -euo pipefail

APPS_DIR="/Applications"
DRY_RUN=0

usage() {
  cat <<'EOF'
Usage:
  bash scripts/cleanup_app_residuals.sh [--apps-dir /Applications] [--dry-run]

Removes only allowed residual folders:
- PostureMirror.app.bak.*
- PostureMirror.app.old*
EOF
}

log() {
  printf '[cleanup] %s\n' "$1"
}

run() {
  if [[ "$DRY_RUN" -eq 1 ]]; then
    log "DRY-RUN: $*"
    return 0
  fi
  "$@"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --apps-dir)
      APPS_DIR="${2:-}"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ "$DRY_RUN" -eq 0 && -d "$APPS_DIR" && ! -w "$APPS_DIR" ]]; then
  echo "No write permission to $APPS_DIR. Re-run with sudo if needed." >&2
  exit 1
fi

removed=0
while IFS= read -r residual; do
  [[ -z "$residual" ]] && continue
  log "Removing residual: $residual"
  run rm -rf "$residual"
  removed=$((removed + 1))
done < <(find "$APPS_DIR" -maxdepth 1 -mindepth 1 \( -name 'PostureMirror.app.bak.*' -o -name 'PostureMirror.app.old*' \) 2>/dev/null)

log "Residual cleanup finished. Removed count: $removed"
