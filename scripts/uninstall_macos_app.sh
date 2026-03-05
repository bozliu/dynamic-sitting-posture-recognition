#!/usr/bin/env bash
set -euo pipefail

APPS_DIR="/Applications"
PURGE_USER_DATA=0
DRY_RUN=0

usage() {
  cat <<'EOF'
Usage:
  bash scripts/uninstall_macos_app.sh [--apps-dir /Applications] [--purge-user-data] [--dry-run]

Options:
  --apps-dir         Applications directory (default: /Applications)
  --purge-user-data  Also remove user data/cache/preferences files
  --dry-run          Print actions without modifying filesystem
EOF
}

log() {
  printf '[uninstall] %s\n' "$1"
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
    --purge-user-data)
      PURGE_USER_DATA=1
      shift
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

TARGET_APP="$APPS_DIR/PostureMirror.app"
if [[ -d "$TARGET_APP" ]]; then
  log "Removing app: $TARGET_APP"
  run rm -rf "$TARGET_APP"
fi

while IFS= read -r residual; do
  [[ -z "$residual" ]] && continue
  log "Removing residual: $residual"
  run rm -rf "$residual"
done < <(find "$APPS_DIR" -maxdepth 1 -mindepth 1 \( -name 'PostureMirror.app.bak.*' -o -name 'PostureMirror.app.old*' \) 2>/dev/null)

if [[ "$PURGE_USER_DATA" -eq 1 ]]; then
  USER_PATHS=(
    "$HOME/Library/Application Support/PostureMirror"
    "$HOME/Library/Caches/com.bozliu.posturemirror"
    "$HOME/Library/Preferences/com.bozliu.posturemirror.plist"
    "$HOME/Library/Saved Application State/com.bozliu.posturemirror.savedState"
  )

  for user_path in "${USER_PATHS[@]}"; do
    if [[ -e "$user_path" ]]; then
      log "Removing user data: $user_path"
      run rm -rf "$user_path"
    fi
  done
else
  log "User data retained. Use --purge-user-data to remove it."
fi

log "Uninstall complete."
