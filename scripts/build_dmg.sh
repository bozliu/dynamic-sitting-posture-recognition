#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

VERSION="${1:-v2.0.0}"
VERSION="${VERSION#v}"
APP_PATH="$ROOT_DIR/dist/PostureMirror.app"

if [[ ! -d "$APP_PATH" ]]; then
  echo "Missing app bundle: $APP_PATH" >&2
  echo "Run scripts/build_macos_app.sh first." >&2
  exit 1
fi

DMG_NAME="PostureMirror-v${VERSION}-macos-arm64.dmg"
DMG_PATH="$ROOT_DIR/dist/$DMG_NAME"
SHA_PATH="$DMG_PATH.sha256"

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

cp -R "$APP_PATH" "$TMP_DIR/PostureMirror.app"
ln -s /Applications "$TMP_DIR/Applications"

hdiutil create \
  -volname "PostureMirror" \
  -srcfolder "$TMP_DIR" \
  -ov \
  -format UDZO \
  "$DMG_PATH"

shasum -a 256 "$DMG_PATH" > "$SHA_PATH"

echo "Built DMG: $DMG_PATH"
echo "Built SHA256: $SHA_PATH"
