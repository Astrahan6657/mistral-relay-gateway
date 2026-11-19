#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET_TRIPLE="${1:-$(rustc -vV | awk '/^host:/ {print $2}')}"

if [[ -z "$TARGET_TRIPLE" ]]; then
  echo "ERROR: Could not detect Rust target triple. Pass it explicitly, e.g. aarch64-apple-darwin" >&2
  exit 1
fi

OUT_DIR="$ROOT_DIR/desktop/src-tauri/binaries"
WORK_DIR="$(mktemp -d "${TMPDIR:-/tmp}/mrg-pyi.XXXXXX")"
trap 'rm -rf "$WORK_DIR"' EXIT

mkdir -p "$OUT_DIR"

echo "==> Building standalone Sidecar for target: $TARGET_TRIPLE"
cd "$ROOT_DIR"

uv sync --frozen 2>/dev/null || uv sync
uv run --with pyinstaller pyinstaller \
  --onefile \
  --clean \
  --noconfirm \
  --name "mistral-relay-gateway-${TARGET_TRIPLE}" \
  --distpath "$OUT_DIR" \
  --workpath "$WORK_DIR" \
  --specpath "$WORK_DIR" \
  --add-data "$ROOT_DIR/src/mistral_relay_gateway/static:mistral_relay_gateway/static" \
  --hidden-import uvicorn.logging \
  --hidden-import uvicorn.loops.auto \
  --hidden-import uvicorn.loops.asyncio \
  --hidden-import uvicorn.loops.uvloop \
  --hidden-import uvicorn.protocols.http.auto \
  --hidden-import uvicorn.protocols.http.h11_impl \
  --hidden-import uvicorn.protocols.http.httptools_impl \
  --hidden-import uvicorn.protocols.websockets.auto \
  --hidden-import uvicorn.lifespan.on \
  --hidden-import uvicorn.lifespan.off \
  "$ROOT_DIR/scripts/sidecar_entry.py"

ls -lh "$OUT_DIR/mistral-relay-gateway-${TARGET_TRIPLE}"*
echo "==> Sidecar ready."
