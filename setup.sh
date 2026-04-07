#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
PYTHON="${PYTHON:-python3}"
MODELS_URL="https://storage.googleapis.com/mediapipe-models"

info()  { printf '\033[1;34m==> %s\033[0m\n' "$*"; }
error() { printf '\033[1;31merror: %s\033[0m\n' "$*" >&2; exit 1; }

command -v "$PYTHON" >/dev/null || error "$PYTHON not found"
command -v curl    >/dev/null || error "curl not found"
command -v ffmpeg  >/dev/null || echo "warning: ffmpeg not found — needed for feed test tool"

# --- server ---
info "setting up server"
cd "$ROOT/server"
"$PYTHON" -m venv .venv
source .venv/bin/activate
pip install --upgrade pip -q
pip install -e . -q
deactivate

info "downloading models"
mkdir -p "$ROOT/server/models"
if [ ! -f "$ROOT/server/models/hand_landmarker.task" ]; then
    curl -sL -o "$ROOT/server/models/hand_landmarker.task" \
        "$MODELS_URL/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
    echo "  hand_landmarker.task downloaded"
else
    echo "  hand_landmarker.task already exists, skipping"
fi

# --- feed ---
info "setting up feed"
cd "$ROOT/feed"
"$PYTHON" -m venv .venv
source .venv/bin/activate
pip install --upgrade pip -q
pip install -e . -q
deactivate

info "done"
echo
echo "  server:  cd server && source .venv/bin/activate && python main.py"
echo "  feed:    cd feed   && source .venv/bin/activate && python main.py"
echo "  test:    cd feed   && source .venv/bin/activate && python test_video.py <video>"
echo
