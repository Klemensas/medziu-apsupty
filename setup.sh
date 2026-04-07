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
for model in \
    "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task" \
    "face_landmarker/face_landmarker/float16/latest/face_landmarker.task"; do
    name="$(basename "$model")"
    if [ ! -f "$ROOT/server/models/$name" ]; then
        curl -sL -o "$ROOT/server/models/$name" "$MODELS_URL/$model"
        echo "  $name downloaded"
    else
        echo "  $name already exists, skipping"
    fi
done

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
