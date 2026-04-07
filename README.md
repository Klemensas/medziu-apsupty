# medziu-apsupty

Interactive audio-visual installation — a camera feed is processed in real time by a server that detects scene features and maps them to visual and audio output.

## Project structure

```
feed/       Edge device program — captures video, plays audio/video output
server/     Processing server  — detection models + Wekinator-style mapping
```

Each component is an independent Python package with its own virtual environment.

## Quick start

```bash
./setup.sh
```

This creates virtual environments for both components, installs all dependencies, and downloads required ML models. Requires Python 3.12+, curl, and (optionally) ffmpeg for the feed test tool. Override the Python binary with `PYTHON=python3.12 ./setup.sh` if needed.

## Processing server

Receives a live video stream over WebSocket, applies a Wekinator-style feature→effect mapping with MediaPipe hand tracking, and streams the transformed video back.

### Run

```bash
cd server
source .venv/bin/activate
python main.py              # listens on 0.0.0.0:8000
```

The feed program connects to `ws://<server-ip>:8000/ws/video` and exchanges binary JPEG frames — send a frame, receive the transformed frame.

A health-check endpoint is available at `GET /health`.

## Feed program

Runs on an edge device (e.g. Raspberry Pi 5). Currently contains a speaker test that plays a melody on loop via direct audio output or Chromecast.

### Run

```bash
cd feed
source .venv/bin/activate

# Direct playback (default sounddevice output)
python main.py

# Direct playback on a specific device
python main.py --device "USB Audio CODEC"

# Scan for Chromecast devices on the network
python main.py --backend chromecast --list-devices

# Cast to a Chromecast by friendly name
python main.py --backend chromecast --device "Kitchen"

# Adjust volume
python main.py --backend chromecast --device "Kitchen" --volume 0.8
```

### Options

| Flag              | Description                                          |
|-------------------|------------------------------------------------------|
| `--backend`       | `sounddevice` (default) or `chromecast`              |
| `--device`        | Device name (sounddevice) or friendly name (cast)    |
| `--list-devices`  | List available outputs and exit                      |
| `--volume`        | Playback volume 0.0–1.0 (default: 0.5)              |
| `--sample-rate`   | Sample rate in Hz (default: 44100)                   |

## Development

Shared tooling config (ruff) lives in the root `pyproject.toml`.

```bash
pip install -r requirements-dev.txt
ruff check feed/ server/
ruff format feed/ server/
```
