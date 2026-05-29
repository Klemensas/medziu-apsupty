#!/usr/bin/env python3
"""
Test tool — reads a local video file or live camera feed, streams frames
to the processing server over WebSocket, and displays the transformed
output in a window.

Uses ffmpeg as a subprocess for video file decode + scale + fps conversion
so the Python side only handles lightweight JPEG encode (at 640x480) and
display.  For camera input, OpenCV's VideoCapture is used directly.

A background reader thread drains the source continuously and keeps only
the latest frame, so if the server can't keep up we skip stale frames
instead of building up lag.

Requires the [test] optional dependencies and ffmpeg on PATH (for files):
    pip install -e ".[test]"
    brew install ffmpeg        # or apt install ffmpeg
"""

import argparse
import asyncio
import json
import shutil
import signal
import struct
import subprocess
import threading
import time

import cv2
import numpy as np
import sounddevice as sd
import websockets
from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import ThreadingOSCUDPServer

from base_speakers.melody import make_tone

WIDTH = 640
HEIGHT = 480
TARGET_FPS = 30
JPEG_QUALITY = 80

SAMPLE_RATE = 44100
OSC_PORT = 9000


class BeatPlayer:
    """Looping audio player driven by OSC beat data.

    Synthesises a melody from note frequencies/durations and loops it
    through a sounddevice OutputStream.  Thread-safe: ``update()`` can
    be called from the OSC handler thread at any time.
    """

    def __init__(self, sample_rate: int = SAMPLE_RATE) -> None:
        self._sr = sample_rate
        self._lock = threading.Lock()
        self._melody = np.zeros(0, dtype=np.float32)
        self._pos = 0
        self._volume = 0.5
        self._active = False
        self._stream = sd.OutputStream(
            samplerate=sample_rate,
            channels=1,
            dtype="float32",
            callback=self._callback,
            blocksize=1024,
        )
        self._stream.start()

    def _callback(self, outdata: np.ndarray, frames: int, _time, _status) -> None:
        with self._lock:
            mel = self._melody
            if not self._active or len(mel) == 0:
                outdata[:] = 0.0
                return
            out = np.empty(frames, dtype=np.float32)
            written = 0
            pos = self._pos
            while written < frames:
                chunk = min(frames - written, len(mel) - pos)
                out[written : written + chunk] = mel[pos : pos + chunk]
                pos = (pos + chunk) % len(mel)
                written += chunk
            self._pos = pos
        outdata[:, 0] = out

    def update(self, notes: list[dict], volume: float) -> None:
        """Rebuild the melody from a list of ``{freq, duration}`` dicts."""
        if not notes or volume <= 0:
            with self._lock:
                self._active = False
                self._melody = np.zeros(0, dtype=np.float32)
                self._pos = 0
            return

        samples = [
            make_tone(n["freq"], n["duration"], self._sr, volume)
            for n in notes
        ]
        melody = np.concatenate(samples).astype(np.float32)
        with self._lock:
            self._melody = melody
            self._pos = 0
            self._volume = volume
            self._active = True

    @property
    def active(self) -> bool:
        with self._lock:
            return self._active

    def stop(self) -> None:
        self._stream.stop()
        self._stream.close()


def _start_osc_listener(player: BeatPlayer, port: int = OSC_PORT) -> ThreadingOSCUDPServer:
    """Start a background OSC server that routes /beat messages to *player*."""

    def _handle_beat(_address: str, *args) -> None:
        active = bool(args[0]) if len(args) > 0 else False
        volume = float(args[3]) if len(args) > 3 else 0.0
        notes_json = args[4] if len(args) > 4 else "[]"

        if active:
            notes = json.loads(notes_json)
            player.update(notes, volume)
        else:
            player.update([], 0.0)

    dispatcher = Dispatcher()
    dispatcher.map("/beat", _handle_beat)
    server = ThreadingOSCUDPServer(("0.0.0.0", port), dispatcher)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


def _probe_video(path: str) -> dict:
    """Use ffprobe to get stream info."""
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate,nb_frames,duration",
        "-show_entries", "format=duration",
        "-of", "json",
        path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    data = json.loads(result.stdout)
    stream = data.get("streams", [{}])[0]
    fmt = data.get("format", {})

    num, den = stream.get("r_frame_rate", "30/1").split("/")
    src_fps = int(num) / int(den)

    duration = float(stream.get("duration") or fmt.get("duration") or 0)
    nb_frames = int(stream.get("nb_frames") or 0)
    if nb_frames == 0 and duration > 0:
        nb_frames = int(duration * src_fps)

    return {
        "src_width": int(stream.get("width", 0)),
        "src_height": int(stream.get("height", 0)),
        "src_fps": src_fps,
        "duration": duration,
        "nb_frames": nb_frames,
        "est_output_frames": int(duration * TARGET_FPS) if duration else nb_frames,
    }


def _open_ffmpeg(path: str) -> subprocess.Popen:
    """Spawn ffmpeg to decode, scale, and fps-convert to raw BGR frames."""
    cmd = [
        "ffmpeg",
        "-v", "error",
        "-re",
        "-i", path,
        "-vf", f"scale={WIDTH}:{HEIGHT},fps={TARGET_FPS}",
        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "-",
    ]
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


class FrameReader:
    """Drains ffmpeg stdout in a background thread, keeping only the latest frame."""

    def __init__(self, pipe, frame_bytes: int) -> None:
        self._pipe = pipe
        self._frame_bytes = frame_bytes
        self._lock = threading.Lock()
        self._latest: np.ndarray | None = None
        self._finished = False
        self._read_count = 0
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while True:
            raw = self._pipe.read(self._frame_bytes)
            if len(raw) < self._frame_bytes:
                with self._lock:
                    self._finished = True
                return
            frame = np.frombuffer(raw, dtype=np.uint8).reshape(HEIGHT, WIDTH, 3).copy()
            with self._lock:
                self._latest = frame
                self._read_count += 1

    def get(self) -> tuple[np.ndarray | None, int]:
        """Return (latest_frame_or_None, total_frames_read).

        Consuming a frame clears the slot so the same frame is never sent twice.
        """
        with self._lock:
            frame = self._latest
            self._latest = None
            return frame, self._read_count

    @property
    def finished(self) -> bool:
        with self._lock:
            return self._finished


class CameraReader:
    """Reads frames from a camera device in a background thread, keeping only the latest."""

    def __init__(self, device: int) -> None:
        self._cap = cv2.VideoCapture(device)
        if not self._cap.isOpened():
            raise RuntimeError(f"cannot open camera device {device}")
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
        self._lock = threading.Lock()
        self._latest: np.ndarray | None = None
        self._finished = False
        self._read_count = 0
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

        self.src_fps = self._cap.get(cv2.CAP_PROP_FPS)

    def _run(self) -> None:
        while True:
            ret, frame = self._cap.read()
            if not ret:
                with self._lock:
                    self._finished = True
                return
            frame = cv2.resize(frame, (WIDTH, HEIGHT))
            with self._lock:
                self._latest = frame
                self._read_count += 1

    def get(self) -> tuple[np.ndarray | None, int]:
        with self._lock:
            frame = self._latest
            self._latest = None
            return frame, self._read_count

    @property
    def finished(self) -> bool:
        with self._lock:
            return self._finished

    def release(self) -> None:
        self._cap.release()


class FpsCounter:
    """Sliding-window FPS tracker."""

    def __init__(self, window: float = 2.0) -> None:
        self._window = window
        self._timestamps: list[float] = []

    def tick(self) -> None:
        self._timestamps.append(time.monotonic())

    def fps(self) -> float:
        now = time.monotonic()
        cutoff = now - self._window
        self._timestamps = [t for t in self._timestamps if t > cutoff]
        if len(self._timestamps) < 2:
            return 0.0
        span = self._timestamps[-1] - self._timestamps[0]
        return (len(self._timestamps) - 1) / span if span > 0 else 0.0


def _unpack_response(data: bytes) -> tuple[bytes, dict]:
    """Decode the server's framed response.

    Format: [4B video_len (uint32 BE)][video_len B JPEG][rest: JSON metadata]

    Returns ``(jpeg_bytes, metadata_dict)``.
    """
    if len(data) < 4:
        return data, {}
    video_len = struct.unpack("!I", data[:4])[0]
    jpeg_bytes = data[4 : 4 + video_len]
    tail = data[4 + video_len :]
    meta = json.loads(tail) if tail else {}
    return jpeg_bytes, meta


async def _stream_loop(
    reader: FrameReader | CameraReader,
    server_url: str,
    player: BeatPlayer,
    est_total: int | None = None,
) -> None:
    """Core streaming loop shared by file and camera modes."""
    counter = FpsCounter()
    sent = 0
    t_start = time.monotonic()
    total_encode_ms = 0.0
    total_rtt_ms = 0.0
    total_decode_ms = 0.0
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]

    async with websockets.connect(server_url, max_size=10 * 1024 * 1024) as ws:
        print("connected — streaming (press q in window to stop)\n")

        while True:
            frame, total_read = reader.get()

            if frame is None:
                if reader.finished:
                    break
                await asyncio.sleep(0.001)
                continue

            t_enc = time.monotonic()
            ok, buf = cv2.imencode(".jpg", frame, encode_params)
            if not ok:
                continue
            jpeg_bytes = buf.tobytes()
            encode_ms = (time.monotonic() - t_enc) * 1000
            total_encode_ms += encode_ms

            t_rtt = time.monotonic()
            await ws.send(jpeg_bytes)
            response = await ws.recv()
            rtt_ms = (time.monotonic() - t_rtt) * 1000
            total_rtt_ms += rtt_ms

            t_dec = time.monotonic()
            video_data, meta = _unpack_response(response)
            result = cv2.imdecode(
                np.frombuffer(video_data, dtype=np.uint8), cv2.IMREAD_COLOR
            )
            decode_ms = (time.monotonic() - t_dec) * 1000
            total_decode_ms += decode_ms

            beat = meta.get("beat")

            counter.tick()
            sent += 1
            dropped = total_read - sent

            if result is not None:
                current_fps = counter.fps()
                overlay = (
                    f"fps: {current_fps:5.1f}  |  rtt: {rtt_ms:5.1f}ms"
                    f"  |  enc: {encode_ms:4.1f}ms"
                    f"  |  drop: {dropped}"
                )
                cv2.putText(
                    result, overlay, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2,
                )
                if beat and beat.get("active"):
                    beat_label = (
                        f"Beat: {beat.get('emotion', '?')} "
                        f"{beat.get('tempo_bpm', 0):.0f}bpm "
                        f"vol={beat.get('volume', 0):.0%}"
                    )
                    color = (0, 220, 200) if player.active else (0, 100, 100)
                    cv2.putText(
                        result, beat_label, (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1,
                    )
                    audio_tag = "PLAYING" if player.active else "MUTED"
                    cv2.putText(
                        result, audio_tag, (10, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1,
                    )
                cv2.imshow("server output", result)

            if sent % 100 == 0:
                elapsed = time.monotonic() - t_start
                avg_fps = sent / elapsed
                total_label = f"/{est_total}" if est_total else ""
                print(
                    f"  sent {sent:>6}  read {total_read:>6}{total_label}"
                    f"  drop {dropped}"
                    f"   fps: {counter.fps():5.1f} (avg {avg_fps:.1f})"
                    f"   rtt: {rtt_ms:.1f}ms (avg {total_rtt_ms / sent:.1f})"
                    f"   enc: {encode_ms:.1f}ms (avg {total_encode_ms / sent:.1f})"
                    f"   dec: {decode_ms:.1f}ms (avg {total_decode_ms / sent:.1f})"
                    f"   out: {len(jpeg_bytes) // 1024}KB"
                    f"   audio: {'on' if player.active else 'off'}"
                )

            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("\ninterrupted by user (q)")
                break

    elapsed = time.monotonic() - t_start
    _, total_read = reader.get()
    total_read = max(total_read, sent)
    if sent > 0:
        print(
            f"\ndone — sent {sent}, read {total_read}, dropped {total_read - sent}"
            f"  in {elapsed:.1f}s  (avg {sent / elapsed:.1f} fps)\n"
            f"  avg rtt:    {total_rtt_ms / sent:.1f}ms\n"
            f"  avg encode: {total_encode_ms / sent:.1f}ms\n"
            f"  avg decode: {total_decode_ms / sent:.1f}ms"
        )


async def stream_video(video_path: str, server_url: str) -> None:
    info = _probe_video(video_path)
    est_total = info["est_output_frames"]
    print(
        f"source : {video_path}\n"
        f"         {info['src_width']}x{info['src_height']} @ {info['src_fps']:.1f} fps"
        f"  ({info['duration']:.1f}s, ~{info['nb_frames']} frames)\n"
        f"output : {WIDTH}x{HEIGHT} @ {TARGET_FPS} fps  (~{est_total} frames)\n"
        f"server : {server_url}\n"
    )

    player = BeatPlayer()
    osc_server = _start_osc_listener(player)
    print(f"OSC listener on :{OSC_PORT}  |  audio via sounddevice\n")

    proc = _open_ffmpeg(video_path)
    frame_bytes = WIDTH * HEIGHT * 3
    reader = FrameReader(proc.stdout, frame_bytes)

    try:
        await _stream_loop(reader, server_url, player, est_total=est_total)
    except (KeyboardInterrupt, asyncio.CancelledError):
        print("\ninterrupted")
    finally:
        proc.terminate()
        proc.wait()
        player.stop()
        osc_server.shutdown()
        cv2.destroyAllWindows()


async def stream_camera(device: int, server_url: str) -> None:
    reader = CameraReader(device)
    print(
        f"source : camera {device} @ {reader.src_fps:.0f} fps\n"
        f"output : {WIDTH}x{HEIGHT}\n"
        f"server : {server_url}\n"
    )

    player = BeatPlayer()
    osc_server = _start_osc_listener(player)
    print(f"OSC listener on :{OSC_PORT}  |  audio via sounddevice\n")

    try:
        await _stream_loop(reader, server_url, player)
    except (KeyboardInterrupt, asyncio.CancelledError):
        print("\ninterrupted")
    finally:
        reader.release()
        player.stop()
        osc_server.shutdown()
        cv2.destroyAllWindows()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Stream a video file or live camera feed to the processing server",
    )
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("video", nargs="?", default=None, help="path to a video file")
    source.add_argument(
        "-c", "--camera",
        type=int,
        metavar="DEVICE",
        help="use a camera device index (e.g. 0 for the default webcam)",
    )
    p.add_argument(
        "--server",
        default="ws://localhost:8000/ws/video",
        help="WebSocket URL of the processing server (default: ws://localhost:8000/ws/video)",
    )
    return p.parse_args()


def main() -> None:
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    args = parse_args()

    if args.camera is not None:
        try:
            asyncio.run(stream_camera(args.camera, args.server))
        except KeyboardInterrupt:
            pass
    else:
        if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
            print("error: ffmpeg and ffprobe must be installed and on PATH")
            raise SystemExit(1)
        try:
            asyncio.run(stream_video(args.video, args.server))
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
