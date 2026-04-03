#!/usr/bin/env python3
"""
Test tool — reads a local video file, streams frames to the processing
server over WebSocket, and displays the transformed output in a window.

Uses ffmpeg as a subprocess for decode + scale + fps conversion so the
Python side only handles lightweight JPEG encode (at 640x480) and display.

A background reader thread drains ffmpeg continuously and keeps only the
latest frame, so if the server can't keep up we skip stale frames instead
of building up lag.

Requires the [test] optional dependencies and ffmpeg on PATH:
    pip install -e ".[test]"
    brew install ffmpeg        # or apt install ffmpeg
"""

import argparse
import asyncio
import json
import shutil
import signal
import subprocess
import threading
import time

import cv2
import numpy as np
import websockets

WIDTH = 640
HEIGHT = 480
TARGET_FPS = 30
JPEG_QUALITY = 80


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
    src_fps = int(num) / int(den) if int(den) else 30.0

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

    proc = _open_ffmpeg(video_path)
    frame_bytes = WIDTH * HEIGHT * 3
    reader = FrameReader(proc.stdout, frame_bytes)
    counter = FpsCounter()
    sent = 0
    t_start = time.monotonic()
    total_encode_ms = 0.0
    total_rtt_ms = 0.0
    total_decode_ms = 0.0
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]

    try:
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
                result = cv2.imdecode(
                    np.frombuffer(response, dtype=np.uint8), cv2.IMREAD_COLOR
                )
                decode_ms = (time.monotonic() - t_dec) * 1000
                total_decode_ms += decode_ms

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
                    cv2.imshow("server output", result)

                if sent % 100 == 0:
                    elapsed = time.monotonic() - t_start
                    avg_fps = sent / elapsed
                    print(
                        f"  sent {sent:>6}  read {total_read:>6}/{est_total}"
                        f"  drop {dropped}"
                        f"   fps: {counter.fps():5.1f} (avg {avg_fps:.1f})"
                        f"   rtt: {rtt_ms:.1f}ms (avg {total_rtt_ms / sent:.1f})"
                        f"   enc: {encode_ms:.1f}ms (avg {total_encode_ms / sent:.1f})"
                        f"   dec: {decode_ms:.1f}ms (avg {total_decode_ms / sent:.1f})"
                        f"   out: {len(jpeg_bytes) // 1024}KB"
                    )

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("\ninterrupted by user (q)")
                    break

    except (KeyboardInterrupt, asyncio.CancelledError):
        print("\ninterrupted")
    finally:
        proc.terminate()
        proc.wait()
        cv2.destroyAllWindows()

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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Send a video file to the processing server and display the result",
    )
    p.add_argument("video", help="path to the video file")
    p.add_argument(
        "--server",
        default="ws://localhost:8000/ws/video",
        help="WebSocket URL of the processing server (default: ws://localhost:8000/ws/video)",
    )
    return p.parse_args()


def main() -> None:
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        print("error: ffmpeg and ffprobe must be installed and on PATH")
        raise SystemExit(1)

    signal.signal(signal.SIGINT, signal.SIG_DFL)
    args = parse_args()
    try:
        asyncio.run(stream_video(args.video, args.server))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
