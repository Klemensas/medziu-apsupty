"""
Processing server — receives JPEG frames over a WebSocket, applies a
Wekinator-style visual transform, and streams the result back.

Protocol (binary WebSocket):
    client → server : raw JPEG bytes (one message per frame)
    server → client : transformed JPEG bytes
"""

import logging
import time

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from hand_tracker import HandTracker
from video_processor import apply_transform, decode_frame, encode_frame
from wekinator import SimpleWekinator

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
log = logging.getLogger("server")

app = FastAPI(title="medziu-apsupty processing server")
wek = SimpleWekinator()
hands = HandTracker()


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.websocket("/ws/video")
async def video_stream(ws: WebSocket):
    await ws.accept()
    log.info("feed connected: %s", ws.client)

    idx = 0
    total_proc_ms = 0.0
    try:
        while True:
            data = await ws.receive_bytes()

            t0 = time.monotonic()
            frame = decode_frame(data)
            if frame is None:
                log.warning("failed to decode frame, skipping")
                continue

            tracking = hands.detect(frame)
            params = wek.process(frame)
            transformed = apply_transform(frame, **params)
            transformed = hands.draw(transformed, tracking)
            out_bytes = encode_frame(transformed)
            proc_ms = (time.monotonic() - t0) * 1000

            await ws.send_bytes(out_bytes)

            idx += 1
            total_proc_ms += proc_ms
            if idx % 100 == 0:
                avg_ms = total_proc_ms / idx
                log.info(
                    "frame %d  proc: %.1fms (avg %.1fms)  size: %dx%d  out: %dKB",
                    idx, proc_ms, avg_ms,
                    frame.shape[1], frame.shape[0],
                    len(out_bytes) // 1024,
                )
    except WebSocketDisconnect:
        if idx > 0:
            avg_ms = total_proc_ms / idx
            log.info(
                "feed disconnected: %s  (%d frames, avg proc %.1fms)",
                ws.client, idx, avg_ms,
            )
        else:
            log.info("feed disconnected: %s", ws.client)


def main():
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
