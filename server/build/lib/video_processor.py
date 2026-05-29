"""Frame-level decode → transform → encode pipeline."""

import cv2
import numpy as np


def decode_frame(data: bytes) -> np.ndarray | None:
    buf = np.frombuffer(data, dtype=np.uint8)
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def apply_transform(
    frame_bgr: np.ndarray,
    *,
    hue_shift: float = 0.0,
    saturation: float = 1.0,
    brightness: float = 1.0,
) -> np.ndarray:
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

    h, s, v = cv2.split(hsv)
    if hue_shift != 0.0:
        h = ((h.astype(np.int16) + int(hue_shift)) % 180).astype(np.uint8)
    if saturation != 1.0:
        s = cv2.multiply(s, saturation, dtype=cv2.CV_8U)
    if brightness != 1.0:
        v = cv2.multiply(v, brightness, dtype=cv2.CV_8U)

    cv2.merge([h, s, v], dst=hsv)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def encode_frame(frame_bgr: np.ndarray, *, quality: int = 80) -> bytes:
    ok, buf = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("JPEG encode failed")
    return buf.tobytes()
