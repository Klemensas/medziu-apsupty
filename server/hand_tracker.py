"""
Hand tracking via MediaPipe Hand Landmarker.

Wraps the MediaPipe task API and exposes a simple per-frame interface that
returns normalised hand positions and draws landmarks onto the frame.

Detection runs in a background thread so the main pipeline never blocks on
inference — it always uses the latest available result.
"""

import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

_MODEL_PATH = Path(__file__).parent / "models" / "hand_landmarker.task"

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

WRIST = 0
INDEX_TIP = 8
MIDDLE_TIP = 12
PINKY_TIP = 20

_CONNECTIONS = frozenset({
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
})

_COLORS = {
    "Left": (255, 100, 100),
    "Right": (100, 100, 255),
}
_DEFAULT_COLOR = (100, 255, 100)

_EMPTY_RESULT: "HandTrackingResult"


@dataclass
class HandPosition:
    label: str
    x: float
    y: float
    landmarks: list[tuple[float, float]] = field(default_factory=list)


@dataclass
class HandTrackingResult:
    hands: list[HandPosition] = field(default_factory=list)

    @property
    def left(self) -> HandPosition | None:
        return next((h for h in self.hands if h.label == "Left"), None)

    @property
    def right(self) -> HandPosition | None:
        return next((h for h in self.hands if h.label == "Right"), None)


_EMPTY_RESULT = HandTrackingResult()


class HandTracker:
    """Non-blocking hand tracker.

    Call ``submit(frame)`` to queue a frame for detection.  The inference runs
    in a background thread.  Call ``latest()`` at any time to get the most
    recent completed result (never blocks).  ``draw()`` renders the result
    onto a frame in-place.
    """

    def __init__(self, model_path: str | Path = _MODEL_PATH) -> None:
        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=VisionRunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._landmarker = HandLandmarker.create_from_options(options)
        self._frame_ts_ms = 0

        self._lock = threading.Lock()
        self._latest_result: HandTrackingResult = _EMPTY_RESULT
        self._pending_frame: np.ndarray | None = None
        self._busy = False
        self._thread: threading.Thread | None = None

        self._submitted = 0
        self._processed = 0
        self._total_detect_ms = 0.0

    # -- public API ----------------------------------------------------------

    def submit(self, frame_bgr: np.ndarray) -> None:
        """Queue *frame_bgr* for detection.  Non-blocking.

        If the detector is busy with a previous frame, this one replaces
        the pending slot so only the freshest frame is processed.
        """
        with self._lock:
            self._pending_frame = frame_bgr
            self._submitted += 1
            if not self._busy:
                self._busy = True
                self._thread = threading.Thread(target=self._run, daemon=True)
                self._thread.start()

    def latest(self) -> HandTrackingResult:
        """Return the most recent completed result (never blocks)."""
        with self._lock:
            return self._latest_result

    def stats(self) -> dict[str, float | int]:
        with self._lock:
            dropped = self._submitted - self._processed
            avg_ms = (self._total_detect_ms / self._processed) if self._processed else 0.0
            return {
                "submitted": self._submitted,
                "processed": self._processed,
                "dropped": dropped,
                "avg_detect_ms": avg_ms,
            }

    def detect_sync(self, frame_bgr: np.ndarray) -> HandTrackingResult:
        """Blocking single-frame detect (useful for testing)."""
        return self._detect(frame_bgr)

    def draw(self, frame_bgr: np.ndarray, result: HandTrackingResult) -> None:
        """Draw hand landmarks onto *frame_bgr* **in-place**."""
        h, w = frame_bgr.shape[:2]

        for hand in result.hands:
            color = _COLORS.get(hand.label, _DEFAULT_COLOR)
            pts = [(int(lx * w), int(ly * h)) for lx, ly in hand.landmarks]

            for start_idx, end_idx in _CONNECTIONS:
                cv2.line(frame_bgr, pts[start_idx], pts[end_idx], color, 2)

            for px, py in pts:
                cv2.circle(frame_bgr, (px, py), 4, color, -1)

            wx, wy = int(hand.x * w), int(hand.y * h)
            cv2.putText(
                frame_bgr, hand.label, (wx + 10, wy - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2,
            )

    def close(self) -> None:
        if self._thread is not None:
            self._thread.join(timeout=2)
        self._landmarker.close()

    # -- internals -----------------------------------------------------------

    def _detect(self, frame_bgr: np.ndarray) -> HandTrackingResult:
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        self._frame_ts_ms += 33
        result = self._landmarker.detect_for_video(mp_image, self._frame_ts_ms)

        hands: list[HandPosition] = []
        for handedness_list, landmarks in zip(
            result.handedness, result.hand_landmarks, strict=True
        ):
            label = handedness_list[0].category_name
            wrist = landmarks[WRIST]
            hands.append(HandPosition(
                label=label,
                x=wrist.x,
                y=wrist.y,
                landmarks=[(lm.x, lm.y) for lm in landmarks],
            ))

        return HandTrackingResult(hands=hands)

    def _run(self) -> None:
        while True:
            with self._lock:
                frame = self._pending_frame
                self._pending_frame = None
                if frame is None:
                    self._busy = False
                    return

            t0 = time.monotonic()
            result = self._detect(frame)
            elapsed_ms = (time.monotonic() - t0) * 1000

            with self._lock:
                self._latest_result = result
                self._processed += 1
                self._total_detect_ms += elapsed_ms
