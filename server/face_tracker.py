"""
Face landmark detection via MediaPipe Face Landmarker.

Same async pattern as hand_tracker: detection runs in a background thread,
the main pipeline always uses the latest completed result without blocking.
"""

import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

_MODEL_PATH = Path(__file__).parent / "models" / "face_landmarker.task"

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

_Conn = mp.tasks.vision.FaceLandmarksConnections
_DRAW_GROUPS: list[tuple[frozenset, tuple[int, int, int], int]] = [
    (_Conn.FACE_LANDMARKS_FACE_OVAL,  (200, 180, 130), 1),
    (_Conn.FACE_LANDMARKS_LEFT_EYE,   (0, 220, 220),   1),
    (_Conn.FACE_LANDMARKS_RIGHT_EYE,  (0, 220, 220),   1),
    (_Conn.FACE_LANDMARKS_LIPS,       (0, 180, 230),    1),
]

_EMPTY_RESULT: "FaceTrackingResult"


@dataclass
class FacePosition:
    landmarks: list[tuple[float, float]] = field(default_factory=list)


@dataclass
class FaceTrackingResult:
    faces: list[FacePosition] = field(default_factory=list)


_EMPTY_RESULT = FaceTrackingResult()


class FaceTracker:
    """Non-blocking face landmark tracker.

    Call ``submit(frame)`` to queue a frame for detection.  The inference runs
    in a background thread.  Call ``latest()`` at any time to get the most
    recent completed result (never blocks).  ``draw()`` renders contours
    onto a frame in-place.
    """

    def __init__(self, model_path: str | Path = _MODEL_PATH) -> None:
        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=VisionRunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._landmarker = FaceLandmarker.create_from_options(options)
        self._frame_ts_ms = 0

        self._lock = threading.Lock()
        self._latest_result: FaceTrackingResult = _EMPTY_RESULT
        self._pending_frame: np.ndarray | None = None
        self._busy = False
        self._thread: threading.Thread | None = None

        self._submitted = 0
        self._processed = 0
        self._total_detect_ms = 0.0

    # -- public API ----------------------------------------------------------

    def submit(self, frame_bgr: np.ndarray) -> None:
        with self._lock:
            self._pending_frame = frame_bgr
            self._submitted += 1
            if not self._busy:
                self._busy = True
                self._thread = threading.Thread(target=self._run, daemon=True)
                self._thread.start()

    def latest(self) -> FaceTrackingResult:
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

    def draw(self, frame_bgr: np.ndarray, result: FaceTrackingResult) -> None:
        """Draw face contours onto *frame_bgr* **in-place**."""
        h, w = frame_bgr.shape[:2]

        for face in result.faces:
            pts = face.landmarks

            for connections, color, thickness in _DRAW_GROUPS:
                for conn in connections:
                    lx0, ly0 = pts[conn.start]
                    lx1, ly1 = pts[conn.end]
                    p0 = (int(lx0 * w), int(ly0 * h))
                    p1 = (int(lx1 * w), int(ly1 * h))
                    cv2.line(frame_bgr, p0, p1, color, thickness)

    def close(self) -> None:
        if self._thread is not None:
            self._thread.join(timeout=2)
        self._landmarker.close()

    # -- internals -----------------------------------------------------------

    def _detect(self, frame_bgr: np.ndarray) -> FaceTrackingResult:
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        self._frame_ts_ms += 33
        result = self._landmarker.detect_for_video(mp_image, self._frame_ts_ms)

        faces: list[FacePosition] = []
        for landmarks in result.face_landmarks:
            faces.append(FacePosition(
                landmarks=[(lm.x, lm.y) for lm in landmarks],
            ))

        return FaceTrackingResult(faces=faces)

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
