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
class Blendshape:
    name: str
    score: float


@dataclass
class FacePosition:
    landmarks: list[tuple[float, float]] = field(default_factory=list)
    blendshapes: list[Blendshape] = field(default_factory=list)

    def top_expressions(self, n: int = 5, threshold: float = 0.15) -> list[Blendshape]:
        return [
            b for b in sorted(self.blendshapes, key=lambda b: b.score, reverse=True)
            if b.name != "_neutral" and b.score >= threshold
        ][:n]

    def dominant_emotion(self) -> tuple[str, float]:
        """Map blendshapes to a high-level emotion. Returns (label, confidence)."""
        bs = {b.name: b.score for b in self.blendshapes}
        g = bs.get

        scores = {
            "Happy": (
                g("mouthSmileLeft", 0) + g("mouthSmileRight", 0)
                + 0.5 * g("cheekSquintLeft", 0) + 0.5 * g("cheekSquintRight", 0)
            ) / 3.0,
            "Surprised": (
                g("eyeWideLeft", 0) + g("eyeWideRight", 0)
                + g("jawOpen", 0) + g("browInnerUp", 0)
            ) / 4.0,
            "Angry": (
                g("browDownLeft", 0) + g("browDownRight", 0)
                + 0.5 * g("mouthFrownLeft", 0) + 0.5 * g("mouthFrownRight", 0)
                + 0.4 * g("noseSneerLeft", 0) + 0.4 * g("noseSneerRight", 0)
            ) / 2.8,
            "Sad": (
                g("mouthFrownLeft", 0) + g("mouthFrownRight", 0)
                + 0.6 * g("browInnerUp", 0)
                + 0.3 * g("mouthPucker", 0)
            ) / 2.9,
            "Disgusted": (
                g("noseSneerLeft", 0) + g("noseSneerRight", 0)
                + 0.5 * g("mouthUpperUpLeft", 0) + 0.5 * g("mouthUpperUpRight", 0)
            ) / 3.0,
        }

        best = max(scores, key=scores.__getitem__)
        return best, scores[best]


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
            output_face_blendshapes=True,
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
        """Draw face contours and top expressions onto *frame_bgr* **in-place**."""
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

            emotion, confidence = face.dominant_emotion()
            nose_x, nose_y = pts[1]
            tx = int(nose_x * w) + 60
            ty = int(nose_y * h) - 60

            if confidence >= 0.08:
                emotion_label = f"{emotion} ({confidence:.0%})"
                cv2.putText(
                    frame_bgr, emotion_label, (tx, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 100), 2,
                )

            top = face.top_expressions()
            base_y = ty + 28
            for i, b in enumerate(top):
                label = f"{b.name}: {b.score:.0%}"
                cv2.putText(
                    frame_bgr, label, (tx, base_y + i * 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 180), 1,
                )

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
        for i, landmarks in enumerate(result.face_landmarks):
            blendshapes: list[Blendshape] = []
            if i < len(result.face_blendshapes):
                blendshapes = [
                    Blendshape(name=b.category_name, score=b.score)
                    for b in result.face_blendshapes[i]
                ]
            faces.append(FacePosition(
                landmarks=[(lm.x, lm.y) for lm in landmarks],
                blendshapes=blendshapes,
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
