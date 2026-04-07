"""
Hand tracking via MediaPipe Hand Landmarker.

Wraps the MediaPipe task API and exposes a simple per-frame interface that
returns normalised hand positions and draws landmarks onto the frame.
"""

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


class HandTracker:
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

    def detect(self, frame_bgr: np.ndarray) -> HandTrackingResult:
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

    def draw(self, frame_bgr: np.ndarray, result: HandTrackingResult) -> np.ndarray:
        h, w = frame_bgr.shape[:2]
        out = frame_bgr.copy()

        for hand in result.hands:
            color = _COLORS.get(hand.label, _DEFAULT_COLOR)
            pts = [(int(lx * w), int(ly * h)) for lx, ly in hand.landmarks]

            for start_idx, end_idx in _CONNECTIONS:
                cv2.line(out, pts[start_idx], pts[end_idx], color, 2)

            for px, py in pts:
                cv2.circle(out, (px, py), 4, color, -1)

            wx, wy = int(hand.x * w), int(hand.y * h)
            cv2.putText(
                out, hand.label, (wx + 10, wy - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2,
            )

        return out

    def close(self) -> None:
        self._landmarker.close()
