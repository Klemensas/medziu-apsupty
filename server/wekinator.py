"""
Minimal Wekinator-style mapper: extracts features from a video frame and
maps them to visual-effect parameters via a simple linear transform.

Input features  (extracted per frame):
    0  mean brightness  [0, 255]
    1  mean red         [0, 255]
    2  mean green       [0, 255]
    3  mean blue        [0, 255]
    4  motion magnitude [0, 1]   (normalised frame diff from previous)

Output parameters (drive the video transform):
    hue_shift   float   [-30, 30]   degrees added to every pixel's hue
    saturation  float   [0.5, 2.0]  multiplicative saturation scale
    brightness  float   [0.7, 1.3]  multiplicative brightness scale
"""

import numpy as np

NUM_FEATURES = 5
NUM_OUTPUTS = 3

_WEIGHTS = np.array(
    [
        # hue_shift: driven mainly by R-B colour difference
        [0.08, 0.15, -0.05, -0.15, 10.0],
        # saturation: boosted by motion, lowered by high brightness
        [-0.002, 0.0, 0.0, 0.0, 1.0],
        # brightness: slight lift when dark, dampen when bright
        [-0.002, 0.0, 0.0, 0.0, 0.3],
    ],
    dtype=np.float64,
)

_BIAS = np.array([0.0, 1.0, 1.0], dtype=np.float64)

_OUT_MIN = np.array([-30.0, 0.5, 0.7])
_OUT_MAX = np.array([30.0, 2.0, 1.3])


class SimpleWekinator:
    """Stateful mapper that tracks the previous frame for motion estimation."""

    def __init__(self) -> None:
        self._prev_gray: np.ndarray | None = None

    def extract_features(self, frame_bgr: np.ndarray) -> np.ndarray:
        gray = frame_bgr.mean(axis=2)
        mean_brightness = gray.mean() / 255.0

        means = frame_bgr.mean(axis=(0, 1))
        mean_b, mean_g, mean_r = means[0], means[1], means[2]

        if self._prev_gray is not None:
            diff = np.abs(gray.astype(np.float32) - self._prev_gray.astype(np.float32))
            motion = float(np.clip(diff.mean() / 40.0, 0.0, 1.0))
        else:
            motion = 0.0
        self._prev_gray = gray.astype(np.float32)

        return np.array(
            [mean_brightness * 255, mean_r, mean_g, mean_b, motion], dtype=np.float64
        )

    def map(self, features: np.ndarray) -> dict[str, float]:
        raw = _WEIGHTS @ features + _BIAS
        clamped = np.clip(raw, _OUT_MIN, _OUT_MAX)
        return {
            "hue_shift": float(clamped[0]),
            "saturation": float(clamped[1]),
            "brightness": float(clamped[2]),
        }

    def process(self, frame_bgr: np.ndarray) -> dict[str, float]:
        return self.map(self.extract_features(frame_bgr))
