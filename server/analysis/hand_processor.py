"""Hand landmark analysis.

Takes a ``HandTrackingResult`` from the hand tracker and computes
spatial metrics: per-hand height in the frame, inter-hand distance,
and height difference.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from hand_tracker import HandPosition, HandTrackingResult


def _hand_height(hand: HandPosition) -> float:
    """Normalised height (1.0 = top, 0.0 = bottom) from the hand centre."""
    return 1.0 - hand.center[1]


@dataclass
class HandAnalysis:
    hand_count: int = 0
    left_height: float | None = None
    right_height: float | None = None
    hand_distance: float | None = None
    height_difference: float | None = None


class HandProcessor:
    """Processor: ``HandTrackingResult`` → ``HandAnalysis``."""

    input_type = HandTrackingResult

    def process(self, data: HandTrackingResult) -> HandAnalysis:
        left = data.left
        right = data.right

        left_h = _hand_height(left) if left else None
        right_h = _hand_height(right) if right else None

        distance: float | None = None
        height_diff: float | None = None

        if left and right:
            (lx, ly) = left.center
            (rx, ry) = right.center
            distance = math.hypot(rx - lx, ry - ly)

        if left_h is not None and right_h is not None:
            height_diff = left_h - right_h

        return HandAnalysis(
            hand_count=len(data.hands),
            left_height=left_h,
            right_height=right_h,
            hand_distance=distance,
            height_difference=height_diff,
        )
