"""Face landmark analysis.

Takes a ``FaceTrackingResult`` from the face tracker and distils it
into a compact analysis with the dominant emotion and top expressions.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from face_tracker import Blendshape, FaceTrackingResult


@dataclass
class FaceAnalysis:
    face_detected: bool = False
    dominant_emotion: str | None = None
    emotion_confidence: float = 0.0
    top_expressions: list[Blendshape] = field(default_factory=list)


class FaceProcessor:
    """Processor: ``FaceTrackingResult`` → ``FaceAnalysis``."""

    input_type = FaceTrackingResult

    def process(self, data: FaceTrackingResult) -> FaceAnalysis:
        if not data.faces:
            return FaceAnalysis()

        face = data.faces[0]
        emotion, confidence = face.dominant_emotion()

        return FaceAnalysis(
            face_detected=True,
            dominant_emotion=emotion,
            emotion_confidence=confidence,
            top_expressions=face.top_expressions(),
        )
