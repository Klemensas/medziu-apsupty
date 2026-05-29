"""Base-beat transformer — generates a simple melody driven by face
emotion and hand presence.

Emotion selects the musical scale, root note, and tempo.  Both hands
being visible activates the full melody; one hand produces a sparse
pattern; no hands yields silence.

The transformer maintains a ``BeatState`` that other components (e.g.
the feed audio system) can read to know what to play.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from analysis.face_processor import FaceAnalysis
from analysis.hand_processor import HandAnalysis

# ---------------------------------------------------------------------------
# Music theory helpers
# ---------------------------------------------------------------------------

_CHROMATIC = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

_SCALE_INTERVALS: dict[str, list[int]] = {
    "major": [0, 2, 4, 5, 7, 9, 11],
    "minor": [0, 2, 3, 5, 7, 8, 10],
    "dorian": [0, 2, 3, 5, 7, 9, 10],
    "phrygian": [0, 1, 3, 5, 7, 8, 10],
}

_EMOTION_PARAMS: dict[str, dict[str, Any]] = {
    "Happy": {"scale": "major", "root": "C", "octave": 4, "tempo": 120, "feel": "bright"},
    "Sad": {"scale": "minor", "root": "A", "octave": 3, "tempo": 76, "feel": "gentle"},
    "Angry": {"scale": "phrygian", "root": "D", "octave": 4, "tempo": 140, "feel": "driving"},
    "Surprised": {"scale": "major", "root": "G", "octave": 4, "tempo": 108, "feel": "playful"},
    "Disgusted": {"scale": "dorian", "root": "E", "octave": 3, "tempo": 90, "feel": "dark"},
}

_DEFAULT_PARAMS = _EMOTION_PARAMS["Happy"]

# Melodic patterns per feel — each entry is (scale_degree, beat_count).
# Scale degree -1 is a rest.
_PATTERNS: dict[str, list[tuple[int, float]]] = {
    "bright": [
        (0, 1), (2, 1), (4, 1), (2, 1),
        (4, 1), (5, 1), (4, 2), (-1, 1),
    ],
    "gentle": [
        (0, 2), (2, 1), (4, 2), (2, 1),
        (0, 2), (-1, 2),
    ],
    "driving": [
        (0, 0.5), (0, 0.5), (3, 0.5), (4, 0.5),
        (0, 0.5), (6, 0.5), (4, 1), (-1, 0.5), (0, 0.5),
    ],
    "playful": [
        (0, 1), (4, 0.5), (2, 0.5), (4, 1),
        (5, 1), (4, 0.5), (2, 0.5), (0, 1), (-1, 1),
    ],
    "dark": [
        (0, 2), (1, 1), (2, 2), (4, 1),
        (3, 2), (-1, 1), (0, 1),
    ],
}

# Sparse pattern when only one hand is visible.
_SPARSE_PATTERN: list[tuple[int, float]] = [
    (0, 2), (-1, 2), (4, 2), (-1, 2),
]

# Minimum consecutive frames before the emotion switch is committed.
_EMOTION_HOLD_FRAMES = 15


def _note_freq(name: str, octave: int) -> float:
    """MIDI-based frequency: A4 = 440 Hz."""
    semitone = _CHROMATIC.index(name) - _CHROMATIC.index("A")
    midi = 69 + semitone + 12 * (octave - 4)
    return 440.0 * (2.0 ** ((midi - 69) / 12.0))


def _build_scale(root: str, octave: int, scale: str) -> list[tuple[str, int, float]]:
    """Return ``[(note_name, octave, freq), ...]`` for one octave."""
    root_idx = _CHROMATIC.index(root)
    result: list[tuple[str, int, float]] = []
    for interval in _SCALE_INTERVALS[scale]:
        abs_idx = root_idx + interval
        oct = octave + abs_idx // 12
        name = _CHROMATIC[abs_idx % 12]
        result.append((name, oct, _note_freq(name, oct)))
    return result


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class BeatNote:
    name: str
    freq: float
    duration: float


@dataclass
class BeatState:
    active: bool = False
    emotion: str | None = None
    tempo_bpm: float = 120.0
    volume: float = 0.5
    notes: list[BeatNote] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Transformer
# ---------------------------------------------------------------------------

class BaseBeatTransformer:
    """Pipeline transformer: ``{FaceAnalysis, HandAnalysis}`` → ``BeatState``.

    ``transform()`` returns a JSON-serialisable dict (via
    ``dataclasses.asdict``) so the pipeline can include it in the
    response payload sent to the feed.
    """

    name = "beat"
    input_types = frozenset({FaceAnalysis, HandAnalysis})

    def __init__(self) -> None:
        self._state = BeatState()
        self._active_emotion: str | None = None
        self._candidate_emotion: str | None = None
        self._candidate_count: int = 0

    @property
    def state(self) -> BeatState:
        return self._state

    # -- Transformer protocol ------------------------------------------------

    def transform(self, analyses: dict[type, Any]) -> dict[str, Any]:
        face: FaceAnalysis | None = analyses.get(FaceAnalysis)
        hand: HandAnalysis | None = analyses.get(HandAnalysis)

        if face is None or not face.face_detected:
            self._state = BeatState()
            return asdict(self._state)

        emotion = self._resolve_emotion(face)
        both_hands = hand is not None and hand.hand_count >= 2
        one_hand = hand is not None and hand.hand_count == 1

        params = _EMOTION_PARAMS.get(emotion, _DEFAULT_PARAMS)
        scale = _build_scale(params["root"], params["octave"], params["scale"])
        beat_dur = 60.0 / params["tempo"]

        volume = min(1.0, 0.3 + face.emotion_confidence)

        if both_hands:
            pattern = _PATTERNS.get(params["feel"], _PATTERNS["bright"])
        elif one_hand:
            pattern = _SPARSE_PATTERN
        else:
            self._state = BeatState(
                active=False, emotion=emotion,
                tempo_bpm=params["tempo"], volume=0.0,
            )
            return asdict(self._state)

        notes: list[BeatNote] = []
        for degree, beats in pattern:
            dur = beat_dur * beats
            if degree < 0 or degree >= len(scale):
                notes.append(BeatNote(name="REST", freq=0.0, duration=dur))
            else:
                name, oct, freq = scale[degree]
                notes.append(BeatNote(name=f"{name}{oct}", freq=freq, duration=dur))

        self._state = BeatState(
            active=True,
            emotion=emotion,
            tempo_bpm=params["tempo"],
            volume=volume,
            notes=notes,
        )
        return asdict(self._state)

    def debug(self, frame_bgr: np.ndarray) -> None:
        """Draw beat state info onto the frame."""
        import cv2

        s = self._state
        color = (0, 200, 200) if s.active else (80, 80, 80)
        label = f"Beat: {s.emotion or 'off'}  {s.tempo_bpm:.0f}bpm  vol={s.volume:.0%}"
        cv2.putText(frame_bgr, label, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        if s.active and s.notes:
            seq = " ".join(n.name for n in s.notes[:8])
            cv2.putText(frame_bgr, seq, (10, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

    # -- internals -----------------------------------------------------------

    def _resolve_emotion(self, face: FaceAnalysis) -> str:
        """Hysteresis: only switch emotion after it's stable for several frames."""
        incoming = face.dominant_emotion or "Happy"

        if incoming == self._active_emotion:
            self._candidate_emotion = None
            self._candidate_count = 0
            return incoming

        if incoming == self._candidate_emotion:
            self._candidate_count += 1
        else:
            self._candidate_emotion = incoming
            self._candidate_count = 1

        if self._candidate_count >= _EMOTION_HOLD_FRAMES:
            self._active_emotion = incoming
            self._candidate_emotion = None
            self._candidate_count = 0
            return incoming

        return self._active_emotion or incoming
