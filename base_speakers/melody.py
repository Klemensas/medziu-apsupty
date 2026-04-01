import numpy as np

from base_speakers.melody_config import MELODY, NOTE_FREQS


def make_tone(freq: float, duration: float, sample_rate: int, volume: float = 0.5) -> np.ndarray:
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    if freq == 0:
        return np.zeros_like(t)
    envelope = np.ones_like(t)
    attack = int(0.01 * sample_rate)
    release = int(0.05 * sample_rate)
    if attack < len(envelope):
        envelope[:attack] = np.linspace(0, 1, attack)
    if release < len(envelope):
        envelope[-release:] = np.linspace(1, 0, release)
    return volume * np.sin(2 * np.pi * freq * t) * envelope


def build_melody(sample_rate: int, volume: float = 0.5) -> np.ndarray:
    samples = [make_tone(NOTE_FREQS[note], dur, sample_rate, volume) for note, dur in MELODY]
    return np.concatenate(samples).astype(np.float32)
