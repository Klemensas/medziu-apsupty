import time
import numpy as np
import sounddevice as sd

SAMPLE_RATE = 44100
DEVICE_NAME = "USB Audio CODEC"

NOTE_FREQS = {
    "C4": 261.63, "D4": 293.66, "E4": 329.63, "F4": 349.23,
    "G4": 392.00, "A4": 440.00, "B4": 493.88, "C5": 523.25,
    "REST": 0,
}

MELODY = [
    ("E4", 0.3), ("E4", 0.3), ("F4", 0.3), ("G4", 0.3),
    ("G4", 0.3), ("F4", 0.3), ("E4", 0.3), ("D4", 0.3),
    ("C4", 0.3), ("C4", 0.3), ("D4", 0.3), ("E4", 0.3),
    ("E4", 0.45), ("D4", 0.15), ("D4", 0.6),
    ("REST", 0.3),
]

PAUSE_BETWEEN_LOOPS = 1.0


def make_tone(freq: float, duration: float) -> np.ndarray:
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), endpoint=False)
    if freq == 0:
        return np.zeros_like(t)
    envelope = np.ones_like(t)
    attack = int(0.01 * SAMPLE_RATE)
    release = int(0.05 * SAMPLE_RATE)
    if attack < len(envelope):
        envelope[:attack] = np.linspace(0, 1, attack)
    if release < len(envelope):
        envelope[-release:] = np.linspace(1, 0, release)
    return 0.4 * np.sin(2 * np.pi * freq * t) * envelope


def build_melody() -> np.ndarray:
    samples = [make_tone(NOTE_FREQS[note], dur) for note, dur in MELODY]
    return np.concatenate(samples).astype(np.float32)


def find_device(name: str) -> int:
    devices = sd.query_devices()
    num_devices = len(devices)
    for i in range(num_devices):
        dev = sd.query_devices(i)
        if name.lower() in dev["name"].lower() and dev["max_output_channels"] > 0:
            return i
    raise RuntimeError(
        f"No output device matching '{name}' found. Available:\n{devices}"
    )


def main():
    device_index = find_device(DEVICE_NAME)
    device_info = sd.query_devices(device_index)
    print(f"Using device {device_index}: {device_info['name']}")

    melody = build_melody()
    print("Playing melody on loop — press Ctrl+C to stop")
    try:
        while True:
            sd.play(melody, samplerate=SAMPLE_RATE, device=device_index)
            sd.wait()
            time.sleep(PAUSE_BETWEEN_LOOPS)
    except KeyboardInterrupt:
        sd.stop()
        print("\nStopped.")


if __name__ == "__main__":
    main()
