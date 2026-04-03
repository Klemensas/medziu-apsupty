import numpy as np

from base_speakers.audio_backend.base import AudioBackend


class SoundDeviceBackend(AudioBackend):
    """Direct playback through PortAudio / sounddevice."""

    def __init__(self, device: str | int | None = None):
        import sounddevice as sd

        self._sd = sd

        if isinstance(device, str):
            device = self._find_device(device)
        self._device = device

        if device is not None:
            info = sd.query_devices(device)
            print(f"[sounddevice] Using: {info['name']}")
        else:
            info = sd.query_devices(kind="output")
            print(f"[sounddevice] Using default output: {info['name']}")

    def _find_device(self, name: str) -> int:
        for i, dev in enumerate(self._sd.query_devices()):
            if name.lower() in dev["name"].lower() and dev["max_output_channels"] > 0:
                return i
        raise RuntimeError(
            f"No output device matching '{name}'. Available:\n"
            + "\n".join(
                f"  [{i}] {d['name']}"
                for i, d in enumerate(self._sd.query_devices())
                if d["max_output_channels"] > 0
            )
        )

    @classmethod
    def list_devices(cls):
        import sounddevice as sd

        for i, dev in enumerate(sd.query_devices()):
            if dev["max_output_channels"] > 0:
                print(f"  [{i}] {dev['name']}")

    def play(self, samples: np.ndarray, sample_rate: int) -> None:
        self._sd.play(samples, samplerate=sample_rate, device=self._device)

    def wait(self) -> None:
        self._sd.wait()

    def stop(self) -> None:
        self._sd.stop()
