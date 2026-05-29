import argparse
import time

from base_speakers.audio_backend.base import AudioBackend
from base_speakers.audio_backend.chromecast import ChromecastBackend
from base_speakers.audio_backend.sound_device import SoundDeviceBackend
from base_speakers.melody import build_melody
from base_speakers.melody_config import PAUSE_BETWEEN_LOOPS, SAMPLE_RATE


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Speaker test — melody playback via direct audio or Chromecast",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  %(prog)s                                        # default local output\n"
            "  %(prog)s --device 'USB Audio CODEC'             # specific local device\n"
            "  %(prog)s --backend chromecast --list-devices     # scan for Chromecasts\n"
            "  %(prog)s --backend chromecast --device 'Kitchen' # cast to a Chromecast\n"
        ),
    )
    p.add_argument(
        "--backend",
        choices=["sounddevice", "chromecast"],
        default="sounddevice",
        help="audio backend (default: sounddevice)",
    )
    p.add_argument(
        "--device",
        default=None,
        help="device name (sounddevice) or friendly name (chromecast)",
    )
    p.add_argument(
        "--list-devices",
        action="store_true",
        help="list available outputs for the chosen backend and exit",
    )
    p.add_argument(
        "--volume",
        type=float,
        default=0.5,
        help="playback volume 0.0-1.0 (default: 0.5)",
    )
    p.add_argument(
        "--sample-rate",
        type=int,
        default=SAMPLE_RATE,
        help=f"sample rate in Hz (default: {SAMPLE_RATE})",
    )
    return p.parse_args()


def main():
    args = parse_args()

    if args.list_devices:
        if args.backend == "chromecast":
            ChromecastBackend.list_devices()
        else:
            print("Sound devices:")
            SoundDeviceBackend.list_devices()
        return

    backend: AudioBackend
    if args.backend == "chromecast":
        backend = ChromecastBackend(device=args.device)
    else:
        backend = SoundDeviceBackend(device=args.device)

    melody = build_melody(args.sample_rate, volume=args.volume)
    print("Playing melody on loop — Ctrl+C to stop")
    try:
        while True:
            backend.play(melody, args.sample_rate)
            backend.wait()
            time.sleep(PAUSE_BETWEEN_LOOPS)
    except KeyboardInterrupt:
        backend.stop()
        print("\nStopped.")
    finally:
        backend.close()
