import socket
import io
import wave
import numpy as np
from http.server import HTTPServer, BaseHTTPRequestHandler
from base_speakers.audio_backend.base import AudioBackend
import threading
import time


def encode_wav(samples: np.ndarray, sample_rate: int) -> bytes:
    pcm = np.clip(samples * 32767, -32768, 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())
    return buf.getvalue()

def _get_local_ip() -> str:
    """Return the local IP reachable from the LAN (for the Chromecast to fetch audio)."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    finally:
        s.close()


class _WavHandler(BaseHTTPRequestHandler):
    """Serves a single WAV blob from the class-level `wav_data` attribute."""

    wav_data: bytes = b""

    def do_GET(self):
        data = self.wav_data
        self.send_response(200)
        self.send_header("Content-Type", "audio/wav")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Accept-Ranges", "bytes")
        self.end_headers()
        self.wfile.write(data)

    def do_HEAD(self):
        data = self.wav_data
        self.send_response(200)
        self.send_header("Content-Type", "audio/wav")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Accept-Ranges", "bytes")
        self.end_headers()

    def log_message(self, format, *args):
        pass


class ChromecastBackend(AudioBackend):
    """Cast audio to a Chromecast device on the local network."""

    def __init__(self, device: str | None = None):
        import pychromecast
        self._pychromecast = pychromecast

        print("[chromecast] Discovering devices…")
        if device:
            chromecasts, browser = pychromecast.get_listed_chromecasts(
                friendly_names=[device],
            )
        else:
            chromecasts, browser = pychromecast.get_chromecasts()

        self._browser = browser

        if not chromecasts:
            browser.stop_discovery()
            if device:
                raise RuntimeError(f"No Chromecast named '{device}' found on the network")
            raise RuntimeError("No Chromecast devices found on the network")

        self._cast = chromecasts[0]
        self._cast.wait()
        print(f"[chromecast] Connected: {self._cast.name}")

        self._server: HTTPServer | None = None
        self._server_thread: threading.Thread | None = None
        self._media_url: str | None = None

    def _ensure_server(self, wav_data: bytes):
        _WavHandler.wav_data = wav_data

        if self._server is not None:
            return

        self._server = HTTPServer(("0.0.0.0", 0), _WavHandler)
        port = self._server.server_address[1]
        local_ip = _get_local_ip()
        self._media_url = f"http://{local_ip}:{port}/melody.wav"

        self._server_thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._server_thread.start()
        print(f"[chromecast] Serving audio at {self._media_url}")

    def _stop_server(self):
        if self._server:
            self._server.shutdown()
            self._server = None
            self._server_thread = None

    def play(self, samples: np.ndarray, sample_rate: int) -> None:
        wav_data = encode_wav(samples, sample_rate)
        self._ensure_server(wav_data)

        mc = self._cast.media_controller
        mc.play_media(self._media_url, "audio/wav")
        mc.block_until_active()

    def wait(self) -> None:
        mc = self._cast.media_controller
        while mc.status.player_state in ("PLAYING", "BUFFERING"):
            time.sleep(0.3)

    def stop(self) -> None:
        try:
            self._cast.media_controller.stop()
        except Exception:
            pass
        self._stop_server()

    def close(self) -> None:
        self.stop()
        self._cast.disconnect()
        self._browser.stop_discovery()

    @classmethod
    def list_devices(cls):
        import pychromecast
        print("[chromecast] Scanning network…")
        chromecasts, browser = pychromecast.get_chromecasts()
        if not chromecasts:
            print("  (no devices found)")
        for cc in chromecasts:
            model = cc.model_name or "unknown model"
            print(f"  • {cc.name}  ({model})")
        browser.stop_discovery()
