# medziu-apsupty

Speaker test — plays a melody on loop via direct audio output or Chromecast.

## Setup

```
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

```
# Direct playback (default sounddevice output)
python speaker_test.py

# Direct playback on a specific device
python speaker_test.py --device "USB Audio CODEC"

# Scan for Chromecast devices on the network
python speaker_test.py --backend chromecast --list-devices

# Cast to a Chromecast by friendly name
python speaker_test.py --backend chromecast --device "Kitchen"

# Adjust volume
python speaker_test.py --backend chromecast --device "Kitchen" --volume 0.8
```

## Options

| Flag              | Description                                          |
|-------------------|------------------------------------------------------|
| `--backend`       | `sounddevice` (default) or `chromecast`              |
| `--device`        | Device name (sounddevice) or friendly name (cast)    |
| `--list-devices`  | List available outputs and exit                      |
| `--volume`        | Playback volume 0.0–1.0 (default: 0.5)              |
| `--sample-rate`   | Sample rate in Hz (default: 44100)                   |
