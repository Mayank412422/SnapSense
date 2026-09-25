# Setup

Status: **IMPLEMENTED locally**.

Requirements: Python 3.10 or newer. No third-party runtime dependency is required for the baseline.

```bash
python3 -m unittest discover -s tests -v
python3 -m snapsense.app
```

Open `http://127.0.0.1:8765` in a browser and select **Enable sensors**. Camera and microphone permissions are controlled by the browser and host OS. Model adapters can be installed later without changing the fusion or decision contracts.

The microphone recorder uses an audio-only `MediaStream` and negotiated WebM/Opus. The browser sends each non-empty four-second blob to `/api/speech`; the server reports container, codec, duration, sample rate, channel count, and non-zero sample count in the speech response. Use `scripts/inspect_speech_audio.py` to inspect a captured blob without retaining it.
