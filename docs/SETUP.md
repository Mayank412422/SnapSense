# Setup

Status: **IMPLEMENTED locally**.

Requirements: Python 3.10 or newer. No third-party runtime dependency is required for the baseline.

```bash
python3 -m unittest discover -s tests -v
python3 -m snapsense.app
```

Open `http://127.0.0.1:8765` in a browser and select **Enable sensors**. Camera and microphone permissions are controlled by the browser and host OS. Model adapters can be installed later without changing the fusion or decision contracts.
