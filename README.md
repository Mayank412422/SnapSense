# SnapSense Edge

SnapSense Edge is a local-first multimodal application baseline. It combines camera and microphone permissions in a browser UI with replaceable vision and speech workers, deterministic context fusion, mode routing, local decisions, and runtime telemetry.

## Status

- **IMPLEMENTED:** local HTTP application, technical UI, sensor permission flow, typed context contracts, deterministic Study/Interview/Coding decisions, benchmark statistics, tests, and runtime registry.
- **IMPLEMENTED:** optional real Whisper-Tiny and YOLOv4-Tiny CPU inference paths, independent worker endpoints, real-media integration tests, and host benchmark scripts.
- **PARTIALLY IMPLEMENTED:** browser camera/microphone ingestion replaces the specification's OpenCV/PortAudio worker inputs; active-window context is API-ready but not collected by the browser.
- **PENDING:** Qualcomm AI Hub compilation, Snapdragon profiling, NPU/CPU mapping, target memory and latency, output parity, and physical-device validation.

## Run

```bash
python3 -m snapsense.app
```

Open `http://127.0.0.1:8765`. The browser captures short audio buffers and JPEG frames and sends them to the loopback service for transient processing. The default workers return `UNKNOWN` until model artifacts and optional dependencies are configured.

Install real local adapters with `python3 -m pip install -e '.[speech,vision]'`. Whisper-Tiny uses the CTranslate2-compatible `Systran/faster-whisper-tiny` artifact and downloads it on first use when `SNAPSENSE_ENABLE_WHISPER=1` is set. Download the YOLOv4-Tiny ONNX candidate with `PYTHONPATH=. python3 scripts/download_models.py --vision`, then set `SNAPSENSE_VISION_MODEL=models/yolov4-tiny.onnx`, `SNAPSENSE_VISION_CANDIDATE=YOLOv4-Tiny`, and `SNAPSENSE_VISION_LABELS=models/coco.names`. RTMDet can use the same interface with a compatible exported ONNX artifact and `SNAPSENSE_VISION_CANDIDATE=RTMDet`. Model downloads and artifact availability are external prerequisites.

## Quick start: fallback mode

This sequence runs without model artifacts and shows honest `UNKNOWN`/`PENDING` states:

```bash
python3 -m pip install -e .
python3 -m unittest discover -s tests -v
PYTHONPATH=. python3 scripts/benchmark_models.py
PYTHONPATH=. python3 scripts/benchmark_app.py
python3 -m snapsense.app
```

Open `http://127.0.0.1:8765` after the final command.

## Real CPU model run

Install optional runtimes, download the YOLOv4-Tiny artifact, configure both workers, then provide real media fixtures. Whisper-Tiny downloads its CTranslate2 artifact on first use:

```bash
python3 -m pip install -e '.[speech,vision]'
PYTHONPATH=. python3 scripts/download_models.py --vision --directory models
export SNAPSENSE_VISION_MODEL=models/yolov4-tiny.onnx
export SNAPSENSE_VISION_CANDIDATE=YOLOv4-Tiny
export SNAPSENSE_VISION_LABELS=models/coco.names
export SNAPSENSE_ENABLE_WHISPER=1
export SNAPSENSE_REAL_VISION_IMAGE=/path/to/real-image.jpg
export SNAPSENSE_REAL_AUDIO=/path/to/real-audio.webm
python3 -m unittest discover -s tests -v
PYTHONPATH=. python3 scripts/benchmark_models.py --image "$SNAPSENSE_REAL_VISION_IMAGE" --audio "$SNAPSENSE_REAL_AUDIO"
PYTHONPATH=. python3 scripts/benchmark_app.py --image "$SNAPSENSE_REAL_VISION_IMAGE" --audio "$SNAPSENSE_REAL_AUDIO"
```

Inspect the exact bytes from a browser capture before transcription:

```bash
PYTHONPATH=. python3 scripts/inspect_speech_audio.py /path/to/browser-capture.webm --transcribe
```

The model directory and media fixtures are ignored and must not be committed. See [docs/REQUIREMENTS_CHECKLIST.md](docs/REQUIREMENTS_CHECKLIST.md) for the complete implementation audit.

Run the minimal checks independently with:

```bash
python3 -m unittest discover -s tests -v
PYTHONPATH=. python3 scripts/benchmark_app.py
PYTHONPATH=. python3 scripts/benchmark_models.py --image /path/to/frame.jpg --audio /path/to/audio.webm
```

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md), [docs/MODELS.md](docs/MODELS.md), and [docs/SNAPDRAGON_DEPLOYMENT.md](docs/SNAPDRAGON_DEPLOYMENT.md) for the boundary between measured local behavior and pending target work.
