# Models

Status: **IMPLEMENTED adapters and interfaces; model artifacts PENDING**.

Vision candidates are RTMDet and YOLOv4-Tiny. The project does not declare a final winner; selection depends on task fit, deployment, latency, memory, and correctness measurements. Speech uses a Whisper-Tiny interface. The current default adapters deliberately return `UNKNOWN` because no model artifacts are present in this Codespace.

The speech adapter uses `faster-whisper`, the `Systran/faster-whisper-tiny` CTranslate2 artifact, and transient WebM files. The vision adapter uses OpenCV DNN and ONNX. `scripts/download_models.py --vision` downloads the YOLOv4-Tiny ONNX candidate and COCO labels; RTMDet requires a compatible exported ONNX artifact supplied by the operator. Both adapters load lazily, so the CPU fallback application remains runnable without optional packages. Qualcomm reference figures in the source specification are external reference evidence, not SnapSense results. They must not be copied into benchmark output.

For a real integration test after setup:

```bash
SNAPSENSE_VISION_MODEL=models/yolov4-tiny.onnx \
SNAPSENSE_VISION_CANDIDATE=YOLOv4-Tiny \
SNAPSENSE_VISION_LABELS=models/coco.names \
SNAPSENSE_REAL_VISION_IMAGE=/path/to/real-image.jpg \
SNAPSENSE_ENABLE_WHISPER=1 \
SNAPSENSE_REAL_AUDIO=/path/to/real-audio.webm \
python3 -m unittest tests.test_real_inference -v
```

The tests are skipped, rather than mocked, when the operator has not supplied real artifacts and media.
