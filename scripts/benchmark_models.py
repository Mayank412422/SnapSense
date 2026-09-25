"""Run model-level measurements only when real artifacts are configured."""

import argparse
import json
from pathlib import Path

from snapsense.benchmark import benchmark
from snapsense.models import ModelInferenceError, ModelUnavailableError, configured_speech_model, configured_vision_model
from snapsense.pipeline import memory_measurement


def run(model, payload: bytes, operation: str) -> dict:
    try:
        def measure():
            result = getattr(model, operation)(payload)
            if getattr(result, "status", "MEASURED") not in {"MEASURED", "IMPLEMENTED"}:
                raise ModelUnavailableError(f"worker returned status {result.status}")
            return result

        report = benchmark(measure)
        artifact = getattr(model, "model_path", None)
        return {"model": model.name, "artifact": artifact, "model_size_bytes": Path(artifact).stat().st_size if artifact and Path(artifact).is_file() else "UNKNOWN", "observed_process_memory": memory_measurement(), **report.to_dict()}
    except (ModelUnavailableError, ModelInferenceError) as error:
        return {"model": model.name, "status": "PENDING", "reason": str(error)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path)
    parser.add_argument("--audio", type=Path)
    args = parser.parse_args()
    reports = []
    if args.image:
        reports.append({"level": "model", "worker": "vision", **run(configured_vision_model(), args.image.read_bytes(), "infer")})
    else:
        reports.append({"level": "model", "worker": "vision", "status": "PENDING", "reason": "pass --image and configure SNAPSENSE_VISION_MODEL"})
    if args.audio:
        reports.append({"level": "model", "worker": "speech", **run(configured_speech_model(), args.audio.read_bytes(), "transcribe")})
    else:
        reports.append({"level": "model", "worker": "speech", "status": "PENDING", "reason": "pass --audio and enable Whisper-Tiny"})
    print(json.dumps(reports, indent=2))


if __name__ == "__main__":
    main()