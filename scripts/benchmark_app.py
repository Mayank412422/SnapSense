import json
import argparse
from pathlib import Path

from snapsense.benchmark import benchmark
from snapsense.models import configured_speech_model, configured_vision_model
from snapsense.pipeline import memory_measurement
from snapsense.pipeline import run_pipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path)
    parser.add_argument("--audio", type=Path)
    args = parser.parse_args()
    frame = args.image.read_bytes() if args.image else None
    audio = args.audio.read_bytes() if args.audio else None
    vision, speech = configured_vision_model(), configured_speech_model()
    report = benchmark(lambda: run_pipeline("study", vision, speech, frame, audio))
    print(json.dumps({"application_pipeline": report.to_dict(), "inputs": {"image": bool(frame), "audio": bool(audio)}, "observed_process_memory": memory_measurement(), "memory_note": "Process high-water observation; not a per-model peak attribution.", "hardware_validation": "PENDING"}, indent=2))


if __name__ == "__main__":
    main()
