"""Inspect transient audio metadata and optionally run the configured Whisper worker."""

import argparse
import json
from pathlib import Path

from snapsense.models import configured_speech_model, inspect_audio_bytes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("audio", type=Path)
    parser.add_argument("--transcribe", action="store_true")
    args = parser.parse_args()
    payload = args.audio.read_bytes()
    result = {"path": str(args.audio), "metadata": inspect_audio_bytes(payload)}
    if args.transcribe:
        speech = configured_speech_model().transcribe(payload)
        result["speech"] = speech.__dict__
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()