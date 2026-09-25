"""Download optional model artifacts into a user-selected local directory."""

import argparse
import shutil
import urllib.request
from pathlib import Path


YOLOV4_TINY_ONNX_URL = "https://huggingface.co/Kalray/yolov4-tiny/resolve/main/yolov4-tiny.onnx"
YOLOV4_TINY_LABELS_URL = "https://raw.githubusercontent.com/AlexeyAB/darknet/master/data/coco.names"


def download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as response, destination.open("wb") as output:
        shutil.copyfileobj(response, output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, default=Path("models"))
    parser.add_argument("--vision", action="store_true", help="download the YOLOv4-Tiny ONNX candidate")
    args = parser.parse_args()
    if args.vision:
        download(YOLOV4_TINY_ONNX_URL, args.directory / "yolov4-tiny.onnx")
        download(YOLOV4_TINY_LABELS_URL, args.directory / "coco.names")
        print(f"Vision artifact: {args.directory / 'yolov4-tiny.onnx'}")
    print("Whisper-Tiny is downloaded by faster-whisper on first use from Systran/faster-whisper-tiny.")


if __name__ == "__main__":
    main()