from __future__ import annotations

from dataclasses import dataclass
import os
import io
import tempfile
from functools import lru_cache
from typing import Any, Protocol

from .contracts import SpeechContext, VisualContext, VisualEntity


class ModelUnavailableError(RuntimeError):
    """Raised when a configured model cannot run in the current environment."""


class ModelInferenceError(RuntimeError):
    """Raised when an installed model fails to process an input."""


class VisionModel(Protocol):
    name: str

    def infer(self, frame: bytes | None) -> VisualContext: ...


class SpeechModel(Protocol):
    name: str

    def transcribe(self, audio: bytes | None) -> SpeechContext: ...


@dataclass
class UnavailableVisionModel:
    name: str = "RTMDet"

    def infer(self, frame: bytes | None) -> VisualContext:
        return VisualContext(model=self.name, status="UNKNOWN")


@dataclass
class UnavailableSpeechModel:
    name: str = "Whisper-Tiny"

    def transcribe(self, audio: bytes | None) -> SpeechContext:
        return SpeechContext(model=self.name, status="UNKNOWN")


@dataclass
class WhisperTinyModel:
    """Real Whisper-Tiny adapter using faster-whisper when installed."""

    model_path: str = "Systran/faster-whisper-tiny"
    name: str = "Whisper-Tiny"
    _model: Any = None

    def _load(self) -> Any:
        if self._model is not None:
            return self._model
        try:
            from faster_whisper import WhisperModel
        except ImportError as error:
            raise ModelUnavailableError("Install the speech extra to run Whisper-Tiny") from error
        device = os.getenv("SNAPSENSE_WHISPER_DEVICE", "cpu")
        compute_type = os.getenv("SNAPSENSE_WHISPER_COMPUTE_TYPE", "float32")
        self._model = WhisperModel(self.model_path, device=device, compute_type=compute_type)
        return self._model

    def transcribe(self, audio: bytes | None) -> SpeechContext:
        if not audio:
            return SpeechContext(model=self.name, status="NO_INPUT")
        try:
            diagnostics = inspect_audio_bytes(audio)
            model = self._load()
            with tempfile.NamedTemporaryFile(suffix=".webm") as audio_file:
                audio_file.write(audio)
                audio_file.flush()
                segments, info = model.transcribe(audio_file.name, vad_filter=True)
                transcript = " ".join(segment.text.strip() for segment in segments).strip()
            confidence = float(info.language_probability) if info.language_probability is not None else None
            status = "MEASURED" if transcript else "NO_SPEECH"
            return SpeechContext(transcript=transcript, confidence=confidence, model=self.name, status=status, **diagnostics)
        except ModelUnavailableError:
            raise
        except Exception as error:
            raise ModelInferenceError(f"Whisper-Tiny failed: {error}") from error


def inspect_audio_bytes(audio: bytes) -> dict[str, Any]:
    """Decode transient audio and return metadata without retaining media bytes."""
    try:
        import av
        import numpy as np
        with av.open(io.BytesIO(audio)) as container:
            stream = next(iter(container.streams.audio), None)
            if stream is None:
                raise ModelInferenceError("audio container has no audio stream")
            frames = list(container.decode(stream))
            sample_count = sum(frame.samples for frame in frames)
            nonzero_samples = sum(int(np.count_nonzero(frame.to_ndarray())) for frame in frames)
            sample_rate = stream.rate or stream.codec_context.sample_rate
            duration_ms = sample_count / sample_rate * 1000 if sample_rate else None
            return {
                "audio_bytes": len(audio),
                "audio_format": container.format.name,
                "audio_codec": stream.codec_context.name,
                "audio_duration_ms": duration_ms,
                "audio_sample_rate": sample_rate,
                "audio_channels": stream.codec_context.channels,
                "audio_nonzero_samples": nonzero_samples,
            }
    except ModelInferenceError:
        raise
    except Exception as error:
        raise ModelInferenceError(f"audio decode failed: {error}") from error


@dataclass
class OnnxVisionModel:
    """OpenCV DNN adapter for a local RTMDet or YOLOv4-Tiny ONNX artifact."""

    model_path: str
    candidate: str = "RTMDet"
    labels_path: str | None = None
    input_size: tuple[int, int] = (640, 640)
    name: str = "RTMDet"
    _net: Any = None

    def __post_init__(self) -> None:
        self.name = self.candidate

    def _load(self) -> Any:
        if self._net is not None:
            return self._net
        try:
            import cv2
        except ImportError as error:
            raise ModelUnavailableError("Install the vision extra to run the ONNX vision adapter") from error
        if not os.path.isfile(self.model_path):
            raise ModelUnavailableError(f"Vision model artifact not found: {self.model_path}")
        self._net = cv2.dnn.readNetFromONNX(self.model_path)
        return self._net

    def infer(self, frame: bytes | None) -> VisualContext:
        if not frame:
            return VisualContext(model=self.name, status="NO_INPUT")
        try:
            import cv2
            import numpy as np
            image = cv2.imdecode(np.frombuffer(frame, dtype=np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                raise ModelInferenceError("Camera payload is not a decodable image")
            net = self._load()
            blob = cv2.dnn.blobFromImage(image, 1 / 255.0, self.input_size, swapRB=True, crop=False)
            net.setInput(blob)
            outputs = net.forward(net.getUnconnectedOutLayersNames())
            entities = self._parse_outputs(outputs, image.shape)
            return VisualContext(tuple(entities), model=self.name, status="MEASURED")
        except (ModelUnavailableError, ModelInferenceError):
            raise
        except Exception as error:
            raise ModelInferenceError(f"{self.name} inference failed: {error}") from error

    def _parse_outputs(self, outputs: Any, image_shape: tuple[int, ...] | None = None) -> list[VisualEntity]:
        if self.candidate.lower() == "yolov4-tiny":
            return self._parse_yolov4_tiny(outputs, image_shape)
        labels = []
        if self.labels_path and os.path.isfile(self.labels_path):
            labels = [line.strip() for line in open(self.labels_path, encoding="utf-8") if line.strip()]
        entities = []
        for output in outputs if isinstance(outputs, (list, tuple)) else [outputs]:
            values = output.reshape(-1, output.shape[-1])
            for row in values:
                if len(row) < 6:
                    continue
                scores = row[5:]
                class_index = int(scores.argmax()) if len(scores) else 0
                confidence = float(scores[class_index]) if len(scores) else float(row[4])
                confidence = max(0.0, min(1.0, confidence))
                if confidence < 0.25:
                    continue
                label = labels[class_index] if class_index < len(labels) else f"class_{class_index}"
                entities.append(VisualEntity(label, confidence, self.name))
        return entities

    def _parse_yolov4_tiny(self, outputs: Any, image_shape: tuple[int, ...] | None) -> list[VisualEntity]:
        """Decode the two raw 3-anchor YOLOv4-Tiny heads emitted by the ONNX graph."""
        import cv2
        import numpy as np

        labels = []
        if self.labels_path and os.path.isfile(self.labels_path):
            with open(self.labels_path, encoding="utf-8") as label_file:
                labels = [line.strip() for line in label_file if line.strip()]
        anchors = ((10, 14), (23, 27), (37, 58), (81, 82), (135, 169), (344, 319))
        masks = ((3, 4, 5), (0, 1, 2))
        boxes: list[list[float]] = []
        scores: list[float] = []
        class_indexes: list[int] = []
        input_width, input_height = self.input_size
        image_height, image_width = image_shape[:2] if image_shape else (input_height, input_width)
        for output, mask in zip(outputs, masks):
            output = np.asarray(output)
            _, channels, grid_height, grid_width = output.shape
            class_count = channels // 3 - 5
            values = output.reshape(3, 5 + class_count, grid_height, grid_width)
            for anchor_offset, anchor_index in enumerate(mask):
                for row in range(grid_height):
                    for column in range(grid_width):
                        cell = values[anchor_offset, :, row, column]
                        objectness = 1.0 / (1.0 + np.exp(-float(cell[4])))
                        class_scores = 1.0 / (1.0 + np.exp(-cell[5:]))
                        class_index = int(np.argmax(class_scores))
                        confidence = objectness * float(class_scores[class_index])
                        if confidence < 0.25:
                            continue
                        center_x = (1.0 / (1.0 + np.exp(-float(cell[0]))) + column) / grid_width * image_width
                        center_y = (1.0 / (1.0 + np.exp(-float(cell[1]))) + row) / grid_height * image_height
                        box_width = np.exp(min(float(cell[2]), 10.0)) * anchors[anchor_index][0] / input_width * image_width
                        box_height = np.exp(min(float(cell[3]), 10.0)) * anchors[anchor_index][1] / input_height * image_height
                        boxes.append([center_x - box_width / 2, center_y - box_height / 2, box_width, box_height])
                        scores.append(confidence)
                        class_indexes.append(class_index)
        kept = cv2.dnn.NMSBoxes(boxes, scores, 0.25, 0.45)
        entities = []
        for index in np.asarray(kept).reshape(-1) if len(kept) else []:
            class_index = class_indexes[int(index)]
            label = labels[class_index] if class_index < len(labels) else f"class_{class_index}"
            entities.append(VisualEntity(label, float(scores[int(index)]), self.name))
        return entities


def configured_vision_model() -> VisionModel:
    path = os.getenv("SNAPSENSE_VISION_MODEL")
    if not path:
        return UnavailableVisionModel()
    candidate = os.getenv("SNAPSENSE_VISION_CANDIDATE", "RTMDet")
    input_size = (416, 416) if candidate.lower() == "yolov4-tiny" else (640, 640)
    return _vision_model(path, candidate, os.getenv("SNAPSENSE_VISION_LABELS"), input_size)


def configured_speech_model() -> SpeechModel:
    if os.getenv("SNAPSENSE_WHISPER_MODEL") or os.getenv("SNAPSENSE_ENABLE_WHISPER") == "1":
        return _speech_model(os.getenv("SNAPSENSE_WHISPER_MODEL", "Systran/faster-whisper-tiny"))
    return UnavailableSpeechModel()


@lru_cache(maxsize=8)
def _vision_model(path: str, candidate: str, labels_path: str | None, input_size: tuple[int, int]) -> VisionModel:
    return OnnxVisionModel(path, candidate, labels_path, input_size)


@lru_cache(maxsize=4)
def _speech_model(path: str) -> SpeechModel:
    return WhisperTinyModel(path)


VISION_CANDIDATES = ("RTMDet", "YOLOv4-Tiny")
SPEECH_CANDIDATES = ("Whisper-Tiny",)


def model_registry() -> dict[str, dict[str, Any]]:
    return {
        "vision": {"candidates": list(VISION_CANDIDATES), "selected": "PENDING_MEASUREMENT", "status": "IMPLEMENTED"},
        "speech": {"candidates": list(SPEECH_CANDIDATES), "selected": "Whisper-Tiny", "status": "IMPLEMENTED_INTERFACE"},
    }
