from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

Mode = Literal["study", "interview", "coding"]
Unknown = "UNKNOWN"


@dataclass(frozen=True)
class SpeechContext:
    transcript: str = ""
    confidence: float | None = None
    model: str = "Whisper-Tiny"
    status: str = Unknown
    timestamp_ms: int | None = None


@dataclass(frozen=True)
class VisualEntity:
    label: str
    confidence: float | None = None
    source: str = "vision"


@dataclass(frozen=True)
class VisualContext:
    entities: tuple[VisualEntity, ...] = ()
    model: str = "RTMDet"
    status: str = Unknown
    timestamp_ms: int | None = None


@dataclass(frozen=True)
class ApplicationContext:
    title: str = ""
    source: str = "user"


@dataclass(frozen=True)
class FusedContext:
    active_mode: Mode
    speech: SpeechContext
    visual: VisualContext
    application: ApplicationContext
    created_at_ms: int
    confidence: float | None
    available_signals: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Decision:
    mode: Mode
    message: str
    priority: Literal["info", "attention", "action"] = "info"
    rationale: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Telemetry:
    preprocessing_ms: float | None = None
    vision_inference_ms: float | None = None
    speech_inference_ms: float | None = None
    fusion_ms: float | None = None
    decision_ms: float | None = None
    ui_response_ms: float | None = None
    end_to_end_ms: float | None = None
    memory: str = Unknown
    runtime: str = "python-local"
    model: str = "UNKNOWN"
    device: str = "local-host"
    compute_layer: str = Unknown
    status: str = "IMPLEMENTED"
    vision_status: str = Unknown
    speech_status: str = Unknown

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
