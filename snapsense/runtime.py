from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RuntimeDescriptor:
    name: str
    available: bool
    execution: str
    status: str


def runtime_registry() -> list[RuntimeDescriptor]:
    return [
        RuntimeDescriptor("python-local", True, "CPU", "IMPLEMENTED"),
        RuntimeDescriptor("onnxruntime-directml", False, "UNKNOWN", "PENDING_EXTERNAL_DEPENDENCY"),
        RuntimeDescriptor("qualcomm-ai-runtime", False, "UNKNOWN", "PENDING_SNAPDRAGON_DEVICE"),
        RuntimeDescriptor("qualcomm-ai-hub", False, "UNKNOWN", "PENDING_EXTERNAL_SERVICE"),
    ]
