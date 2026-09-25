from __future__ import annotations

from datetime import datetime, timezone

from .contracts import ApplicationContext, FusedContext, Mode, SpeechContext, VisualContext


def now_ms() -> int:
    return int(datetime.now(timezone.utc).timestamp() * 1000)


def fuse_context(mode: Mode, speech: SpeechContext, visual: VisualContext, application: ApplicationContext, timestamp_ms: int | None = None) -> FusedContext:
    """Merge compact worker outputs; no model or generative step is used."""
    signals: list[str] = []
    if speech.transcript.strip():
        signals.append("speech")
    if visual.entities:
        signals.append("vision")
    if application.title.strip():
        signals.append("application")
    confidences = [value for value in (speech.confidence, *(entity.confidence for entity in visual.entities)) if value is not None]
    confidence = sum(confidences) / len(confidences) if confidences else None
    return FusedContext(mode, speech, visual, application, timestamp_ms or now_ms(), confidence, tuple(signals))
