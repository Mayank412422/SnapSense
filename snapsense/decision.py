from __future__ import annotations

from .contracts import Decision, FusedContext


def decide(context: FusedContext) -> Decision:
    """Deterministic mode rules, intentionally small and inspectable."""
    transcript = context.speech.transcript.lower()
    entities = {entity.label.lower() for entity in context.visual.entities}
    if context.active_mode == "study":
        if not context.speech.transcript and not entities:
            return Decision("study", "Waiting for a question or visual signal.", "info", ("no active signals",))
        return Decision("study", "Capture the key concept and connect it to the visible context.", "action", tuple(context.available_signals))
    if context.active_mode == "interview":
        if "question" in transcript or "why" in transcript:
            return Decision("interview", "Pause briefly, then answer with a claim and one concrete example.", "action", ("question-like transcript",))
        return Decision("interview", "Keep a steady pace and make your next point explicit.", "info", tuple(context.available_signals))
    if "error" in transcript or "bug" in transcript or "terminal" in entities:
        return Decision("coding", "Inspect the smallest failing boundary before changing code.", "attention", ("debugging signal",))
    return Decision("coding", "State the invariant, then make the smallest testable change.", "action", tuple(context.available_signals))
