from __future__ import annotations

from time import perf_counter
from concurrent.futures import ThreadPoolExecutor

try:
    import resource
except ImportError:
    resource = None

from .contracts import ApplicationContext, Mode, SpeechContext, Telemetry, VisualContext
from .decision import decide
from .fusion import fuse_context
from .models import ModelInferenceError, ModelUnavailableError, SpeechModel, VisionModel


def memory_measurement() -> str:
    if resource is None:
        return "UNKNOWN"
    return f"{resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024:.1f} MB"


def run_vision_worker(vision: VisionModel, frame: bytes | None) -> tuple[VisualContext, float, str]:
    started = perf_counter()
    try:
        result = vision.infer(frame)
        return result, (perf_counter() - started) * 1000, result.status
    except (ModelUnavailableError, ModelInferenceError):
        return VisualContext(model=vision.name, status="ERROR"), (perf_counter() - started) * 1000, "ERROR"
    except Exception:
        return VisualContext(model=vision.name, status="ERROR"), (perf_counter() - started) * 1000, "ERROR"


def run_speech_worker(speech: SpeechModel, audio: bytes | None) -> tuple[SpeechContext, float, str]:
    started = perf_counter()
    try:
        result = speech.transcribe(audio)
        return result, (perf_counter() - started) * 1000, result.status
    except (ModelUnavailableError, ModelInferenceError):
        return SpeechContext(model=speech.name, status="ERROR"), (perf_counter() - started) * 1000, "ERROR"
    except Exception:
        return SpeechContext(model=speech.name, status="ERROR"), (perf_counter() - started) * 1000, "ERROR"


def fuse_outputs(mode: Mode, spoken: SpeechContext, visual: VisualContext, application: ApplicationContext | None = None) -> tuple[dict, dict, float, float]:
    fusion_started = perf_counter()
    context = fuse_context(mode, spoken, visual, application or ApplicationContext())
    fusion_ms = (perf_counter() - fusion_started) * 1000
    decision_started = perf_counter()
    decision = decide(context)
    decision_ms = (perf_counter() - decision_started) * 1000
    return context.to_dict(), decision.to_dict(), fusion_ms, decision_ms


def run_pipeline(mode: Mode, vision: VisionModel, speech: SpeechModel, frame: bytes | None = None, audio: bytes | None = None, application: ApplicationContext | None = None) -> tuple[dict, dict, dict]:
    started = perf_counter()
    telemetry = Telemetry(model=f"{vision.name}+{speech.name}", compute_layer="CPU")
    telemetry.memory = memory_measurement()
    preprocess_started = perf_counter()
    frame = frame if frame else None
    audio = audio if audio else None
    telemetry.preprocessing_ms = (perf_counter() - preprocess_started) * 1000
    with ThreadPoolExecutor(max_workers=2, thread_name_prefix="snapsense-worker") as workers:
        vision_future = workers.submit(run_vision_worker, vision, frame)
        speech_future = workers.submit(run_speech_worker, speech, audio)
        visual, telemetry.vision_inference_ms, vision_status = vision_future.result()
        spoken, telemetry.speech_inference_ms, speech_status = speech_future.result()
    if vision_status == "ERROR" or speech_status == "ERROR":
        telemetry.status = "ERROR"
    telemetry.vision_status = vision_status
    telemetry.speech_status = speech_status
    context_dict, decision_dict, telemetry.fusion_ms, telemetry.decision_ms = fuse_outputs(mode, spoken, visual, application)
    telemetry.end_to_end_ms = (perf_counter() - started) * 1000
    return context_dict, decision_dict, telemetry.to_dict()
