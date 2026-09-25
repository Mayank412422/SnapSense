from __future__ import annotations

import json
import base64
import binascii
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from .contracts import ApplicationContext, SpeechContext, VisualContext, VisualEntity
from .models import configured_speech_model, configured_vision_model, model_registry
from .pipeline import fuse_outputs, memory_measurement, run_pipeline, run_speech_worker, run_vision_worker
from .runtime import runtime_registry

ROOT = Path(__file__).parent
VALID_MODES = {"study", "interview", "coding"}
MAX_REQUEST_BYTES = 32 * 1024 * 1024


class Handler(BaseHTTPRequestHandler):
    def _send(self, payload: bytes, content_type: str, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path == "/api/health":
            self._send(json.dumps({"status": "ok", "project": "SnapSense Edge"}).encode(), "application/json")
        elif path == "/api/capabilities":
            payload = {"models": model_registry(), "runtimes": [item.__dict__ for item in runtime_registry()], "hardware_validation": "PENDING"}
            self._send(json.dumps(payload).encode(), "application/json")
        else:
            filename = "index.html" if path == "/" else path.removeprefix("/")
            file_path = (ROOT / "web" / filename).resolve()
            if ROOT.joinpath("web").resolve() not in file_path.parents or not file_path.is_file():
                self._send(b"Not found", "text/plain", 404)
                return
            content_type = "text/html" if file_path.suffix == ".html" else "text/css" if file_path.suffix == ".css" else "application/javascript"
            self._send(file_path.read_bytes(), content_type)

    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path not in {"/api/analyze", "/api/vision", "/api/speech", "/api/fuse"}:
            self._send(b"Not found", "text/plain", 404)
            return
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            if content_length > MAX_REQUEST_BYTES:
                raise ValueError("request exceeds the 32 MiB local media limit")
            body = json.loads(self.rfile.read(content_length))
            mode = body.get("mode", "study")
            if mode not in VALID_MODES:
                raise ValueError(f"mode must be one of: {', '.join(sorted(VALID_MODES))}")
            frame = base64.b64decode(body["frame_b64"], validate=True) if body.get("frame_b64") else None
            audio = base64.b64decode(body["audio_b64"], validate=True) if body.get("audio_b64") else None
            application = ApplicationContext(body.get("application_title", ""))
            if path == "/api/vision":
                visual, inference_ms, worker_status = run_vision_worker(configured_vision_model(), frame)
                self._send(json.dumps({"visual": visual_to_dict(visual), "telemetry": worker_telemetry("vision", visual.model, inference_ms, worker_status)}).encode(), "application/json")
                return
            if path == "/api/speech":
                spoken, inference_ms, worker_status = run_speech_worker(configured_speech_model(), audio)
                self._send(json.dumps({"speech": spoken.__dict__, "telemetry": worker_telemetry("speech", spoken.model, inference_ms, worker_status)}).encode(), "application/json")
                return
            if path == "/api/fuse":
                spoken = SpeechContext(**body.get("speech", {}))
                visual_payload = body.get("visual", {})
                entities = tuple(VisualEntity(**entity) for entity in visual_payload.get("entities", []))
                visual = VisualContext(entities=entities, model=visual_payload.get("model", "UNKNOWN"), status=visual_payload.get("status", "UNKNOWN"))
                context, decision, fusion_ms, decision_ms = fuse_outputs(mode, spoken, visual, application)
                self._send(json.dumps({"context": context, "decision": decision, "telemetry": {"fusion_ms": fusion_ms, "decision_ms": decision_ms, "memory": memory_measurement(), "status": "IMPLEMENTED"}}).encode(), "application/json")
                return
            context, decision, telemetry = run_pipeline(mode, configured_vision_model(), configured_speech_model(), frame, audio, application)
            self._send(json.dumps({"context": context, "decision": decision, "telemetry": telemetry}).encode(), "application/json")
        except (ValueError, TypeError, KeyError, binascii.Error) as error:
            self._send(json.dumps({"error": str(error)}).encode(), "application/json", 400)
        except Exception as error:
            self._send(json.dumps({"error": f"pipeline failure: {error}"}).encode(), "application/json", 500)

    def log_message(self, format: str, *args: object) -> None:
        return


def visual_to_dict(visual: VisualContext) -> dict:
    return {"entities": [entity.__dict__ for entity in visual.entities], "model": visual.model, "status": visual.status}


def worker_telemetry(worker: str, model: str, inference_ms: float, status: str) -> dict:
    return {"worker": worker, "model": model, "inference_ms": inference_ms, "memory": memory_measurement(), "runtime": "python-local", "device": "local-host", "compute_layer": "CPU", "status": status}


def main() -> None:
    host, port = "127.0.0.1", 8765
    print(f"SnapSense Edge running at http://{host}:{port}")
    ThreadingHTTPServer((host, port), Handler).serve_forever()


if __name__ == "__main__":
    main()
