# SnapSense Edge Requirements Checklist

This checklist compares the current repository with the supplied SnapSense Edge specification. Status is based on code and executed checks, not intended future behavior.

| Requirement | Status | Evidence / boundary |
|---|---|---|
| Camera ingestion | PARTIALLY IMPLEMENTED | Browser `getUserMedia` capture and JPEG frame upload are implemented. The specification's OpenCV camera worker is not implemented. |
| Microphone ingestion | PARTIALLY IMPLEMENTED | Browser `MediaRecorder` WebM capture and upload are implemented. The specification's PortAudio ingestion path is not implemented. |
| Vision worker | IMPLEMENTED | Configurable ONNX worker runs YOLOv4-Tiny on CPU; RTMDet-compatible ONNX path is available. Real fixture tests pass when configured. |
| Speech worker | IMPLEMENTED | `faster-whisper` runs the CTranslate2 Whisper-Tiny artifact on CPU. Real transcript fixture tests pass when configured. |
| Active application context | PARTIALLY IMPLEMENTED | Structured `ApplicationContext` is supported by the API, but the browser does not yet collect active-window metadata. |
| Deterministic fusion | IMPLEMENTED | Structured speech/visual/application inputs are merged without another AI model. |
| Study mode | IMPLEMENTED | Deterministic Study decision rules and UI mode control are tested. |
| Interview mode | IMPLEMENTED | Deterministic pacing/question guidance and UI mode control are tested. |
| Coding mode | IMPLEMENTED | Deterministic debugging/invariant guidance and UI mode control are tested. |
| Telemetry | PARTIALLY IMPLEMENTED | Worker, fusion, decision, end-to-end, CPU runtime, and observed process memory are reported. Browser UI response timing is measured in the browser but not persisted server-side; target-device metrics are pending. |
| Model-level benchmarking | IMPLEMENTED | Five warm-ups and min/mean/median/p95/max are measured for configured real workers; model size and observed process memory are reported where available. |
| Application-level benchmarking | IMPLEMENTED | Real vision, speech, concurrent worker, fusion, decision, and end-to-end timings are benchmarkable; observed process high-water memory is labeled as such. |
| Privacy / local-first behavior | PARTIALLY IMPLEMENTED | Loopback-only transient media processing and no raw-media persistence are implemented. Host permissions, endpoint integrity, and absolute security remain outside scope. |
| Runtime abstraction | PARTIALLY IMPLEMENTED | Python CPU runtime and registries for DirectML, Qualcomm AI Runtime, and AI Hub exist. Non-CPU runtimes are not executed in this Codespace. |
| Snapdragon deployment path | PARTIALLY IMPLEMENTED | SELECT/BASELINE/COMPILE/PROFILE/VALIDATE/INTEGRATE scaffolding and configuration exist. AI Hub compilation, NPU traces, parity, and physical integration are pending. |
| Snapdragon measurements | PENDING | No Snapdragon latency, memory, NPU utilization, quantization, INT8, or Qualcomm performance result is claimed. |

## Deliberate Non-Claims

The repository does not include model weights, credentials, `.env` files, downloaded fixtures, or generated benchmark results. CPU measurements from this Codespace are not Snapdragon measurements.
