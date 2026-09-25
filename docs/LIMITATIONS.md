# Limitations

Status: **KNOWN / PENDING**.

- Real Whisper-Tiny and ONNX vision adapters are implemented, but the default vision and speech workers remain explicit `UNKNOWN` until model artifacts and optional packages are installed.
- No Snapdragon device, Qualcomm AI Hub job, NPU trace, target driver, or target OS is available in this Codespace.
- Physical latency, memory, compute mapping, quantization, and output parity are **PENDING**, not estimated.
- The browser sensor flow demonstrates local capture but does not yet stream frames into a compiled vision model or audio into Whisper-Tiny.
- UI response timing and peak memory are not claimed unless measured by a future instrumented adapter.
