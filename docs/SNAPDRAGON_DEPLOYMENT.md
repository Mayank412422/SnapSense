# Snapdragon Deployment

Status: **IN PROGRESS scaffolding; PENDING hardware validation**.

The deployment path is intentionally staged: SELECT, BASELINE, COMPILE through Qualcomm AI Hub, PROFILE, VALIDATE output parity, then INTEGRATE. `deployment/runtime.example.json` is a configuration record, not evidence of a successful compile or execution.

The runtime registry leaves Qualcomm AI Runtime, Qualcomm AI Hub, and DirectML unavailable until their external dependencies and target device are present. Do not report NPU utilization, INT8/quantization, latency, memory, or compilation success from this environment.
