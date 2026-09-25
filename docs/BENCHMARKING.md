# Benchmarking

Status: **IMPLEMENTED local framework; PENDING model and target measurements**.

`snapsense.benchmark` discards five warm-up cycles by default and reports min, mean, median, p95, and max for measured operations. `scripts/benchmark_models.py` measures configured model workers only when they return `MEASURED`; missing artifacts, packages, or inputs report `PENDING`. `scripts/benchmark_app.py` measures the complete local pipeline. Measurements are produced at runtime and are not checked into the repository.

Model-level measurements record inference latency, model size, observed process memory where measurable, output status/correctness, and verified compute mapping. Application-level measurements cover preprocessing, vision, speech, fusion, decision, UI response, end-to-end time, and observed process memory. The current memory value is a process high-water observation and is not falsely attributed as a per-model peak. Snapdragon values remain `PENDING` until physically profiled.
