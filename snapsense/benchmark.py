from __future__ import annotations

import json
import statistics
from dataclasses import dataclass
from time import perf_counter
from typing import Callable


@dataclass(frozen=True)
class BenchmarkReport:
    cycles: int
    warmup_cycles: int
    samples_ms: tuple[float, ...]
    minimum_ms: float | None
    mean_ms: float | None
    median_ms: float | None
    p95_ms: float | None
    maximum_ms: float | None
    status: str = "MEASURED_LOCAL"

    def to_dict(self) -> dict:
        return {"cycles": self.cycles, "warmup_cycles": self.warmup_cycles, "samples_ms": list(self.samples_ms), "min_ms": self.minimum_ms, "mean_ms": self.mean_ms, "median_ms": self.median_ms, "p95_ms": self.p95_ms, "max_ms": self.maximum_ms, "status": self.status}


def benchmark(operation: Callable[[], object], cycles: int = 20, warmup_cycles: int = 5) -> BenchmarkReport:
    if cycles < 1:
        raise ValueError("cycles must be at least 1")
    if warmup_cycles < 0:
        raise ValueError("warmup_cycles cannot be negative")
    for _ in range(warmup_cycles):
        operation()
    samples = []
    for _ in range(cycles):
        started = perf_counter()
        operation()
        samples.append((perf_counter() - started) * 1000)
    ordered = sorted(samples)
    percentile_index = min(len(ordered) - 1, max(0, int((len(ordered) - 1) * 0.95)))
    return BenchmarkReport(cycles, warmup_cycles, tuple(samples), min(samples), statistics.mean(samples), statistics.median(samples), ordered[percentile_index], max(samples))


def main() -> None:
    print(json.dumps(benchmark(lambda: None).to_dict(), indent=2))


if __name__ == "__main__":
    main()
