import unittest

from snapsense.benchmark import benchmark
from snapsense.contracts import ApplicationContext, SpeechContext, VisualContext
from snapsense.fusion import fuse_context
from snapsense.pipeline import run_pipeline
from snapsense.contracts import SpeechContext, VisualContext


class FailureTests(unittest.TestCase):
    def test_empty_inputs_do_not_fabricate_confidence(self):
        context = fuse_context("study", SpeechContext(), VisualContext(), ApplicationContext(), timestamp_ms=1)
        self.assertIsNone(context.confidence)
        self.assertEqual(context.available_signals, ())

    def test_benchmark_rejects_zero_cycles(self):
        with self.assertRaises(ValueError):
            benchmark(lambda: None, cycles=0)

    def test_benchmark_rejects_negative_warmup(self):
        with self.assertRaises(ValueError):
            benchmark(lambda: None, warmup_cycles=-1)

    def test_worker_failure_does_not_mark_other_worker_failed(self):
        class BrokenVision:
            name = "broken-vision"

            def infer(self, frame):
                raise RuntimeError("vision exploded")

        class HealthySpeech:
            name = "test-speech"

            def transcribe(self, audio):
                return SpeechContext("real worker output", 0.9, model=self.name, status="MEASURED")

        context, _, telemetry = run_pipeline("study", BrokenVision(), HealthySpeech())
        self.assertEqual(context["visual"]["status"], "ERROR")
        self.assertEqual(context["speech"]["status"], "MEASURED")
        self.assertEqual(telemetry["vision_status"], "ERROR")
        self.assertEqual(telemetry["speech_status"], "MEASURED")


if __name__ == "__main__":
    unittest.main()
