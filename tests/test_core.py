import unittest

from snapsense.contracts import ApplicationContext, SpeechContext, VisualContext, VisualEntity
from snapsense.decision import decide
from snapsense.fusion import fuse_context
from snapsense.models import UnavailableSpeechModel, UnavailableVisionModel


class FusionTests(unittest.TestCase):
    def test_fusion_is_structured_and_deterministic(self):
        speech = SpeechContext("Explain this bug", 0.8, status="MEASURED")
        visual = VisualContext((VisualEntity("terminal", 0.9),), status="MEASURED")
        first = fuse_context("coding", speech, visual, ApplicationContext("VS Code"), timestamp_ms=123)
        second = fuse_context("coding", speech, visual, ApplicationContext("VS Code"), timestamp_ms=123)
        self.assertEqual(first, second)
        self.assertEqual(first.available_signals, ("speech", "vision", "application"))
        self.assertAlmostEqual(first.confidence, 0.85)

    def test_interview_rule_uses_transcript_without_another_model(self):
        context = fuse_context("interview", SpeechContext("Why did you choose this design?"), VisualContext(), ApplicationContext())
        result = decide(context)
        self.assertEqual(result.priority, "action")
        self.assertIn("claim", result.message)


class ModelContractTests(unittest.TestCase):
    def test_unavailable_models_preserve_unknown_status(self):
        self.assertEqual(UnavailableVisionModel().infer(None).status, "UNKNOWN")
        self.assertEqual(UnavailableSpeechModel().transcribe(None).status, "UNKNOWN")


if __name__ == "__main__":
    unittest.main()
