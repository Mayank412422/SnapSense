import os
import unittest
from pathlib import Path

from snapsense.contracts import ApplicationContext, SpeechContext, VisualContext
from snapsense.fusion import fuse_context
from snapsense.models import configured_speech_model, configured_vision_model


class RealModelIntegrationTests(unittest.TestCase):
    """Run only when the operator supplies real artifacts and real media."""

    @unittest.skipUnless(os.getenv("SNAPSENSE_REAL_VISION_IMAGE") and os.getenv("SNAPSENSE_VISION_MODEL"), "real vision artifact and image not configured")
    def test_real_vision_output_flows_to_fusion(self):
        result = configured_vision_model().infer(Path(os.environ["SNAPSENSE_REAL_VISION_IMAGE"]).read_bytes())
        self.assertEqual(result.status, "MEASURED")
        self.assertGreater(len(result.entities), 0)
        context = fuse_context("study", speech=SpeechContext(), visual=result, application=ApplicationContext("real vision test"), timestamp_ms=1)
        self.assertIn("vision", context.available_signals)

    @unittest.skipUnless(os.getenv("SNAPSENSE_REAL_AUDIO") and (os.getenv("SNAPSENSE_ENABLE_WHISPER") or os.getenv("SNAPSENSE_WHISPER_MODEL")), "real Whisper artifact and audio not configured")
    def test_real_speech_output_flows_to_fusion(self):
        result = configured_speech_model().transcribe(Path(os.environ["SNAPSENSE_REAL_AUDIO"]).read_bytes())
        self.assertEqual(result.status, "MEASURED")
        self.assertTrue(result.transcript.strip())
        context = fuse_context("study", speech=result, visual=VisualContext(), application=ApplicationContext("real speech test"), timestamp_ms=1)
        self.assertIn("speech", context.available_signals)


if __name__ == "__main__":
    unittest.main()