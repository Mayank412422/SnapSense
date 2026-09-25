import os
import unittest
from pathlib import Path

from snapsense.contracts import ApplicationContext, SpeechContext, VisualContext
from snapsense.fusion import fuse_context
from snapsense.models import configured_speech_model, configured_vision_model, inspect_audio_bytes


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

    @unittest.skipUnless(os.getenv("SNAPSENSE_BROWSER_AUDIO_FIXTURE") and (os.getenv("SNAPSENSE_ENABLE_WHISPER") or os.getenv("SNAPSENSE_WHISPER_MODEL")), "browser audio fixture and Whisper not configured")
    def test_browser_audio_decodes_and_reaches_whisper(self):
        payload = Path(os.environ["SNAPSENSE_BROWSER_AUDIO_FIXTURE"]).read_bytes()
        metadata = inspect_audio_bytes(payload)
        self.assertEqual(metadata["audio_codec"], "opus")
        self.assertGreater(metadata["audio_bytes"], 0)
        self.assertGreater(metadata["audio_nonzero_samples"], 0)
        self.assertGreater(metadata["audio_duration_ms"], 1000)
        result = configured_speech_model().transcribe(payload)
        self.assertEqual(result.status, "MEASURED")
        self.assertTrue(result.transcript.strip())


if __name__ == "__main__":
    unittest.main()