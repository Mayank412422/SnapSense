import json
import base64
import os
import threading
import unittest
from http.client import HTTPConnection

from snapsense.app import Handler
from http.server import ThreadingHTTPServer
from pathlib import Path


class ApplicationIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.port = cls.server.server_port

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.thread.join()

    def test_health_and_capability_endpoints(self):
        connection = HTTPConnection("127.0.0.1", self.port)
        connection.request("GET", "/api/health")
        self.assertEqual(connection.getresponse().status, 200)
        connection.request("GET", "/api/capabilities")
        payload = json.loads(connection.getresponse().read())
        self.assertEqual(payload["hardware_validation"], "PENDING")
        self.assertIn("RTMDet", payload["models"]["vision"]["candidates"])

    def test_analysis_endpoint_returns_all_pipeline_stages(self):
        connection = HTTPConnection("127.0.0.1", self.port)
        body = json.dumps({"mode": "coding"})
        connection.request("POST", "/api/analyze", body, {"Content-Type": "application/json"})
        result = json.loads(connection.getresponse().read())
        self.assertIn("context", result)
        self.assertIn("decision", result)
        self.assertIsNotNone(result["telemetry"]["end_to_end_ms"])

    def test_analysis_endpoint_rejects_unknown_mode(self):
        connection = HTTPConnection("127.0.0.1", self.port)
        connection.request("POST", "/api/analyze", '{"mode":"unknown"}', {"Content-Type": "application/json"})
        self.assertEqual(connection.getresponse().status, 400)

    def test_analysis_endpoint_rejects_invalid_media_encoding(self):
        connection = HTTPConnection("127.0.0.1", self.port)
        connection.request("POST", "/api/analyze", '{"mode":"study","audio_b64":"not-base64"}', {"Content-Type": "application/json"})
        self.assertEqual(connection.getresponse().status, 400)

    @unittest.skipUnless(os.getenv("SNAPSENSE_REAL_VISION_IMAGE") and os.getenv("SNAPSENSE_VISION_MODEL"), "real vision fixture not configured")
    def test_real_vision_endpoint_returns_fixture_detections(self):
        connection = HTTPConnection("127.0.0.1", self.port)
        encoded = base64.b64encode(Path(os.environ["SNAPSENSE_REAL_VISION_IMAGE"]).read_bytes()).decode()
        body = json.dumps({"mode": "study", "frame_b64": encoded})
        connection.request("POST", "/api/vision", body, {"Content-Type": "application/json"})
        result = json.loads(connection.getresponse().read())
        self.assertEqual(result["visual"]["status"], "MEASURED")
        self.assertGreater(len(result["visual"]["entities"]), 0)

    @unittest.skipUnless(os.getenv("SNAPSENSE_REAL_AUDIO") and (os.getenv("SNAPSENSE_ENABLE_WHISPER") or os.getenv("SNAPSENSE_WHISPER_MODEL")), "real speech fixture not configured")
    def test_real_speech_endpoint_returns_fixture_transcript(self):
        connection = HTTPConnection("127.0.0.1", self.port)
        encoded = base64.b64encode(Path(os.environ["SNAPSENSE_REAL_AUDIO"]).read_bytes()).decode()
        body = json.dumps({"mode": "study", "audio_b64": encoded})
        connection.request("POST", "/api/speech", body, {"Content-Type": "application/json"})
        result = json.loads(connection.getresponse().read())
        self.assertEqual(result["speech"]["status"], "MEASURED")
        self.assertTrue(result["speech"]["transcript"].strip())


if __name__ == "__main__":
    unittest.main()
