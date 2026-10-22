import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from server import app as api_app
from server.speech_jobs import SpeechJobs
from training import uploaded_audio


class ContainerRecoveryTest(unittest.TestCase):
    def test_interrupted_upload_training_can_be_retried(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            voice_id = "c" * 32
            with patch.object(api_app, "DATA", root), \
                 patch.object(uploaded_audio, "WORK", root / "upload_training"), \
                 patch.object(api_app, "_speech_jobs", SpeechJobs(root / "speech_jobs")), \
                 patch.dict(os.environ, {"VOICE_STUDIO_WORKER_MODE": "subprocess"}):
                uploaded_audio.save(voice_id, status="training")
                with TestClient(api_app.app):
                    self.assertEqual(uploaded_audio.load(voice_id)["status"], "failed")
                    self.assertIn("khởi động lại", uploaded_audio.load(voice_id)["error"])
