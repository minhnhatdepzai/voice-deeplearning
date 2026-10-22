import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from server.speech_jobs import SpeechJobs
from server import app as api_app


class SpeechJobsTest(unittest.TestCase):
    def test_persists_and_recovers_unfinished_job(self):
        with tempfile.TemporaryDirectory() as temporary:
            job_id = "a" * 32
            store = SpeechJobs(Path(temporary))
            store.put(job_id, status="queued", voice_id="b" * 32)
            restarted = SpeechJobs(Path(temporary))
            self.assertEqual(restarted.unfinished()[0]["id"], job_id)
            restarted.put(job_id, status="complete", result={"text": "hello"})
            self.assertEqual(restarted.unfinished(), [])
            self.assertEqual(restarted.get(job_id)["result"]["text"], "hello")
            restarted.delete(job_id)
            self.assertIsNone(restarted.get(job_id))

    def test_http_job_completes_and_can_be_deleted(self):
        with tempfile.TemporaryDirectory() as temporary:
            async def fake_convert(file, voice_id, language, quality, tts_profile, denoise, progress=None):
                self.assertEqual(await file.read(), b"fake audio")
                progress("transcribing")
                return {"text": "Xin chào", "audio_url": ""}

            with patch.object(api_app, "_speech_jobs", SpeechJobs(Path(temporary))), \
                 patch.object(api_app, "_get_voice", return_value={"id": "b" * 32}), \
                 patch.object(api_app, "_speech_to_speech_impl", side_effect=fake_convert):
                with TestClient(api_app.app) as client:
                    response = client.post("/api/speech-to-speech/jobs", files={"file": ("voice.webm", b"fake audio", "audio/webm")},
                                           data={"voice_id": "b" * 32})
                    self.assertEqual(response.status_code, 202, response.text)
                    job_id = response.json()["id"]
                    for _ in range(50):
                        state = client.get(f"/api/speech-to-speech/jobs/{job_id}").json()
                        if state["status"] == "complete":
                            break
                        time.sleep(.02)
                    self.assertEqual(state["status"], "complete")
                    self.assertEqual(state["result"]["text"], "Xin chào")
                    self.assertEqual(client.delete(f"/api/speech-to-speech/jobs/{job_id}").status_code, 200)
