import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import soundfile as sf

from training import upload_train, uploaded_audio
from training.upload_train import evaluation_dataset


class TrainingPrecheckTest(unittest.TestCase):
    def test_precheck_holds_out_multiple_disjoint_sentences(self):
        with tempfile.TemporaryDirectory() as directory:
            source, target = Path(directory) / "full", Path(directory) / "precheck"
            (source / "raw_audio").mkdir(parents=True)
            rows = []
            for index in range(50):
                name = f"{index:06d}.wav"
                (source / "raw_audio" / name).write_bytes(b"clip")
                rows.append({"file_name": name, "text": f"Câu {index}", "split": "train"})
            (source / "train.jsonl").write_text("\n".join(json.dumps(row) for row in rows))
            (source / "manifest.json").write_text(json.dumps({"full_training": True, "counts": {"train": 50}}))
            manifest = evaluation_dataset(source, target)
            self.assertGreaterEqual(manifest["counts"]["test"], 5)
            self.assertEqual(sum(manifest["counts"].values()), 50)
            groups = [set(json.loads(line)["file_name"] for line in (target / f"{split}.jsonl").read_text().splitlines())
                      for split in ("train", "validation", "test")]
            self.assertFalse(groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])

    def test_extra_recordings_are_combined_but_short_data_cannot_replace_voice(self):
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory)
            voice_id = "a" * 32
            original = data / "voices/sources" / f"{voice_id}.wav"
            extra = data / "voices/training" / voice_id / "extra.wav"
            reference = data / "voices" / f"{voice_id}.wav"
            for path in (original, extra, reference):
                path.parent.mkdir(parents=True, exist_ok=True)
                sf.write(path, np.full(16000 * 6, .1, dtype=np.float32), 16000)
            voice = {"id": voice_id, "name": "test", "source": "original.wav", "source_file": original.name,
                     "duration": 6, "locale": "vi-VN", "training_files": [{"file": extra.name, "duration": 6}]}
            (data / "voices.json").write_text(json.dumps([voice]))
            with patch.object(upload_train, "DATA", data), patch.object(uploaded_audio, "WORK", data / "upload_training"):
                uploaded_audio.save(voice_id, status="queued", attempt=1, clean_before_training=True)

                def fake_extract(source, sample, workspace, **kwargs):
                    workspace.mkdir(parents=True)
                    target = workspace / "target.wav"
                    sf.write(target, np.full(16000 * 6, .1, dtype=np.float32), 16000)
                    return {"target": str(target), "selected_chunks": 1, "total_chunks": 1, "method": "test"}

                def fake_build(voice, source, output, progress):
                    output.mkdir(parents=True)
                    self.assertEqual(sf.info(source).duration, 12.5)
                    return {"counts": {"train": 4, "validation": 0, "test": 0}, "source_seconds": 12.5,
                            "usable_seconds": 12, "rejected": {}, "minutes": {"train": .2}}

                with patch.object(upload_train, "extract_target_voice", side_effect=fake_extract), \
                     patch.object(upload_train, "build", side_effect=fake_build), \
                     patch.object(upload_train.urllib.request, "urlopen", side_effect=OSError("offline")):
                    self.assertEqual(upload_train.run(voice_id), 0)
                state = uploaded_audio.load(voice_id)
                self.assertEqual(state["status"], "needs_more_data")
                self.assertEqual(state["source_file_count"], 2)
                self.assertIsNone(state.get("trained_voice_id"))


if __name__ == "__main__":
    unittest.main()
