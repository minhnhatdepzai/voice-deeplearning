import json
import tempfile
import unittest
from pathlib import Path

from training import dataset_review


class DatasetReviewTest(unittest.TestCase):
    def test_edit_exclude_and_approve_rebuilds_training_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            dataset = Path(temporary)
            rows = [{"file_name": f"{index:06}.wav", "text": f"Line {index}", "duration": 4.0,
                     "split": "train"} for index in range(3)]
            (dataset / "train.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
            (dataset / "manifest.json").write_text(json.dumps({"counts": {"train": 3, "validation": 0, "test": 0},
                                                                  "minutes": {"train": .2, "validation": 0, "test": 0},
                                                                  "usable_seconds": 12}))
            self.assertEqual(dataset_review.load(dataset, limit=2)["total"], 3)
            dataset_review.edit(dataset, "000000.wav", "Corrected words", True)
            dataset_review.edit(dataset, "000001.wav", "Wrong speaker", False)
            dataset_review.approve(dataset)
            selected = [json.loads(line) for line in (dataset / "train.jsonl").read_text().splitlines()]
            self.assertEqual([row["file_name"] for row in selected], ["000000.wav", "000002.wav"])
            self.assertEqual(selected[0]["text"], "Corrected words")
            manifest = json.loads((dataset / "manifest.json").read_text())
            self.assertEqual(manifest["counts"]["train"], 2)
            self.assertEqual(manifest["usable_seconds"], 8.0)
            self.assertTrue(manifest["review_sha256"])
            self.assertTrue(dataset_review.load(dataset)["approved"])

    def test_cannot_approve_empty_dataset(self):
        with tempfile.TemporaryDirectory() as temporary:
            dataset = Path(temporary)
            (dataset / "train.jsonl").write_text(json.dumps({"file_name": "000000.wav", "text": "Hi", "duration": 4}) + "\n")
            dataset_review.edit(dataset, "000000.wav", "Hi", False)
            with self.assertRaises(ValueError):
                dataset_review.approve(dataset)
