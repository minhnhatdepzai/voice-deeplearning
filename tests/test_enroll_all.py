import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import pyarrow as pa
import pyarrow.parquet as pq

from training.enroll_all import enroll


class EnrollmentTest(unittest.TestCase):
    def test_uses_every_embedding_without_changing_model_scale(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            parquet = root / "train.parquet"
            pq.write_table(pa.table({"speaker_embedding": [[12.0, 0.0], [14.0, 2.0]]}), parquet)
            self.assertEqual(enroll(parquet, root, "TEST", "all utterances"), 2)
            preset = json.loads((root / "voices_v3_turbo.json").read_text())["presets"]["TEST"]
            self.assertEqual(preset["speaker_emb"], [13.0, 1.0])
            self.assertIsNone(preset["codes"])


if __name__ == "__main__":
    unittest.main()
