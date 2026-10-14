from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
import wave

from training.uploaded_audio import transcribe_full, word_groups


class ChunkCoverageTest(unittest.TestCase):
    def test_boundary_words_are_included_once(self):
        class Model:
            def transcribe(self, filename, **kwargs):
                index = int(Path(filename).stem.rsplit("_", 1)[1])
                read_start = max(0, index * 4 - 3)
                words = [SimpleNamespace(start=t - read_start, end=t + .2 - read_start, word=f" {t}")
                         for t in (0.5, 3.8, 4.2, 7.8, 8.2) if read_start <= t < index * 4 + 7]
                return [SimpleNamespace(words=words, start=words[0].start,
                                        end=words[-1].end, text=" ".join(w.word for w in words))], None

        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "source.wav"
            with wave.open(str(path), "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(24000)
                wav.writeframes(b"\x00\x00" * 24000 * 10)
            segments = transcribe_full(Model(), path, "vi", chunk_seconds=4)
            words = [word.word.strip() for segment in segments for word in segment.words]
            self.assertEqual(words, ["0.5", "3.8", "4.2", "7.8", "8.2"])
            self.assertEqual(" ".join(group[2] for group in word_groups(segments)).split(), words)


if __name__ == "__main__":
    unittest.main()
