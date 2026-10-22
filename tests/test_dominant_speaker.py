import unittest

import numpy as np

from server.dominant_speaker import MAX_SECONDS, cluster_embeddings


def unit(rows):
    values = np.asarray(rows, dtype=np.float32)
    return values / np.linalg.norm(values, axis=1, keepdims=True)


class DominantSpeakerTests(unittest.TestCase):
    def test_duration_selects_dominant_cluster(self):
        vectors = unit([[1, .02], [.98, -.03], [.97, .05], [.01, 1], [-.04, .98]])
        result = cluster_embeddings(vectors, np.asarray([9, 8, 7, 4, 4], dtype=np.float32))
        self.assertEqual(result["clusters"], 2)
        labels = result["labels"]
        main = result["dominant"]
        self.assertEqual(int(np.sum(labels == main)), 3)

    def test_nearly_identical_centers_are_one_voice(self):
        vectors = unit([[1, .00, .01], [1, .02, -.01], [.99, -.02, .01], [1, .01, .02]])
        result = cluster_embeddings(vectors, np.ones(4, dtype=np.float32) * 4)
        self.assertEqual(result["clusters"], 1)

    def test_similar_but_distinct_voices_are_not_merged(self):
        vectors = unit([[1, .02, 0], [.99, -.02, .01], [.88, .47, 0], [.86, .50, .02]])
        result = cluster_embeddings(vectors, np.ones(4, dtype=np.float32) * 8)
        self.assertEqual(result["clusters"], 2)

    def test_long_recording_contract_is_one_hour(self):
        self.assertEqual(MAX_SECONDS, 3600)


if __name__ == "__main__":
    unittest.main()
