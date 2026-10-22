import unittest

from server.target_voice import choose_stem


class SpeakerDecisionTest(unittest.TestCase):
    def test_rejects_unrelated_and_ambiguous_speech(self):
        self.assertIsNone(choose_stem([.30, .12], .19, allow_fallback=True))
        self.assertIsNone(choose_stem([.60, .59], .72, allow_fallback=True))

    def test_uses_mixture_and_margin_when_separation_colors_voice(self):
        self.assertEqual(choose_stem([.52, .16], .67, allow_fallback=False), 0)
        self.assertIsNone(choose_stem([.52, .16], .31, allow_fallback=False))

    def test_denoised_source_only_when_explicitly_allowed(self):
        self.assertEqual(choose_stem([.42, .15], .86, allow_fallback=True), -1)
        self.assertIsNone(choose_stem([.42, .15], .86, allow_fallback=False))


if __name__ == "__main__":
    unittest.main()
