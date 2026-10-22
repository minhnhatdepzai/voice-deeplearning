import unittest

from server.speech_to_speech import postprocess_filter, recognition_profile, synthesis_profile, validate_utterance


class SpeechToSpeechTests(unittest.TestCase):
    def test_profiles_make_accuracy_cost_explicit(self):
        self.assertEqual((recognition_profile("fast").model, recognition_profile("fast").beam_size), ("small", 1))
        self.assertEqual((recognition_profile("balanced").model, recognition_profile("balanced").beam_size), ("turbo", 3))
        self.assertEqual((recognition_profile("accurate").model, recognition_profile("accurate").beam_size), ("turbo", 5))
        with self.assertRaises(ValueError):
            recognition_profile("unknown")

    def test_interactive_turn_has_bounded_duration_and_size(self):
        validate_utterance(0.5, 1000)
        validate_utterance(20, 25 * 1024 * 1024)
        for duration in (0.1, 20.01):
            with self.assertRaises(ValueError):
                validate_utterance(duration, 1000)
        with self.assertRaises(ValueError):
            validate_utterance(2, 25 * 1024 * 1024 + 1)

    def test_synthesis_profiles_are_bounded_and_validated(self):
        self.assertEqual(synthesis_profile("natural")["temperature"], 0.8)
        self.assertLess(synthesis_profile("stable")["temperature"], synthesis_profile("clear")["temperature"])
        with self.assertRaises(ValueError):
            synthesis_profile("random")
        self.assertIn("afftdn", postprocess_filter("light"))
        self.assertIsNone(postprocess_filter("off"))
        with self.assertRaises(ValueError):
            postprocess_filter("maximum")


if __name__ == "__main__":
    unittest.main()
