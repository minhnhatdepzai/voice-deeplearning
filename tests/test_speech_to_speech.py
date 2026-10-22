import unittest

from server.speech_to_speech import (MAX_SPEECH_TO_SPEECH_BYTES, postprocess_filter,
                                     recognition_profile, split_synthesis_text,
                                     synthesis_profile, validate_utterance)


class SpeechToSpeechTests(unittest.TestCase):
    def test_profiles_make_accuracy_cost_explicit(self):
        self.assertEqual((recognition_profile("fast").model, recognition_profile("fast").beam_size), ("small", 1))
        self.assertEqual((recognition_profile("balanced").model, recognition_profile("balanced").beam_size), ("turbo", 3))
        self.assertEqual((recognition_profile("accurate").model, recognition_profile("accurate").beam_size), ("turbo", 5))
        with self.assertRaises(ValueError):
            recognition_profile("unknown")

    def test_manual_recording_has_no_duration_cap_but_keeps_upload_bound(self):
        validate_utterance(0.5, 1000)
        validate_utterance(8 * 60 * 60, MAX_SPEECH_TO_SPEECH_BYTES)
        with self.assertRaises(ValueError):
            validate_utterance(0.1, 1000)
        with self.assertRaises(ValueError):
            validate_utterance(2, MAX_SPEECH_TO_SPEECH_BYTES + 1)

    def test_long_transcript_is_split_without_losing_words(self):
        text = "Câu đầu tiên khá ngắn. " + " ".join(f"từ{i}" for i in range(180)) + "! Câu cuối."
        chunks = split_synthesis_text(text, max_chars=80)
        self.assertGreater(len(chunks), 3)
        self.assertTrue(all(len(chunk) <= 80 for chunk in chunks))
        self.assertEqual(" ".join(chunks), " ".join(text.split()))

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
