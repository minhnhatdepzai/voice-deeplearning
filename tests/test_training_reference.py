import pytest

from training.run import select_reference


def test_select_reference_prefers_clean_scored_clip():
    rows = [
        {"file_name": "separated.wav", "duration": 5.0, "extraction_cosine": 0.91},
        {"file_name": "clean.wav", "duration": 5.7, "reference_score": 1.0},
    ]
    assert select_reference(rows, {})["file_name"] == "clean.wav"


def test_select_reference_honors_manifest_override():
    rows = [
        {"file_name": "a.wav", "duration": 5.0, "reference_score": 1.0},
        {"file_name": "chosen.wav", "duration": 6.0, "reference_score": 0.8},
    ]
    assert select_reference(rows, {"reference_file": "chosen.wav"})["file_name"] == "chosen.wav"


def test_select_reference_rejects_invalid_override():
    with pytest.raises(ValueError, match="missing or ineligible"):
        select_reference([{"file_name": "a.wav", "duration": 5.0}], {"reference_file": "missing.wav"})
