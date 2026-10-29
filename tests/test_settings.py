import pytest

from config.settings import Settings


def test_defaults_suit_a_cpu_only_english_setup():
    s = Settings(_env_file=None)
    assert s.whisper_language == "en" and s.whisper_model == "base" and s.similarity_threshold == 0.363
    assert s.whisper_threads == 0 and s.detect_max_fps == 10.0 and "Aether" in s.whisper_prompt


@pytest.mark.parametrize("raw,expected", [("auto", None), ("AUTO", None), ("", None), ("none", None),
                                          ("FR", "fr"), (" de ", "de"), ("en", "en")])
def test_language_values(monkeypatch, raw, expected):
    monkeypatch.setenv("AETHER_WHISPER_LANGUAGE", raw)
    assert Settings(_env_file=None).whisper_language == expected


def test_invalid_rates_are_rejected():
    with pytest.raises(ValueError):
        Settings(_env_file=None, detect_max_fps=0)
    with pytest.raises(ValueError):
        Settings(_env_file=None, whisper_threads=-1)
