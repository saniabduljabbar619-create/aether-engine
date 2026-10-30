import pytest

from config.settings import Settings


def test_defaults_suit_a_cpu_only_english_setup():
    s = Settings(_env_file=None)
    assert s.whisper_language == "en" and s.whisper_model == "base" and s.similarity_threshold == 0.363
    assert s.whisper_threads == 0 and s.detect_max_fps == 10.0 and s.follow_up_seconds == 10.0
    assert s.wake_word_list == ["aether"] and "Aether" in s.effective_whisper_prompt


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


def test_wake_words_are_a_comma_separated_list(monkeypatch):
    monkeypatch.setenv("AETHER_WAKE_WORDS", " Computer , hey jarvis ,")
    s = Settings(_env_file=None)
    assert s.wake_word_list == ["computer", "hey jarvis"]
    assert "Computer" in s.effective_whisper_prompt  # the hint follows the wake word
    assert Settings(_env_file=None, wake_words=" , ").wake_word_list == ["aether"]  # never empty


def test_an_explicit_prompt_wins():
    assert Settings(_env_file=None, whisper_prompt="  my own hint ").effective_whisper_prompt == "my own hint"
