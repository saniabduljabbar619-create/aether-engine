import os
import sys
import types
from dataclasses import dataclass

import numpy as np

from audio.stt_engine import FasterWhisperEngine, default_threads, is_phantom


@dataclass
class Seg:
    text: str
    no_speech_prob: float = 0.0
    avg_logprob: float = -0.3
    compression_ratio: float = 1.2


class FakeModel:
    def __init__(self, segments, language="en"):
        self.segments, self.language, self.kwargs = segments, language, None

    def transcribe(self, audio, **kwargs):
        self.kwargs = kwargs
        return iter(self.segments), types.SimpleNamespace(language=self.language)


AUDIO = np.zeros(16000, np.float32)


def run(segments, **engine_kwargs):
    model = FakeModel(segments)
    return FasterWhisperEngine(_model=model, **engine_kwargs).transcribe(AUDIO), model


def test_language_prompt_and_speed_options_reach_whisper():
    (text, lang), model = run([Seg("Aether, what time is it?")], language="en", prompt="Hey Aether")
    k = model.kwargs
    assert (text, lang) == ("Aether, what time is it?", "en")
    assert k["language"] == "en" and k["initial_prompt"] == "Hey Aether"
    assert k["beam_size"] == 1 and k["without_timestamps"] is True and k["condition_on_previous_text"] is False


def test_auto_language_passes_none():
    _, model = run([Seg("hola")], language=None)
    assert model.kwargs["language"] is None and model.kwargs["initial_prompt"] is None


def test_doubtful_segments_are_dropped():
    (text, _), _ = run([
        Seg("Aether who am I"),
        Seg("ghost words", no_speech_prob=0.9, avg_logprob=-2.0),  # probably silence
        Seg("la la la la la la", compression_ratio=3.0),  # looping garbage
        Seg("is quiet but sure", no_speech_prob=0.9, avg_logprob=-0.3),  # confident words are kept
    ])
    assert text == "Aether who am I is quiet but sure"


def test_phantom_phrases_only_when_they_are_the_whole_text():
    assert is_phantom("Thank you.") and is_phantom("I'm sorry.") and is_phantom("  Okay! ")
    assert not is_phantom("Aether, thank you") and not is_phantom("thank you for the help")
    (text, _), _ = run([Seg("Thank you.")])
    assert text == ""
    (text, _), _ = run([Seg("Aether, thank you.")])
    assert text == "Aether, thank you."


def test_default_threads_leaves_cores_free(monkeypatch):
    for cores, expected in ((8, 4), (4, 2), (2, 1), (1, 1), (None, 1)):
        monkeypatch.setattr(os, "cpu_count", lambda c=cores: c)
        assert default_threads() == expected


def test_threads_setting_reaches_the_model(monkeypatch):
    made = {}

    class FakeWhisperModel:
        def __init__(self, name, **kw):
            made.update(name=name, **kw)

    monkeypatch.setitem(sys.modules, "faster_whisper", types.SimpleNamespace(WhisperModel=FakeWhisperModel))
    monkeypatch.setattr(os, "cpu_count", lambda: 4)
    FasterWhisperEngine("base")
    assert made["cpu_threads"] == 2 and made["name"] == "base" and made["compute_type"] == "int8"
    FasterWhisperEngine("base", threads=3)
    assert made["cpu_threads"] == 3
