import numpy as np

from core.commands import CommandRouter
from examples.wake_test import Trial, evaluate, summarize


class Scripted:
    """Stands in for Whisper: returns the next scripted text for each clip."""

    def __init__(self, texts):
        self.texts = iter(texts)

    def transcribe(self, audio):
        return next(self.texts), "en"


CLIP = np.zeros(16000, np.float32)


def test_evaluate_scores_wake_word_and_command():
    trials = evaluate([CLIP, CLIP, None, CLIP], Scripted(["Aether, what time is it?", "The police and the camera",
                                                          "Aether who am I"]), CommandRouter())
    assert [t.woke for t in trials] == [True, False, False, True]
    assert [t.intent for t in trials] == ["time", None, None, "who_am_i"]
    assert trials[2].heard == ""  # nothing was said: counted as a miss, not a crash


def test_summary_counts_and_verdicts():
    def trials(hits, n=10):
        return [Trial("Aether what time is it", True, "time", 1.5)] * hits + [Trial("we shot the camera", False, None, 1.2)] * (n - hits)

    good = summarize(trials(10), "Aether", "time")
    assert "10/10" in good and "excellent" in good
    ok = summarize(trials(8), "Aether", "time")
    assert "8/10" in ok and "workable" in ok and '"we shot the camera"' in ok
    bad = summarize(trials(3), "Aether", "time")
    assert "3/10" in bad and "poor" in bad and "--wake computer" in bad and "F9" in bad


def test_command_must_match_the_expected_one():
    t = [Trial("Aether who am I", True, "who_am_i", 1.0)]
    assert "Whole command understood:    0/1" in summarize(t, "Aether", "time")
    assert "recognized:   1/1" in summarize(t, "Aether", "time")


def test_a_different_wake_word_is_used_end_to_end():
    r = CommandRouter(wake_words=["computer"])
    trials = evaluate([CLIP, CLIP], Scripted(["Computer, what time is it", "Aether what time is it"]), r)
    assert [t.woke for t in trials] == [True, False]
