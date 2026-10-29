import asyncio
import sys
import types

import numpy as np
import pytest

from audio.recorder import (FRAME, SAMPLE_RATE, EnergyVAD, MicRecorder, SpeechStart, Utterance,
                            UtteranceSegmenter, level_db)
from core.event_bus import EventBus, Topic

rng = np.random.default_rng(0)


def noise(sec, amp=0.001):
    return (rng.normal(size=int(sec * SAMPLE_RATE)) * amp).astype(np.float32)


def speech(sec, amp=0.1):
    t = np.arange(int(sec * SAMPLE_RATE)) / SAMPLE_RATE
    tone = np.sin(2 * np.pi * 220 * t) + 0.5 * np.sin(2 * np.pi * 660 * t)
    return (amp * tone * (0.6 + 0.4 * np.sin(2 * np.pi * 4 * t))).astype(np.float32) + noise(sec)


def run(seg, audio, chunk=FRAME):
    events = []
    for i in range(0, len(audio), chunk):
        events += seg.push(audio[i:i + chunk], now=i / SAMPLE_RATE)
    return events


def utterances(events):
    return [e for e in events if isinstance(e, Utterance)]


def test_level_db():
    assert abs(level_db(np.ones(480, np.float32))) < 0.01
    assert level_db(np.zeros(480, np.float32)) < -150


def test_one_phrase_one_utterance_with_preroll_and_hangover():
    ev = run(UtteranceSegmenter(), np.concatenate([noise(1), speech(1.0), noise(1.5)]))
    assert sum(isinstance(e, SpeechStart) for e in ev) == 1
    (u,) = utterances(ev)
    assert 1.5 < u.duration < 2.2  # 1.0 speech + 0.3 pre-roll + 0.7 hangover


def test_pause_inside_phrase_does_not_split():
    audio = np.concatenate([noise(1), speech(0.6), noise(0.4), speech(0.6), noise(1.5)])
    assert len(utterances(run(UtteranceSegmenter(), audio))) == 1


def test_two_phrases_two_utterances():
    audio = np.concatenate([noise(1), speech(0.8), noise(1.5), speech(0.8), noise(1.5)])
    assert len(utterances(run(UtteranceSegmenter(), audio))) == 2


def test_click_is_ignored():
    audio = np.concatenate([noise(1), speech(0.05, amp=0.5), noise(2)])
    assert run(UtteranceSegmenter(), audio) == []


def test_short_blip_discarded():
    audio = np.concatenate([noise(1), speech(0.12), noise(2)])
    assert utterances(run(UtteranceSegmenter(), audio)) == []


def test_chunk_size_does_not_matter():
    audio = np.concatenate([noise(1), speech(1.0), noise(1.5), speech(0.7), noise(1.5)])
    counts = {len(utterances(run(UtteranceSegmenter(), audio, chunk=c))) for c in (FRAME, 160, 1000, 4096, 7)}
    assert counts == {2}


def test_long_speech_is_split_at_max():
    audio = np.concatenate([noise(1), speech(7.0), noise(1.5)])
    assert len(utterances(run(UtteranceSegmenter(max_seconds=3.0), audio))) >= 2


def test_steady_noise_is_absorbed_into_the_floor():
    vad = EnergyVAD()
    hum = noise(30, amp=0.02)  # fan: about -34 dBFS, well above the quiet-room floor
    flags = [vad.is_speech(hum[i:i + FRAME]) for i in range(0, len(hum) - FRAME, FRAME)]
    assert flags[0] and not any(flags[-100:])  # starts as "speech", adapts away


def test_speech_still_detected_over_adapted_hum():
    seg = UtteranceSegmenter()
    audio = np.concatenate([noise(25, amp=0.02), speech(1.0, amp=0.3), noise(1.5, amp=0.02)])
    assert len(utterances(run(seg, audio))) >= 1


async def test_mic_recorder_publishes_via_fake_sounddevice(monkeypatch):
    captured = {}

    class FakeStream:
        def __init__(self, **kw):
            captured.update(kw)

        def start(self): ...
        def stop(self): ...
        def close(self): ...

    monkeypatch.setitem(sys.modules, "sounddevice", types.SimpleNamespace(InputStream=FakeStream))
    bus = EventBus()
    await bus.start()
    started, got = [], []

    async def on_start(e):
        started.append(e.payload)

    async def on_utt(e):
        got.append(e.payload)

    bus.subscribe(Topic.SPEECH_STARTED, on_start)
    bus.subscribe(Topic.SPEECH_CAPTURED, on_utt)
    mic = MicRecorder(bus)
    mic.start()
    assert captured["samplerate"] == SAMPLE_RATE and captured["channels"] == 1
    audio = np.concatenate([noise(1), speech(1.0), noise(1.5)])
    for i in range(0, len(audio) - FRAME + 1, FRAME):  # what PortAudio's thread would do
        captured["callback"](audio[i:i + FRAME, None], FRAME, None, None)
    await asyncio.sleep(0.3)
    mic.stop()
    await bus.stop()
    assert len(started) == 1 and len(got) == 1 and got[0].duration > 1.0


async def test_mic_open_failure_raises_clear_error(monkeypatch):
    def boom(**kw):
        raise OSError("no device")

    monkeypatch.setitem(sys.modules, "sounddevice", types.SimpleNamespace(InputStream=boom))
    from audio.recorder import MicrophoneError

    with pytest.raises(MicrophoneError, match="no device"):
        MicRecorder(EventBus()).start()
