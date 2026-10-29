import asyncio
import time

import numpy as np
import pytest

from audio.recorder import Utterance
from audio.stt_engine import SttService, Transcript
from core.event_bus import EventBus, Topic


class FakeEngine:
    def __init__(self, text="lock the front door", fail=False, delay=0.0):
        self.text, self.fail, self.delay, self.calls = text, fail, delay, 0

    def transcribe(self, audio):
        self.calls += 1
        time.sleep(self.delay)
        if self.fail:
            raise RuntimeError("model crashed")
        return self.text, "en"


@pytest.fixture
async def bus():
    b = EventBus()
    await b.start()
    yield b
    await b.stop()


def utt(i=0):
    return Utterance(np.zeros(16000, np.float32), float(i))


async def test_publishes_transcript(bus):
    got = []

    async def h(e):
        got.append(e.payload)

    bus.subscribe(Topic.SPEECH_TRANSCRIBED, h)
    SttService(bus, FakeEngine()).start()
    bus.publish(Topic.SPEECH_CAPTURED, utt())
    await asyncio.sleep(0.05)
    assert len(got) == 1 and isinstance(got[0], Transcript)
    assert got[0].text == "lock the front door" and got[0].language == "en" and got[0].audio_seconds == 1.0


async def test_empty_text_is_not_published(bus):
    got = []

    async def h(e):
        got.append(e)

    bus.subscribe(Topic.SPEECH_TRANSCRIBED, h)
    SttService(bus, FakeEngine(text="")).start()
    bus.publish(Topic.SPEECH_CAPTURED, utt())
    await asyncio.sleep(0.05)
    assert got == []


async def test_engine_failure_does_not_kill_service(bus):
    got = []

    async def h(e):
        got.append(e)

    bus.subscribe(Topic.SPEECH_TRANSCRIBED, h)
    eng = FakeEngine(fail=True)
    SttService(bus, eng).start()
    bus.publish(Topic.SPEECH_CAPTURED, utt(1))
    await asyncio.sleep(0.05)
    eng.fail = False
    bus.publish(Topic.SPEECH_CAPTURED, utt(2))
    await asyncio.sleep(0.05)
    assert len(got) == 1


async def test_phrases_are_processed_in_order(bus):
    got = []

    async def h(e):
        got.append(e.payload.started_at)

    bus.subscribe(Topic.SPEECH_TRANSCRIBED, h)
    SttService(bus, FakeEngine(delay=0.02)).start()
    for i in range(4):
        bus.publish(Topic.SPEECH_CAPTURED, utt(i))
    await asyncio.sleep(0.4)
    assert got == [0.0, 1.0, 2.0, 3.0]
