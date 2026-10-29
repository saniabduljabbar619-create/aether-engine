import asyncio
import threading
import time

import numpy as np
import pytest

from core.event_bus import EventBus, Topic
from vision.capture import CameraError, FrameCapture


class FakeSource:
    def __init__(self, frames=None, opened=True, fail_after=None, delay=0.002):
        self._opened, self._n, self._fail_after, self._delay = opened, 0, fail_after, delay
        self.released = threading.Event()

    def isOpened(self):
        return self._opened

    def read(self):
        time.sleep(self._delay)
        self._n += 1
        if self._fail_after is not None and self._n > self._fail_after:
            return False, None
        return True, np.full((4, 4, 3), self._n % 256, dtype=np.uint8)

    def release(self):
        self.released.set()


@pytest.fixture
async def bus():
    b = EventBus()
    await b.start()
    yield b
    await b.stop()


async def test_publishes_frames_and_stops_cleanly(bus):
    got = []

    async def h(e):
        got.append(e.payload)

    bus.subscribe(Topic.FRAME_CAPTURED, h)
    src = FakeSource()
    cap = FrameCapture(bus, source_factory=lambda: src)
    await cap.start()
    await asyncio.sleep(0.15)
    await cap.stop()
    assert len(got) > 10
    assert [f.frame_id for f in got] == sorted(f.frame_id for f in got)
    assert src.released.is_set()
    assert cap.latest() is not None and cap.fps > 0


async def test_open_failure_raises(bus):
    cap = FrameCapture(bus, source_factory=lambda: FakeSource(opened=False))
    with pytest.raises(CameraError):
        await cap.start()


async def test_camera_loss_reports_error_and_reconnects(bus):
    errors, frames = [], []

    async def on_err(e):
        errors.append(e.payload)

    async def on_frame(e):
        frames.append(e.payload)

    bus.subscribe(Topic.SYSTEM_ERROR, on_err)
    bus.subscribe(Topic.FRAME_CAPTURED, on_frame, max_queue=1000)
    sources = iter([FakeSource(fail_after=5), FakeSource()])
    cap = FrameCapture(bus, source_factory=lambda: next(sources), max_failures=3, reconnect_delay=0.02)
    await cap.start()
    await asyncio.sleep(0.4)
    await cap.stop()
    assert errors == ["camera lost, reconnecting"]
    assert len(frames) > 10  # frames kept flowing after the reconnect


async def test_slow_consumer_does_not_slow_capture(bus):
    async def slow(e):
        await asyncio.sleep(0.2)

    sub = bus.subscribe(Topic.FRAME_CAPTURED, slow, max_queue=1)
    cap = FrameCapture(bus, source_factory=lambda: FakeSource())
    await cap.start()
    await asyncio.sleep(0.3)
    await cap.stop()
    assert cap.latest().frame_id > 20 and sub.dropped > 10
