import asyncio
import threading

import pytest

from core.event_bus import WILDCARD, EventBus, Overflow, Topic


@pytest.fixture
async def bus():
    b = EventBus()
    await b.start()
    yield b
    await b.stop()


async def test_delivers_to_topic_subscribers(bus):
    got = []

    async def h(e):
        got.append(e.payload)

    bus.subscribe(Topic.FACE_RECOGNIZED, h)
    bus.publish(Topic.FACE_RECOGNIZED, "alice")
    bus.publish(Topic.SPEECH_TRANSCRIBED, "ignored")
    await bus.stop()
    assert got == ["alice"]


async def test_wildcard_receives_everything(bus):
    seen = []

    async def h(e):
        seen.append(e.topic)

    bus.subscribe(WILDCARD, h)
    bus.publish(Topic.FACE_DETECTED)
    bus.publish(Topic.SPEECH_STARTED)
    await bus.stop()
    assert seen == [Topic.FACE_DETECTED.value, Topic.SPEECH_STARTED.value]


async def test_slow_subscriber_does_not_block_publisher_or_others(bus):
    fast = []
    release = asyncio.Event()

    async def slow(e):
        await release.wait()

    async def quick(e):
        fast.append(e.payload)

    bus.subscribe(Topic.FRAME_CAPTURED, slow, max_queue=2)
    bus.subscribe(Topic.FRAME_CAPTURED, quick)
    for i in range(10):
        bus.publish(Topic.FRAME_CAPTURED, i)  # never awaits, never blocks
    await asyncio.sleep(0.01)
    assert fast == list(range(10))
    release.set()


async def test_drop_oldest_keeps_newest(bus):
    got = []
    gate = asyncio.Event()

    async def h(e):
        await gate.wait()
        got.append(e.payload)

    sub = bus.subscribe(Topic.FRAME_CAPTURED, h, max_queue=3, overflow=Overflow.DROP_OLDEST)
    for i in range(10):
        bus.publish(Topic.FRAME_CAPTURED, i)
        await asyncio.sleep(0)
    gate.set()
    await bus.stop()
    assert got[-3:] == [7, 8, 9] and sub.dropped > 0


async def test_drop_newest_keeps_oldest(bus):
    got = []
    gate = asyncio.Event()

    async def h(e):
        await gate.wait()
        got.append(e.payload)

    bus.subscribe(Topic.FRAME_CAPTURED, h, max_queue=2, overflow=Overflow.DROP_NEWEST)
    for i in range(6):
        bus.publish(Topic.FRAME_CAPTURED, i)
        await asyncio.sleep(0)
    gate.set()
    await bus.stop()
    assert 0 in got and 5 not in got


async def test_handler_exception_is_isolated(bus):
    ok = []

    async def bad(e):
        raise ValueError("boom")

    async def good(e):
        ok.append(e.payload)

    bad_sub = bus.subscribe(Topic.SYSTEM_ERROR, bad)
    bus.subscribe(Topic.SYSTEM_ERROR, good)
    bus.publish(Topic.SYSTEM_ERROR, 1)
    bus.publish(Topic.SYSTEM_ERROR, 2)
    await bus.stop()
    assert ok == [1, 2] and bad_sub.errors == 2


async def test_publish_threadsafe(bus):
    got = []

    async def h(e):
        got.append(e.payload)

    bus.subscribe(Topic.FRAME_CAPTURED, h)
    t = threading.Thread(target=lambda: [bus.publish_threadsafe(Topic.FRAME_CAPTURED, i) for i in range(5)])
    t.start()
    t.join()
    await asyncio.sleep(0.05)
    assert got == list(range(5))


async def test_subscribe_after_start_and_unsubscribe(bus):
    got = []

    async def h(e):
        got.append(e.payload)

    sub = bus.subscribe(Topic.COMMAND_ISSUED, h)
    bus.publish(Topic.COMMAND_ISSUED, "a")
    await asyncio.sleep(0.01)
    bus.unsubscribe(sub)
    bus.publish(Topic.COMMAND_ISSUED, "b")
    await asyncio.sleep(0.01)
    assert got == ["a"]


async def test_late_publish_during_stop_cannot_evict_the_shutdown_marker():
    """Regression: a full max_queue=1 queue holding the stop marker used to lose it to drop_oldest."""
    bus = EventBus()
    await bus.start()
    seen = []

    async def slow(e):
        await asyncio.sleep(0.05)
        seen.append(e.payload)

    bus.subscribe(Topic.FRAME_CAPTURED, slow, max_queue=1)
    bus.publish(Topic.FRAME_CAPTURED, 1)  # being handled
    await asyncio.sleep(0.01)
    bus.publish(Topic.FRAME_CAPTURED, 2)  # waiting: queue is full
    stopping = asyncio.create_task(bus.stop())
    await asyncio.sleep(0.07)  # worker has taken 2, so the stop marker now sits in the 1-slot queue
    bus.publish(Topic.FRAME_CAPTURED, 3)  # a late publish, e.g. from another handler
    await asyncio.wait_for(stopping, 2)  # must not hang
    assert seen == [1, 2]  # queued events were handled; the late one was refused


async def test_publish_after_stop_is_ignored():
    bus = EventBus()
    await bus.start()
    got = []

    async def h(e):
        got.append(e)

    sub = bus.subscribe(Topic.SYSTEM_ERROR, h)
    await bus.stop()
    bus.publish(Topic.SYSTEM_ERROR, "late")
    assert sub.queue.qsize() == 0 and got == []


async def test_stop_gives_up_on_a_stuck_handler():
    bus = EventBus()
    await bus.start()
    never = asyncio.Event()

    async def stuck(e):
        await never.wait()

    bus.subscribe(Topic.SYSTEM_ERROR, stuck)
    bus.publish(Topic.SYSTEM_ERROR, 1)
    await asyncio.sleep(0.02)
    t0 = asyncio.get_running_loop().time()
    await asyncio.wait_for(bus.stop(timeout=0.3), 3)
    assert asyncio.get_running_loop().time() - t0 < 1.5


async def test_bus_can_be_restarted_after_stop():
    bus = EventBus()
    await bus.start()
    await bus.stop()
    got = []

    async def h(e):
        got.append(e.payload)

    bus.subscribe(Topic.SYSTEM_ERROR, h)
    await bus.start()
    bus.publish(Topic.SYSTEM_ERROR, "again")
    await asyncio.sleep(0.02)
    await bus.stop()
    assert got == ["again"]
