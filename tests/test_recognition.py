import asyncio

import numpy as np
import pytest

from core.event_bus import EventBus, Topic
from vision.capture import Frame
from vision.detector import FaceDetection, FaceEvent
from vision.recognition import RecognitionService
from vision.spatial_engine import VectorIdentityMatcher

rng = np.random.default_rng(3)
ALICE, BOB, STRANGER = (rng.normal(size=128).astype(np.float32) for _ in range(3))


def ev(i, *embs, left=10):
    faces = [FaceDetection((10, left + 100 + k * 300, 110, left + k * 300), 0.9,
                           np.zeros((5, 2), np.float32), e) for k, e in enumerate(embs)]
    return FaceEvent(Frame(i, i * 0.1, None), faces, 5.0)


@pytest.fixture
async def setup():
    bus = EventBus()
    await bus.start()
    m = VectorIdentityMatcher()
    m.register_identities({"alice": [ALICE], "bob": [BOB]})
    rec, upd = [], []

    async def on_rec(e):
        rec.append(e.payload)

    async def on_upd(e):
        upd.append(e.payload)

    bus.subscribe(Topic.FACE_RECOGNIZED, on_rec, max_queue=1000)
    bus.subscribe(Topic.FACES_UPDATED, on_upd, max_queue=1000)
    svc = RecognitionService(bus, m, greet_cooldown=30.0)
    yield bus, svc, rec, upd
    await bus.stop()


async def push(bus, event):
    bus.publish(Topic.FACE_DETECTED, event)
    await asyncio.sleep(0.01)


async def test_recognizes_known_person_once(setup):
    bus, svc, rec, upd = setup
    svc.start()
    for i in range(10):
        await push(bus, ev(i, ALICE * 2))  # scaled copy: same direction, cosine = 1
    assert [r.name for r in rec] == ["alice"] and rec[0].score > 0.99
    assert upd[-1].tracks[0].identity == "alice"


async def test_two_people_get_two_names(setup):
    bus, svc, rec, upd = setup
    svc.start()
    for i in range(10):
        await push(bus, ev(i, ALICE, BOB))
    assert sorted(r.name for r in rec) == ["alice", "bob"]


async def test_stranger_is_unknown_and_not_announced(setup):
    bus, svc, rec, upd = setup
    svc.start()
    for i in range(10):
        await push(bus, ev(i, STRANGER))
    assert rec == [] and upd[-1].tracks[0].identity == "Unknown"


async def test_cooldown_suppresses_repeat_greeting(setup):
    bus, svc, rec, upd = setup
    svc.start()
    for i in range(6):
        await push(bus, ev(i, ALICE))
    await push(bus, ev(6))  # leaves the frame
    await push(bus, ev(30))  # long gap: old track expires
    for i in range(31, 38):  # returns 3 s later (< 30 s cooldown)
        await push(bus, ev(i, ALICE))
    assert len(rec) == 1


async def test_no_faces_still_publishes_update(setup):
    bus, svc, rec, upd = setup
    svc.start()
    await push(bus, ev(1))
    assert len(upd) == 1 and upd[0].tracks == []
