import asyncio

import numpy as np
import pytest

import core.orchestrator as orch_mod
from audio.stt_engine import Transcript
from core.event_bus import EventBus, Topic
from core.orchestrator import Orchestrator
from storage.face_store import FaceStore
from storage.local_db import AuditLog
from vision.capture import Frame
from vision.detector import FaceDetection, FaceEvent
from vision.recognition import FacesUpdate, Recognition
from vision.spatial_engine import VectorIdentityMatcher
from vision.tracker import TrackState

rng = np.random.default_rng(5)
ALICE, BOB = (rng.normal(size=128).astype(np.float32) for _ in range(2))


class FakeSpeaker:
    def __init__(self):
        self.said, self.speaking = [], False

    def say(self, text):
        self.said.append(text)

    def was_speaking_at(self, t):
        return self.speaking

    def close(self): ...


class Rig:
    def __init__(self, bus, store, matcher, audit, speaker, replies, orch):
        self.bus, self.store, self.matcher, self.audit = bus, store, matcher, audit
        self.speaker, self.replies, self.orch = speaker, replies, orch

    async def hear(self, text):
        self.bus.publish(Topic.SPEECH_TRANSCRIBED, Transcript(text, "en", 1.0, 10.0, 0.0))
        await asyncio.sleep(0.03)

    async def see(self, *identities):
        """identities: names, 'Unknown', or None (still verifying)."""
        tracks = [TrackState(i + 1, (10, 100 + 120 * i, 110, 10 + 120 * i), ident, 0.8, False)
                  for i, ident in enumerate(identities)]
        self.bus.publish(Topic.FACES_UPDATED, FacesUpdate(Frame(1, 0.0, None), tracks, 5.0))
        await asyncio.sleep(0.02)

    async def face_sample(self, emb=ALICE, score=0.95, n=1):
        faces = [FaceDetection((10, 100, 110, 10), score, np.zeros((5, 2), np.float32), emb)] * n
        self.bus.publish(Topic.FACE_DETECTED, FaceEvent(Frame(1, 0.0, None), faces, 5.0))
        await asyncio.sleep(0.02)


@pytest.fixture
async def rig(tmp_path, monkeypatch):
    monkeypatch.setattr(orch_mod, "ENROLL_MIN_GAP", 0.0)
    bus = EventBus()
    await bus.start()
    store = FaceStore(tmp_path)
    matcher = VectorIdentityMatcher()
    audit = AuditLog(":memory:")
    speaker, replies = FakeSpeaker(), []
    o = Orchestrator(bus, matcher, store, audit, speaker, status_fn=lambda: "Camera is running.",
                     on_reply=replies.append)
    o.start()
    yield Rig(bus, store, matcher, audit, speaker, replies, o)
    await bus.stop()


def enrolled(rig, name="Alice", emb=ALICE):
    rig.store.add(name, emb)
    rig.matcher.register_identities(rig.store.load_all())


async def test_greets_recognized_person(rig):
    rig.bus.publish(Topic.FACE_RECOGNIZED, Recognition("Sani_Abdul", 0.8, 1, 0.0))
    await asyncio.sleep(0.03)
    assert rig.replies == ["Hello Sani Abdul."] and rig.speaker.said == rig.replies


async def test_ignores_speech_without_wake_word(rig):
    await rig.hear("what time is it")
    assert rig.replies == []


async def test_answers_time_and_help_and_unknown(rig):
    await rig.hear("Aether, what time is it?")
    await rig.hear("Aether help")
    await rig.hear("Aether juggle")
    assert rig.replies[0].startswith("It's ") and rig.replies[0].endswith(("AM.", "PM."))
    assert "who is at the camera" in rig.replies[1] and "didn't understand" in rig.replies[2]


async def test_never_obeys_its_own_voice(rig):
    rig.speaker.speaking = True
    await rig.hear("Aether goodbye")
    assert rig.replies == [] and not rig.orch.shutdown.is_set()


async def test_who_is_here_variants(rig):
    await rig.hear("Aether who is at the camera")
    assert rig.replies[-1] == "I don't see anyone."
    await rig.see("Alice", "Unknown")
    await rig.hear("Aether who is at the camera")
    assert rig.replies[-1] == "I can see Alice, plus 1 person I don't recognize."
    await rig.see("Unknown", None)
    await rig.hear("Aether who is at the camera")
    assert rig.replies[-1] == "I see 1 person I don't recognize and 1 I'm still checking."
    await rig.see("Alice")
    await rig.hear("Aether who is at the camera")
    assert rig.replies[-1] == "I can see Alice."


async def test_who_am_i(rig):
    await rig.see("Sani_Abdul")
    await rig.hear("Aether who am I")
    await rig.see("Unknown")
    await rig.hear("Aether who am I")
    await rig.see("Alice", "Unknown")
    await rig.hear("Aether who am I")
    assert rig.replies[0] == "You're Sani Abdul."
    assert "don't know you yet" in rig.replies[1] and "exactly one person" in rig.replies[2]


async def test_stale_view_is_ignored(rig, monkeypatch):
    monkeypatch.setattr(orch_mod, "VIEW_FRESH", 0.01)
    await rig.see("Alice")
    await asyncio.sleep(0.05)
    await rig.hear("Aether who is at the camera")
    assert rig.replies[-1] == "I don't see anyone."


async def test_bootstrap_enrollment_when_nobody_is_enrolled(rig):
    await rig.hear("Aether, enroll me as Sani")
    assert rig.replies[-1].startswith("Okay Sani")
    for _ in range(orch_mod.ENROLL_SAMPLES):
        await rig.face_sample(BOB)
    assert rig.replies[-1] == "Done. I'll remember you, Sani."
    assert len(rig.store.load_all()["Sani"]) == orch_mod.ENROLL_SAMPLES
    (label, _), = rig.matcher.search_batch(BOB[None])
    assert label == "Sani"  # the live matcher already knows the new face


async def test_enrollment_ignores_bad_frames(rig):
    await rig.hear("Aether enroll me as Sani")
    await rig.face_sample(BOB, n=2)  # two faces
    await rig.face_sample(BOB, score=0.5)  # not confident
    assert rig.store.load_all() == {} and rig.orch._enroll.samples == []


async def test_enrollment_needs_a_name(rig):
    await rig.hear("Aether enroll me")
    assert "followed by your name" in rig.replies[-1]


async def test_enrollment_times_out_and_saves_nothing(rig, monkeypatch):
    monkeypatch.setattr(orch_mod, "ENROLL_TIMEOUT", 0.05)
    await rig.hear("Aether enroll me as Sani")
    await rig.face_sample(BOB)
    await asyncio.sleep(0.15)
    assert "cancelled" in rig.replies[-1] and rig.store.load_all() == {} and rig.orch._enroll is None


async def test_enrollment_refused_if_nobody_known_in_view(rig):
    enrolled(rig)
    await rig.see("Unknown")
    await rig.hear("Aether enroll me as Mallory")
    assert "only someone I already recognize" in rig.replies[-1] and rig.orch._enroll is None


async def test_recognized_person_may_enroll_a_new_person(rig):
    enrolled(rig)
    await rig.see("Alice")
    await rig.hear("Aether enroll me as Bob")
    assert rig.replies[-1].startswith("Okay Bob") and rig.orch._enroll is not None


async def test_cannot_add_samples_to_someone_elses_name(rig):
    enrolled(rig, "Alice")
    enrolled(rig, "Bob", BOB)
    await rig.see("Bob")
    await rig.hear("Aether enroll me as Alice")
    assert "Only they can add" in rig.replies[-1] and rig.orch._enroll is None


async def test_forget_me_removes_only_that_person(rig):
    enrolled(rig, "Alice")
    enrolled(rig, "Bob", BOB)
    await rig.see("Alice")
    await rig.hear("Aether forget me")
    assert list(rig.store.load_all()) == ["Bob"]
    assert rig.matcher.search_batch(ALICE[None])[0][0] != "Alice"


async def test_forgetting_the_last_person_empties_the_matcher(rig):
    enrolled(rig, "Alice")
    await rig.see("Alice")
    await rig.hear("Aether forget me")
    assert rig.store.load_all() == {} and rig.matcher.search_batch(ALICE[None]) == [(None, 0.0)]


async def test_forget_refused_when_unrecognized_or_ambiguous(rig):
    enrolled(rig, "Alice")
    enrolled(rig, "Bob", BOB)
    await rig.hear("Aether forget me")
    assert "see and recognize" in rig.replies[-1]
    await rig.see("Alice", "Bob")
    await rig.hear("Aether forget me")
    assert "more than one" in rig.replies[-1] and set(rig.store.load_all()) == {"Alice", "Bob"}


async def test_who_known_and_status(rig):
    await rig.hear("Aether who do you know")
    enrolled(rig, "Sani_Abdul")
    await rig.hear("Aether who do you know")
    await rig.hear("Aether status")
    assert rig.replies == ["I don't know anyone yet.", "I know Sani Abdul.", "Camera is running. I know 1 person."]


async def test_shutdown_sets_event(rig):
    await rig.hear("Aether goodbye")
    assert rig.orch.shutdown.is_set() and rig.replies[-1] == "Goodbye."


async def test_custom_handler_and_handler_failure(rig):
    async def lock(cmd):
        return "Front door locked."

    async def broken(cmd):
        raise RuntimeError("hardware offline")

    rig.orch.register_handler("lock_door", [r"\block\b.*\bdoor\b"], lock)
    rig.orch.register_handler("fan", [r"\bfan\b"], broken)
    await rig.hear("Aether lock the front door")
    await rig.hear("Aether turn on the fan")
    await rig.hear("Aether what time is it")  # still alive afterwards
    assert rig.replies[0] == "Front door locked." and "went wrong" in rig.replies[1]
    assert rig.replies[2].startswith("It's ")


async def test_audit_trail_and_recent_activity(rig):
    rig.bus.publish(Topic.FACE_RECOGNIZED, Recognition("Alice", 0.8, 1, 0.0))
    await rig.hear("Aether what time is it")
    await rig.hear("Aether recent activity")
    kinds = [k for _, k, _ in rig.audit.recent(10)]
    assert "face_recognized" in kinds and "command" in kinds
    assert rig.replies[-1].startswith("Recently:") and "face recognized Alice" in rig.replies[-1]
