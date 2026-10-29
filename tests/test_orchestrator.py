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
ALICE, BOB, CAROL = (rng.normal(size=128).astype(np.float32) for _ in range(3))
# yaw of each frame: straight-on, a turn to each side, then straight again (a cooperative person)
POSES = [0.0] * 4 + [-0.3] * 3 + [0.3] * 3 + [0.0] * 2


def view(base):
    """A slightly different view of the same person."""
    return (base + rng.normal(size=128) * 0.7).astype(np.float32)


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

    async def face_sample(self, emb=ALICE, score=0.95, n=1, yaw=0.0):
        lm = np.array([[80, 100], [120, 100], [100 + yaw * 40, 125], [88, 150], [112, 150]], np.float32)
        faces = [FaceDetection((10, 100, 110, 10), score, lm, view(emb))] * n
        self.bus.publish(Topic.FACE_DETECTED, FaceEvent(Frame(1, 0.0, None), faces, 5.0))
        await asyncio.sleep(0.02)

    async def enroll_frames(self, base, poses=POSES):
        """Show one person turning their head through the given yaws."""
        for yaw in poses:
            await self.face_sample(base, yaw=yaw)


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
    await rig.enroll_frames(BOB)
    assert rig.replies[-1] == "Done. I'll remember you, Sani."
    assert len(rig.store.load_all()["Sani"]) == orch_mod.ENROLL_SAMPLES
    (label, _), = rig.matcher.search_batch(BOB[None])
    assert label == "Sani"  # the live matcher already knows the new face


async def test_enrollment_ignores_bad_frames(rig):
    await rig.hear("Aether enroll me as Sani")
    await rig.face_sample(BOB, n=2)  # two faces
    await rig.face_sample(BOB, score=0.5)  # not confident
    assert rig.store.load_all() == {} and rig.orch._enroll.samples == []
    assert "isn't clear" in rig.orch.enrollment_hint


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


async def test_enrolling_an_already_enrolled_face_under_a_new_name_is_stopped(rig):
    """The mistake from the first real run: one face enrolled twice under two names."""
    enrolled(rig, "Alice", ALICE)
    await rig.see("Alice")
    await rig.hear("Aether enroll me as Bob")
    assert rig.orch._enroll is not None
    await rig.enroll_frames(ALICE)  # it is actually Alice
    assert "already matches Alice" in rig.replies[-1] and "Nothing was saved" in rig.replies[-1]
    assert rig.orch._enroll is None and list(rig.store.load_all()) == ["Alice"]
    assert len(rig.store.load_all()["Alice"]) == 1


async def test_adding_samples_to_your_own_name_is_allowed(rig):
    enrolled(rig, "Alice", ALICE)
    await rig.see("Alice")
    await rig.hear("Aether enroll me as Alice")
    await rig.enroll_frames(ALICE)
    assert rig.replies[-1] == "Done. I'll remember you, Alice."
    assert len(rig.store.load_all()["Alice"]) == 1 + orch_mod.ENROLL_SAMPLES


async def test_two_people_taking_turns_are_rejected(rig):
    await rig.hear("Aether enroll me as Sani")
    for i, yaw in enumerate(POSES):
        await rig.face_sample(BOB if i % 2 else CAROL, yaw=yaw)  # alternating faces
    assert "don't look like one person" in rig.replies[-1]
    assert rig.store.load_all() == {} and rig.orch._enroll is None


async def test_ui_can_start_enrollment_and_read_progress_and_hint(rig):
    assert rig.orch.enrollment_progress is None and rig.orch.enrollment_hint == ""
    rig.orch.begin_enrollment("Sani")
    assert rig.orch.enrollment_progress == ("Sani", 0, orch_mod.ENROLL_SAMPLES)
    assert "straight at the camera" in rig.orch.enrollment_hint
    for yaw in POSES[:4]:
        await rig.face_sample(BOB, yaw=yaw)
    assert rig.orch.enrollment_progress[1] == 4 and "turn your head" in rig.orch.enrollment_hint
    await rig.enroll_frames(BOB, POSES[4:])
    assert rig.orch.enrollment_progress is None and rig.orch.enrollment_hint == ""


async def test_spoken_hints_are_throttled(rig, monkeypatch):
    monkeypatch.setattr(orch_mod, "HINT_EVERY", 1000.0)
    await rig.hear("Aether enroll me as Sani")
    await rig.enroll_frames(BOB)
    hints = [t for t in rig.speaker.said if t in ("Look straight at the camera.",) or "turn your head" in t.lower()]
    assert len(hints) <= 2  # one reply carries the first instruction; the rest are not repeated aloud


async def test_closing_cancels_an_enrollment_in_progress(rig):
    rig.orch.begin_enrollment("Sani")
    task = rig.orch._enroll_task
    rig.orch.close()
    await asyncio.sleep(0.01)
    assert rig.orch.enrollment_progress is None and task.cancelled()
