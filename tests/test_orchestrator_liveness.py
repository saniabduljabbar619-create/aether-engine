"""The "verify me" flow through the orchestrator: who is checked, what a photo does, what it unlocks."""
import asyncio


from core.event_bus import Topic
from tests import test_orchestrator as base
from tests.test_orchestrator import ALICE, BOB, enrolled, view
from vision.capture import Frame
from vision.detector import FaceDetection, FaceEvent
from vision.enrollment import USER_RIGHT
from vision.pose import simulate_landmarks

rig = base.rig  # reuse the orchestrator fixture


async def show(rig, emb, turn_deg, planar=False):
    face = FaceDetection((10, 100, 110, 10), 0.95, simulate_landmarks(turn_deg, planar=planar), view(emb))
    rig.bus.publish(Topic.FACE_DETECTED, FaceEvent(Frame(1, 0.0, None), [face], 5.0))
    await asyncio.sleep(0.02)


async def perform(rig, emb, planar=False, turn=35.0):
    """Look straight, then turn toward the requested side (as a person would)."""
    toward = turn if rig.orch._live.session.direction == USER_RIGHT else -turn
    for _ in range(6):
        await show(rig, emb, 0.0, planar)
    for _ in range(8):
        if rig.orch._live is None:
            break
        await show(rig, emb, toward, planar)


async def test_real_turn_verifies(rig):
    enrolled(rig)
    await rig.see("Alice")
    await rig.hear("aether verify me")
    assert rig.orch.verifying == "Alice" and "turn your head" in rig.replies[-1]
    await perform(rig, ALICE)
    assert rig.orch.verified_names() == ["Alice"]
    assert "Verified" in rig.replies[-1]


async def test_photo_is_refused(rig):
    enrolled(rig)
    await rig.see("Alice")
    await rig.hear("aether verify me")
    for _ in range(6):
        await show(rig, ALICE, 0.0, planar=True)
    for _ in range(8):
        await show(rig, ALICE, 35.0, planar=True)
        await show(rig, ALICE, -35.0, planar=True)
    rig.orch._live.session.offer([], 1e6)  # let the clock run out
    await show(rig, ALICE, 0.0, planar=True)
    assert rig.orch.verified_names() == []


async def test_needs_someone_recognized_and_only_one(rig):
    await rig.hear("aether verify me")
    assert "recognize" in rig.replies[-1] and rig.orch.verifying is None
    enrolled(rig)
    enrolled(rig, "Bob", BOB)
    await rig.see("Alice", "Bob")
    await rig.hear("aether verify me")
    assert "one person" in rig.replies[-1]


async def test_different_face_fails(rig):
    enrolled(rig)
    enrolled(rig, "Bob", BOB)
    await rig.see("Alice")
    await rig.hear("aether verify me")
    for _ in range(4):
        await show(rig, BOB, 0.0)  # someone else steps in
    assert rig.orch.verifying is None and rig.orch.verified_names() == []
    assert "not Alice" in rig.replies[-1]


async def test_verification_expires(rig):
    rig.orch.verified_seconds = 0.05
    enrolled(rig)
    await rig.see("Alice")
    await rig.hear("aether verify me")
    await perform(rig, ALICE)
    assert rig.orch.verified_names() == ["Alice"]
    await asyncio.sleep(0.1)
    assert rig.orch.verified_names() == []


async def test_policy_guards_forget_and_custom_commands(rig):
    rig.orch.require_liveness = True
    enrolled(rig)
    ran = []

    async def unlock(cmd):
        ran.append(1)
        return "Unlocked."

    rig.orch.register_handler("unlock", [r"\bunlock\b"], unlock, requires_verified=True)
    await rig.see("Alice")
    await rig.hear("aether unlock the door")
    assert not ran and "verify" in rig.replies[-1]
    await rig.hear("aether forget me")
    assert sorted(rig.store.load_all()) == ["Alice"]
    await rig.hear("aether verify me")
    await perform(rig, ALICE)
    await rig.hear("aether unlock the door")
    assert ran == [1]


async def test_policy_off_by_default(rig):
    enrolled(rig)
    await rig.see("Alice")
    await rig.hear("aether forget me")
    assert sorted(rig.store.load_all()) == []
