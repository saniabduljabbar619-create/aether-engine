import itertools
import random
from types import SimpleNamespace

import numpy as np
import pytest

from vision.enrollment import USER_LEFT, USER_RIGHT
from vision.liveness import HOLD_FRAMES, MIN_TURN, Phase, LivenessSession
from vision.pose import simulate_landmarks

DT = 0.1  # seconds between frames
SWEEP = [0] * 6 + [4, 8, 12, 16, 20, 20, 20, 20]  # a person turning their head up to 20 degrees and holding


def face(turn_right_deg=0.0, **kw):
    return SimpleNamespace(landmarks=simulate_landmarks(turn_right_deg, **kw))


def drive(session, frames, start=0.0, dt=DT):
    """Feed frames (each a list of faces); return the last status."""
    t, status = start, None
    for faces in frames:
        status = session.offer(faces, t)
        t += dt
        if status.passed is not None:
            break
    return status


def sweep(side_sign, planar=False, **kw):
    """side_sign +1 = the person turns to their right, -1 = to their left."""
    return [[face(side_sign * d, planar=planar, **kw)] for d in SWEEP]


@pytest.mark.parametrize("asked,sign", [(USER_RIGHT, +1), (USER_LEFT, -1)])
def test_a_real_turn_in_the_requested_direction_passes(asked, sign):
    status = drive(LivenessSession(asked), sweep(sign))
    assert status.phase is Phase.PASSED and status.passed is True and status.turn > MIN_TURN


@pytest.mark.parametrize("asked,sign", [(USER_RIGHT, -1), (USER_LEFT, +1)])
def test_turning_the_wrong_way_does_not_pass_but_asks_for_the_other_way(asked, sign):
    s = LivenessSession(asked)
    status = drive(s, sweep(sign))
    assert status.passed is None and "other way" in status.hint
    # ...and the person can still correct themselves
    assert drive(s, sweep(-sign), start=5.0).passed is True


def test_the_min_turn_is_well_clear_of_the_worst_photo():
    assert MIN_TURN >= 3 * 0.035  # 0.035 = the worst planar deviation measured in tests/test_pose.py


def test_a_flat_photo_never_passes_however_it_is_moved():
    """Rotate, tilt, spin and move a photo through every pose the model covers, at several distances."""
    for asked, dist, roll, pitch in itertools.product((USER_RIGHT, USER_LEFT), (25, 40, 60), (-40, 0, 40), (-30, 30)):
        frames = [[face(min(3 * d, 45), planar=True, roll_deg=roll, pitch_deg=pitch, distance_cm=dist)] for d in SWEEP]
        s = LivenessSession(asked, timeout=60)
        assert drive(s, frames * 3).passed is None, (asked, dist, roll, pitch)  # never decided "real"
    # and when time runs out it is an explicit failure, not a pass
    s = LivenessSession(USER_RIGHT)
    status = drive(s, [[face(45, planar=True)]] * 200)
    assert status.phase is Phase.FAILED and status.reason == "no head turn in time"


def test_with_detector_noise_real_heads_pass_and_photos_do_not():
    rng = np.random.default_rng(3)
    real = photo = 0
    for i in range(80):
        asked, sign = (USER_RIGHT, +1) if i % 2 == 0 else (USER_LEFT, -1)
        s = LivenessSession(asked)
        real += drive(s, [[face(sign * d, noise_px=1.0, rng=rng)] for d in SWEEP]).passed is True
        s = LivenessSession(asked, timeout=60)
        photo += drive(s, [[face(sign * min(3 * d, 45), planar=True, noise_px=1.0, rng=rng)] for d in SWEEP] * 3).passed is True
    assert real == 80  # every real turn passes
    assert photo == 0  # not one photo does


def test_the_baseline_must_be_steady_so_a_moving_head_cannot_set_it():
    s = LivenessSession(USER_RIGHT, timeout=60)
    frames = [[face(15 if i % 2 else -15)] for i in range(60)]  # never steady
    status = drive(s, frames)
    assert s.phase is Phase.BASELINE and s.baseline is None and status.passed is None


def test_one_or_two_glitch_frames_cannot_pass_but_three_in_a_row_can():
    def with_burst(n):
        frames = [[face(0)]] * 8 + [[face(20)]] * n + [[face(0)]] * 3
        return drive(LivenessSession(USER_RIGHT), frames)

    assert HOLD_FRAMES == 3
    assert with_burst(1).passed is None and with_burst(2).passed is None
    assert with_burst(3).passed is True


def test_timeout_without_any_turn_fails():
    status = drive(LivenessSession(USER_RIGHT, timeout=5), [[face(0)]] * 100)
    assert status.phase is Phase.FAILED and status.reason == "no head turn in time" and status.passed is False


def test_a_brief_loss_of_the_face_is_tolerated_a_long_one_fails():
    s = LivenessSession(USER_RIGHT)
    frames = sweep(+1)
    gap = [[]] * 10  # one second with no face, as when a fast turn confuses the detector
    assert drive(s, frames[:9] + gap + frames[9:]).passed is True
    long_gap = drive(LivenessSession(USER_RIGHT), frames[:8] + [[]] * 30)
    assert long_gap.phase is Phase.FAILED and long_gap.reason == "face lost"


def test_more_than_one_face_makes_no_progress_and_says_so():
    s = LivenessSession(USER_RIGHT)
    status = drive(s, [[face(0), face(0)]] * 10)
    assert "Only the person being verified" in status.hint and s.baseline is None


def test_unreadable_landmarks_do_not_crash_and_ask_for_a_clearer_view():
    broken = SimpleNamespace(landmarks=np.zeros((5, 2), np.float32))
    status = LivenessSession(USER_RIGHT).offer([broken], 0.0)
    assert "Face the camera" in status.hint


def test_a_decision_is_final():
    s = LivenessSession(USER_RIGHT)
    assert drive(s, sweep(+1)).passed is True
    assert s.offer([face(0)], 99.0).passed is True  # even much later, even with a different face
    f = LivenessSession(USER_RIGHT, timeout=1)
    assert drive(f, [[face(0)]] * 30).passed is False
    assert f.offer([face(20)], 99.0).passed is False


def test_the_direction_is_random_each_time_and_can_be_fixed_for_tests():
    seen = {LivenessSession().direction for _ in range(100)}
    assert seen == {USER_RIGHT, USER_LEFT}
    assert LivenessSession(rng=random.Random(1)).direction == LivenessSession(rng=random.Random(1)).direction


def test_the_instruction_names_the_side():
    assert "right" in LivenessSession(USER_RIGHT).instruction
    assert "left" in LivenessSession(USER_LEFT).instruction
    s = LivenessSession(USER_LEFT)
    assert s.offer([face(0)], 0.0).hint == "Look straight at the camera."
