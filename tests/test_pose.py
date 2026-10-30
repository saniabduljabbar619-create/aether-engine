import itertools

import numpy as np
import pytest

from vision.pose import estimate_yaw, simulate_landmarks

FRONTAL = estimate_yaw(simulate_landmarks(0.0))


def turn_of(**kw) -> float:
    """How far the head-turn reading moves from the frontal one."""
    return estimate_yaw(simulate_landmarks(**kw)) - FRONTAL


def test_a_frontal_head_reads_zero_and_degenerate_eyes_read_none():
    assert FRONTAL == pytest.approx(0.0, abs=1e-6)
    assert estimate_yaw(np.zeros((5, 2), np.float32)) is None


def test_sign_convention_right_is_negative_and_it_is_symmetric():
    assert turn_of(turn_right_deg=20) < 0 < turn_of(turn_right_deg=-20)
    assert turn_of(turn_right_deg=20) == pytest.approx(-turn_of(turn_right_deg=-20), abs=1e-6)


@pytest.mark.parametrize("deg", [10, 15, 20, 30])
def test_a_real_head_turn_changes_the_reading_a_lot(deg):
    assert abs(turn_of(turn_right_deg=deg)) > 0.008 * deg  # about 0.0097 per degree: the nose slides sideways


def test_the_reading_grows_steadily_with_the_turn():
    readings = [abs(turn_of(turn_right_deg=d)) for d in (5, 10, 15, 20, 25, 30, 40)]
    assert readings == sorted(readings)


def test_a_flat_photo_cannot_fake_it_whatever_you_do_with_it():
    """Rotate, tilt, spin and move a photo through every pose and distance: the reading barely changes,
    because a photo can only move rigidly; its nose cannot slide across its face."""
    worst = 0.0
    for dist, turn, roll, pitch in itertools.product((25, 30, 40, 50, 80), range(-45, 46, 15),
                                                    range(-40, 41, 20), (-30, 0, 30)):
        base = estimate_yaw(simulate_landmarks(0, planar=True, distance_cm=dist))
        got = estimate_yaw(simulate_landmarks(turn, planar=True, roll_deg=roll, pitch_deg=pitch, distance_cm=dist))
        worst = max(worst, abs(got - base))
    assert worst < 0.04
    assert abs(turn_of(turn_right_deg=15)) > 3 * worst  # a modest real turn clears the worst photo by a wide margin


def test_tilting_sideways_does_not_look_like_a_turn_for_photo_or_head():
    """The old estimator read raw x offsets, so spinning a photo like a steering wheel moved it."""
    for planar in (True, False):
        base = estimate_yaw(simulate_landmarks(0, planar=planar))
        for roll in (-40, -20, 20, 40):
            assert estimate_yaw(simulate_landmarks(0, planar=planar, roll_deg=roll)) == pytest.approx(base, abs=1e-6)


def test_nodding_does_not_change_it():
    assert turn_of(pitch_deg=25) == pytest.approx(0.0, abs=1e-6)


def test_the_margin_survives_realistic_detector_noise():
    """With about 1 px of landmark jitter, a 20 degree turn is still far outside the photo range."""
    rng = np.random.default_rng(0)
    photo = [abs(estimate_yaw(simulate_landmarks(30, planar=True, noise_px=1.0, rng=rng)) - FRONTAL) for _ in range(400)]
    head = [abs(estimate_yaw(simulate_landmarks(20, noise_px=1.0, rng=rng)) - FRONTAL) for _ in range(400)]
    assert max(photo) < 0.11 < min(head)


def test_liveness_selftest_demo_passes_a_head_and_refuses_a_photo():
    from examples.liveness_demo import selftest
    assert selftest() == 0
