import numpy as np
import pytest

from vision.detector import FaceDetection
from vision.enrollment import (CENTER, DIM_WARNING, PROMPTS, USER_LEFT, USER_RIGHT, EnrollmentSession, cohesion,
                               estimate_yaw, pose_of)
from vision.spatial_engine import VectorIdentityMatcher

rng = np.random.default_rng(21)
BASE = rng.normal(size=128).astype(np.float32)
OTHER = rng.normal(size=128).astype(np.float32)


def landmarks(yaw=0.0):
    """right eye, left eye, nose, mouth right, mouth left. Eye distance is 40 px."""
    return np.array([[80, 100], [120, 100], [100 + yaw * 40, 125], [88, 150], [112, 150]], np.float32)


def variant(base=BASE, noise=0.7):
    return (base + rng.normal(size=128) * noise).astype(np.float32)  # same person, slightly different view


def face(yaw=0.0, emb=None, score=0.95, height=110):
    return FaceDetection((10, 200, 10 + height, 60), score, landmarks(yaw), variant() if emb is None else emb)


def test_yaw_sign_and_pose_bins():
    assert estimate_yaw(landmarks(0.0)) == pytest.approx(0.0)
    assert estimate_yaw(landmarks(-0.3)) == pytest.approx(-0.3)  # nose toward image-left
    assert pose_of(-0.3) == USER_RIGHT and pose_of(0.3) == USER_LEFT and pose_of(0.05) == CENTER
    assert estimate_yaw(np.zeros((5, 2), np.float32)) is None  # degenerate eyes


def test_cohesion_one_person_vs_two():
    same = [variant(BASE) for _ in range(8)]
    mixed = [variant(BASE) if i % 2 else variant(OTHER) for i in range(8)]
    assert cohesion(same) > 0.5 > 0.45 > cohesion(mixed)
    assert cohesion([BASE]) == 1.0


@pytest.mark.parametrize("faces,fragment", [
    ([], "can't see a face"),
    ([face(), face()], "Only the person enrolling"),
    ([face(score=0.5)], "isn't clear"),
    ([face(height=40)], "closer"),
    ([FaceDetection((10, 100, 120, 10), 0.95, np.zeros((5, 2), np.float32), BASE)], "make out"),
])
def test_unusable_frames_are_refused_with_a_reason(faces, fragment):
    fb = EnrollmentSession("a").offer(faces, 120.0, 0.0)
    assert not fb.accepted and fragment in fb.text


def test_frames_closer_than_the_gap_and_near_duplicates_are_skipped():
    s = EnrollmentSession("a", min_gap=0.4)
    first = face()
    assert s.offer([first], 120, 0.0).accepted
    assert not s.offer([face()], 120, 0.2).accepted  # too soon
    dup = s.offer([FaceDetection(first.bbox, 0.95, first.landmarks, first.embedding * 2)], 120, 1.0)
    assert not dup.accepted and "different views" in dup.text  # identical face, nothing new
    assert s.offer([face()], 120, 1.5).accepted and len(s.samples) == 2


def test_prompts_follow_what_is_missing_and_it_completes():
    s, t = EnrollmentSession("a", needed=10), 0.0
    assert s.offer([face(0)], 120, t).text == PROMPTS[CENTER]
    for _ in range(3):  # 4 straight-on samples in total
        t += 0.5
        fb = s.offer([face(0)], 120, t)
    assert fb.accepted and fb.text == PROMPTS[USER_RIGHT]
    t += 0.5
    assert not s.offer([face(0)], 120, t).accepted  # enough straight-on, so the next one must be a turn
    for _ in range(2):
        t += 0.5
        fb = s.offer([face(-0.3)], 120, t)
    assert fb.text == PROMPTS[USER_LEFT]
    done = None
    for _ in range(4):
        t += 0.5
        done = s.offer([face(0.3)], 120, t)
    assert done.accepted and done.done and len(s.samples) == 10
    assert s.counts == {CENTER: 4, USER_RIGHT: 2, USER_LEFT: 4}


def test_head_turns_are_waived_after_a_while():
    """Someone who cannot turn their head is not stuck forever."""
    s, t, last = EnrollmentSession("a", needed=6, relax_after=10.0), 0.0, None
    for _ in range(20):
        last = s.offer([face(0)], 120, t)
        if last.done:
            break
        t += 2.0
    assert last.done and len(s.samples) == 6 and t >= 10.0 and s.counts[USER_RIGHT] == 0


def test_darkness_warns_but_never_rejects():
    """An absolute brightness cut-off would penalize darker skin tones and dim rooms."""
    for brightness in (3.0, 30.0, 59.0):
        fb = EnrollmentSession("a").offer([face()], brightness, 0.0)
        assert fb.accepted and DIM_WARNING in fb.text
    bright = EnrollmentSession("a").offer([face()], 140.0, 0.0)
    assert bright.accepted and DIM_WARNING not in bright.text


def test_overlapping_identities_finds_the_same_person_enrolled_twice():
    m = VectorIdentityMatcher()
    m.register_identities({"Asak": [variant(BASE) for _ in range(5)],
                           "YourName": [variant(BASE) for _ in range(5)],  # same person, other name
                           "Bob": [variant(OTHER) for _ in range(5)]})
    pairs = m.overlapping_identities()
    assert [(a, b) for a, b, _ in pairs] == [("Asak", "YourName")] and pairs[0][2] > 0.8
    assert m.overlapping_identities(threshold=0.999) == []


def test_overlapping_identities_edge_cases():
    m = VectorIdentityMatcher()
    assert m.overlapping_identities() == []
    m.register_identities({"Only": [BASE]})
    assert m.overlapping_identities() == []
