import numpy as np

from vision.tracker import UNKNOWN, FaceTracker, iou_matrix


def box(left, top, size=100):
    return (top, left + size, top + size, left)  # (top, right, bottom, left)


def feed(tr, t, *dets):
    return tr.update(t, list(dets))


def test_iou_matrix():
    a = np.array([[0, 0, 10, 10]], np.float32)
    b = np.array([[0, 0, 10, 10], [5, 0, 15, 10], [20, 20, 30, 30]], np.float32)
    iou = iou_matrix(a, b)[0]
    assert iou[0] == 1.0 and abs(iou[1] - 1 / 3) < 1e-6 and iou[2] == 0.0


def test_two_faces_keep_separate_tracks():
    tr = FaceTracker()
    ids = set()
    for i in range(5):
        s = feed(tr, i * 0.1, (box(10, 10), UNKNOWN, 0.1), (box(400, 10), UNKNOWN, 0.1))
        ids |= {x.track_id for x in s}
    assert len(ids) == 2


def test_identity_needs_votes_and_announces_once():
    tr = FaceTracker(min_votes=3)
    out = [feed(tr, i * 0.1, (box(10, 10), "alice", 0.8))[0] for i in range(6)]
    assert [o.identity for o in out[:2]] == [None, None]  # still verifying
    assert out[2].identity == "alice" and out[2].newly_identified
    assert not any(o.newly_identified for o in out[3:])


def test_single_bad_frame_does_not_flip_identity():
    tr = FaceTracker()
    for i in range(5):
        feed(tr, i * 0.1, (box(10, 10), "alice", 0.8))
    s = feed(tr, 0.5, (box(10, 10), "bob", 0.5))[0]
    assert s.identity == "alice" and not s.newly_identified


def test_unknown_is_never_announced():
    tr = FaceTracker()
    out = [feed(tr, i * 0.1, (box(10, 10), UNKNOWN, 0.1))[0] for i in range(6)]
    assert out[-1].identity == UNKNOWN and not any(o.newly_identified for o in out)


def test_box_is_smoothed():
    tr = FaceTracker(alpha=0.5)
    feed(tr, 0.0, (box(0, 0), "a", 0.9))
    s = feed(tr, 0.1, (box(40, 0), "a", 0.9))[0]
    assert s.bbox[3] == 20  # halfway between left=0 and left=40


def test_track_expires_and_new_id_is_issued():
    tr = FaceTracker(ttl_seconds=1.0)
    first = feed(tr, 0.0, (box(10, 10), "a", 0.9))[0].track_id
    feed(tr, 5.0)  # nothing visible for 5 s
    again = feed(tr, 5.1, (box(10, 10), "a", 0.9))[0].track_id
    assert again != first
