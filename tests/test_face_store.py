import numpy as np
import pytest

from storage.face_store import FaceStore


def test_add_load_remove(tmp_path):
    s = FaceStore(tmp_path)
    s.add("alice", np.ones(128))
    s.add("alice", np.zeros(128))
    s.add("bob", np.ones(128))
    db = s.load_all()
    assert {k: len(v) for k, v in db.items()} == {"alice": 2, "bob": 1}
    assert db["alice"][0].dtype == np.float32 and db["alice"][0].shape == (128,)
    assert s.remove("alice") == 2 and "alice" not in s.load_all()


def test_empty_store(tmp_path):
    assert FaceStore(tmp_path / "missing").load_all() == {}


@pytest.mark.parametrize("bad", ["../evil", "a/b", "", " x", "a" * 65])
def test_rejects_unsafe_names(tmp_path, bad):
    with pytest.raises(ValueError):
        FaceStore(tmp_path).add(bad, np.ones(128))


def test_samples_saved_in_a_burst_are_never_overwritten(tmp_path, monkeypatch):
    """Regression: files were named after the clock. Windows (Python <= 3.12) ticks every ~15 ms, so the 10
    samples saved in a tight loop at the end of an enrollment shared names and overwrote each other."""
    import storage.face_store as fs

    monkeypatch.setattr(fs.time, "time_ns", lambda: 1_700_000_000_000_000_000)  # a clock that never advances
    store = FaceStore(tmp_path)
    samples = [np.full(128, float(i), np.float32) for i in range(20)]
    for s in samples:
        store.add("alice", s)
    saved = store.load_all()["alice"]
    assert len(saved) == 20
    assert sorted(float(v[0]) for v in saved) == [float(i) for i in range(20)]  # all distinct, none lost


def test_samples_keep_their_save_order_and_old_files_still_load(tmp_path):
    store = FaceStore(tmp_path)
    (tmp_path / "alice").mkdir()
    np.save(tmp_path / "alice" / "1700000000000000000.npy", np.full(128, -1.0, np.float32))  # an old-style name
    for i in range(5):
        store.add("alice", np.full(128, float(i), np.float32))
    assert [float(v[0]) for v in store.load_all()["alice"]] == [-1.0, 0.0, 1.0, 2.0, 3.0, 4.0]
