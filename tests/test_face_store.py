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
