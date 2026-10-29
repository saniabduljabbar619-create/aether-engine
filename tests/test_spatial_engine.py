import time

import numpy as np

from vision.spatial_engine import IdentityTemporalFilter, VectorIdentityMatcher


def _matcher():
    rng = np.random.default_rng(0)
    db = {"alice": [rng.normal(size=128)], "bob": [rng.normal(size=128)]}
    m = VectorIdentityMatcher()
    m.register_identities(db)
    return m, db


def test_matches_known_and_rejects_unknown():
    m, db = _matcher()
    q = np.stack([db["alice"][0] * 3, np.random.default_rng(9).normal(size=128)])
    (l1, s1), (l2, _) = m.search_batch(q)
    assert l1 == "alice" and s1 > 0.99
    assert l2 == "Unknown"


def test_empty_database():
    assert VectorIdentityMatcher().search_batch(np.ones((2, 128))) == [(None, 0.0)] * 2


def test_search_is_fast():
    rng = np.random.default_rng(1)
    m = VectorIdentityMatcher()
    m.register_identities({f"p{i}": [rng.normal(size=128)] for i in range(1000)})
    q = rng.normal(size=(4, 128)).astype(np.float32)
    t = time.perf_counter()
    m.search_batch(q)
    assert time.perf_counter() - t < 0.05


def test_temporal_filter_smooths_and_expires():
    f = IdentityTemporalFilter(alpha=0.5, ttl_seconds=1.0)
    f.update(0.0, "a", (0, 100, 100, 0), 1.0)
    bbox, conf = f.update(0.5, "a", (0, 200, 100, 100), 0.0)
    assert bbox == (0, 150, 100, 50) and conf == 0.5
    f.cleanup_stale_tracks(5.0)
    assert not f.tracks
