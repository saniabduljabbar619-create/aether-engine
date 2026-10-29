"""Vectorized identity matcher and temporal stabilizer. Pure NumPy, no OpenCV/UI deps."""
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

BBox = Tuple[int, int, int, int]  # (top, right, bottom, left)


@dataclass
class TrackedIdentity:
    identity_id: str
    confidence: float
    bbox: BBox
    last_seen: float


class VectorIdentityMatcher:
    """Cosine-similarity search over L2-normalized embeddings.

    The 0.363 default is the cosine threshold recommended for SFace embeddings.
    """

    def __init__(self, similarity_threshold: float = 0.363):
        self.similarity_threshold = similarity_threshold
        self.known_embeddings: Optional[np.ndarray] = None  # (N, D)
        self.known_labels: List[str] = []

    def register_identities(self, database: Dict[str, List[np.ndarray]]) -> None:
        labels, vectors = [], []
        for name, embeds in database.items():
            for e in embeds:
                vectors.append(e / np.linalg.norm(e))
                labels.append(name)
        self.known_embeddings = np.asarray(vectors, dtype=np.float32) if vectors else None
        self.known_labels = labels

    def overlapping_identities(self, threshold: Optional[float] = None) -> List[Tuple[str, str, float]]:
        """Pairs of enrolled names whose average faces match each other, i.e. probably the same
        person enrolled twice. Returns [(name_a, name_b, similarity)], most similar first."""
        if self.known_embeddings is None:
            return []
        threshold = self.similarity_threshold if threshold is None else threshold
        names = sorted(set(self.known_labels))
        centroids = []
        for name in names:
            rows = self.known_embeddings[[i for i, lbl in enumerate(self.known_labels) if lbl == name]]
            c = rows.mean(axis=0)
            centroids.append(c / (np.linalg.norm(c) + 1e-12))
        sims = np.stack(centroids) @ np.stack(centroids).T
        pairs = [(names[i], names[j], float(sims[i, j]))
                 for i in range(len(names)) for j in range(i + 1, len(names)) if sims[i, j] >= threshold]
        return sorted(pairs, key=lambda t: -t[2])

    def search_batch(self, queries: np.ndarray) -> List[Tuple[Optional[str], float]]:
        """queries: (M, D). Returns [(label | 'Unknown', score)] per query."""
        if self.known_embeddings is None or len(self.known_embeddings) == 0:
            return [(None, 0.0)] * len(queries)

        norms = np.linalg.norm(queries, axis=1, keepdims=True)
        norms[norms == 0] = 1e-10
        sims = (queries / norms).astype(np.float32) @ self.known_embeddings.T  # (M, N)

        best = sims.argmax(axis=1)
        scores = sims[np.arange(len(sims)), best]
        return [
            (self.known_labels[i] if s >= self.similarity_threshold else "Unknown", float(s))
            for i, s in zip(best, scores)
        ]


class IdentityTemporalFilter:
    """EMA smoothing of bbox and confidence, with TTL-based track expiry."""

    def __init__(self, alpha: float = 0.35, ttl_seconds: float = 1.5):
        self.alpha = alpha
        self.ttl_seconds = ttl_seconds
        self.tracks: Dict[str, TrackedIdentity] = {}

    def update(self, now: float, identity: str, bbox: BBox, confidence: float) -> Tuple[BBox, float]:
        prev = self.tracks.get(identity)
        if prev is not None and now - prev.last_seen <= self.ttl_seconds:
            a = self.alpha
            smoothed = a * np.array(bbox, np.float32) + (1 - a) * np.array(prev.bbox, np.float32)
            bbox = tuple(int(v) for v in smoothed)
            confidence = a * confidence + (1 - a) * prev.confidence
        self.tracks[identity] = TrackedIdentity(identity, confidence, bbox, now)
        return bbox, confidence

    def cleanup_stale_tracks(self, now: float) -> None:
        for k in [k for k, v in self.tracks.items() if now - v.last_seen > self.ttl_seconds]:
            del self.tracks[k]
