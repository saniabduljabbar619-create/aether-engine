"""On-disk store of reference face embeddings: <root>/<name>/<timestamp>.npy.

Embeddings are biometric data. They stay on this machine and *.npy is git-ignored.
"""
from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Dict, List

import numpy as np

_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")


class FaceStore:
    def __init__(self, root: "str | Path" = "storage/reference_faces") -> None:
        self.root = Path(root)

    @staticmethod
    def valid_name(name: str) -> bool:
        return bool(_NAME.match(name))

    def add(self, name: str, embedding: np.ndarray) -> Path:
        if not _NAME.match(name):
            raise ValueError("name must be letters, digits, '-' or '_' (max 64 chars)")
        folder = self.root / name
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{time.time_ns()}.npy"
        np.save(path, np.asarray(embedding, dtype=np.float32).reshape(-1))
        return path

    def load_all(self) -> Dict[str, List[np.ndarray]]:
        db: Dict[str, List[np.ndarray]] = {}
        if self.root.is_dir():
            for folder in sorted(p for p in self.root.iterdir() if p.is_dir()):
                vecs = [np.load(f) for f in sorted(folder.glob("*.npy"))]
                if vecs:
                    db[folder.name] = vecs
        return db

    def remove(self, name: str) -> int:
        """Delete all samples of a person. Returns how many files were removed."""
        folder = self.root / name
        if not _NAME.match(name) or not folder.is_dir():
            return 0
        files = list(folder.glob("*.npy"))
        for f in files:
            f.unlink()
        if not any(folder.iterdir()):
            folder.rmdir()
        return len(files)
