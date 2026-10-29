"""Local SQLite audit log of what the system saw and did. Stays on this machine."""
from __future__ import annotations

import sqlite3
import threading
from datetime import datetime
from pathlib import Path
from typing import List, Tuple


class AuditLog:
    def __init__(self, path: "str | Path" = "storage/aether.db") -> None:
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(path), check_same_thread=False)
        self._lock = threading.Lock()
        with self._lock:
            self._db.execute("CREATE TABLE IF NOT EXISTS events ("
                             "id INTEGER PRIMARY KEY, ts TEXT NOT NULL, kind TEXT NOT NULL, detail TEXT NOT NULL)")
            self._db.commit()

    def record(self, kind: str, detail: str) -> None:
        with self._lock:
            self._db.execute("INSERT INTO events (ts, kind, detail) VALUES (?, ?, ?)",
                             (datetime.now().isoformat(timespec="seconds"), kind, detail))
            self._db.commit()

    def recent(self, n: int = 5) -> List[Tuple[str, str, str]]:
        """Newest first: (timestamp, kind, detail)."""
        with self._lock:
            return self._db.execute(
                "SELECT ts, kind, detail FROM events ORDER BY id DESC LIMIT ?", (n,)).fetchall()

    def close(self) -> None:
        with self._lock:
            self._db.close()
