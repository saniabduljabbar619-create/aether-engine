"""Local SQLite storage for audit events, notes, and timers/reminders. Stays on this machine."""
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
            self._db.execute("CREATE TABLE IF NOT EXISTS notes ("
                             "id INTEGER PRIMARY KEY, ts TEXT NOT NULL, author TEXT NOT NULL, text TEXT NOT NULL)")
            self._db.execute("CREATE TABLE IF NOT EXISTS reminders ("
                             "id INTEGER PRIMARY KEY, created_at TEXT NOT NULL, due_at REAL NOT NULL, "
                             "author TEXT NOT NULL, text TEXT NOT NULL, completed INTEGER DEFAULT 0)")
            self._db.commit()

    # ---- audit events ----------------------------------------------------
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

    # ---- notes -----------------------------------------------------------
    def add_note(self, text: str, author: str = "") -> int:
        with self._lock:
            cur = self._db.execute(
                "INSERT INTO notes (ts, author, text) VALUES (?, ?, ?)",
                (datetime.now().isoformat(timespec="seconds"), author, text.strip()),
            )
            self._db.commit()
            return int(cur.lastrowid or 0)

    def get_notes(self, limit: int = 5) -> List[Tuple[int, str, str, str]]:
        """Newest first: (id, timestamp, author, text)."""
        with self._lock:
            return self._db.execute(
                "SELECT id, ts, author, text FROM notes ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()

    def clear_notes(self) -> int:
        with self._lock:
            cur = self._db.execute("DELETE FROM notes")
            self._db.commit()
            return cur.rowcount

    # ---- reminders -------------------------------------------------------
    def add_reminder(self, text: str, due_at: float, author: str = "") -> int:
        with self._lock:
            cur = self._db.execute(
                "INSERT INTO reminders (created_at, due_at, author, text, completed) VALUES (?, ?, ?, ?, 0)",
                (datetime.now().isoformat(timespec="seconds"), float(due_at), author, text.strip()),
            )
            self._db.commit()
            return int(cur.lastrowid or 0)

    def get_due_reminders(self, now: float) -> List[Tuple[int, str, str, str]]:
        """Active reminders that have reached their due_at timestamp: (id, created_at, author, text)."""
        with self._lock:
            return self._db.execute(
                "SELECT id, created_at, author, text FROM reminders WHERE completed = 0 AND due_at <= ? ORDER BY due_at ASC",
                (now,),
            ).fetchall()

    def get_pending_reminders(self, now: float, limit: int = 5) -> List[Tuple[int, float, str, str]]:
        """Upcoming reminders: (id, due_at, author, text) ordered by nearest deadline."""
        with self._lock:
            return self._db.execute(
                "SELECT id, due_at, author, text FROM reminders WHERE completed = 0 AND due_at > ? ORDER BY due_at ASC LIMIT ?",
                (now, limit),
            ).fetchall()

    def mark_reminder_completed(self, reminder_id: int) -> None:
        with self._lock:
            self._db.execute("UPDATE reminders SET completed = 1 WHERE id = ?", (reminder_id,))
            self._db.commit()

    def close(self) -> None:
        with self._lock:
            self._db.close()
