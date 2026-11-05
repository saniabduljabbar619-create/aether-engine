from storage.local_db import AuditLog


def test_record_and_recent_newest_first():
    db = AuditLog(":memory:")
    for i in range(7):
        db.record("command", f"c{i}")
    rows = db.recent(3)
    assert [r[2] for r in rows] == ["c6", "c5", "c4"] and rows[0][1] == "command"
    assert len(rows[0][0]) == 19  # ISO timestamp
    db.close()


def test_empty_and_persistent(tmp_path):
    p = tmp_path / "sub" / "a.db"  # parent folder is created
    db = AuditLog(p)
    assert db.recent() == []
    db.record("x", "y")
    db.close()
    assert AuditLog(p).recent()[0][1:] == ("x", "y")


def test_notes_storage():
    db = AuditLog(":memory:")
    assert db.get_notes() == []
    id1 = db.add_note("buy groceries", author="Alice")
    id2 = db.add_note("read book", author="Bob")
    notes = db.get_notes()
    assert len(notes) == 2
    assert notes[0][0] == id2 and notes[0][3] == "read book" and notes[0][2] == "Bob"
    assert notes[1][0] == id1 and notes[1][3] == "buy groceries" and notes[1][2] == "Alice"

    cleared = db.clear_notes()
    assert cleared == 2
    assert db.get_notes() == []
    db.close()


def test_reminders_storage():
    db = AuditLog(":memory:")
    now = 1000.0
    r1 = db.add_reminder("take medicine", due_at=now + 50, author="Alice")
    r2 = db.add_reminder("stretch", due_at=now + 10, author="Alice")

    # Pending
    pending = db.get_pending_reminders(now)
    assert len(pending) == 2
    assert pending[0][0] == r2  # sorted by earliest due_at

    # Due at now + 20
    due = db.get_due_reminders(now + 20)
    assert len(due) == 1
    assert due[0][0] == r2

    db.mark_reminder_completed(r2)
    due_after_complete = db.get_due_reminders(now + 20)
    assert len(due_after_complete) == 0

    pending_after = db.get_pending_reminders(now)
    assert len(pending_after) == 1
    assert pending_after[0][0] == r1
    db.close()


def test_csv_exports():
    db = AuditLog(":memory:")
    db.record("presence", "Saw Alice")
    db.record("command", "what time is it")
    db.record("enrolled", "Bob enrolled")

    # Full events export
    csv_all = db.export_events_csv()
    assert "ID,Timestamp,Event Type,Details" in csv_all
    assert "Saw Alice" in csv_all
    assert "what time is it" in csv_all

    # Filtered by kind
    csv_filtered = db.export_events_csv(kind="presence")
    assert "Saw Alice" in csv_filtered
    assert "what time is it" not in csv_filtered

    # Visitor attendance export
    csv_visitors = db.export_visitor_attendance_csv()
    assert "ID,Timestamp,Activity,Description" in csv_visitors
    assert "Saw Alice" in csv_visitors
    assert "Bob enrolled" in csv_visitors
    assert "what time is it" not in csv_visitors
    db.close()

