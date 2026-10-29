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
