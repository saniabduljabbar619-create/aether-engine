"""Tests for Aether commercial FastAPI REST & WebSocket API."""
import asyncio
import pytest
from httpx import ASGITransport, AsyncClient

from config.settings import Settings
from core.engine import Engine
from core.event_bus import Topic
from server.api import create_app


@pytest.fixture
def make_engine(tmp_path):
    async def _make():
        cfg = Settings(db_path=str(tmp_path / "test.db"), faces_dir=str(tmp_path / "faces"))
        eng = Engine(cfg=cfg, camera=False, voice=False, tts=False)
        await eng.start()
        return eng
    return _make


async def test_openapi_and_docs_available(make_engine):
    engine = await make_engine()
    app = create_app(engine)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.get("/")
        assert r.status_code == 200
        assert "Aether Edge AI" in r.text
        r = await client.get("/docs")
        assert r.status_code == 200
        r = await client.get("/openapi.json")
        assert r.status_code == 200
        spec = r.json()
        assert spec["info"]["title"] == "Aether-Vision Engine API"
        assert "/api/presence" in spec["paths"]
        assert "/api/speak" in spec["paths"]
        assert "/api/notes" in spec["paths"]
        assert "/api/reminders" in spec["paths"]
    await engine.stop()


async def test_health_endpoint(make_engine):
    engine = await make_engine()
    app = create_app(engine)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.get("/api/health")
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "online"
        assert "battery" in data
        assert "resources" in data
    await engine.stop()


async def test_presence_endpoint(make_engine):
    from vision.tracker import TrackState

    engine = await make_engine()
    app = create_app(engine)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        # Empty presence
        r = await client.get("/api/presence")
        assert r.status_code == 200
        data = r.json()
        assert data["count"] == 0
        assert data["people"] == []

        # With active tracks in view
        mock_track = TrackState(
            track_id=42,
            bbox=(10, 50, 50, 10),
            identity="Alice_Smith",
            score=0.92,
            newly_identified=False,
        )
        engine.orch._visible = lambda: [mock_track]
        engine.orch._visible_known = lambda: ["Alice_Smith"]

        r = await client.get("/api/presence")
        assert r.status_code == 200
        data = r.json()
        assert data["count"] == 1
        assert len(data["people"]) == 1
        assert data["people"][0]["track_id"] == 42
        assert data["people"][0]["identity"] == "Alice Smith"
        assert data["people"][0]["confidence"] == 0.92
        assert data["people"][0]["is_recognized"] is True

    await engine.stop()


async def test_identities_endpoints(make_engine):
    engine = await make_engine()
    app = create_app(engine)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        # Initially empty
        r = await client.get("/api/identities")
        assert r.status_code == 200
        assert r.json()["count"] == 0

        # Delete non-existent identity -> 404
        r = await client.delete("/api/identities/NonExistent")
        assert r.status_code == 404
    await engine.stop()


async def test_speak_endpoint(make_engine):
    engine = await make_engine()
    app = create_app(engine)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.post("/api/speak", json={"text": "Hello, world!"})
        assert r.status_code == 200
        assert r.json()["status"] == "spoken"
    await engine.stop()


async def test_command_endpoint(make_engine):
    engine = await make_engine()
    app = create_app(engine)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.post("/api/command", json={"text": "what time is it"})
        assert r.status_code == 200
        assert r.json()["status"] == "accepted"
    await engine.stop()


async def test_notes_endpoints(make_engine):
    engine = await make_engine()
    app = create_app(engine)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        # Add note
        r = await client.post("/api/notes", json={"text": "Order new RFID tags", "author": "Alice"})
        assert r.status_code == 200
        assert r.json()["status"] == "created"

        # Get notes
        r = await client.get("/api/notes")
        assert r.status_code == 200
        data = r.json()
        assert data["count"] == 1
        assert data["notes"][0]["text"] == "Order new RFID tags"
        assert data["notes"][0]["author"] == "Alice"

        # Clear notes
        r = await client.delete("/api/notes")
        assert r.status_code == 200
        assert r.json()["deleted_count"] == 1

        r = await client.get("/api/notes")
        assert r.json()["count"] == 0
    await engine.stop()


async def test_reminders_endpoints(make_engine):
    engine = await make_engine()
    app = create_app(engine)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.post("/api/reminders", json={"text": "Review security audit", "seconds": 300, "author": "Admin"})
        assert r.status_code == 200
        assert r.json()["status"] == "scheduled"

        r = await client.get("/api/reminders")
        assert r.status_code == 200
        data = r.json()
        assert data["count"] == 1
        assert data["reminders"][0]["text"] == "Review security audit"
        assert data["reminders"][0]["remaining_seconds"] > 0
    await engine.stop()


async def test_audit_endpoint(make_engine):
    engine = await make_engine()
    engine.audit.record("test_event", "system test execution")
    app = create_app(engine)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.get("/api/audit?limit=5")
        assert r.status_code == 200
        data = r.json()
        assert data["count"] >= 1
        assert any(e["kind"] == "test_event" for e in data["events"])
    await engine.stop()


async def test_audit_csv_export(make_engine):
    engine = await make_engine()
    engine.audit.record("presence", "Saw Alice at main lobby")
    engine.audit.record("command", "unlock gate")
    app = create_app(engine)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        # Full export
        r = await client.get("/api/audit/export.csv")
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/csv")
        assert "ID,Timestamp,Event Type,Details" in r.text
        assert "Saw Alice at main lobby" in r.text

        # Attendance format
        r = await client.get("/api/audit/export.csv?format=attendance")
        assert r.status_code == 200
        assert "ID,Timestamp,Activity,Description" in r.text
        assert "Saw Alice at main lobby" in r.text
        assert "unlock gate" not in r.text
    await engine.stop()


async def test_api_key_authentication(tmp_path):
    cfg = Settings(
        db_path=str(tmp_path / "test.db"),
        faces_dir=str(tmp_path / "faces"),
        api_key="secret-token-123",
    )
    engine = Engine(cfg=cfg, camera=False, voice=False, tts=False)
    await engine.start()
    app = create_app(engine)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        # Health check is public and reports auth_required
        r = await client.get("/api/health")
        assert r.status_code == 200
        assert r.json()["auth_required"] is True

        # Unauthenticated request to protected endpoint -> 401
        r = await client.get("/api/presence")
        assert r.status_code == 401

        # Invalid key -> 401
        r = await client.get("/api/presence", headers={"X-API-Key": "wrong-token"})
        assert r.status_code == 401

        # Valid X-API-Key header -> 200
        r = await client.get("/api/presence", headers={"X-API-Key": "secret-token-123"})
        assert r.status_code == 200

        # Valid Bearer token -> 200
        r = await client.post(
            "/api/speak",
            json={"text": "Authorized speak"},
            headers={"Authorization": "Bearer secret-token-123"},
        )
        assert r.status_code == 200

        # Valid query parameter for CSV export -> 200
        r = await client.get("/api/audit/export.csv?api_key=secret-token-123")
        assert r.status_code == 200

    await engine.stop()


async def test_chat_endpoint(make_engine):
    engine = await make_engine()
    app = create_app(engine)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        # Command query
        r = await client.post("/api/chat", json={"message": "what time is it"})
        assert r.status_code == 200
        data = r.json()
        assert "reply" in data
        assert "time" in data["intent"] or len(data["reply"]) > 0

        # Conversational greeting
        r = await client.post("/api/chat", json={"message": "Hello Aether"})
        assert r.status_code == 200
        data = r.json()
        assert "reply" in data
        assert "listening" in data["reply"].lower() or "hello" in data["reply"].lower()

    await engine.stop()
