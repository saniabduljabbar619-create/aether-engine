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
    engine = await make_engine()
    app = create_app(engine)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.get("/api/presence")
        assert r.status_code == 200
        data = r.json()
        assert data["count"] == 0
        assert data["people"] == []
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
