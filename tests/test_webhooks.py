"""Tests for outgoing WebhookDispatcher."""
import asyncio
import json
import httpx
import pytest

from core.event_bus import EventBus, Topic
from core.webhooks import WebhookDispatcher
from storage.local_db import AuditLog
from vision.recognition import Recognition


async def test_webhook_disabled_when_no_url():
    bus = EventBus()
    await bus.start()
    disp = WebhookDispatcher(url=None, bus=bus)
    disp.start()
    assert await disp.dispatch("test", {"key": "val"}) is False
    await disp.close()
    await bus.stop()


async def test_webhook_dispatches_with_payload_and_secret(tmp_path):
    bus = EventBus()
    audit = AuditLog(str(tmp_path / "a.db"))
    await bus.start()

    requests = []

    def handler(request):
        body = json.loads(request.read())
        requests.append({"url": str(request.url), "headers": dict(request.headers), "body": body})
        return httpx.Response(200, text='{"status":"ok"}')

    disp = WebhookDispatcher(url="http://webhook.test/events", secret="super-secret-123", bus=bus, audit=audit)
    disp.start()
    disp._client._transport = httpx.MockTransport(handler)

    ok = await disp.dispatch("custom_event", {"hello": "world"})
    assert ok is True
    assert len(requests) == 1

    req = requests[0]
    assert req["url"] == "http://webhook.test/events"
    assert req["headers"].get("x-aether-secret") == "super-secret-123"
    assert req["body"]["event"] == "custom_event"
    assert req["body"]["data"]["hello"] == "world"

    # Audit logged
    events = audit.recent(2)
    assert any("webhook_dispatched" == e[1] for e in events)

    await disp.close()
    await bus.stop()


async def test_webhook_listens_to_face_recognized(tmp_path):
    bus = EventBus()
    await bus.start()

    requests = []

    def handler(request):
        body = json.loads(request.read())
        requests.append({"url": str(request.url), "headers": dict(request.headers), "body": body})
        return httpx.Response(200, text='{"status":"ok"}')

    disp = WebhookDispatcher(url="http://webhook.test/events", bus=bus)
    disp.start()
    disp._client._transport = httpx.MockTransport(handler)

    # Publish face recognized on bus
    bus.publish(Topic.FACE_RECOGNIZED, Recognition("Alice", 0.85, 1, 0.0))
    await asyncio.sleep(0.08)

    assert len(requests) == 1
    req = requests[0]
    assert req["body"]["event"] == "face_recognized"
    assert req["body"]["data"]["identity"] == "Alice"

    await disp.close()
    await bus.stop()
