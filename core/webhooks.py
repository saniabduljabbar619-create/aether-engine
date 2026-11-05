"""Outbound Webhook Dispatcher.

Asynchronously notifies external systems (Slack, Zapier, Make, Home Assistant,
custom backends) when events occur (face recognized, command issued, etc.).
Non-blocking and failure-isolated.
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Dict, Optional

import httpx

from core.event_bus import Event, EventBus, Topic
from core.orchestrator import display
from storage.local_db import AuditLog
from vision.recognition import Recognition

log = logging.getLogger("aether.webhooks")


class WebhookDispatcher:
    def __init__(
        self,
        url: Optional[str] = None,
        secret: Optional[str] = None,
        bus: Optional[EventBus] = None,
        audit: Optional[AuditLog] = None,
        timeout: float = 5.0,
    ) -> None:
        self.url = url
        self.secret = secret
        self.bus = bus
        self.audit = audit
        self.timeout = timeout
        self._client: Optional[httpx.AsyncClient] = None
        self._subs = []

    def start(self) -> None:
        """Start listening to event bus topics if a webhook URL is configured."""
        if not self.url or not self.bus:
            return
        self._client = httpx.AsyncClient(timeout=self.timeout)
        self._subs.append(self.bus.subscribe(Topic.FACE_RECOGNIZED, self._on_recognized))
        self._subs.append(self.bus.subscribe(Topic.COMMAND_ISSUED, self._on_command))
        log.info("Webhook dispatcher active: %s", self.url)

    async def close(self) -> None:
        """Close HTTP client and unsubscribe from the event bus."""
        if self.bus:
            for sub in self._subs:
                try:
                    self.bus.unsubscribe(sub)
                except Exception:
                    pass
            self._subs.clear()
        if self._client:
            await self._client.aclose()
            self._client = None

    async def dispatch(self, event_type: str, data: Dict[str, Any]) -> bool:
        """Send a JSON webhook payload asynchronously."""
        if not self.url:
            return False
        client = self._client or httpx.AsyncClient(timeout=self.timeout)
        should_close = self._client is None

        payload = {
            "source": "aether-engine",
            "event": event_type,
            "timestamp": time.time(),
            "data": data,
        }
        headers = {"Content-Type": "application/json", "User-Agent": "Aether-Engine-Webhook/0.2.0"}
        if self.secret:
            headers["X-Aether-Secret"] = self.secret

        try:
            resp = await client.post(self.url, json=payload, headers=headers)
            ok = resp.status_code < 400
            if self.audit:
                status_str = f"HTTP {resp.status_code}"
                self.audit.record("webhook_dispatched", f"{event_type} -> {status_str}")
            return ok
        except Exception as exc:
            log.warning("Webhook dispatch failed (%s): %s", self.url, exc)
            if self.audit:
                self.audit.record("webhook_error", f"{event_type}: {exc}")
            return False
        finally:
            if should_close:
                await client.aclose()

    async def _on_recognized(self, e: Event) -> None:
        r: Recognition = e.payload
        asyncio.create_task(
            self.dispatch(
                "face_recognized",
                {"identity": r.name, "display_name": display(r.name), "confidence": float(r.score)},
            )
        )

    async def _on_command(self, e: Event) -> None:
        cmd = e.payload
        asyncio.create_task(
            self.dispatch(
                "command_issued",
                {"intent": getattr(cmd, "intent", "unknown"), "text": getattr(cmd, "text", "")},
            )
        )
