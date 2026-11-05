"""Aether Edge AI Engine - Commercial REST & WebSocket API Server.

Provides a production-grade, headless API with automatic OpenAPI / Swagger docs,
real-time WebSockets, MJPEG camera streaming, and remote control for vision/voice.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Response, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from core.engine import Engine
from core.event_bus import Event, Topic
from core.orchestrator import display
from vision.tracker import UNKNOWN

log = logging.getLogger("aether.server")


# ---- Pydantic Schemas ----------------------------------------------------
class SpeakRequest(BaseModel):
    text: str = Field(..., description="Text to speak via text-to-speech")


class CommandRequest(BaseModel):
    text: str = Field(..., description="Command to execute (e.g. 'what time is it', 'enroll me as Alice')")


class NoteRequest(BaseModel):
    text: str = Field(..., description="Note text to save")
    author: Optional[str] = Field(default="", description="Optional author or person tag")


class ReminderRequest(BaseModel):
    text: str = Field(..., description="Task or reminder description")
    seconds: float = Field(..., gt=0, description="Duration in seconds until reminder fires")
    author: Optional[str] = Field(default="", description="Optional author tag")


def create_app(engine: Engine) -> FastAPI:
    """Create and configure the FastAPI application bound to an active Engine instance."""
    # Active WebSocket clients connected to the event feed
    ws_clients: List[WebSocket] = []

    async def _on_bus_event(e: Event) -> None:
        """Broadcast selected event bus updates to connected WebSockets."""
        if not ws_clients:
            return
        payload_data = None
        if isinstance(e.payload, (str, int, float, bool, dict, list)):
            payload_data = e.payload
        elif hasattr(e.payload, "to_dict"):
            payload_data = e.payload.to_dict()
        elif hasattr(e.payload, "__dict__"):
            try:
                payload_data = {k: v for k, v in e.payload.__dict__.items() if not k.startswith("_")}
            except Exception:
                payload_data = str(e.payload)
        else:
            payload_data = str(e.payload)

        msg = json.dumps({"topic": e.topic, "timestamp": e.timestamp, "payload": payload_data})
        dead = []
        for ws in ws_clients:
            try:
                await ws.send_text(msg)
            except Exception:
                dead.append(ws)
        for d in dead:
            if d in ws_clients:
                ws_clients.remove(d)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        subs = []
        for topic in (Topic.FACE_RECOGNIZED, Topic.COMMAND_ISSUED, Topic.PEOPLE_CHANGED, Topic.SYSTEM_ERROR):
            subs.append(engine.bus.subscribe(topic, _on_bus_event))
        yield
        for s in subs:
            try:
                engine.bus.unsubscribe(s)
            except Exception:
                pass

    app = FastAPI(
        title="Aether-Vision Engine API",
        description="Commercial Edge AI Multimodal API: Biometrics, Liveness, Local Speech & Conversation",
        version="0.2.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # Enable CORS for easy integration with web dashboards and frontends
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ---- API Routes ------------------------------------------------------
    async def verify_auth(
        x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
        authorization: Optional[str] = Header(None),
        api_key: Optional[str] = Query(None),
    ) -> None:
        """Enforce API Key authentication if configured in settings (dev mode if None)."""
        expected = getattr(engine.cfg, "api_key", None)
        if not expected:
            return  # No authentication configured (open local dev mode)
        token = x_api_key or api_key
        if not token and authorization and authorization.lower().startswith("bearer "):
            token = authorization[7:].strip()
        if not token or token != expected:
            raise HTTPException(
                status_code=401,
                detail="Unauthorized: invalid or missing API key. Provide via X-API-Key header, Authorization: Bearer, or ?api_key= query parameter.",
                headers={"WWW-Authenticate": "Bearer"},
            )

    @app.get("/", response_class=HTMLResponse, tags=["Dashboard"], include_in_schema=False)
    async def get_dashboard() -> str:
        """Serve the modern dark-mode web management dashboard."""
        dash_path = Path(__file__).parent / "dashboard.html"
        if dash_path.exists():
            return dash_path.read_text(encoding="utf-8")
        return "<h1>Aether Edge AI API</h1><p>Visit <a href='/docs'>/docs</a> for Swagger UI.</p>"

    @app.get("/api/health", tags=["System"])
    async def get_health() -> Dict[str, Any]:
        """Health check and high-level operational status."""
        from core.system_info import get_battery_status, get_system_resources
        known_count = len(engine.store.load_all())
        return {
            "status": "online",
            "auth_required": bool(engine.cfg.api_key),
            "camera_active": engine.cap is not None,
            "voice_active": engine.mic is not None,
            "tts_muted": bool(engine.speaker.muted if engine.speaker else False),
            "enrolled_identities": known_count,
            "battery": get_battery_status(),
            "resources": get_system_resources(),
        }

    @app.get("/api/presence", tags=["Vision"], dependencies=[Depends(verify_auth)])
    async def get_presence() -> Dict[str, Any]:
        """Query real-time presence of people in front of the camera."""
        orch = engine.orch
        if orch is None:
            return {"visible": [], "count": 0, "verified": []}
        tracks = orch._visible()
        known = orch._visible_known()
        verified = orch.verified_names()
        details = []
        for t in tracks:
            ident = t.identity
            is_known = bool(ident and ident != UNKNOWN)
            details.append({
                "track_id": t.id,
                "identity": display(ident) if is_known else "Unknown",
                "is_recognized": is_known,
                "is_verified": ident in verified if is_known else False,
                "confidence": round(float(t.score), 3),
            })
        return {
            "count": len(tracks),
            "known": [display(n) for n in known],
            "verified": [display(n) for n in verified],
            "people": details,
        }

    @app.get("/api/identities", tags=["Biometrics"], dependencies=[Depends(verify_auth)])
    async def list_identities() -> Dict[str, Any]:
        """List all enrolled identities and their sample counts."""
        names = engine.store.load_all()
        return {
            "count": len(names),
            "identities": [{"name": n, "display": display(n)} for n in sorted(names)],
        }

    @app.delete("/api/identities/{name}", tags=["Biometrics"], dependencies=[Depends(verify_auth)])
    async def delete_identity(name: str) -> Dict[str, Any]:
        """Delete an enrolled identity and their vector embeddings."""
        n = engine.store.remove(name)
        if not n:
            raise HTTPException(status_code=404, detail=f"Identity '{name}' not found")
        engine.matcher.register_identities(engine.store.load_all())
        engine.audit.record("api_identity_removed", name)
        engine.bus.publish(Topic.PEOPLE_CHANGED, name)
        return {"status": "deleted", "name": name, "samples_removed": n}

    @app.post("/api/speak", tags=["Voice"], dependencies=[Depends(verify_auth)])
    async def speak(req: SpeakRequest) -> Dict[str, Any]:
        """Trigger text-to-speech output on the engine's speaker."""
        if not engine.speaker:
            raise HTTPException(status_code=503, detail="TTS engine not initialized")
        engine.speaker.say(req.text)
        return {"status": "spoken", "text": req.text}

    @app.post("/api/command", tags=["Orchestration"], dependencies=[Depends(verify_auth)])
    async def execute_command(req: CommandRequest) -> Dict[str, Any]:
        """Execute a text-based or simulated spoken command directly through the orchestrator."""
        from audio.stt_engine import Transcript
        orch = engine.orch
        if orch is None:
            raise HTTPException(status_code=503, detail="Orchestrator not ready")
        engine.bus.publish(Topic.SPEECH_TRANSCRIBED, Transcript(req.text, "typed", 0.0, 0.0, 0.0))
        return {"status": "accepted", "command": req.text}

    @app.get("/api/audit", tags=["Audit & Logs"], dependencies=[Depends(verify_auth)])
    async def get_audit(limit: int = Query(default=10, ge=1, le=100)) -> Dict[str, Any]:
        """Retrieve recent audit events from the local database."""
        events = engine.audit.recent(n=limit)
        return {
            "count": len(events),
            "events": [{"timestamp": ts, "kind": kind, "detail": detail} for ts, kind, detail in events],
        }

    @app.get("/api/audit/export.csv", tags=["Audit & Logs"], dependencies=[Depends(verify_auth)])
    async def export_audit_csv(
        limit: int = Query(default=1000, ge=1, le=10000),
        format: str = Query(default="all"),
        kind: Optional[str] = Query(default=None),
    ) -> Response:
        """Export audit events or visitor attendance as a standard downloadable CSV file."""
        if format == "attendance":
            csv_text = engine.audit.export_visitor_attendance_csv(limit=limit)
            filename = "aether_visitor_attendance.csv"
        else:
            csv_text = engine.audit.export_events_csv(limit=limit, kind=kind)
            filename = "aether_audit_log.csv"
        return Response(
            content=csv_text,
            media_type="text/csv",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )

    @app.get("/api/notes", tags=["Notes"], dependencies=[Depends(verify_auth)])
    async def get_notes(limit: int = Query(default=10, ge=1, le=50)) -> Dict[str, Any]:
        """Retrieve saved notes."""
        notes = engine.audit.get_notes(limit=limit)
        return {
            "count": len(notes),
            "notes": [{"id": nid, "timestamp": ts, "author": author, "text": text}
                      for nid, ts, author, text in notes],
        }

    @app.post("/api/notes", tags=["Notes"], dependencies=[Depends(verify_auth)])
    async def create_note(req: NoteRequest) -> Dict[str, Any]:
        """Save a new note."""
        nid = engine.audit.add_note(req.text, author=req.author or "")
        engine.audit.record("api_note_added", f"{req.text} (by {req.author or 'api'})")
        return {"status": "created", "id": nid, "text": req.text, "author": req.author}

    @app.delete("/api/notes", tags=["Notes"], dependencies=[Depends(verify_auth)])
    async def clear_notes() -> Dict[str, Any]:
        """Clear all stored notes."""
        deleted = engine.audit.clear_notes()
        engine.audit.record("api_notes_cleared", f"{deleted} notes deleted")
        return {"status": "cleared", "deleted_count": deleted}

    @app.get("/api/reminders", tags=["Reminders"], dependencies=[Depends(verify_auth)])
    async def get_reminders() -> Dict[str, Any]:
        """Retrieve active pending reminders."""
        now = time.time()
        pending = engine.audit.get_pending_reminders(now, limit=20)
        return {
            "count": len(pending),
            "reminders": [
                {
                    "id": rid,
                    "remaining_seconds": max(0.0, round(due - now, 1)),
                    "author": author,
                    "text": text,
                }
                for rid, due, author, text in pending
            ],
        }

    @app.post("/api/reminders", tags=["Reminders"], dependencies=[Depends(verify_auth)])
    async def create_reminder(req: ReminderRequest) -> Dict[str, Any]:
        """Schedule a new reminder."""
        due_at = time.time() + req.seconds
        rid = engine.audit.add_reminder(req.text, due_at, author=req.author or "")
        engine.audit.record("api_reminder_set", f"{req.text} in {req.seconds:.0f}s")
        return {"status": "scheduled", "id": rid, "due_in_seconds": req.seconds, "text": req.text}

    @app.get("/api/stream", tags=["Vision"], summary="Live MJPEG Camera Stream", dependencies=[Depends(verify_auth)])
    async def mjpeg_stream():
        """Returns a multipart/x-mixed-replace live JPEG stream of the camera feed with bounding boxes."""
        import cv2
        from ui.overlay import draw

        async def generator():
            # Queue to receive latest frames
            q: asyncio.Queue = asyncio.Queue(maxsize=1)

            async def on_frame(e: Event):
                if q.full():
                    try:
                        q.get_nowait()
                    except asyncio.QueueEmpty:
                        pass
                await q.put(e.payload)

            sub = engine.bus.subscribe(Topic.FACES_UPDATED, on_frame, max_queue=1)
            try:
                while True:
                    u = await q.get()
                    if u.frame.image is None:
                        continue
                    status_line = f"{engine.cap.fps if engine.cap else 0:4.1f} FPS | {len(u.tracks)} face(s)"
                    frame_img = draw(u.frame.image.copy(), u.tracks, status_line)
                    ok, buf = cv2.imencode(".jpg", frame_img, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    if not ok:
                        continue
                    yield (b"--frame\r\n"
                           b"Content-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")
            finally:
                engine.bus.unsubscribe(sub)

        return StreamingResponse(generator(), media_type="multipart/x-mixed-replace; boundary=frame")

    @app.websocket("/ws/events")
    async def websocket_events(websocket: WebSocket, api_key: Optional[str] = Query(None)):
        """Live WebSocket event stream for building external dashboards, home automation, or integrations."""
        expected = getattr(engine.cfg, "api_key", None)
        if expected and api_key != expected:
            await websocket.close(code=1008)  # 1008 = Policy violation
            return
        await websocket.accept()
        ws_clients.append(websocket)
        try:
            while True:
                # Keep alive / ping loop
                await websocket.receive_text()
        except WebSocketDisconnect:
            if websocket in ws_clients:
                ws_clients.remove(websocket)
        except Exception:
            if websocket in ws_clients:
                ws_clients.remove(websocket)

    return app
