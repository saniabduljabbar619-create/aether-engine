"""The orchestrator: reacts to bus events and turns them into behaviour.

    FACE_RECOGNIZED      -> greet
    SPEECH_TRANSCRIBED   -> wake word + intent -> handler -> spoken/printed reply
    FACES_UPDATED        -> remembers who is currently in view
    FACE_DETECTED        -> collects samples while a voice-triggered enrollment is running
    COMMAND_ISSUED / FACE_RECOGNIZED / SYSTEM_ERROR -> audit log

Safety rules (this system manages biometric data):
  * Audio captured while we were speaking is ignored, so the assistant never obeys itself.
  * Enrolling a face needs an already-recognized person in view, unless nobody is enrolled yet.
  * Adding samples to an existing name needs that person in view, so nobody can poison it.
  * "Forget me" only works on a recognized person who is in view, and only removes their own data.
"""
from __future__ import annotations

import asyncio
import logging
from collections import Counter
import time
from datetime import datetime
from typing import Awaitable, Callable, Dict, List, Optional, Tuple

import numpy as np

from audio.stt_engine import TYPED, Transcript
from audio.tts import NullSpeaker, Speaker
from core.commands import SAFE_WITHOUT_WAKE, Addressing, Command, CommandRouter
from core.event_bus import Event, EventBus, Topic
from storage.face_store import FaceStore
from storage.local_db import AuditLog
from vision.detector import FaceEvent
from vision.enrollment import CENTER, PROMPTS, EnrollmentSession, cohesion
from vision.recognition import FacesUpdate, Recognition
from vision.spatial_engine import VectorIdentityMatcher
from vision.tracker import UNKNOWN

log = logging.getLogger("aether.orchestrator")

ENROLL_SAMPLES = 10
ENROLL_MIN_GAP = 0.4  # seconds between accepted samples
ENROLL_TIMEOUT = 40.0
ENROLL_COHESION_MIN = 0.45  # samples of one person agree well above this; two people mixed fall below
DUPLICATE_CHECK_AT = 3  # after this many samples, check the face is not someone already enrolled
HINT_EVERY = 3.0  # seconds between spoken enrollment hints
TALK_SECONDS = 8.0  # how long push-to-talk listens for a command
VIEW_FRESH = 1.5  # a FACES_UPDATED older than this no longer describes "who is here"

Handler = Callable[[Command], Awaitable[str]]


def display(name: str) -> str:
    return name.replace("_", " ")


class Orchestrator:
    def __init__(self, bus: EventBus, matcher: VectorIdentityMatcher, store: FaceStore, audit: AuditLog,
                 speaker: Optional[Speaker] = None, router: Optional[CommandRouter] = None,
                 status_fn: Optional[Callable[[], str]] = None, on_reply: Optional[Callable[[str], None]] = None,
                 follow_up_seconds: float = 10.0):
        self.bus, self.matcher, self.store, self.audit = bus, matcher, store, audit
        self.speaker = speaker or NullSpeaker()
        self.router = router or CommandRouter()
        self.status_fn = status_fn
        self.on_reply = on_reply or (lambda text: print(f"Aether: {text}"))
        self.shutdown = asyncio.Event()
        self.listening = True  # the UI can pause voice commands
        self.follow_up_seconds = follow_up_seconds
        self._follow_until = 0.0  # after we speak, safe commands need no wake word until then
        self._talk_until = 0.0  # push-to-talk: the next thing said is for us
        self._safe_intents = set(SAFE_WITHOUT_WAKE)
        self._view: Optional[FacesUpdate] = None
        self._view_at = 0.0
        self._enroll: Optional[EnrollmentSession] = None
        self._enroll_hint = ""
        self._hint_said = ("", float("-inf"))
        self._dup_checked = False
        self._enroll_task: Optional[asyncio.Task] = None
        self._handlers: Dict[str, Handler] = {
            "attention": self._h_attention, "unknown": self._h_unknown, "help": self._h_help,
            "thanks": self._h_thanks, "time": self._h_time, "date": self._h_date,
            "status": self._h_status, "recent": self._h_recent, "who_here": self._h_who_here,
            "who_am_i": self._h_who_am_i, "who_known": self._h_who_known, "enroll": self._h_enroll,
            "forget_me": self._h_forget, "shutdown": self._h_shutdown,
        }

    def close(self) -> None:
        """Stop background work (an enrollment in progress) before the bus shuts down."""
        self._end_enrollment()

    def register_handler(self, intent: str, patterns: List[str], handler: Handler,
                         safe_without_wake: bool = False) -> None:
        """Add a custom voice command, e.g. unlocking a door via your own hardware code.

        It needs the wake word unless ``safe_without_wake`` is set, which only suits harmless
        read-only commands: otherwise any chatter after a reply could trigger it."""
        self.router.add_intent(intent, patterns)
        self._handlers[intent] = handler
        if safe_without_wake:
            self._safe_intents.add(intent)

    def start(self) -> None:
        sub = self.bus.subscribe
        sub(Topic.FACES_UPDATED, self._on_view, max_queue=1)
        sub(Topic.FACE_DETECTED, self._on_detected, max_queue=1)
        sub(Topic.FACE_RECOGNIZED, self._on_recognized)
        sub(Topic.SPEECH_TRANSCRIBED, self._on_transcript, max_queue=8)
        for topic in (Topic.FACE_RECOGNIZED, Topic.COMMAND_ISSUED, Topic.SYSTEM_ERROR):
            sub(topic, self._on_audit)

    # ---- bus handlers ----------------------------------------------------
    async def _on_view(self, e: Event) -> None:
        self._view, self._view_at = e.payload, time.monotonic()

    async def _on_recognized(self, e: Event) -> None:
        r: Recognition = e.payload
        self._reply(f"Hello {display(r.name)}.")

    async def _on_audit(self, e: Event) -> None:
        p = e.payload
        if e.topic == Topic.FACE_RECOGNIZED.value:
            self.audit.record("face_recognized", f"{p.name} ({p.score:.2f})")
        elif e.topic == Topic.COMMAND_ISSUED.value:
            self.audit.record("command", f"{p.intent}: {p.text}")
        else:
            self.audit.record("error", str(p))

    # ---- who is being addressed ------------------------------------------
    def listen_now(self, seconds: float = TALK_SECONDS) -> None:
        """Push-to-talk: for a few seconds the next thing said is a command, wake word or not."""
        self._talk_until = time.monotonic() + seconds

    def attention(self) -> Optional[Tuple[str, float]]:
        """('talk' | 'follow_up', seconds left) while we are listening without a wake word, else None."""
        now = time.monotonic()
        if now < self._talk_until:
            return "talk", self._talk_until - now
        if now < self._follow_until:
            return "follow_up", self._follow_until - now
        return None

    def _addressing(self, language: str) -> Addressing:
        now = time.monotonic()
        if language == TYPED or now < self._talk_until:
            return Addressing.DIRECT
        return Addressing.FOLLOW_UP if now < self._follow_until else Addressing.WAKE

    def _decide(self, text: str, language: str = "en") -> Optional[Command]:
        """The command this speech should run, or None if it is not for us."""
        mode = self._addressing(language)
        cmd = self.router.parse(text, mode)
        if cmd is not None and mode is Addressing.FOLLOW_UP and not cmd.woke and cmd.intent not in self._safe_intents:
            log.info("ignored %r: %s needs the wake word", text, cmd.intent)
            return None
        return cmd

    def would_handle(self, text: str, language: str = "en") -> bool:
        """Whether this speech would currently be treated as a command (for showing it in the UI)."""
        return self._decide(text, language) is not None

    async def _on_transcript(self, e: Event) -> None:
        t: Transcript = e.payload
        if t.language != TYPED and (not self.listening or self.speaker.was_speaking_at(t.started_at)):
            return
        cmd = self._decide(t.text, t.language)
        if cmd is None:
            return  # not addressed to us
        self._talk_until = 0.0  # push-to-talk is used up by one command
        log.info("heard %r -> %s", t.text, cmd.intent)
        try:
            reply = await self._handlers[cmd.intent](cmd)
        except Exception:
            log.exception("handler %s failed", cmd.intent)
            reply = "Sorry, something went wrong with that."
        self.bus.publish(Topic.COMMAND_ISSUED, cmd)
        self._reply(reply)

    def _reply(self, text: str) -> None:
        self.on_reply(text)
        self.speaker.say(text)
        self._follow_until = time.monotonic() + self.follow_up_seconds  # a natural conversation continues

    # ---- who is in view --------------------------------------------------
    def _visible(self):
        if self._view is None or time.monotonic() - self._view_at > VIEW_FRESH:
            return []
        return self._view.tracks

    def _visible_known(self) -> List[str]:
        return sorted({t.identity for t in self._visible() if t.identity and t.identity != UNKNOWN})

    # ---- intent handlers -------------------------------------------------
    async def _h_attention(self, c: Command) -> str:
        return "Yes?"

    async def _h_unknown(self, c: Command) -> str:
        return f"Sorry, I didn't understand. Say '{self.router.wake_name}, help' to hear what I can do."

    async def _h_help(self, c: Command) -> str:
        return ("You can ask: who is at the camera, who am I, what time is it, what's the date, status, "
                "recent activity, who do you know. Or: enroll me as, then your name; forget me; goodbye.")

    async def _h_thanks(self, c: Command) -> str:
        return "You're welcome."

    async def _h_time(self, c: Command) -> str:
        n = datetime.now()
        return f"It's {n.hour % 12 or 12}:{n.minute:02d} {'PM' if n.hour >= 12 else 'AM'}."

    async def _h_date(self, c: Command) -> str:
        n = datetime.now()
        return f"Today is {n.strftime('%A')}, {n.strftime('%B')} {n.day}, {n.year}."

    async def _h_status(self, c: Command) -> str:
        people = len(self.store.load_all())
        base = f"I know {people} {'person' if people == 1 else 'people'}."
        return f"{self.status_fn()} {base}" if self.status_fn else base

    async def _h_recent(self, c: Command) -> str:
        rows = self.audit.recent(5)
        if not rows:
            return "Nothing has happened yet."
        return "Recently: " + "; ".join(f"{ts[11:16]} {kind.replace('_', ' ')} {detail}" for ts, kind, detail in rows)

    async def _h_who_here(self, c: Command) -> str:
        tracks = self._visible()
        if not tracks:
            return "I don't see anyone."
        known = self._visible_known()
        unknown = sum(1 for t in tracks if t.identity == UNKNOWN)
        pending = sum(1 for t in tracks if t.identity is None)
        others = []
        if unknown:
            others.append(f"{unknown} {'person' if unknown == 1 else 'people'} I don't recognize")
        if pending:
            others.append(f"{pending} I'm still checking")
        if known:
            names = " and ".join(display(n) for n in known)
            return f"I can see {names}" + (f", plus {' and '.join(others)}." if others else ".")
        return "I see " + " and ".join(others) + "."

    async def _h_who_am_i(self, c: Command) -> str:
        tracks = self._visible()
        if len(tracks) != 1:
            return "I can't tell, I need to see exactly one person." if tracks else "I don't see anyone."
        t = tracks[0]
        if t.identity is None:
            return "One moment, I'm still checking."
        if t.identity == UNKNOWN:
            return f"I don't know you yet. Say '{self.router.wake_name}, enroll me as' and then your name."
        return f"You're {display(t.identity)}."

    async def _h_who_known(self, c: Command) -> str:
        names = [display(n) for n in self.store.load_all()]
        return ("I know " + ", ".join(names) + ".") if names else "I don't know anyone yet."

    async def _h_enroll(self, c: Command) -> str:
        name = c.args.get("name")
        if not name:
            return f"Say: {self.router.wake_name}, enroll me as, followed by your name."
        people, known = self.store.load_all(), self._visible_known()
        if self._enroll is None:  # (the "already enrolling" message comes from begin_enrollment)
            if people and not known:
                return "For safety, only someone I already recognize can enroll a face. Please let me recognize you first."
            if name in people and name not in known:
                return f"I already know a {display(name)}. Only they can add more samples of their face."
        try:
            return self.begin_enrollment(name)
        except ValueError as exc:
            return str(exc)

    def begin_enrollment(self, name: str) -> str:
        """Start capturing samples for ``name``. No permission check: voice commands apply the
        rules first, while the UI calls this directly for someone at the keyboard."""
        if not self.store.valid_name(name):
            raise ValueError("Names can only use letters, digits, '-' and '_'.")
        if self._enroll is not None:
            raise ValueError("I'm already enrolling someone. One moment.")
        self._enroll = EnrollmentSession(name, needed=ENROLL_SAMPLES, min_gap=ENROLL_MIN_GAP)
        self._enroll_hint, self._dup_checked = PROMPTS[CENTER], False
        self._enroll_task = asyncio.create_task(self._enroll_timeout(self._enroll))
        return f"Okay {display(name)}. {PROMPTS[CENTER]} Then slowly turn your head a little to each side."

    @property
    def enrollment_progress(self) -> Optional[tuple]:
        """(name, samples so far, samples needed) while enrolling, else None."""
        en = self._enroll
        return (en.name, len(en.samples), en.needed) if en else None

    @property
    def enrollment_hint(self) -> str:
        """What the person enrolling should do right now (for the UI)."""
        return self._enroll_hint if self._enroll else ""

    def forget_person(self, name: str) -> int:
        """Delete everything stored about ``name``. Returns the number of samples removed."""
        n = self.store.remove(name)
        self.matcher.register_identities(self.store.load_all())
        if n:
            self.audit.record("forgotten", name)
            self.bus.publish(Topic.PEOPLE_CHANGED, name)
        return n

    async def _h_forget(self, c: Command) -> str:
        known = self._visible_known()
        if not known:
            return "I can only forget you if I can see and recognize you."
        if len(known) > 1:
            return "I see more than one person I know. Please make sure only you are in view."
        self.forget_person(known[0])
        return f"Done. I've deleted everything I stored about {display(known[0])}."

    async def _h_shutdown(self, c: Command) -> str:
        self.shutdown.set()
        return "Goodbye."

    # ---- enrollment (voice or UI) ----------------------------------------
    async def _on_detected(self, e: Event) -> None:
        en = self._enroll
        if en is None:
            return
        ev: FaceEvent = e.payload
        now = time.monotonic()
        img = ev.frame.image
        brightness = float(img[::16, ::16].mean()) if img is not None else 255.0
        fb = en.offer(ev.faces, brightness, now)
        self._enroll_hint = fb.text
        self._say_hint(fb.text, now)
        if fb.accepted and not self._dup_checked and len(en.samples) >= DUPLICATE_CHECK_AT:
            self._dup_checked = True
            other = self._matches_someone_else(en)
            if other:
                name, score = other
                self._end_enrollment()
                self._reply(f"That face already matches {display(name)} ({score:.2f}). If that's you, enroll as "
                            f"{display(name)} to add samples, otherwise forget {display(name)} first. Nothing was saved.")
                return
        if fb.done:
            self._finish_enrollment(en)

    def _say_hint(self, text: str, now: float) -> None:
        """Speak a hint only when it changes, and not more often than every few seconds."""
        last_text, last_time = self._hint_said
        if text != last_text and now - last_time >= HINT_EVERY:
            self._hint_said = (text, now)
            self.speaker.say(text)

    def _matches_someone_else(self, en: EnrollmentSession):
        """(name, score) if the new samples are clearly an already-enrolled OTHER person."""
        results = self.matcher.search_batch(np.stack(en.samples))
        votes = Counter(lbl for lbl, _ in results if lbl and lbl != UNKNOWN and lbl != en.name)
        if not votes:
            return None
        name, n = votes.most_common(1)[0]
        if n < 2:
            return None
        return name, float(np.mean([s for lbl, s in results if lbl == name]))

    def _end_enrollment(self) -> None:
        self._enroll, self._enroll_hint = None, ""
        if self._enroll_task:
            self._enroll_task.cancel()

    def _finish_enrollment(self, en: EnrollmentSession) -> None:
        self._end_enrollment()
        consistency = cohesion(en.samples)
        if consistency < ENROLL_COHESION_MIN:
            self._reply("Those samples don't look like one person, so I haven't saved them. "
                        "Please try again with only you in front of the camera.")
            return
        for emb in en.samples:
            self.store.add(en.name, emb)
        self.matcher.register_identities(self.store.load_all())
        self.audit.record("enrolled", f"{en.name} ({len(en.samples)} samples, consistency {consistency:.2f})")
        self.bus.publish(Topic.PEOPLE_CHANGED, en.name)
        self._reply(f"Done. I'll remember you, {display(en.name)}.")

    async def _enroll_timeout(self, en: EnrollmentSession) -> None:
        await asyncio.sleep(ENROLL_TIMEOUT)
        if self._enroll is en:
            self._end_enrollment()
            self._reply("I couldn't get enough clear views of one face. Enrollment cancelled; nothing was saved.")
