"""Spoken-command understanding: wake word + rule-based intent matching.

Whisper output is messy ("Hey, Aether!  Who's at the camera?"), so text is normalized first.
The wake word is matched fuzzily because Whisper often hears "ether" or "either"; it can be any
word or short phrase (a common word like "computer" is far easier for a speech model than "Aether").
Intents are checked in order and the first match wins, so put specific ones before general ones.

How a transcript is read depends on who is assumed to be talking to us (``Addressing``):
  WAKE       normal listening: the wake word is required.
  FOLLOW_UP  just after we spoke: no wake word needed, but chatter we cannot map to a command is
             ignored silently, so background conversation does not get answered.
  DIRECT     typed text or push-to-talk: someone deliberately addressed us; no wake word needed.
"""
from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Pattern, Sequence, Tuple

WAKE_WORD = "aether"
_FILLERS = {"hey", "hi", "hello", "ok", "okay", "yo", "please"}


class Addressing(str, Enum):
    WAKE = "wake"
    FOLLOW_UP = "follow_up"
    DIRECT = "direct"


# Harmless, read-only intents that may run without the wake word in a follow-up window. Anything that
# changes data (enroll, forget), stops the system, or reveals names or activity needs the wake word,
# so a "goodbye" said to someone else on the phone cannot shut the system down.
SAFE_WITHOUT_WAKE = frozenset({"time", "date", "status", "help", "thanks", "who_here", "who_am_i"})


@dataclass(frozen=True)
class Command:
    intent: str
    text: str  # normalized text after the wake word
    args: Dict[str, str] = field(default_factory=dict)
    woke: bool = True  # was the wake word actually said?


def normalize(text: str) -> str:
    text = text.lower().replace("’", "'")
    return " ".join(re.sub(r"[^a-z0-9' ]+", " ", text).split())


# Real words that look enough like "aether" to pass the fuzzy match but are not the wake word.
_NOT_WAKE = {"weather", "whether", "feather", "leather", "father", "rather", "other"}


def is_wake_word(token: str, wake: str = WAKE_WORD, threshold: float = 0.75) -> bool:
    if token in _NOT_WAKE:
        return False
    return difflib.SequenceMatcher(None, token, wake).ratio() >= threshold


# (intent, patterns). Order matters.
_BUILTIN: List[Tuple[str, List[str]]] = [
    ("shutdown", [r"\b(shut ?down|good ?bye|exit|quit|stop listening|turn off)\b"]),
    ("verify", [r"\b(verify|authenticate)\b.*\b(me|my identity|that i'?m|i am)\b", r"\bam i (real|alive|a real person)\b",
                r"\bliveness\b", r"\b(prove|confirm|check)\b.*\b(i'?m|i am|it'?s|that it'?s)\b.*\b(real|really|alive|me)\b",
                r"\bit'?s really me\b"]),
    ("forget_me", [r"\b(forget|delete|remove)\b.*\b(me|my face|my data)\b"]),
    ("enroll", [r"\b(enrol|enroll|register|remember|learn)\b.*\b(me|my face)\b"]),
    ("who_am_i", [r"\bwho am i\b", r"\bdo you (know|recogni[sz]e) me\b"]),
    ("who_known", [r"\bwho do you (know|remember)\b", r"\bwho is enrolled\b", r"\blist (the )?(people|faces)\b"]),
    ("who_here", [r"\bwho\b.*\b(camera|here|there|see|front)\b", r"\bwho is (this|that)\b",
                  r"\bis (anyone|anybody|someone) (here|there)\b"]),
    ("time", [r"\bwhat(?:'s| is)? the time\b", r"\bwhat time\b", r"\btime is it\b"]),
    ("date", [r"\bwhat(?:'s| is)? (?:the |today's )?date\b", r"\bwhat day\b"]),
    ("recent", [r"\b(recent|latest) (activity|events)\b", r"\bwhat happened\b"]),
    ("status", [r"\bstatus\b", r"\bhow are you\b", r"\bare you (working|ok|okay)\b", r"\bsystem check\b"]),
    ("help", [r"\bhelp\b", r"\bwhat can you do\b"]),
    ("thanks", [r"\bthank(s| you)\b"]),
]

_NAME = re.compile(r"(?:\bas\b|\bname is\b|\bcall me\b|\bi am\b|\bi'm\b)\s+([a-z0-9' ]+)$")


def extract_name(text: str) -> Optional[str]:
    """'enroll me as sani abdul' -> 'Sani_Abdul'. Max two words, letters and digits only."""
    m = _NAME.search(text)
    if not m:
        return None
    words = [re.sub(r"[^a-z0-9]", "", w) for w in m.group(1).split()]
    words = [w for w in words if w][:2]
    return "_".join(w.capitalize() for w in words) or None


def match_wake(tokens: Sequence[str], wake_words: Sequence[str]) -> int:
    """How many leading tokens form a wake word (0 if none). Wake words may have several words."""
    for wake in wake_words:
        n = len(wake.split())
        if len(tokens) >= n and is_wake_word(" ".join(tokens[:n]), wake):
            return n
    return 0


class CommandRouter:
    def __init__(self, require_wake_word: bool = True, wake_words: Sequence[str] = (WAKE_WORD,)) -> None:
        self.require_wake_word = require_wake_word
        self.wake_words = tuple(w for w in (normalize(x) for x in wake_words) if w) or (WAKE_WORD,)
        self._intents: List[Tuple[str, List[Pattern]]] = [
            (name, [re.compile(p) for p in pats]) for name, pats in _BUILTIN]

    @property
    def wake_name(self) -> str:
        """The wake word as shown to people, e.g. 'Aether' or 'Computer'."""
        return self.wake_words[0].title()

    def add_intent(self, name: str, patterns: List[str], first: bool = True) -> None:
        """Register a custom intent, e.g. ('lock_door', [r'\\block\\b.*\\bdoor\\b'])."""
        entry = (name, [re.compile(p) for p in patterns])
        self._intents.insert(0, entry) if first else self._intents.append(entry)

    def parse(self, transcript: str, mode: Addressing = Addressing.WAKE) -> Optional[Command]:
        """None = not for us. Otherwise a Command (intent 'unknown' when we were addressed but could not tell what for)."""
        if not self.require_wake_word:
            mode = Addressing.DIRECT
        tokens = normalize(transcript).split()
        n = match_wake(tokens, self.wake_words)
        if n:
            rest, woke = tokens[n:], True
        else:  # "hey computer": the filler may be part of a multi-word wake word, so try both ways
            stripped = list(tokens)
            while stripped and stripped[0] in _FILLERS:
                stripped.pop(0)
            n = match_wake(stripped, self.wake_words)
            rest, woke = (stripped[n:], True) if n else (stripped, False)
        if not woke and mode is Addressing.WAKE:
            return None
        text = " ".join(rest)
        if not text:
            return Command("attention", text) if woke else None
        for name, patterns in self._intents:
            if any(p.search(text) for p in patterns):
                args = {"name": n} if name == "enroll" and (n := extract_name(text)) else {}
                return Command(name, text, args, woke)
        if not woke and mode is Addressing.FOLLOW_UP:
            return None  # background chatter, not a command
        return Command("unknown", text, woke=woke)
