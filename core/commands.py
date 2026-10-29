"""Spoken-command understanding: wake word + rule-based intent matching.

Whisper output is messy ("Hey, Aether!  Who's at the camera?"), so text is normalized first.
The wake word is matched fuzzily because Whisper often hears "ether" or "either".
Intents are checked in order and the first match wins, so put specific ones before general ones.
"""
from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Pattern, Tuple

WAKE_WORD = "aether"
_FILLERS = {"hey", "hi", "hello", "ok", "okay", "yo", "please"}


@dataclass(frozen=True)
class Command:
    intent: str
    text: str  # normalized text after the wake word
    args: Dict[str, str] = field(default_factory=dict)


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


class CommandRouter:
    def __init__(self, require_wake_word: bool = True) -> None:
        self.require_wake_word = require_wake_word
        self._intents: List[Tuple[str, List[Pattern]]] = [
            (name, [re.compile(p) for p in pats]) for name, pats in _BUILTIN]

    def add_intent(self, name: str, patterns: List[str], first: bool = True) -> None:
        """Register a custom intent, e.g. ('lock_door', [r'\\block\\b.*\\bdoor\\b'])."""
        entry = (name, [re.compile(p) for p in patterns])
        self._intents.insert(0, entry) if first else self._intents.append(entry)

    def parse(self, transcript: str) -> Optional[Command]:
        """None = not addressed to us (no wake word). Otherwise a Command; intent 'unknown' if unclear."""
        tokens = normalize(transcript).split()
        while tokens and tokens[0] in _FILLERS:
            tokens.pop(0)
        woke = bool(tokens) and is_wake_word(tokens[0])
        if woke:
            tokens.pop(0)
        elif self.require_wake_word:
            return None
        text = " ".join(tokens)
        if not text:
            return Command("attention", text)
        for name, patterns in self._intents:
            if any(p.search(text) for p in patterns):
                args = {"name": n} if name == "enroll" and (n := extract_name(text)) else {}
                return Command(name, text, args)
        return Command("unknown", text)
