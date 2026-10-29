import pytest

from core.commands import CommandRouter, extract_name, is_wake_word, normalize


def intent(text, router=None):
    c = (router or CommandRouter()).parse(text)
    return c.intent if c else None


@pytest.mark.parametrize("text,expected", [
    ("Hey Aether, who is at the camera?", "who_here"),
    ("Aether, who's in front of me", "who_here"),
    ("Aether is anyone there", "who_here"),
    ("Aether who am I?", "who_am_i"),
    ("Aether, do you recognize me", "who_am_i"),
    ("Aether who do you know", "who_known"),
    ("Aether, what time is it?", "time"),
    ("Aether what's the time", "time"),
    ("Aether, what's the date today", "date"),
    ("Aether what day is it", "date"),
    ("Aether status", "status"),
    ("Aether how are you", "status"),
    ("Aether recent activity", "recent"),
    ("Aether what can you do", "help"),
    ("Aether thank you", "thanks"),
    ("Aether goodbye", "shutdown"),
    ("Aether, shut down", "shutdown"),
    ("Aether enroll me as Sani", "enroll"),
    ("Aether, remember my face", "enroll"),
    ("Aether forget me", "forget_me"),
    ("Aether delete my data", "forget_me"),
    ("Aether sing me a song", "unknown"),
    ("Aether", "attention"),
])
def test_intents(text, expected):
    assert intent(text) == expected


def test_who_am_i_is_not_swallowed_by_who_here():
    assert intent("Aether who am I") == "who_am_i"
    assert intent("Aether who do you know") == "who_known"


@pytest.mark.parametrize("text", ["who is at the camera", "what time is it", "lock the door", "thanks everyone"])
def test_ignored_without_wake_word(text):
    assert CommandRouter().parse(text) is None


@pytest.mark.parametrize("wake", ["Aether", "ether", "Eather", "aether,", "Hey Aether", "OK ether", "okay, Aether!"])
def test_wake_word_variants_whisper_produces(wake):
    assert intent(f"{wake} what time is it") == "time"


@pytest.mark.parametrize("word", ["weather", "whether", "father", "other", "the", "hello", "time", "camera", "mother"])
def test_wake_word_does_not_fire_on_ordinary_words(word):
    assert not is_wake_word(word)


def test_wake_word_optional_mode():
    assert intent("what time is it", CommandRouter(require_wake_word=False)) == "time"


def test_extract_name():
    assert extract_name("enroll me as sani") == "Sani"
    assert extract_name("remember my face my name is abdul jabbar") == "Abdul_Jabbar"
    assert extract_name("call me o'brien") == "Obrien"
    assert extract_name("enroll me as a b c d") == "A_B"  # max two words
    assert extract_name("enroll me") is None
    assert extract_name(normalize("enroll me as ../etc")) == "Etc"  # nothing unsafe survives


def test_enroll_carries_name_argument():
    c = CommandRouter().parse("Hey Aether, enroll me as Sani.")
    assert c.intent == "enroll" and c.args == {"name": "Sani"}


def test_custom_intent_takes_priority():
    r = CommandRouter()
    r.add_intent("lock_door", [r"\block\b.*\bdoor\b"])
    assert intent("Aether lock the front door", r) == "lock_door"
    assert intent("Aether what time is it", r) == "time"


def test_normalize():
    assert normalize("  Hey, Aether!!  What's the TIME?  ") == "hey aether what's the time"
    assert normalize("it’s") == "it's"
