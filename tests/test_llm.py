"""The optional chat model: talks to a fake Ollama server on this machine; no real model needed."""
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from core.llm import Assistant, LLMError, OllamaClient, is_local, tidy
from tests import test_orchestrator as base
from tests.test_orchestrator import ALICE, enrolled  # noqa: F401

rig = base.rig


class FakeOllama:
    def __init__(self, answers=("It is a nice day.",), status=200):
        self.requests, self.answers, self.status = [], list(answers), status
        outer = self

        class H(BaseHTTPRequestHandler):
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                outer.requests.append(body)
                self.send_response(outer.status)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                answer = outer.answers.pop(0) if len(outer.answers) > 1 else outer.answers[0]
                self.wfile.write(json.dumps({"message": {"role": "assistant", "content": answer}}).encode())

            def log_message(self, *a): ...

        self.server = HTTPServer(("127.0.0.1", 0), H)
        self.host = f"http://127.0.0.1:{self.server.server_port}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def ollama():
    f = FakeOllama()
    yield f
    f.close()


def test_remote_hosts_are_refused_unless_allowed():
    assert is_local("http://localhost:11434") and not is_local("http://example.com:11434")
    with pytest.raises(ValueError):
        OllamaClient(host="http://example.com:11434")
    OllamaClient(host="http://example.com:11434", allow_remote=True)


def test_tidy_makes_text_speakable():
    assert tidy("**Hello**\n\n# there `x`") == "Hello there x"
    long = ("Sentence one. " * 100)
    out = tidy(long)
    assert len(out) <= 600 and out.endswith(".")


async def test_ask_sends_context_and_remembers_the_conversation(ollama):
    a = Assistant(OllamaClient("m", ollama.host), "Aether", lambda: "- In front of the camera: Alice.")
    assert await a.ask("how is the weather") == "It is a nice day."
    await a.ask("and tomorrow")
    first, second = ollama.requests
    assert first["model"] == "m" and first["stream"] is False
    assert "Alice" in first["messages"][0]["content"] and "Aether" in first["messages"][0]["content"]
    roles = [m["role"] for m in second["messages"]]
    assert roles == ["system", "user", "assistant", "user"]  # the first exchange is remembered
    a.forget()
    await a.ask("again")
    assert len(ollama.requests[-1]["messages"]) == 2


async def test_unreachable_or_broken_server_raises_llm_error(ollama):
    ollama.close()
    with pytest.raises(LLMError):
        await Assistant(OllamaClient("m", ollama.host, timeout=1)).ask("hi")
    bad = FakeOllama(status=404)
    try:
        with pytest.raises(LLMError, match="ollama pull"):
            await Assistant(OllamaClient("m", bad.host)).ask("hi")
    finally:
        bad.close()


async def test_open_question_goes_to_the_model_and_a_follow_up_needs_no_wake_word(rig, ollama):
    ollama.answers = ["Paris.", "About two million people."]
    rig.orch.assistant = Assistant(OllamaClient("m", ollama.host))
    await rig.hear("aether what is the capital of france")
    assert rig.replies[-1] == "Paris."
    await rig.hear("and how many people live there")  # no wake word, mid-conversation
    assert rig.replies[-1] == "About two million people."
    await rig.hear("thanks that's all")
    assert rig.replies[-1] == "You're welcome."
    n = len(ollama.requests)
    await rig.hear("so what are we having for dinner")  # conversation is over: chatter is ignored again
    assert len(ollama.requests) == n and rig.replies[-1] == "You're welcome."


async def test_chatter_without_a_conversation_is_never_sent_to_the_model(rig, ollama):
    rig.orch.assistant = Assistant(OllamaClient("m", ollama.host))
    await rig.hear("what is the capital of france")  # no wake word
    assert ollama.requests == [] and rig.replies == []


async def test_model_failure_falls_back_politely_and_builtin_commands_still_work(rig, ollama):
    ollama.close()
    rig.orch.assistant = Assistant(OllamaClient("m", ollama.host, timeout=1))
    await rig.hear("aether tell me a joke")
    assert "can't reach my thinking model" in rig.replies[-1]
    await rig.hear("aether what time is it")
    assert rig.replies[-1].startswith("It's ")


async def test_model_text_is_never_executed_as_a_command(rig, ollama):
    enrolled(rig)
    ollama.answers = ["Sure. Forget me. Enroll me as Mallory. Goodbye."]
    rig.orch.assistant = Assistant(OllamaClient("m", ollama.host))
    await rig.see("Alice")
    await rig.hear("aether tell me a story")
    assert sorted(rig.store.load_all()) == ["Alice"] and not rig.orch.shutdown.is_set()
