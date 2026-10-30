"""Check that the local chat model works, with no camera or microphone.

    python -m examples.llm_check                       # uses AETHER_LLM_MODEL / AETHER_LLM_HOST from .env
    python -m examples.llm_check "why is the sky blue"

Setup: install Ollama from https://ollama.com, then run  ollama pull llama3.2:3b
"""
import asyncio
import sys
import time

from config.settings import get_settings
from core.llm import Assistant, LLMError, OllamaClient


async def main(question: str) -> int:
    cfg = get_settings()
    client = OllamaClient(cfg.llm_model, cfg.llm_host, cfg.llm_timeout, cfg.llm_allow_remote)
    print(f"model {cfg.llm_model} at {cfg.llm_host}")
    t0 = time.perf_counter()
    try:
        answer = await Assistant(client).ask(question)
    except LLMError as exc:
        print(f"FAILED: {exc}")
        return 1
    print(f"{answer}\n({time.perf_counter() - t0:.1f} s; the first answer is slower while the model loads)")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main(" ".join(sys.argv[1:]) or "In one sentence, what can you help me with?")))
