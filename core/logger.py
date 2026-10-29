"""Logging setup using the stdlib, so the bus and modules need no extra dependency."""
import logging


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=level.upper(),
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    # Third-party libraries log routine details at INFO (every request, every transcription).
    for noisy in ("httpx", "httpx2", "httpcore", "huggingface_hub", "comtypes", "faster_whisper", "PIL"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
