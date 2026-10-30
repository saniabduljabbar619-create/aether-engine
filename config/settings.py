"""Centralized configuration. Values come from env vars (prefix AETHER_) or a .env file."""
from functools import lru_cache
from typing import List, Literal, Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="AETHER_", env_file=".env", extra="ignore")

    camera_index: int = 0
    frame_width: int = 640
    frame_height: int = 480
    similarity_threshold: float = Field(0.363, ge=0.0, le=1.0)
    whisper_model: str = "base"
    whisper_device: Literal["auto", "cpu", "cuda"] = "auto"
    # Fixing the language skips Whisper's detection pass (faster) and stops short clips being
    # "detected" as Greek or Polish. Use AETHER_WHISPER_LANGUAGE=auto to detect automatically.
    whisper_language: Optional[str] = "en"
    whisper_prompt: str = ""  # empty = a hint built from the wake word, to bias decoding toward it
    # Say this first. Several words can be given, comma-separated. A common word ("computer") is far
    # easier for a speech model to hear than a rare one ("Aether").
    wake_words: str = "aether"
    # When on, enrolling, forgetting and "requires_verified" custom commands need a recent "verify me"
    # (a head-turn challenge that a photo cannot pass). Try "verify me" yourself before switching it on.
    require_liveness: bool = False
    verified_seconds: float = Field(60.0, gt=0)  # how long a passed check stays valid
    follow_up_seconds: float = Field(10.0, ge=0)  # after Aether speaks, safe commands need no wake word for this long
    whisper_threads: int = Field(0, ge=0)  # 0 = half of the CPU threads, so the camera stays smooth
    detect_max_fps: float = Field(10.0, gt=0)  # face detection rate cap; frees CPU for speech
    mic_device: Optional[int] = None  # None = system default
    faces_dir: str = "storage/reference_faces"
    db_path: str = "storage/aether.db"
    log_level: str = "INFO"
    bus_queue_size: int = Field(100, ge=1)

    @property
    def wake_word_list(self) -> List[str]:
        return [w.strip().lower() for w in self.wake_words.split(",") if w.strip()] or ["aether"]

    @property
    def effective_whisper_prompt(self) -> str:
        if self.whisper_prompt.strip():
            return self.whisper_prompt.strip()
        w = self.wake_word_list[0].title()
        return f"Hey {w}, who is at the camera? {w}, what time is it?"

    @field_validator("whisper_language", mode="before")
    @classmethod
    def _auto_means_none(cls, v):
        return None if v is None or str(v).strip().lower() in ("", "auto", "none") else str(v).strip().lower()


@lru_cache
def get_settings() -> Settings:
    return Settings()
