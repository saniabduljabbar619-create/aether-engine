"""Centralized configuration. Values come from env vars (prefix AETHER_) or a .env file."""
from functools import lru_cache
from typing import Literal, Optional

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
    whisper_prompt: str = "Hey Aether, who is at the camera? Aether, what time is it?"  # biases decoding to the wake word
    whisper_threads: int = Field(0, ge=0)  # 0 = half of the CPU threads, so the camera stays smooth
    detect_max_fps: float = Field(10.0, gt=0)  # face detection rate cap; frees CPU for speech
    mic_device: Optional[int] = None  # None = system default
    faces_dir: str = "storage/reference_faces"
    db_path: str = "storage/aether.db"
    log_level: str = "INFO"
    bus_queue_size: int = Field(100, ge=1)

    @field_validator("whisper_language", mode="before")
    @classmethod
    def _auto_means_none(cls, v):
        return None if v is None or str(v).strip().lower() in ("", "auto", "none") else str(v).strip().lower()


@lru_cache
def get_settings() -> Settings:
    return Settings()
