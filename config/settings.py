"""Centralized configuration. Values come from env vars (prefix AETHER_) or a .env file."""
from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="AETHER_", env_file=".env", extra="ignore")

    camera_index: int = 0
    frame_width: int = 640
    frame_height: int = 480
    similarity_threshold: float = Field(0.68, ge=0.0, le=1.0)
    whisper_model: str = "small"
    whisper_device: Literal["auto", "cpu", "cuda"] = "auto"
    db_path: str = "storage/aether.db"
    log_level: str = "INFO"
    bus_queue_size: int = Field(100, ge=1)


@lru_cache
def get_settings() -> Settings:
    return Settings()
