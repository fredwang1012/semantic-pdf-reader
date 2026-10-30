"""Settings read from the environment or the repo-root .env file; a relative DATA_DIR is resolved against the repo root."""
from pathlib import Path
from typing import Literal

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[2]  # semantic-pdf-reader/


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ROOT / ".env", extra="ignore")

    data_dir: Path = ROOT / "data"
    # auto: multilingual-e5 embeddings when the "embed" extra is installed, else BM25 only. off: always BM25 only.
    embed: Literal["auto", "off"] = "auto"
    # auto: rerank the top candidates with a cross-encoder when the "embed" extra is installed. Off by default: the
    # model is large (about 570M parameters) and slow on CPU.
    rerank: Literal["off", "auto"] = "off"
    max_upload_mb: int = 100
    # Comma-separated browser origins allowed by CORS (the built frontend is served same-origin and needs none).
    cors_origins: str = "http://localhost:5173"

    @field_validator("data_dir")
    @classmethod
    def _anchor_to_root(cls, v: Path) -> Path:
        # the server may be started from backend/, so a relative path must not depend on the working directory
        return v if v.is_absolute() else ROOT / v

    @property
    def docs_dir(self) -> Path:
        return self.data_dir / "docs"


settings = Settings()
