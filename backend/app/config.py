from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[2]  # semantic-pdf-reader/


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ROOT / ".env", extra="ignore")

    data_dir: Path = ROOT / "data"
    # auto: multilingual-e5 embeddings when the "embed" extra is installed, else BM25 only. off: always BM25 only.
    embed: Literal["auto", "off"] = "auto"
    max_upload_mb: int = 100
    # Comma-separated browser origins allowed by CORS (the built frontend is served same-origin and needs none).
    cors_origins: str = "http://localhost:5173"

    @property
    def docs_dir(self) -> Path:
        return self.data_dir / "docs"


settings = Settings()
