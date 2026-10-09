from pathlib import Path

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Everything is configured from `.env` (see `.env.example`). One model, LLM_MODEL, drives every LLM call:
    chat answers, the brief, the relevance check and the topic gate."""

    model_config = SettingsConfigDict(
        env_file=Path(__file__).resolve().parent.parent / ".env", extra="ignore"
    )

    td_host: str
    td_user: str
    td_password: str
    openai_api_key: str = ""
    llm_model: str = "gpt-5-mini"
    reasoning_effort: str = "low"  # gpt-5 models only: minimal | low | medium | high
    embed_model: str = "text-embedding-3-small"  # changing it needs `python scripts/ingest_policies.py` to re-embed the documents
    # Optional overrides. Blank (the default) means "use LLM_MODEL", so every call runs on the same model.
    verify_model: str = ""  # the check that a quoted passage really answers a brief question
    gate_model: str = ""  # the on-topic check before the chat
    demo_cache: bool = True
    cors_origins: list[str] = ["http://localhost:3000"]

    @model_validator(mode="after")
    def _default_to_the_main_model(self):
        self.verify_model = self.verify_model.strip() or self.llm_model
        self.gate_model = self.gate_model.strip() or self.llm_model
        return self


def get_settings() -> Settings:
    return Settings()
