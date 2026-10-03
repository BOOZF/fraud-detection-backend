from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=Path(__file__).resolve().parent.parent / ".env", extra="ignore"
    )

    td_host: str
    td_user: str
    td_password: str
    openai_api_key: str = ""
    llm_model: str = "gpt-4o-mini"
    embed_model: str = "text-embedding-3-small"
    demo_cache: bool = True
    cors_origins: list[str] = ["http://localhost:3000"]


def get_settings() -> Settings:
    return Settings()
