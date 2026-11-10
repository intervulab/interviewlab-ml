"""Environment-driven configuration (mirrors Component 4 patterns)."""

from functools import lru_cache

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    APP_ENV: str = "development"
    APP_PORT: int = 8001
    DEBUG: bool = True

    ALLOWED_ORIGINS: str = "http://localhost:3000,http://localhost:4000"

    DEFAULT_SAMPLE_RATE: int = 16000
    HESITATION_MFCC_DELTA_THRESHOLD: float = 8.0
    FILLER_WORDS: str = "um,uh,like,you know,basically,actually,kind of,sort of"

    RUBRIC_INFERENCE_MODE: str = "heuristic"  # heuristic | transformer
    RUBRIC_MODEL_NAME: str = "distilbert-base-uncased"
    RUBRIC_MODEL_PATH: str = ""

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

    @property
    def origins_list(self) -> list[str]:
        return [o.strip() for o in self.ALLOWED_ORIGINS.split(",") if o.strip()]

    @property
    def filler_word_list(self) -> list[str]:
        return [w.strip().lower() for w in self.FILLER_WORDS.split(",") if w.strip()]


@lru_cache()
def get_settings() -> Settings:
    return Settings()
