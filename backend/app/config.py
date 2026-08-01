from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    data_dir: Path = BACKEND_DIR / "data"
    artifacts_dir: Path = BACKEND_DIR / "artifacts"
    cache_enabled: bool = True
    cors_origins: str = "http://localhost:3000,http://127.0.0.1:3000"
    max_points: int = 2200
    shap_sample_size: int = 600
    xgb_n_estimators: int = 4000
    xgb_early_stopping_rounds: int = 100

    model_config = SettingsConfigDict(env_prefix="ECOLOGIC_", env_file=".env")

    @property
    def cors_origin_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
