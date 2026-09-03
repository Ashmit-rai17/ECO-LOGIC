from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    data_dir: Path = BACKEND_DIR / "data"
    artifacts_dir: Path = BACKEND_DIR / "artifacts"
    cache_enabled: bool = True
    cors_origins: str = "http://localhost:3000,http://127.0.0.1:3000"
    max_points: int = 800
    shap_sample_size: int = 100
    xgb_n_estimators: int = 200
    xgb_early_stopping_rounds: int = 25
    enable_model_comparison: bool = True
    enable_shap: bool = True
    walk_forward_folds: int = 3
    walk_forward_min_train_years: int = 2
    walk_forward_test_years: int = 1
    enable_walk_forward: bool = True
    enable_weather: bool = True
    weather_latitude: float = 39.9612
    weather_longitude: float = -82.9988

    model_config = SettingsConfigDict(env_prefix="ECOLOGIC_", env_file=".env")

    @property
    def cors_origin_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
