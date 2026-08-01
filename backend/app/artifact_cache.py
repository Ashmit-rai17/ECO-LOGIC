from __future__ import annotations

import hashlib
import json
import pickle
from dataclasses import asdict
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

from app.config import Settings
from app.ml.data_loader import DatasetInfo

CACHE_VERSION = "2026-08-01.1"


def dataset_cache_key(info: DatasetInfo, settings: Settings) -> str:
    stat = info.path.stat()
    payload = {
        "cacheVersion": CACHE_VERSION,
        "dataset": asdict(info),
        "source": {
            "name": info.path.name,
            "size": stat.st_size,
            "mtimeNs": stat.st_mtime_ns,
        },
        "settings": {
            "maxPoints": settings.max_points,
            "shapSampleSize": settings.shap_sample_size,
            "xgbNEstimators": settings.xgb_n_estimators,
            "xgbEarlyStoppingRounds": settings.xgb_early_stopping_rounds,
        },
    }
    encoded = json.dumps(payload, default=str, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:24]


class ArtifactCache:
    def __init__(self, artifacts_dir: Path, enabled: bool = True) -> None:
        self.artifacts_dir = artifacts_dir
        self.enabled = enabled

    def load(self, dataset_key: str, cache_key: str, name: str) -> Any | None:
        if not self.enabled:
            return None
        path = self._path(dataset_key, cache_key, name)
        if not path.exists():
            return None
        try:
            with path.open("rb") as file:
                return pickle.load(file)
        except Exception:
            path.unlink(missing_ok=True)
            return None

    def save(self, dataset_key: str, cache_key: str, name: str, value: Any) -> None:
        if not self.enabled:
            return
        path = self._path(dataset_key, cache_key, name)
        path.parent.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile("wb", delete=False, dir=path.parent, suffix=".tmp") as file:
            pickle.dump(value, file, protocol=pickle.HIGHEST_PROTOCOL)
            temporary_path = Path(file.name)
        temporary_path.replace(path)

    def _path(self, dataset_key: str, cache_key: str, name: str) -> Path:
        safe_key = "".join(char for char in dataset_key.upper() if char.isalnum() or char in {"-", "_"})
        return self.artifacts_dir / safe_key / cache_key / f"{name}.pkl"
