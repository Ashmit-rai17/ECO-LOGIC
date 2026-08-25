from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from zipfile import ZipFile

import pandas as pd


@dataclass(frozen=True)
class DatasetInfo:
    key: str
    label: str
    path: Path
    value_column: str
    zipped: bool


def discover_datasets(data_dir: Path) -> list[DatasetInfo]:
    candidates: dict[str, DatasetInfo] = {}
    for path in sorted(data_dir.glob("*_hourly*")):
        if path.suffix not in {".csv", ".zip"}:
            continue
        key = path.name.split("_hourly")[0].replace(" (1)", "").upper()
        if key in candidates and candidates[key].path.suffix == ".zip":
            continue
        value_column = f"{key}_MW"
        candidates[key] = DatasetInfo(
            key=key,
            label=key,
            path=path,
            value_column=value_column,
            zipped=path.suffix == ".zip",
        )
    return sorted(candidates.values(), key=lambda item: item.key)


def load_dataset(info: DatasetInfo) -> pd.DataFrame:
    if info.zipped:
        with ZipFile(info.path) as archive:
            csv_name = next(name for name in archive.namelist() if name.lower().endswith(".csv"))
            with archive.open(csv_name) as file:
                df = pd.read_csv(file)
    else:
        df = pd.read_csv(info.path)

    if "Datetime" not in df.columns:
        raise ValueError(f"{info.path.name} does not contain a Datetime column.")

    value_columns = [col for col in df.columns if col != "Datetime"]
    if not value_columns:
        raise ValueError(f"{info.path.name} does not contain a demand value column.")

    value_column = info.value_column if info.value_column in df.columns else value_columns[0]
    df = df[["Datetime", value_column]].rename(columns={value_column: "demand_mw"})
    return df
