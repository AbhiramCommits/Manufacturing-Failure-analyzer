import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional

EXPECTED_COLUMNS = {
    "test_id": "str",
    "component_id": "str",
    "timestamp": "str",
    "test_type": "str",
    "failure_code": "int64",
    "error_message": "str",
    "duration_ms": "float64",
    "temperature_c": "float64",
    "voltage_v": "float64",
}


class DataLoader:
    def __init__(self, filepath: str):
        self.filepath = Path(filepath)

    def load(self) -> pd.DataFrame:
        if not self.filepath.exists():
            raise FileNotFoundError(f"CSV file not found: {self.filepath}")

        df = pd.read_csv(self.filepath)
        df = self._validate(df)
        df = self._clean(df)
        return df

    def _validate(self, df: pd.DataFrame) -> pd.DataFrame:
        actual_cols = set(df.columns)
        expected_cols = set(EXPECTED_COLUMNS.keys())
        missing = expected_cols - actual_cols
        extra = actual_cols - expected_cols

        if missing:
            raise ValueError(f"Missing columns: {missing}")
        if extra:
            raise ValueError(f"Unexpected columns: {extra}")

        for col, expected_type in EXPECTED_COLUMNS.items():
            if col == "error_message":
                continue
            actual_type = str(df[col].dtype)
            if actual_type != expected_type:
                raise ValueError(
                    f"Column '{col}' has dtype '{actual_type}', expected '{expected_type}'"
                )

        return df

    def _clean(self, df: pd.DataFrame) -> pd.DataFrame:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        invalid_ts = df["timestamp"].isna()
        if invalid_ts.any():
            df = df.loc[~invalid_ts].copy()
            print(f"Warning: Dropped {invalid_ts.sum()} rows with invalid timestamps")

        df["error_message"] = df["error_message"].replace("", np.nan)
        df = df.dropna(subset=["test_id", "component_id"])

        return df.reset_index(drop=True)
