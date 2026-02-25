"""Base dataset loader — all loaders must inherit from this."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


class BaseDatasetLoader(ABC):
    """Abstract base for all HAIIP dataset loaders.

    Subclasses must implement:
        - load() -> pd.DataFrame
        - feature_columns property
        - label_column property

    Optional to override:
        - preprocess(df) -> pd.DataFrame
        - get_train_test_split(test_size, random_state) -> tuple
    """

    def __init__(self, cache_dir: Path | str | None = None) -> None:
        self.cache_dir = Path(cache_dir) if cache_dir else Path("./data_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._df: pd.DataFrame | None = None

    @abstractmethod
    def load(self) -> pd.DataFrame:
        """Load and return the dataset as a DataFrame."""

    @property
    @abstractmethod
    def feature_columns(self) -> list[str]:
        """Return list of feature column names."""

    @property
    @abstractmethod
    def label_column(self) -> str:
        """Return the name of the target/label column."""

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optional preprocessing hook — override in subclasses."""
        return df

    def get_X_y(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (X, y) numpy arrays."""
        if self._df is None:
            self._df = self.load()
        df = self.preprocess(self._df)
        X = df[self.feature_columns].values.astype(np.float64)
        y = df[self.label_column].values
        return X, y

    def get_train_test_split(
        self,
        test_size: float = 0.2,
        random_state: int = 42,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return (X_train, X_test, y_train, y_test)."""
        from sklearn.model_selection import train_test_split

        X, y = self.get_X_y()
        return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=None)

    def info(self) -> dict[str, Any]:
        """Return dataset metadata."""
        if self._df is None:
            self._df = self.load()
        return {
            "name": self.__class__.__name__,
            "n_samples": len(self._df),
            "n_features": len(self.feature_columns),
            "feature_columns": self.feature_columns,
            "label_column": self.label_column,
            "dtypes": self._df.dtypes.astype(str).to_dict(),
        }
