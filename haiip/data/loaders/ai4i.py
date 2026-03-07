"""AI4I 2020 Predictive Maintenance Dataset loader.

Dataset: UCI ML Repository ID 601
License: CC BY 4.0 — legal for research and commercial use
URL: https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset

Features (5 numeric):
    air_temperature      — K (normalised, ~300 K)
    process_temperature  — K (always > air_temperature)
    rotational_speed     — RPM (around 1500 rpm)
    torque               — Nm (normally distributed ~40 Nm)
    tool_wear            — min (H/L products add 5/2.5 min per cycle)

Labels:
    machine_failure      — binary (0/1)
    failure_type         — categorical: no_failure, TWF, HDF, PWF, OSF, RNF
    product_type         — L, M, H (quality variant)

Usage:
    loader = AI4ILoader()
    df = loader.load()
    X, y = loader.get_X_y()
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from haiip.data.loaders.base import BaseDatasetLoader

logger = logging.getLogger(__name__)

FEATURE_COLS = [
    "air_temperature",
    "process_temperature",
    "rotational_speed",
    "torque",
    "tool_wear",
]

FAILURE_MODES = ["TWF", "HDF", "PWF", "OSF", "RNF"]


class AI4ILoader(BaseDatasetLoader):
    """Loader for the AI4I 2020 Predictive Maintenance dataset.

    Downloads automatically from UCI on first use; caches locally.
    Falls back to synthetic data if ucimlrepo is not installed.
    """

    DATASET_ID = 601

    def __init__(self, cache_dir: Path | str | None = None) -> None:
        super().__init__(cache_dir)
        self._cache_file = self.cache_dir / "ai4i_2020.parquet"

    def load(self) -> pd.DataFrame:
        """Load dataset — from cache if available, otherwise download."""
        if self._cache_file.exists():
            logger.info("AI4I: loading from cache %s", self._cache_file)
            return pd.read_parquet(self._cache_file)

        try:
            return self._download()
        except Exception as exc:
            logger.warning("AI4I download failed (%s) — using synthetic fallback", exc)
            return self._synthetic_fallback()

    def _download(self) -> pd.DataFrame:
        """Download from UCI ML Repository using ucimlrepo."""
        from ucimlrepo import fetch_ucirepo

        logger.info("AI4I: downloading from UCI (id=%d)", self.DATASET_ID)
        dataset = fetch_ucirepo(id=self.DATASET_ID)

        X = dataset.data.features
        y = dataset.data.targets

        df = pd.concat([X, y], axis=1)
        df = self._normalise_columns(df)

        df.to_parquet(self._cache_file, index=False)
        logger.info("AI4I: saved to cache (%d rows)", len(df))
        return df

    def _normalise_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Rename columns to snake_case and derive failure_type label."""
        rename_map = {
            "Air temperature [K]": "air_temperature",
            "Process temperature [K]": "process_temperature",
            "Rotational speed [rpm]": "rotational_speed",
            "Torque [Nm]": "torque",
            "Tool wear [min]": "tool_wear",
            "Machine failure": "machine_failure",
            "Type": "product_type",
        }
        df = df.rename(columns=rename_map)

        # Lowercase any remaining uppercase column names
        df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

        # Derive failure_type from individual failure mode columns
        if "failure_type" not in df.columns:
            df["failure_type"] = "no_failure"
            for mode in FAILURE_MODES:
                mode_col = mode.lower()
                if mode_col in df.columns:
                    df.loc[df[mode_col] == 1, "failure_type"] = mode

        return df

    @staticmethod
    def _synthetic_fallback(n_samples: int = 10_000, seed: int = 42) -> pd.DataFrame:
        """Generate synthetic data matching AI4I distributions.

        Used when the network is unavailable or ucimlrepo is not installed.
        Distributions derived from the published dataset statistics.
        """
        rng = np.random.default_rng(seed)

        n_normal = int(n_samples * 0.97)
        n_anomaly = n_samples - n_normal

        def _normal_batch(n: int) -> dict:
            return {
                "air_temperature": rng.normal(300.0, 2.0, n),
                "process_temperature": rng.normal(310.0, 1.5, n),
                "rotational_speed": rng.normal(1538.0, 179.0, n),
                "torque": rng.normal(40.0, 9.8, n),
                "tool_wear": rng.uniform(0, 253, n),
            }

        def _anomaly_batch(n: int) -> dict:
            return {
                "air_temperature": rng.normal(303.0, 3.0, n),
                "process_temperature": rng.normal(313.0, 2.5, n),
                "rotational_speed": rng.normal(1400.0, 300.0, n),
                "torque": rng.normal(60.0, 15.0, n),
                "tool_wear": rng.uniform(200, 253, n),
            }

        normal_data = _normal_batch(n_normal)
        anomaly_data = _anomaly_batch(n_anomaly)

        rows = []
        for i in range(n_normal):
            rows.append(
                {
                    **{k: v[i] for k, v in normal_data.items()},
                    "machine_failure": 0,
                    "failure_type": "no_failure",
                    "product_type": rng.choice(["L", "M", "H"], p=[0.5, 0.3, 0.2]),
                }
            )

        failure_modes = ["TWF", "HDF", "PWF", "OSF", "RNF"]
        for i in range(n_anomaly):
            rows.append(
                {
                    **{k: v[i] for k, v in anomaly_data.items()},
                    "machine_failure": 1,
                    "failure_type": rng.choice(failure_modes),
                    "product_type": rng.choice(["L", "M", "H"], p=[0.5, 0.3, 0.2]),
                }
            )

        df = pd.DataFrame(rows)
        df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
        logger.info("AI4I: synthetic fallback generated (%d rows)", len(df))
        return df

    # ── BaseDatasetLoader interface ───────────────────────────────────────────

    @property
    def feature_columns(self) -> list[str]:
        return FEATURE_COLS

    @property
    def label_column(self) -> str:
        return "failure_type"

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove rows with missing values and clip outliers."""
        df = df.dropna(subset=FEATURE_COLS)

        # Clip physically impossible values
        df = df[df["rotational_speed"] > 0]
        df = df[df["torque"] >= 0]
        df = df[df["tool_wear"] >= 0]

        return df.reset_index(drop=True)

    def get_normal_data(self) -> pd.DataFrame:
        """Return only normal (no failure) rows — for unsupervised training."""
        if self._df is None:
            self._df = self.load()
        df = self.preprocess(self._df)
        return df[df["failure_type"] == "no_failure"].reset_index(drop=True)
