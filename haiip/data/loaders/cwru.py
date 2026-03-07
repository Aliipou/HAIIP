"""CWRU Bearing Dataset loader.

Source: Case Western Reserve University Bearing Data Center
URL: https://engineering.case.edu/bearingdatacenter
License: Public domain (research use)

Data: SKF bearings at 2hp load, 3 fault sizes (0.007", 0.014", 0.021")
Sensors: Drive-end and fan-end accelerometers at 12kHz and 48kHz
Fault locations: Inner race, ball, outer race (centered, orthogonal, opposite)

Extracts statistical features from raw vibration signals:
    RMS, Peak, Crest Factor, Kurtosis, Skewness, Variance
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from haiip.data.loaders.base import BaseDatasetLoader

logger = logging.getLogger(__name__)

FEATURE_COLS = [
    "rms",
    "peak",
    "crest_factor",
    "kurtosis",
    "skewness",
    "variance",
    "rms_fan",
    "kurtosis_fan",
]

FAULT_LABELS = [
    "normal",
    "inner_race",
    "ball",
    "outer_race_centered",
    "outer_race_orthogonal",
    "outer_race_opposite",
]

SAMPLE_RATES = {12000, 48000}
DEFAULT_WINDOW = 12000
OVERLAP = 6000


class CWRULoader(BaseDatasetLoader):
    """CWRU Bearing Dataset loader with feature extraction from raw vibration.

    Loads from .mat files if present in cache_dir.
    Falls back to synthetic vibration features otherwise.
    """

    def __init__(
        self,
        cache_dir: Path | str | None = None,
        sample_rate: int = 12000,
        window_size: int = DEFAULT_WINDOW,
    ) -> None:
        super().__init__(cache_dir)
        if sample_rate not in SAMPLE_RATES:
            raise ValueError(f"sample_rate must be one of {SAMPLE_RATES}")
        self.sample_rate = sample_rate
        self.window_size = window_size
        self._cache_file = self.cache_dir / "cwru_features.parquet"

    def load(self) -> pd.DataFrame:
        if self._cache_file.exists():
            logger.info("CWRU: loading from cache %s", self._cache_file)
            return pd.read_parquet(self._cache_file)

        mat_files = list(self.cache_dir.glob("*.mat"))
        if mat_files:
            return self._load_from_mat(mat_files)

        logger.warning("CWRU .mat files not found — using synthetic fallback")
        return self._synthetic_fallback()

    def _load_from_mat(self, mat_files: list[Path]) -> pd.DataFrame:
        try:
            from scipy.io import loadmat
        except ImportError:
            logger.warning("scipy not installed — using synthetic fallback")
            return self._synthetic_fallback()

        records: list[dict[str, Any]] = []
        for mat_path in mat_files:
            try:
                data = loadmat(str(mat_path))
                de_keys = [k for k in data.keys() if "DE_time" in k]
                fe_keys = [k for k in data.keys() if "FE_time" in k]
                label = self._infer_label(mat_path.stem)

                for de_key in de_keys:
                    signal_de = data[de_key].flatten()
                    signal_fe = data[fe_keys[0]].flatten() if fe_keys else np.zeros_like(signal_de)
                    feats = self._extract_features(signal_de, signal_fe)
                    for feat in feats:
                        feat["label"] = label
                        feat["source_file"] = mat_path.name
                    records.extend(feats)
            except Exception as exc:
                logger.warning("Failed to load %s: %s", mat_path, exc)

        df = pd.DataFrame(records)
        df.to_parquet(self._cache_file, index=False)
        return df

    def _extract_features(
        self, signal_de: np.ndarray, signal_fe: np.ndarray
    ) -> list[dict[str, Any]]:
        features_list: list[dict[str, Any]] = []
        step = self.window_size - OVERLAP

        for start in range(0, len(signal_de) - self.window_size, step):
            w_de = signal_de[start : start + self.window_size]
            w_fe = signal_fe[start : start + self.window_size]

            rms = float(np.sqrt(np.mean(w_de**2)))
            peak = float(np.max(np.abs(w_de)))
            features_list.append(
                {
                    "rms": rms,
                    "peak": peak,
                    "crest_factor": round(peak / (rms + 1e-8), 4),
                    "kurtosis": round(self._kurtosis(w_de), 4),
                    "skewness": round(self._skewness(w_de), 4),
                    "variance": round(float(np.var(w_de)), 8),
                    "rms_fan": round(float(np.sqrt(np.mean(w_fe**2))), 6),
                    "kurtosis_fan": round(self._kurtosis(w_fe), 4),
                }
            )

        return features_list

    @staticmethod
    def _kurtosis(signal: np.ndarray) -> float:
        mu, sigma = np.mean(signal), np.std(signal) + 1e-8
        return float(np.mean(((signal - mu) / sigma) ** 4))

    @staticmethod
    def _skewness(signal: np.ndarray) -> float:
        mu, sigma = np.mean(signal), np.std(signal) + 1e-8
        return float(np.mean(((signal - mu) / sigma) ** 3))

    @staticmethod
    def _infer_label(filename: str) -> str:
        fn = filename.lower()
        if "normal" in fn or "baseline" in fn:
            return "normal"
        if "ir" in fn or "inner" in fn:
            return "inner_race"
        if "ball" in fn or "_ba" in fn:
            return "ball"
        return "outer_race_centered"

    def _synthetic_fallback(self, n_windows: int = 1200, seed: int = 42) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        records: list[dict[str, Any]] = []
        n_per = n_windows // len(FAULT_LABELS)

        for label in FAULT_LABELS:
            is_faulty = label != "normal"
            rms_vals = rng.normal(0.15 if is_faulty else 0.05, 0.05 if is_faulty else 0.01, n_per)
            kurt_vals = rng.normal(6.0 if is_faulty else 3.0, 2.0 if is_faulty else 0.5, n_per)

            for i in range(n_per):
                r = abs(rms_vals[i])
                records.append(
                    {
                        "rms": round(r, 6),
                        "peak": round(r * rng.uniform(3, 5), 6),
                        "crest_factor": round(rng.uniform(3, 8), 4),
                        "kurtosis": round(abs(kurt_vals[i]), 4),
                        "skewness": round(float(rng.normal(0, 0.5)), 4),
                        "variance": round(r**2, 8),
                        "rms_fan": round(r * 0.8, 6),
                        "kurtosis_fan": round(abs(kurt_vals[i]) * 0.9, 4),
                        "label": label,
                        "source_file": "synthetic",
                    }
                )

        df = pd.DataFrame(records).sample(frac=1, random_state=seed).reset_index(drop=True)
        logger.info("CWRU: synthetic generated (%d rows)", len(df))
        return df

    @property
    def feature_columns(self) -> list[str]:
        return FEATURE_COLS

    @property
    def label_column(self) -> str:
        return "label"
