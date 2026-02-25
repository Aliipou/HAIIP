"""MIMII Dataset loader — unsupervised anomaly detection from industrial sounds.

Source: Hitachi / Zenodo
URL: https://zenodo.org/record/3384388
License: Creative Commons Attribution 4.0 (CC BY 4.0)

Machine types: valve, pump, fan, slide_rail
Each type has multiple models with normal and anomalous audio.
Key property: UNSUPERVISED — only normal data available during training.
This exactly mirrors the real SME scenario where failure labels don't exist.

Feature extraction: Log-Mel spectrogram (64 bands, 512 FFT, 512 hop)
Features per frame: 64-dimensional log-Mel vector
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from haiip.data.loaders.base import BaseDatasetLoader

logger = logging.getLogger(__name__)

MACHINE_TYPES = ["valve", "pump", "fan", "slide_rail"]
N_MELS = 64
N_FFT = 1024
HOP_LENGTH = 512
SAMPLE_RATE = 16000
FRAMES_PER_CLIP = 5  # context window for feature vector

FEATURE_COLS = [f"mel_{i:03d}" for i in range(N_MELS * FRAMES_PER_CLIP)]


class MIMIILoader(BaseDatasetLoader):
    """MIMII Industrial Sound Anomaly Detection loader.

    Extracts log-Mel features from .wav files if present.
    Generates synthetic log-Mel features for dev/test if not.

    Training: only normal data (unsupervised / one-class)
    Evaluation: normal + anomalous data
    """

    def __init__(
        self,
        cache_dir: Path | str | None = None,
        machine_type: str = "pump",
        model_id: str = "id_00",
        snr_db: int = 6,
    ) -> None:
        super().__init__(cache_dir)
        if machine_type not in MACHINE_TYPES:
            raise ValueError(f"machine_type must be one of {MACHINE_TYPES}")
        self.machine_type = machine_type
        self.model_id = model_id
        self.snr_db = snr_db
        self._cache_file = self.cache_dir / f"mimii_{machine_type}_{model_id}.parquet"

    def load(self) -> pd.DataFrame:
        if self._cache_file.exists():
            logger.info("MIMII: loading from cache %s", self._cache_file)
            return pd.read_parquet(self._cache_file)

        wav_dir = self.cache_dir / self.machine_type / self.model_id
        if wav_dir.exists() and list(wav_dir.glob("**/*.wav")):
            return self._load_from_wav(wav_dir)

        logger.warning("MIMII .wav files not found — using synthetic fallback")
        return self._synthetic_fallback()

    def _load_from_wav(self, wav_dir: Path) -> pd.DataFrame:
        try:
            import librosa
        except ImportError:
            logger.warning("librosa not installed — using synthetic fallback")
            return self._synthetic_fallback()

        records: list[dict[str, Any]] = []
        for wav_path in sorted(wav_dir.glob("**/*.wav")):
            is_anomaly = "abnormal" in wav_path.parts or "anomaly" in wav_path.name
            try:
                y, sr = librosa.load(str(wav_path), sr=SAMPLE_RATE, mono=True)
                mel = librosa.feature.melspectrogram(
                    y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS
                )
                log_mel = librosa.power_to_db(mel, ref=np.max)  # (64, T)

                # Slide window over time frames
                for t in range(log_mel.shape[1] - FRAMES_PER_CLIP):
                    frame = log_mel[:, t : t + FRAMES_PER_CLIP].flatten()
                    records.append({
                        **{f"mel_{i:03d}": float(v) for i, v in enumerate(frame)},
                        "label": "anomaly" if is_anomaly else "normal",
                        "source_file": wav_path.name,
                    })
            except Exception as exc:
                logger.warning("Failed to process %s: %s", wav_path, exc)

        df = pd.DataFrame(records)
        df.to_parquet(self._cache_file, index=False)
        logger.info("MIMII: loaded %d frames from %s", len(df), wav_dir)
        return df

    def _synthetic_fallback(self, n_samples: int = 2000, seed: int = 42) -> pd.DataFrame:
        """Synthetic log-Mel features matching MIMII statistics.

        Normal: lower energy, narrower spectral spread
        Anomaly: higher energy peaks, broader spread
        """
        rng = np.random.default_rng(seed)
        n_normal = int(n_samples * 0.9)
        n_anomaly = n_samples - n_normal
        n_features = N_MELS * FRAMES_PER_CLIP

        normal = rng.normal(-40.0, 8.0, (n_normal, n_features))
        anomaly = rng.normal(-30.0, 15.0, (n_anomaly, n_features))

        records = []
        for row in normal:
            records.append({
                **{f"mel_{i:03d}": round(float(v), 4) for i, v in enumerate(row)},
                "label": "normal",
                "source_file": "synthetic",
            })
        for row in anomaly:
            records.append({
                **{f"mel_{i:03d}": round(float(v), 4) for i, v in enumerate(row)},
                "label": "anomaly",
                "source_file": "synthetic",
            })

        df = pd.DataFrame(records).sample(frac=1, random_state=seed).reset_index(drop=True)
        logger.info("MIMII: synthetic generated (%d rows)", len(df))
        return df

    def get_normal_only(self) -> pd.DataFrame:
        """Return only normal samples — for unsupervised/one-class training."""
        if self._df is None:
            self._df = self.load()
        return self._df[self._df["label"] == "normal"].reset_index(drop=True)

    @property
    def feature_columns(self) -> list[str]:
        return FEATURE_COLS

    @property
    def label_column(self) -> str:
        return "label"
