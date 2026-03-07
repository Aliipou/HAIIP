"""Tests for data loaders — AI4I, CWRU, CMAPSS, MIMII."""

import numpy as np
import pytest

from haiip.data.loaders.ai4i import AI4ILoader
from haiip.data.loaders.cmapss import CMAPSSLoader
from haiip.data.loaders.cwru import CWRULoader
from haiip.data.loaders.mimii import MIMIILoader

# ── AI4I Loader ───────────────────────────────────────────────────────────────


class TestAI4ILoader:
    def test_synthetic_fallback_shape(self, tmp_path):
        loader = AI4ILoader(cache_dir=tmp_path)
        df = loader._synthetic_fallback(n_samples=500)
        assert len(df) == 500
        assert "air_temperature" in df.columns
        assert "failure_type" in df.columns

    def test_synthetic_has_all_failure_modes(self, tmp_path):
        loader = AI4ILoader(cache_dir=tmp_path)
        df = loader._synthetic_fallback(n_samples=2000)
        assert "no_failure" in df["failure_type"].values
        assert "TWF" in df["failure_type"].values

    def test_load_uses_fallback_when_no_network(self, tmp_path):
        loader = AI4ILoader(cache_dir=tmp_path)
        df = loader.load()
        assert len(df) > 0
        assert set(loader.feature_columns).issubset(df.columns)

    def test_get_X_y_shapes(self, tmp_path):
        loader = AI4ILoader(cache_dir=tmp_path)
        X, y = loader.get_X_y()
        assert X.ndim == 2
        assert X.shape[1] == len(loader.feature_columns)
        assert len(X) == len(y)

    def test_get_normal_data(self, tmp_path):
        loader = AI4ILoader(cache_dir=tmp_path)
        normal_df = loader.get_normal_data()
        assert all(normal_df["failure_type"] == "no_failure")
        assert len(normal_df) > 0

    def test_preprocess_removes_negatives(self, tmp_path):
        loader = AI4ILoader(cache_dir=tmp_path)
        df = loader._synthetic_fallback(n_samples=100)
        df.loc[0, "rotational_speed"] = -1.0  # inject invalid
        cleaned = loader.preprocess(df)
        assert (cleaned["rotational_speed"] > 0).all()

    def test_train_test_split(self, tmp_path):
        loader = AI4ILoader(cache_dir=tmp_path)
        X_tr, X_te, y_tr, y_te = loader.get_train_test_split(test_size=0.2)
        total = len(X_tr) + len(X_te)
        assert abs(len(X_te) / total - 0.2) < 0.05

    def test_feature_column_property(self, tmp_path):
        loader = AI4ILoader(cache_dir=tmp_path)
        assert "air_temperature" in loader.feature_columns
        assert len(loader.feature_columns) == 5

    def test_info_returns_dict(self, tmp_path):
        loader = AI4ILoader(cache_dir=tmp_path)
        info = loader.info()
        assert "n_samples" in info
        assert "n_features" in info


# ── CWRU Loader ───────────────────────────────────────────────────────────────


class TestCWRULoader:
    def test_invalid_sample_rate(self, tmp_path):
        with pytest.raises(ValueError, match="sample_rate"):
            CWRULoader(cache_dir=tmp_path, sample_rate=999)

    def test_synthetic_fallback_shape(self, tmp_path):
        loader = CWRULoader(cache_dir=tmp_path)
        df = loader._synthetic_fallback(n_windows=600)
        assert len(df) > 0
        assert "rms" in df.columns
        assert "kurtosis" in df.columns
        assert "label" in df.columns

    def test_synthetic_has_all_labels(self, tmp_path):
        loader = CWRULoader(cache_dir=tmp_path)
        df = loader._synthetic_fallback(n_windows=1200)
        assert "normal" in df["label"].values
        assert "inner_race" in df["label"].values

    def test_load_returns_dataframe(self, tmp_path):
        loader = CWRULoader(cache_dir=tmp_path)
        df = loader.load()
        assert len(df) > 0

    def test_kurtosis_normal_signal(self, tmp_path):
        loader = CWRULoader(cache_dir=tmp_path)
        signal = np.random.default_rng(0).normal(0, 1, 1000)
        k = loader._kurtosis(signal)
        assert 2.0 < k < 5.0  # normal kurtosis ~3

    def test_kurtosis_impulsive_signal(self, tmp_path):
        loader = CWRULoader(cache_dir=tmp_path)
        signal = np.zeros(1000)
        signal[50] = 100.0  # impulse
        k = loader._kurtosis(signal)
        assert k > 10.0  # impulsive signals have high kurtosis

    def test_feature_columns(self, tmp_path):
        loader = CWRULoader(cache_dir=tmp_path)
        assert "rms" in loader.feature_columns
        assert "kurtosis" in loader.feature_columns

    def test_infer_label_normal(self, tmp_path):
        loader = CWRULoader(cache_dir=tmp_path)
        assert loader._infer_label("normal_baseline_1797") == "normal"
        assert loader._infer_label("IR021_1797_0") == "inner_race"


# ── CMAPSS Loader ─────────────────────────────────────────────────────────────


class TestCMAPSSLoader:
    def test_invalid_subset(self, tmp_path):
        with pytest.raises(ValueError, match="subset"):
            CMAPSSLoader(cache_dir=tmp_path, subset="FD999")

    def test_synthetic_fallback(self, tmp_path):
        loader = CMAPSSLoader(cache_dir=tmp_path)
        df = loader._synthetic_fallback(n_units=10)
        assert len(df) > 0
        assert "rul" in df.columns
        assert "unit_id" in df.columns
        assert "cycle" in df.columns

    def test_rul_non_negative(self, tmp_path):
        loader = CMAPSSLoader(cache_dir=tmp_path)
        df = loader.load()
        assert (df["rul"] >= 0).all()

    def test_preprocess_clips_rul(self, tmp_path):
        loader = CMAPSSLoader(cache_dir=tmp_path, max_rul=125)
        df = loader.load()
        df_proc = loader.preprocess(df)
        assert (df_proc["rul"] <= 125).all()

    def test_get_sequences_shape(self, tmp_path):
        loader = CMAPSSLoader(cache_dir=tmp_path)
        X, y = loader.get_sequences(seq_len=10)
        assert X.ndim == 3
        assert X.shape[1] == 10  # seq_len
        assert len(X) == len(y)

    def test_label_column(self, tmp_path):
        loader = CMAPSSLoader(cache_dir=tmp_path)
        assert loader.label_column == "rul"


# ── MIMII Loader ──────────────────────────────────────────────────────────────


class TestMIMIILoader:
    def test_invalid_machine_type(self, tmp_path):
        with pytest.raises(ValueError, match="machine_type"):
            MIMIILoader(cache_dir=tmp_path, machine_type="invalid_machine")

    def test_synthetic_fallback(self, tmp_path):
        loader = MIMIILoader(cache_dir=tmp_path)
        df = loader._synthetic_fallback(n_samples=200)
        assert len(df) == 200
        assert "label" in df.columns
        assert "mel_000" in df.columns

    def test_normal_abnormal_ratio(self, tmp_path):
        loader = MIMIILoader(cache_dir=tmp_path)
        df = loader._synthetic_fallback(n_samples=1000)
        normal_count = (df["label"] == "normal").sum()
        assert normal_count > 800  # 90% normal

    def test_get_normal_only(self, tmp_path):
        loader = MIMIILoader(cache_dir=tmp_path)
        loader._df = loader._synthetic_fallback(n_samples=500)
        normal_df = loader.get_normal_only()
        assert all(normal_df["label"] == "normal")

    def test_feature_columns_count(self, tmp_path):
        loader = MIMIILoader(cache_dir=tmp_path)
        from haiip.data.loaders.mimii import FRAMES_PER_CLIP, N_MELS

        assert len(loader.feature_columns) == N_MELS * FRAMES_PER_CLIP

    def test_load_returns_dataframe(self, tmp_path):
        loader = MIMIILoader(cache_dir=tmp_path)
        df = loader.load()
        assert len(df) > 0
