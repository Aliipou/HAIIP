# HAIIP — Model Architecture & Design Decisions

## Model Selection Rationale

### Why IsolationForest for anomaly detection (baseline)?

Industrial sensor data is predominantly unlabelled. Failure events are rare (< 1% of readings).
IsolationForest is well-suited because:
- Unsupervised: no labels required
- O(n log n) training, O(log n) inference — fits real-time edge constraints
- Contamination parameter maps directly to expected anomaly rate (set to 0.05 by default)
- Interpretable: path length in tree ensemble is a proxy for normality

**Trade-off**: IsolationForest cannot model temporal dependencies. A gradual bearing degradation
spanning 48 hours will be missed if individual readings look normal.

### Why LSTM Autoencoder for deep anomaly detection?

Addresses the temporal blind spot above:
- Encoder-decoder reconstructs a sliding window (seq_len=10 readings)
- Reconstruction error > 95th percentile of training errors = anomaly
- Captures multi-variate correlations across time (e.g. torque rises as tool wears)
- Trained with PyTorch Lightning for reproducible, checkpointed training

**Trade-off**: Requires ~1,000+ healthy sequences to learn a reliable baseline.
Deployment on new machines requires 7–14 days of warm-up data collection.

### Why Bidirectional LSTM for RUL estimation?

Remaining Useful Life (RUL) depends on past degradation trajectory AND the rate of change:
- Bidirectional processing captures both forward and backward context in the window
- Dual head: classification (failure / no-failure) + regression (RUL in cycles via Softplus)
- Softplus on RUL head guarantees non-negative output without hard clipping

**Trade-off**: BiLSTM is ~2× the parameters of a unidirectional LSTM. On constrained edge
hardware (Raspberry Pi 4), inference takes ~18ms vs ~9ms. Within the 50ms SLA either way.

---

## Hyperparameters

### AnomalyAutoencoder

| Parameter | Value | Justification |
|---|---|---|
| `n_features` | 5 (default) | AI4I 2020 dataset: temperature×2, speed, torque, wear |
| `seq_len` | 10 | 10 readings at 1Hz = 10s window; captures single-cycle anomalies |
| `hidden_size` | 64 | Ablation: 32 underfit, 128 gave no F1 gain on AI4I validation |
| `n_layers` | 2 | Stacked LSTM; 3+ layers showed overfitting on small datasets |
| `threshold_pct` | 95.0 | P95 of training reconstruction error; 1-in-20 false positive rate |
| `max_epochs` | 30 | Early stopping at val_loss plateau (patience=5); typically stops at 12–18 |
| `batch_size` | 32 | Stable gradient on datasets of 1k–50k sequences |
| `learning_rate` | 1e-3 | Adam; cosine annealing LR scheduler |

### MaintenanceLSTM

| Parameter | Value | Justification |
|---|---|---|
| `hidden_size` | 128 | Larger than autoencoder — dual-task learning needs more capacity |
| `n_layers` | 2 | Same as autoencoder |
| `dropout` | 0.2 | Applied between LSTM layers; reduces overfitting on CMAPSS (8k sequences) |
| `seq_len` | 10 | Aligned with autoencoder for unified preprocessing pipeline |

### IsolationForest (sklearn baseline)

| Parameter | Value | Justification |
|---|---|---|
| `n_estimators` | 100 | Standard; diminishing returns beyond 150 on this dataset size |
| `contamination` | 0.05 | ~5% anomaly rate observed in AI4I 2020 |
| `max_samples` | `auto` | `min(256, n_samples)`; fast training, good coverage |
| `random_state` | 42 | Reproducibility |

### AutoRetrainPipeline

| Parameter | Value | Justification |
|---|---|---|
| `drift_feature_threshold` | 2 | Retrain if ≥ 2 features have PSI > 0.2 (industry standard for significant drift) |
| `accuracy_threshold` | 0.80 | Below 80% operator-confirmed accuracy triggers retraining |
| `cooldown_minutes` | 60 | Prevents retrain storms after a single bad data batch |
| `min_improvement` | 0.01 | Challenger must beat champion by ≥ 1% F1 to be promoted |
| `auc_tolerance` | 0.02 | Challenger AUC must not regress more than 2% vs champion |

---

## Evaluation Results (AI4I 2020 Predictive Maintenance Dataset)

Dataset: 10,000 samples, 339 failures (~3.4%), 5 failure modes.
Split: 70% train, 15% val, 15% test. Stratified by failure mode.

### Anomaly Detection

| Model | F1-macro | AUC-ROC | Precision | Recall | Inference (p99) |
|---|---|---|---|---|---|
| IsolationForest (baseline) | 0.78 | 0.84 | 0.81 | 0.76 | < 1ms |
| AnomalyAutoencoder (LSTM) | 0.86 | 0.91 | 0.88 | 0.84 | 12ms (CPU) |
| ONNX AnomalyAutoencoder | 0.86 | 0.91 | 0.88 | 0.84 | 8ms (CPU) |

### RUL Estimation (CMAPSS FD001 subset)

| Model | RMSE (cycles) | MAE (cycles) | Failure F1 |
|---|---|---|---|
| GradientBoosting (baseline) | 28.4 | 19.1 | 0.82 |
| MaintenanceLSTM (BiLSTM) | 18.7 | 12.3 | 0.89 |
| ONNX MaintenanceLSTM | 18.7 | 12.3 | 0.89 |

### Champion-Challenger Promotion History (simulated 30-day run)

| Round | Champion F1 | Challenger F1 | Promoted? | Trigger |
|---|---|---|---|---|
| 1 | — | 0.83 | Yes (first model) | manual |
| 2 | 0.83 | 0.86 | Yes (+3.6%) | drift_critical |
| 3 | 0.86 | 0.87 | Yes (+1.2%) | scheduled |
| 4 | 0.87 | 0.87 | No (< 1% gain) | scheduled |

---

## Trade-offs Summary

| Decision | Benefit | Cost |
|---|---|---|
| ONNX export for edge | ≤ 50ms inference, no Python runtime | Opset 17 required; dynamic shapes need careful testing |
| LSTM over Transformer | Fewer parameters, faster on CPU | Cannot attend to distant history beyond seq_len |
| Multi-tenant isolation | No cross-tenant data leakage | Each tenant needs its own retraining cycle |
| Unsupervised baseline | Works day-1 with zero labels | Lower F1 than supervised; needs operator feedback to improve |
| FedAvg federated learning | Data sovereignty, more training data | Communication rounds add hours; assumes IID data |
