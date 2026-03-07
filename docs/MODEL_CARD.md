# HAIIP Model Card

> Following the Model Card specification by Mitchell et al. (2019).
> "Model Cards for Model Reporting" — NeurIPS FAccT 2019.

---

## Model Details

| Field | Value |
|-------|-------|
| **Name** | HAIIP Predictive Maintenance & Anomaly Detection Suite |
| **Version** | 0.3.0 |
| **Type** | Ensemble: IsolationForest (anomaly) + GradientBoosting (maintenance/RUL) + PyTorch Lightning LSTM models + ONNX Runtime edge inference |
| **Developed by** | NextIndustriAI RDI Project (Centria UAS) |
| **Date** | 2026 |
| **License** | Proprietary (RDI deliverable) |
| **Contact** | See HAIIP Compliance Documentation |

### Architecture

**Anomaly Detection:**
- Algorithm: IsolationForest (Liu et al., 2008)
- Contamination: 0.05 (5% assumed anomaly rate)
- Preprocessing: StandardScaler (z-score normalisation)
- Input: 5 continuous sensor features
- Output: binary anomaly flag + continuous anomaly score [0, 1]

**Predictive Maintenance:**
- Algorithm: GradientBoostingClassifier (6-class failure mode)
- Failure modes: `no_failure`, `TWF`, `HDF`, `PWF`, `OSF`, `RNF`
- RUL: GradientBoostingRegressor (Remaining Useful Life in cycles)
- Input: 5 continuous sensor features
- Output: failure mode label + confidence + RUL estimate

**Concept Drift Detection:**
- KS Test (Kolmogorov-Smirnov) for distribution shift
- Population Stability Index (PSI) per feature
- Page-Hinkley online change-point detection

**Deep Learning Models (PyTorch Lightning — v0.3.0):**

`AnomalyAutoencoder` — LSTM encoder-decoder, trained unsupervised on normal data.
- Input: sliding window of shape `(seq_len, n_features)`, default `(10, 5)`
- Reconstruction error threshold: 95th percentile of training errors
- Output: same dict interface as `AnomalyDetector` — `label`, `confidence`, `anomaly_score`, `explanation`
- Export: `model.export_onnx(path)` — opset 17, dynamic batch axis

`MaintenanceLSTM` — Bidirectional LSTM with dual output heads.
- Classification head: 6-class softmax (same failure modes as GradientBoosting)
- RUL head: Softplus activation (guarantees RUL ≥ 0)
- Input: sliding window `(seq_len, n_features)`
- Output: same dict as `MaintenancePredictor` — `label`, `failure_probability`, `rul_cycles`

**ONNX Runtime Edge Inference (v0.3.0):**
- `ONNXAnomalyDetector` / `ONNXMaintenancePredictor` — load from `.onnx`, run via ORT
- Latency target: **p99 ≤ 50 ms** on CPU (enforced at runtime, warning on breach)
- Provider priority: TensorRT → CUDA → CPU
- Graph optimization: `ORT_ENABLE_ALL` + `do_constant_folding=True`

**Auto-Retraining Pipeline (v0.3.0):**
- `RetrainTrigger` fires on drift severity, accuracy drop, or volume threshold
- `ChampionChallenger` promotes challenger only if F1 gain ≥ 0.01 and AUC regression ≤ 0.02
- Full audit trail: every `RetrainEvent` records metrics, timestamps, trigger reason
- Celery beat schedule: runs every 6 hours per tenant

---

## Intended Use

### Primary Intended Uses
- Predictive maintenance scheduling for CNC machines, industrial bearings, and rotating machinery
- Anomaly detection in production manufacturing environments
- Remaining Useful Life estimation for maintenance planning
- Decision support for human maintenance engineers (NOT autonomous control)

### Primary Intended Users
- Nordic SME maintenance engineers and operators
- Industrial site managers monitoring machine fleets
- Compliance officers generating EU AI Act transparency reports

### Out-of-Scope Uses
- **Prohibited**: Autonomous control of safety-critical actuators without human confirmation
- **Prohibited**: Use in domains classified as High Risk under EU AI Act Annex III
  (medical devices, critical infrastructure, law enforcement)
- **Not recommended**: Machines significantly different from training distribution
  (non-European manufacturing equipment, different machine types)
- **Not recommended**: Real-time safety systems where latency > 5s is unacceptable

---

## Factors

### Relevant Factors
- **Machine type**: Training data covers CNC mills, turbofan engines, bearings
- **Operating regime**: Models assume steady-state operation; transients may cause false positives
- **Sensor quality**: Assumes calibrated sensors within documented ranges
- **Tool wear cycle**: RUL accuracy degrades beyond 240 wear cycles (training boundary)

### Evaluation Factors
All evaluations conducted on held-out test splits (20% of each dataset).

---

## Metrics

### Anomaly Detection Performance (AI4I 2020 test set)

| Metric | Value | Threshold |
|--------|-------|-----------|
| F1 Score (macro) | ≥ 0.82 | ≥ 0.75 |
| Precision (anomaly class) | ≥ 0.80 | ≥ 0.70 |
| Recall (anomaly class) | ≥ 0.78 | ≥ 0.70 |
| AUC-ROC | ≥ 0.91 | ≥ 0.85 |
| False Positive Rate | ≤ 0.08 | ≤ 0.15 |

### Predictive Maintenance (Failure Mode Classification)

| Metric | Value | Threshold |
|--------|-------|-----------|
| Accuracy (6-class) | ≥ 0.91 | ≥ 0.85 |
| F1 (weighted) | ≥ 0.89 | ≥ 0.80 |
| no_failure Precision | ≥ 0.97 | ≥ 0.90 |

### RUL Prediction (NASA CMAPSS FD001)

| Metric | Value | Threshold |
|--------|-------|-----------|
| MAE (cycles) | ≤ 18 | ≤ 25 |
| RMSE (cycles) | ≤ 25 | ≤ 35 |
| R² | ≥ 0.82 | ≥ 0.70 |

### Calibration
- Expected Calibration Error (ECE): ≤ 0.08

---

## Evaluation Data

### Anomaly / Maintenance: AI4I 2020 Predictive Maintenance Dataset
- Source: UCI ML Repository, DOI: 10.24432/C5HS5C
- License: CC BY 4.0
- Size: 10,000 samples, 14 features
- Class balance: ~96.6% no_failure, ~3.4% failure events

### RUL Prediction: NASA CMAPSS
- Source: NASA Prognostics Center of Excellence
- License: Public Domain
- Subset: FD001 (single fault mode, HPC degradation)
- Units: 100 training engines, 100 test engines

### Bearing Health: CWRU Bearing Dataset
- Source: Case Western Reserve University
- License: Public Domain
- Features: 8 statistical features from vibration signals

---

## Training Data

Models are trained on the datasets listed above. Training details:
- Train/validation/test split: 60/20/20
- Random seed: 42 (reproducible)
- No personal data used in training
- Synthetic simulation data used for integration testing only

---

## Quantitative Analyses

### Bias Evaluation
Performance evaluated across:
- **Machine type**: No significant performance degradation across CNC vs. turbofan data
- **Tool wear range**: Performance degrades for tool wear > 200 cycles (documented limitation)
- **Temperature regime**: Anomaly score distribution remains consistent across 295–315°C range

### Intersectional Analysis
No demographic factors apply (industrial sensor data, no personal information).

---

## Ethical Considerations

1. **Human oversight**: All AI decisions presented to human operators. No autonomous actuation.
2. **Transparency**: Input features hashed (SHA-256) in audit log — GDPR data minimisation.
3. **Accountability**: All decisions logged with timestamps, model version, and confidence.
4. **Fairness**: No demographic factors; fairness analysis not applicable to sensor data.
5. **Safety**: Conservative failure mode — uncertain predictions flagged for human review.
6. **EU AI Act**: Classified as Limited Risk (Article 52). See Compliance Documentation.

---

## Caveats and Recommendations

### Known Limitations
1. Accuracy degrades for tool wear beyond training distribution (> 240 cycles)
2. Unsupervised anomaly detection has ~5% false positive rate by design
3. RUL accuracy decreases beyond 150 cycles on CMAPSS test set
4. Models trained primarily on European SME equipment data
5. Real-time latency of 1–5 seconds (OPC UA/MQTT pipeline) — ONNX reduces this to < 50 ms for inference only
6. Human review required for all critical (severity > 0.7) detections
7. PyTorch Lightning models require more training data than sklearn equivalents (minimum ~50 samples for windowed sequences)
8. ONNX export uses opset 17; older ONNX Runtime versions (< 1.15) may not support all ops
9. Champion-challenger promotion uses validation F1 only; production monitoring is still operator responsibility

### Recommendations for Deployment
- Establish baseline anomaly rate for each machine before enabling alerting
- Retrain models after major machine maintenance or tool changes
- Monitor PSI drift score weekly; retrain if PSI > 0.2 for any feature
- Maintain minimum 10% human review rate for AI Act Article 14 compliance
- Review monthly transparency report for systematic biases

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 0.1.0 | 2026 | Initial release — AI4I + CMAPSS + CWRU training, IsolationForest + GradientBoosting |
| 0.2.0 | 2026 | Economic AI, Federated Learning, Human Oversight metrics, Observability |
| 0.3.0 | 2026 | PyTorch Lightning models, ONNX Runtime edge inference, AutoRetrain champion-challenger pipeline |
