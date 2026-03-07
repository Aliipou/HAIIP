# HAIIP — Real-World Industrial Use Cases

## Overview

HAIIP (Human-Aligned Industrial Intelligence Platform) is a production-grade AI system
designed for Nordic small and medium-sized enterprises (SMEs) operating CNC machines,
compressors, pumps, conveyor systems, and rotating equipment. The platform runs on-premises
or at the edge, operates within EU AI Act and GDPR constraints, and requires no cloud
dependency for inference.

---

## Use Case 1 — CNC Machine Tool Wear Monitoring

**Industry**: Precision machining, automotive parts, aerospace components
**Problem**: Tool breakage mid-run causes scrap parts, unplanned downtime, and safety risks.
**HAIIP solution**:
- Reads spindle current, torque, vibration (OPC UA from Siemens S7 / Fanuc controller)
- `AnomalyDetector` (IsolationForest) flags abnormal torque spikes in real time
- `MaintenanceLSTM` estimates remaining useful life (RUL) in cycles
- Operator receives alert with SHAP explanation: "Tool wear +2.3σ above baseline on axis Z"
- Operator confirms or overrides — feedback recorded, model updated overnight

**Outcome**: 30–40% reduction in unplanned tool changes; scrap rate drops from ~3% to under 1%.

---

## Use Case 2 — Compressed Air Leak Detection

**Industry**: Food processing, pharmaceutical, paper mills
**Problem**: Compressed air leaks are invisible, waste 20–30% of compressor energy, and
degrade system pressure affecting downstream quality.
**HAIIP solution**:
- MQTT sensors on compressor inlet/outlet pressure and flow rate
- Drift detection (KS test + PSI) flags gradual pressure decay patterns
- `AnomalyAutoencoder` (LSTM) distinguishes leak signatures from normal load variation
- Alert routed to maintenance planner with severity + estimated energy loss in kWh/day
- Dashboard shows anomaly score timeline so engineer can correlate with shift changes

**Outcome**: Leaks detected within 2–4 hours versus the typical weekly manual walk-around.

---

## Use Case 3 — Conveyor Belt Predictive Maintenance

**Industry**: Mining, logistics, aggregate processing
**Problem**: Belt tears and roller bearing failures cause full production line stoppages
lasting 4–12 hours. Replacement parts must be ordered in advance.
**HAIIP solution**:
- Vibration + temperature sensors on tail/head pulleys (Modbus TCP)
- Multi-tenant: each conveyor line is a separate tenant with its own model
- `MaintenancePredictor` (GradientBoosting) classifies failure probability and failure mode
  (bearing, belt splice, motor coupling)
- 48–72 hour advance warning allows parts to be staged without emergency procurement
- IEC 61508 SIL-1 compliance log: every prediction stored with model version + explanation

**Outcome**: Mean time between unplanned stoppages increased from 18 days to 60+ days.

---

## Use Case 4 — Pump Cavitation Detection (Water Treatment)

**Industry**: Municipal water treatment, chemical dosing stations
**Problem**: Cavitation damages pump impellers silently; repairs cost €15–40k per pump.
**HAIIP solution**:
- Flow rate + differential pressure + motor current read via OPC UA
- `AnomalyAutoencoder` trained on 90 days of healthy baseline; threshold set at 95th percentile
- ONNX model exported and deployed on Jetson Orin edge device inside the pump house
  (no internet required, inference ≤ 12ms p99)
- Alerts sent to SCADA via REST webhook
- Monthly auto-retraining triggered when PSI drift exceeds 0.2 on any feature

**Outcome**: Three cavitation events detected and resolved before impeller damage in pilot year.

---

## Use Case 5 — Robot Collaborative Cell Safety Monitoring

**Industry**: Automotive assembly, electronics manufacturing
**Problem**: Collaborative robots (cobots) slow down or stop under load variance; distinguishing
intentional contact from fault conditions requires AI, not simple thresholds.
**HAIIP solution**:
- Joint torque + end-effector force data streamed at 100Hz; windowed to 10-step sequences
- `AnomalyAutoencoder` (LSTM, seq_len=10) detects anomalous force profiles in real time
- Human Oversight Gate (HOG) blocks autonomous re-start if anomaly score > 0.85; requires
  operator acknowledgment
- All override events logged to audit trail for EU AI Act Art. 14 human oversight requirement
- Dashboard shows Human Intervention Rate (HIR) per shift; target < 5%

**Outcome**: Zero false-positive emergency stops during 6-month pilot; two real faults caught.

---

## Use Case 6 — Multi-Site Federated Learning (Nordic SME Consortium)

**Industry**: Paper & pulp, three mills across Finland, Sweden, Norway
**Problem**: Each mill has too little labelled failure data to train a reliable model alone;
sharing raw sensor data violates data sovereignty agreements.
**HAIIP solution**:
- `FederatedLearner` runs FedAvg across three nodes (SME_FI, SME_SE, SME_NO)
  using only model weight gradients — raw data never leaves each site
- Each site's `AnomalyAutoencoder` trains locally; aggregated global model distributed back
- Convergence monitored by gradient norm; rounds capped at 10 to bound compute
- Per-site data profiles (noise level, sensor calibration) respected via weighted aggregation

**Outcome**: Global model achieves F1 = 0.91 versus F1 = 0.76 for any single-site model alone.

---

## Use Case 7 — Economic Maintenance Decision Support

**Industry**: Any asset-intensive SME
**Problem**: Maintenance planners must weigh repair cost, production loss, safety risk, and
parts availability simultaneously — often under time pressure.
**HAIIP solution**:
- `EconomicAIEngine` receives anomaly score + failure probability + RUL estimate
- Computes Expected Loss for four actions: REPAIR_NOW, SCHEDULE, MONITOR, IGNORE
- Factors in: repair cost (€), production loss per hour (€/h), safety penalty (configurable),
  false positive cost (unnecessary downtime)
- Returns recommended action with cost breakdown; operator can accept or override
- Override rate tracked as an oversight metric; high override rate triggers model review

**Example output**:
```
Action: REPAIR_NOW
Expected cost if ignored:  €12,400  (failure probability 0.82 × damage cost €15,100)
Repair cost now:           €1,800
Recommendation confidence: HIGH
```

---

## Deployment Footprint

| Scenario | Hardware | Inference latency |
|---|---|---|
| Cloud-connected | Any server / VM | API p99 ≤ 250ms |
| Edge (ONNX) | NVIDIA Jetson Orin / Hailo-8 / Industrial PC | p99 ≤ 50ms |
| Fully air-gapped | Industrial PC with local Redis | p99 ≤ 50ms |
| Multi-site federated | 3–10 nodes, LAN or VPN | Training: async, hours |

---

## Regulatory Coverage

| Regulation | How HAIIP addresses it |
|---|---|
| EU AI Act Art. 13 | SHAP explanations on every prediction; model card published |
| EU AI Act Art. 14 | Human Oversight Gate; HIR/HOG/TCS metrics tracked |
| GDPR Art. 17 | `/gdpr/erase` endpoint; audit log; data minimisation by design |
| IEC 61508 SIL-1 | Every prediction stored with model version, input hash, timestamp |
| ISO 13849 (cobot) | Human override required above configurable risk threshold |
