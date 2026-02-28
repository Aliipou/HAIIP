# HAIIP — Research Results Register

> **Status**: All results are *(pending)* until measured in CI on a clean checkout.
> A result is only recorded here after it is reproduced with the environment below.
> Do not add numbers to pending rows. Do not mark a row pending after adding a number.

---

## Environment

All results below are measured in this environment unless otherwise noted.

| Parameter | Value |
|-----------|-------|
| Python version | 3.11 |
| OS | Ubuntu 22.04 (GitHub Actions runner) |
| Hardware | GitHub-hosted runner (2 vCPU, 7 GB RAM) |
| Dataset — AI4I 2020 | v1.0.0, UCI ML Repository, CC BY 4.0 |
| Dataset — NASA CMAPSS | FD001 subset, NASA Prognostics COE |
| Dataset — CWRU Bearing | v1.0, Case Western Reserve U. |
| Date range | 2025-2026 |

---

## RQ1 — Anomaly Detection Accuracy

**Question**: Can IsolationForest + GradientBoosting achieve >= 85% F1 on Nordic SME vibration data?

| Metric | Target | Result | Dataset | Date |
|--------|--------|--------|---------|------|
| F1-macro (AnomalyDetector) | >= 0.85 | *(pending)* | AI4I 2020 | — |
| AUC-ROC (AnomalyDetector) | >= 0.91 | *(pending)* | AI4I 2020 | — |
| Accuracy (MaintenancePredictor) | >= 0.85 | *(pending)* | AI4I 2020 | — |
| MAE — RUL cycles | <= 25 | *(pending)* | NASA CMAPSS FD001 | — |
| R2 — RUL | >= 0.70 | *(pending)* | NASA CMAPSS FD001 | — |

**Notes**: Thresholds are design targets, not measured results. Numbers will be added after CI benchmark run.

---

## RQ2 — Human Oversight Rate

**Question**: What minimum HIR satisfies EU AI Act Article 14 without degrading throughput?

| Metric | Target | Result | Method | Date |
|--------|--------|--------|--------|------|
| Optimal HIR band | 0.05–0.15 | *(pending)* | Simulation + compliance analysis | — |
| Throughput degradation at HIR=0.15 | < 5% | *(pending)* | Load test | — |

**Notes**: HIR target is derived from EU AI Act Art. 14 interpretation. Field validation required.

---

## RQ3 — RAG vs Manual Lookup

**Question**: Does RAG-based document querying reduce time-to-answer vs manual lookup?

| Metric | Target | Result | Method | Date |
|--------|--------|--------|--------|------|
| RAG answer latency (p50) | < 30 s | *(pending)* | Integration test | — |
| Hallucination rate | < 5% | *(pending)* | test_rag_hallucination.py | — |
| Source citation rate | = 100% | *(pending)* | test_rag_hallucination.py | — |

**Notes**: Hallucination is measured via keyword-overlap heuristic, not LLM-as-judge.

---

## RQ4 — Privacy-Preserving Logging

**Question**: What is the SHA-256 hashing overhead in audit logs?

| Metric | Target | Result | Method | Date |
|--------|--------|--------|--------|------|
| Hash overhead per write | < 0.1 ms | *(pending)* | Microbenchmark | — |
| Re-identification risk | zero | *(pending)* | Security analysis | — |

---

## RQ5 — Economic AI (Experimental Branch)

**Question**: How much downtime cost does ELM avoid vs naive thresholding?

| Metric | Target | Result | Method | Date |
|--------|--------|--------|--------|------|
| False positive rate reduction | 30–50% | *(pending)* | Simulation on AI4I 2020 | — |
| Net benefit per avoided stop (default profile) | > EUR 0 | *(pending)* | economic_ai.py ELM | — |

**Notes**: Cost figures depend on SiteEconomicProfile defaults (Nordic manufacturing median).
Results will change at any real site. Run sensitivity_analysis() before citing EUR figures.

---

## RQ6 — Federated Learning (Experimental Branch)

**Question**: Can FedAvg achieve F1 within 15% of centralised baseline?

| Metric | Target | Result | Method | Date |
|--------|--------|--------|--------|------|
| federated_gap (IID split) | < 0.15 | *(pending)* | FederatedLearner.run() | — |
| federated_gap (non-IID) | < 0.15 | *(pending)* | RealisticFederatedScenario | — |
| federated_gap (non-IID + dropout) | documented | *(pending)* | RealisticFederatedScenario | — |
| Convergence round | <= 10 | *(pending)* | FederatedLearner.run() | — |

**Notes**: Non-IID results require RealisticFederatedScenario with AI4I 2020.
Gap under non-IID + dropout is expected to be larger than IID gap — document honestly.

---

## RQ7 — Human Oversight Quantification (Experimental Branch)

**Question**: What are the measurable HOG and TCS?

| Metric | Target | Result | Confidence | Date |
|--------|--------|--------|-----------|------|
| HOG | > 0.02 | *(pending)* | LOW (simulation) | — |
| TCS | >= 0.80 | *(pending)* | LOW (simulation) | — |
| ECE | < 0.10 | *(pending)* | LOW (simulation) | — |
| HIR | 0.05–0.15 | *(pending)* | LOW (simulation) | — |

**WARNING**: All metrics in this section are simulated. simulation_confidence=LOW.
field_study_required=True. Do not cite these numbers in external publications
without a field study with real operators.

---

## RQ8 — Per-Prediction ROI (Experimental Branch)

**Question**: What is compute cost per prediction vs avoided downtime at fleet scale?

| Metric | Target | Result | Environment | Date |
|--------|--------|--------|-------------|------|
| Cost per prediction | < EUR 0.001 | *(pending)* | AWS eu-north-1 t3.medium | — |
| Avoided cost per correct REPAIR_NOW | > EUR 1000 | *(pending)* | Default SiteEconomicProfile | — |
| ROI ratio at 10 machines | > 1000x | *(pending)* | Model estimate | — |

---

## How to Add a Result

1. Run the measurement in CI on a clean checkout with the environment above.
2. Add the measured value to the Result column.
3. Add the date.
4. Do NOT mark the row `*(pending)*` after adding a number.
5. If the environment differs from the table above, note it in the row.
