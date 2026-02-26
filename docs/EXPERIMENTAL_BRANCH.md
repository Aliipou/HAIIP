# HAIIP Experimental Branch — Phase 6 Research Features

> **Status**: Research-grade — validated, tested, not production-hardened.
> **Branch**: `main` (Phase 6 features are included in main after Phase 4/5 foundation)

---

## Overview

Phase 6 extends HAIIP with four research-grade features that push the platform
from a production-ready SME tool toward an RDI-grade academic contribution
suitable for peer-reviewed publication.

---

## Feature Comparison: Main vs Phase 6

| Capability | Main (Phase 1–5) | Phase 6 (Experimental) |
|------------|------------------|------------------------|
| Anomaly Detection | ✅ Production | ✅ Same |
| Predictive Maintenance | ✅ Production | ✅ Same |
| RAG Q&A | ✅ Production | ✅ Same |
| Agentic RAG | ✅ ReAct agent | ✅ Same |
| Economic Decision | ❌ | ✅ Expected Loss Minimization |
| Federated Learning | ❌ | ✅ FedAvg (3 Nordic nodes, simulated) |
| Human Oversight Metrics | Qualitative (Art. 14) | ✅ HIR / HOG / TCS quantified |
| OpenTelemetry Tracing | ❌ | ✅ Distributed tracing + SLA checks |
| Per-Prediction Cost Model | ❌ | ✅ Compute cost vs avoided downtime |
| Kubernetes | Docker only | ✅ EKS deployment |
| Helm Chart | ❌ | ✅ Production-ready chart |
| Terraform IaC | ❌ | ✅ AWS EKS + RDS + ElastiCache |
| Research Notebooks | ❌ | ✅ 2 reproducibility notebooks |

---

## Research Questions Addressed

### RQ5 — Economic AI
> How much downtime cost does the Expected Loss Minimization engine avoid
> compared to naive probability thresholding?

**Implementation**: `haiip/core/economic_ai.py`
**Test coverage**: `haiip/tests/core/test_economic_ai.py` (35+ tests)
**Notebook**: `notebooks/01_economic_decision.ipynb`

**Formula**:
```
E[Cost_wait]   = P(failure) × C_downtime × safety_factor
E[Cost_action] = P(no_failure) × C_false_positive + C_maintenance
Net_benefit    = E[Cost_wait] − E[Cost_action]
```

**Decision rules**:
- `REPAIR_NOW` : P(failure) ≥ 0.75
- `SCHEDULE`   : P(failure) ≥ 0.50
- `MONITOR`    : anomaly_score ≥ 0.20
- `IGNORE`     : below noise floor

---

### RQ6 — Federated Learning
> Can FedAvg across 3 Nordic SME nodes achieve F1 within 15% of the centralized
> baseline while preserving privacy?

**Implementation**: `haiip/core/federated.py`
**Test coverage**: `haiip/tests/core/test_federated.py` (20+ tests)
**Notebook**: `notebooks/02_federated_learning.ipynb`

**Nodes**:
| Node | Country | Industry | n_samples | failure_rate |
|------|---------|----------|-----------|--------------|
| SME_FI | Finland | Paper mill | 800 | 12% |
| SME_SE | Sweden | Automotive stamping | 1200 | 8% |
| SME_NO | Norway | Offshore pumps | 600 | 18% |

**Privacy guarantees**: Only weight deltas transmitted — no raw data leaves node boundary.

---

### RQ7 — Human Oversight Quantification
> What is the measurable Human Override Gain (HOG) and Trust Calibration Score (TCS)?

**Implementation**: `haiip/core/human_oversight.py`
**Test coverage**: `haiip/tests/core/test_human_oversight.py` (30+ tests)

**Metrics**:
| Metric | Formula | Target |
|--------|---------|--------|
| HIR (Human Intervention Rate) | `|reviewed| / |decisions|` | 0.05–0.15 |
| HOG (Human Override Gain) | `F1(corrected) − F1(ai_only)` | > 0.02 |
| TCS (Trust Calibration Score) | `1 − ECE` | ≥ 0.80 |
| ECE (Expected Calibration Error) | Guo et al. (2017) | < 0.10 |

---

### RQ8 — Per-Prediction Cost Model
> What is the compute cost per prediction vs avoided downtime at fleet scale?

**Implementation**: `haiip/observability/cost_model.py`
**Test coverage**: `haiip/tests/core/test_observability.py`

**At SME defaults (AWS eu-north-1)**:
- Inference cost: ~€0.000007 per prediction (t3.medium)
- Avoided downtime per correct REPAIR_NOW: €4,000+
- ROI ratio: >10,000× for high-accuracy models

---

## Limitations and Threats to Validity

1. **Federated simulation**: Federation is simulated in-process (no actual network). Real-world
   communication delays, Byzantine failures, and gradient attacks are not modelled.

2. **Synthetic non-IID data**: Node data is synthesised from Gaussian distributions with
   configurable shift. Real SME sensor data may have different statistical properties.

3. **Economic parameters**: Default cost parameters (€500/hr, 8h MTTR) are representative
   Nordic SME estimates — actual values vary by site and must be calibrated per deployment.

4. **Oversight simulation**: HOG/TCS metrics are computed on simulated human decisions,
   not real operator data. A longitudinal field study is required for validation.

---

## How to Run Phase 6 Features

```bash
# Economic decision
python -c "
from haiip.core.economic_ai import EconomicDecisionEngine
engine = EconomicDecisionEngine()
d = engine.decide(anomaly_score=0.85, failure_probability=0.80)
print(d.action, d.net_benefit)
"

# Federated learning
python -c "
from haiip.core.federated import FederatedLearner
result = FederatedLearner().run(n_rounds=5, local_epochs=2)
print(f'F1: {result.final_global_f1:.4f}, gap: {result.federated_gap:+.4f}')
"

# Human oversight
python -c "
from haiip.core.human_oversight import HumanOversightEngine, OversightEvent
eng = HumanOversightEngine()
eng.record(OversightEvent.create('d1','failure',0.8,'normal',True,True,'normal'))
m = eng.compute_metrics()
print(f'HIR={m.hir:.2f}, HOG={m.hog:+.4f}, TCS={m.tcs:.4f}')
"

# Notebooks
jupyter lab notebooks/
```

---

## Citation

If using Phase 6 experimental features in research, please cite:

```bibtex
@techreport{haiip2025,
  title  = {HAIIP: Human-Aligned Industrial Intelligence Platform},
  author = {NextIndustriAI Team},
  year   = {2025},
  institution = {Centria University of Applied Sciences},
  note   = {RDI deliverable — NextIndustriAI project (Jakobstad/Sundsvall/Narvik)}
}
```
