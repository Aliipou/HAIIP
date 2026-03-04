# HAIIP — Novel Contributions Statement

> Use this document when writing the Introduction of any paper submission.
> Do not claim a contribution unless it is supported by code + tests in this repo.
> Do not soften or expand claims without updating the supporting code.

---

## C1 — Expected Loss Minimization Decision Engine

**Claim**: We introduce a four-action decision engine that replaces naive probability
thresholding with site-calibrated economic reasoning for industrial predictive maintenance.

**What is novel**:
- The four-action decision space (REPAIR_NOW / SCHEDULE / MONITOR / IGNORE) is not
  new, but the combination of site-calibrated ELM with a structured calibration
  interview and a sensitivity analysis is not present in prior work (verified by
  literature search, March 2026).
- The calibration interview maps 15 minutes of operator input to all monetary
  parameters, making ELM deployable by non-economists.
- `sensitivity_analysis()` identifies which parameters drive the decision, enabling
  focused calibration effort.

**Supported by**:
- `haiip/core/economic_ai.py` — EconomicDecisionEngine
- `haiip/core/economic_calibration.py` — SiteEconomicProfile, calibration_interview()
- `haiip/tests/core/test_economic_ai.py` (35+ tests)
- `haiip/tests/core/test_economic_calibration.py` (17+ tests)
- `notebooks/01_economic_decision.ipynb`

**Limitation**: EUR figures depend on SiteEconomicProfile defaults. Any paper claiming
EUR savings must show calibrated profile results or explicitly label values as "using
Nordic manufacturing median defaults."

---

## C2 — Quantifiable Human Oversight Metrics for EU AI Act Article 14

**Claim**: We define and implement three computable human oversight metrics — HIR, HOG,
and TCS — that provide a quantitative answer to EU AI Act Article 14's oversight
requirement, and we show that a HIR of 5–15% satisfies regulatory thresholds without
degrading operational throughput.

**What is novel**:
- HIR, HOG, and TCS are not defined in prior industrial AI literature (verified by
  search of IEEE TII, EAAI, and arXiv, March 2026).
- The simulation confidence tier system (HIGH / MEDIUM / LOW / VERY LOW / NONE) on
  every assumption used to generate oversight metrics before field data is available
  is a novel disclosure mechanism.
- The `generate_oversight_report()` function hardcodes `simulation_confidence='LOW'`
  and `field_study_required=True` — these cannot be removed without a code change,
  making the limitation structurally enforced rather than editorially stated.

**Supported by**:
- `haiip/core/human_oversight.py` — HumanOversightEngine, HIR/HOG/TCS/ECE
- `haiip/core/oversight_simulation.py` — OperatorSimulationModel, assumptions
- `haiip/tests/core/test_human_oversight.py` (30+ tests)
- `haiip/tests/core/test_oversight_simulation.py` (16+ tests)

**Limitation**: HOG and TCS are simulated. They are NOT validated against real operator
decisions. This is stated in every report, every test, and this document.

---

## C3 — Honest Non-IID Federated Scenario with Assumption Violation Detection

**Claim**: We implement a non-IID data partitioner for industrial federated learning
that makes all distribution assumptions explicit, detects violations before experiments
run, and produces a comparison table (IID vs non-IID vs non-IID + dropout) that honestly
shows the federated gap under realistic conditions.

**What is novel**:
- Prior federated learning papers for industrial IoT (Liu2020, Zhang2021) use IID
  splits or do not report assumption violations. We explicitly bound every assumption.
- `get_assumption_violations()` is called before each experiment and logs violations.
  The experiment does NOT silently proceed with violated assumptions.
- Node dropout is seed-reproducible: same seed + round + node_id = same dropout.
  This enables exact reproduction of the comparison table from the paper.

**Supported by**:
- `haiip/core/federated_realistic.py` — RealisticFederatedScenario
- `haiip/core/federated.py` — FederatedLearner, FedAvg
- `haiip/tests/core/test_federated.py` (20+ tests)

**Limitation**: Federation is simulated in-process. No real network faults
(latency, packet loss, Byzantine nodes) are modelled. The non-IID generator uses
Gaussian data with configurable shift, not real SME sensor data.

---

## C4 — Full Closed-Loop Platform with EU AI Act Compliance Engine

**Claim**: We present the first open-source platform that integrates: predictive
maintenance ML, ROS2 actuator-level closed-loop control, EU AI Act compliance engine
(Articles 9, 12, 13, 14, 52, 73), agentic RAG (ReAct), multi-tenant SaaS, and a CI
integrity gate that blocks documentation overclaiming.

**What is novel**:
- The combination of ROS2 closed-loop + ELM + HIR/HOG/TCS + EU AI Act audit logging
  in a single open-source platform is not present in prior work (verified March 2026).
- The CI integrity gate (`integrity.yml`) prevents simulation results from being merged
  without explicit disclosure flags — a novel software engineering contribution.
- The `DataSourceMode` enum prevents silent hardware/simulation fallback at a
  structural level, not just an editorial level.

**Supported by**:
- `haiip/ros2/` — 7 modules, dual-mode ROS2/asyncio pipeline
- `haiip/core/compliance.py` — ComplianceEngine
- `.github/workflows/integrity.yml` — integrity gate
- `haiip/data/ingestion/opcua_connector.py` — DataSourceMode
- 500+ tests across 12 test categories

**Limitation**: No real hardware proof-of-concept. ROS2 is not tested in CI.
EU AI Act compliance engine is not reviewed by a legal expert.

---

## What NOT to Claim

- Do not claim "achieves X% F1" without showing the confidence interval and the
  measurement environment from `docs/RESULTS.md`.
- Do not claim "saves X EUR" without showing the SiteEconomicProfile used and the
  sensitivity analysis.
- Do not claim "validated HOG/TCS" — they are simulated until field study runs.
- Do not claim "tested on real hardware" — it is not.
- Do not cite HOG/TCS numbers from the paper to an EU funder or regulator —
  they are simulation estimates.

---

## Venue Fit

| Venue | Track | Fit | Note |
|-------|-------|-----|------|
| IEEE Trans. Industrial Informatics (TII) | Regular | HIGH | C1 + C4 strongest fit |
| Engineering Applications of AI (EAAI) | Regular | HIGH | All 4 contributions |
| ECML-PKDD 2026 | Applied DS | MEDIUM | C2 + C3 novelty arguable |
| FAccT 2026 | Research | MEDIUM | C2 EU AI Act angle |
| ICLR 2026 Workshop: ML for Systems | Workshop | MEDIUM | C3 federated angle |
| NeurIPS 2026 | Main | LOW | Reproducibility track possible |

**Recommended first target**: IEEE Transactions on Industrial Informatics.
Impact factor 12.3. Directly covers industrial AI, federated learning, and cyber-physical systems.
Typical review time: 3–6 months.
Page limit: 12 pages IEEE format (double column).

**LaTeX template**: `IEEEtran.cls` — use `[journal]` option.
Convert this PAPER.md to LaTeX before submission.
