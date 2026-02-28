# HAIIP — Known Limitations

> This document does not soften language. Every limitation listed here is real.
> The purpose is to make every weakness into a documented, testable guard —
> not to hide weaknesses but to make the system fail loudly when its assumptions
> are violated.

---

## L1 — Federated Learning: Non-IID and Network Conditions Not Validated

**What the system claims**: FedAvg across 3 Nordic SME nodes achieves F1 within 15%
of the centralised baseline while preserving privacy.

**What is actually tested**: IID data split from AI4I 2020, no network faults,
all nodes always available, federation simulated in-process (no real network).

**What a real deployment would need**:
- Non-IID data reflecting different machine types per site (CNC / pumps / conveyors)
  -> use `haiip/core/federated_realistic.py` (`RealisticFederatedScenario`)
- Network fault simulation (node dropout, reconnect, backoff)
  -> see `haiip/tests/robustness/test_network_faults.py`
- Real connectivity testing between Jakobstad / Sundsvall / Narvik nodes
- Byzantine fault tolerance (a node sending malicious weight updates)

**Risk if deployed without addressing**:
- F1 gap may exceed 15% under non-IID conditions with dropout
- FedAvg is known to diverge under high data heterogeneity (Li et al. 2020)

**Severity**: HIGH for research claims. MEDIUM for internal operational use.

**Guard**: `RealisticFederatedScenario.get_assumption_violations()` logs violations
before every experiment. Experiment does NOT silently proceed when assumptions are violated.

---

## L2 — Economic Parameters: Defaults Are Not Calibrated

**What the system claims**: The Expected Loss Minimization engine avoids X EUR
of downtime per avoided stop compared to naive probability thresholding.

**What is actually tested**: Fixed defaults from Nordic manufacturing median
(€850/hr downtime, €65/hr labour, 4h MTTR). These are wrong for every real site.

**What a real deployment would need**:
- `SiteEconomicProfile.calibration_interview()` administered to site operators
  (15-minute questionnaire, maps to all monetary parameters)
- Sensitivity analysis per site: `profile.sensitivity_analysis()` shows which
  parameters drive the ELM decision before any EUR figure is quoted
- Calibration interview responses stored and linked to any ROI report

**Risk if deployed without addressing**:
- ELM decisions optimised for the wrong cost model
- EUR figures in reports will be wrong (could be 10x off for an automotive line)
- False sense of economic precision

**Severity**: HIGH for any financial reporting or ROI claims. MEDIUM for operational use
where decisions are validated by operators before actuation.

**Guard**: `SiteEconomicProfile.validate()` checks all parameters against physical ranges.
`calibrated=False` on the default profile — must be set to True only via
`from_interview_responses()`.

---

## L3 — Human Oversight Metrics: Simulation, Not Field Data

**What the system claims**: HOG (Human Override Gain) and TCS (Trust Calibration Score)
quantify the value of human oversight in the HAIIP pipeline.

**What is actually tested**: Simulated operator behaviour using
`OperatorSimulationModel` with LOW-confidence assumptions. Every assumption
is a named constant with a citation or explicit "Source: model assumption, no citation" marker.

**Low-confidence assumptions in the simulation**:
- `ASSUMPTION_ACCEPT_RATE_NOVICE = 0.54` — Confidence: LOW (extrapolated, no direct citation)
- `ASSUMPTION_FATIGUE_FACTOR = 0.91` — Confidence: LOW (general HCI, not industrial-specific)
- `ASSUMPTION_FALSE_POSITIVE_LEARNING = 0.03` — Confidence: VERY LOW (model assumption)
- `ASSUMPTION_EXPLANATION_BOOST = 0.08` — Confidence: NONE (this is what RQ3 measures)

**What a real deployment would need**:
- Minimum 200 real operator decisions per role type (expert / novice)
- Ground truth labels (was the fault real? confirmed post-hoc from maintenance logs)
- IRB-approved field study protocol at one of the three Nordic sites
- Longitudinal study to measure fatigue and false-positive learning effects

**Risk if deployed without addressing**:
- HOG and TCS numbers are not externally valid
- Reporting them in a paper or to EU funders without disclosure is misleading

**Severity**: HIGH for academic publication or EU funding reports.
LOW for internal operational use where HOG/TCS are used as internal improvement signals.

**Guard**: All oversight reports carry `simulation_confidence: 'LOW'` and
`field_study_required: True` hardcoded. These cannot be changed without modifying
`haiip/core/oversight_simulation.py`. The dashboard shows a visible simulation banner.

---

## L4 — No Hardware Proof-of-Concept

**What the system claims**: Connects to real factory hardware via OPC UA.

**What is actually tested**: OPC UA client (`haiip/data/ingestion/opcua_connector.py`)
connects to `asyncua.SimulatedOPCUAServer` (in-process simulation). No real PLC,
no real industrial network, no real safety review.

**What a real deployment would need**:
- At least one real PLC connection test at a participating site
- Latency measurement on a real industrial network (OPC UA round-trip < 100ms target)
- Safety review: any AI recommendation that reaches a machine actuator must pass
  IEC 62443-3-3 security and IEC 61508 safety analysis
- Operator training before connecting AI to any actuator output
- Emergency stop override that physically bypasses the AI system

**Risk if deployed without addressing**:
- Real hardware may have timing constraints incompatible with the current asyncua polling model
- OPC UA namespace / node IDs differ per PLC vendor and site — require configuration
- Safety-critical actuation without hardware validation is a liability risk

**Severity**: HIGH for any production deployment. LOW for software-only demo use.

**Guard**: `DataSourceMode` enum — every reading is tagged `REAL_HARDWARE`,
`SIMULATION`, or `HARDWARE_FALLBACK`. There is no silent fallback.
`OPCUAConnector.assert_real_hardware()` raises `HardwareNotConnectedError`
if not truly connected. Dashboard shows a visible banner for non-hardware modes.

---

## L5 — ROS2 Pipeline: Software Only, No Hardware-in-Loop Test

**What the system claims**: The ROS2 closed-loop pipeline (VibrationPublisher ->
InferenceNode -> EconomicNode -> ActionNode -> HumanOverride) controls machine actuators.

**What is actually tested**: Standalone asyncio pipeline (`haiip/ros2/pipeline.py`)
with synthetic vibration data. No real ROS2 runtime tested in CI. No real actuator.
No hardware-in-loop (HIL) test.

**What a real deployment would need**:
- ROS2 Humble or Jazzy runtime installed and tested in CI (currently skipped)
- Real sensor hardware publishing to `/haiip/vibration/{machine_id}` topic
- Real actuator subscribed to `/haiip/command/{machine_id}` with fail-safe logic
- End-to-end latency test: sensor -> AI -> command in < 100ms at 50Hz

**Risk if deployed without addressing**:
- ROS2 QoS settings (BEST_EFFORT vs RELIABLE) may need tuning per network
- Human override TTL (600s default) may be wrong for real operational context

**Severity**: MEDIUM for research claims. HIGH for any actuation deployment.

---

## Reading This Document

Each limitation follows this structure:
- **What the system claims** — the capability documented elsewhere
- **What is actually tested** — the honest scope of current tests
- **What a real deployment would need** — concrete engineering steps to address the gap
- **Risk if deployed without addressing** — consequences of ignoring the limitation
- **Severity** — context-dependent assessment
- **Guard** — the code-level mechanism that makes the limitation explicit at runtime

This document is tested by `haiip/tests/test_documentation_honesty.py`.
If you remove or soften a limitation, the test will fail.
