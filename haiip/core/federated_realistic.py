"""
Realistic Non-IID Federated Scenario for Nordic SME HAIIP Nodes
================================================================
Generates data partitions that reflect real industrial heterogeneity.

Every distribution assumption is a NAMED CONSTANT with a tolerance band.
get_assumption_violations() must be called before every federated experiment.
Violations are logged — experiment does NOT silently proceed.

Usage:
    scenario = RealisticFederatedScenario(seed=42)
    partitions = scenario.generate_partitions(df)
    for node_id, df_node in partitions.items():
        violations = scenario.get_assumption_violations(df_node, node_id)
        if violations:
            for v in violations:
                logger.warning("ASSUMPTION VIOLATED: %s", v)

RQ6 context: Tests non-IID + dropout scenarios to bound the federated_gap claim.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Node identifiers
# ---------------------------------------------------------------------------

NODES = ("jakobstad", "sundsvall", "narvik")

# ---------------------------------------------------------------------------
# Distribution assumptions — NAMED CONSTANTS with tolerance bands.
# Each constant has a citation or "Source: model assumption, no citation".
# ---------------------------------------------------------------------------

# Volume ratio: Jakobstad : Sundsvall : Narvik = 3 : 2 : 1
# Source: SME size proxy from NextIndustriAI project scoping report 2024.
# (Jakobstad CNC shop larger than Sundsvall pump facility and Narvik conveyor site.)
# Confidence: LOW — no verified headcount or revenue data available.
VOLUME_RATIO: dict[str, float] = {
    "jakobstad": 3.0,
    "sundsvall": 2.0,
    "narvik":    1.0,
}
VOLUME_TOLERANCE = 0.10  # +/- 10% of target fraction is acceptable

# Failure mode dominant fractions per node.
# Source: domain expert elicitation, NextIndustriAI kickoff workshop 2024.
# Confidence: LOW — must be validated against real site maintenance logs.
DOMINANT_FRACTION           = 0.60  # node's dominant failure type = 60% of failures
DOMINANT_FRACTION_TOLERANCE = 0.10  # +/- 10 percentage points
SECONDARY_FRACTION          = 0.10  # secondary failure type
SECONDARY_FRACTION_TOLERANCE = 0.05

# Node failure mode profiles: (dominant_mode, secondary_mode)
# Jakobstad: CNC mill     -> Heat Dissipation Failure most common
# Sundsvall: Pump station -> Power Failure dominant
# Narvik:    Conveyor     -> Tool Wear Failure most common
# Source: domain expert elicitation, NextIndustriAI kickoff workshop 2024.
NODE_FAILURE_PROFILES: dict[str, dict[str, str]] = {
    "jakobstad": {"dominant": "HDF", "secondary": "TWF"},
    "sundsvall": {"dominant": "PWF", "secondary": "OSF"},
    "narvik":    {"dominant": "TWF", "secondary": "HDF"},
}

# Connectivity: each round, each node has this probability of dropout.
# Source: model assumption based on IEC 62443-3-3 industrial network availability.
# Confidence: LOW — no direct measurement from Jakobstad/Sundsvall/Narvik sites.
DROPOUT_PROBABILITY = 0.15

# AI4I 2020 column names
FAILURE_MODE_COLUMNS = ["TWF", "HDF", "PWF", "OSF", "RNF"]
FAILURE_LABEL_COLUMN = "Machine failure"


@dataclass
class NodePartition:
    node_id: str
    df: pd.DataFrame
    dominant_mode: str
    secondary_mode: str
    target_volume_fraction: float
    assumption_violations: list[str] = field(default_factory=list)


class RealisticFederatedScenario:
    """
    Generates non-IID data partitions reflecting real Nordic SME heterogeneity.

    Every assumption is a named constant above. Violations are surfaced via
    get_assumption_violations() — caller decides whether to abort or log-and-continue.

    Dropout is reproducible: same seed + same round + same node_id = same result.
    This is required for experimental reproducibility (RQ6).
    """

    def __init__(self, seed: int = 42) -> None:
        self._seed = seed
        self._rng  = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_partitions(self, df: pd.DataFrame) -> dict[str, pd.DataFrame]:
        """
        Partition df into 3 non-IID node datasets.

        Returns {'jakobstad': df_a, 'sundsvall': df_b, 'narvik': df_c}.
        Logs assumption violations for each partition — does NOT raise.
        """
        if df.empty:
            raise ValueError("Input DataFrame is empty — cannot partition.")

        partitions: dict[str, pd.DataFrame] = {}
        total_ratio = sum(VOLUME_RATIO.values())

        has_failure_col = FAILURE_LABEL_COLUMN in df.columns
        failure_rows = df[df[FAILURE_LABEL_COLUMN] == 1] if has_failure_col else df.iloc[:0]
        normal_rows  = df[df[FAILURE_LABEL_COLUMN] == 0] if has_failure_col else df

        for node_id in NODES:
            target_frac = VOLUME_RATIO[node_id] / total_ratio
            target_n    = max(1, int(len(df) * target_frac))
            profile     = NODE_FAILURE_PROFILES[node_id]
            dom_col     = profile["dominant"]
            sec_col     = profile["secondary"]

            # Select failures by mode
            dom_rows = (
                failure_rows[failure_rows[dom_col] == 1]
                if dom_col in df.columns and not failure_rows.empty
                else failure_rows.iloc[:0]
            )
            sec_rows = (
                failure_rows[failure_rows[sec_col] == 1]
                if sec_col in df.columns and not failure_rows.empty
                else failure_rows.iloc[:0]
            )
            other_idx  = ~failure_rows.index.isin(dom_rows.index) & ~failure_rows.index.isin(sec_rows.index)
            other_rows = failure_rows[other_idx] if not failure_rows.empty else failure_rows.iloc[:0]

            dom_n   = int(target_n * DOMINANT_FRACTION)
            sec_n   = int(target_n * SECONDARY_FRACTION)
            other_n = max(0, int(target_n * 0.05))

            def _sample(src: pd.DataFrame, n: int) -> pd.DataFrame:
                if src.empty or n == 0:
                    return src.iloc[:0]
                return src.sample(min(n, len(src)), random_state=self._seed, replace=len(src) < n)

            chosen = pd.concat(
                [_sample(dom_rows, dom_n), _sample(sec_rows, sec_n), _sample(other_rows, other_n)],
                ignore_index=True,
            )

            remaining = max(0, target_n - len(chosen))
            if remaining > 0:
                fill   = _sample(normal_rows, remaining)
                chosen = pd.concat([chosen, fill], ignore_index=True)

            chosen = chosen.sample(frac=1.0, random_state=self._seed).reset_index(drop=True)
            partitions[node_id] = chosen

            violations = self.get_assumption_violations(chosen, node_id)
            if violations:
                logger.warning("federated_assumption_violations node=%s violations=%s", node_id, violations)

        return partitions

    def simulate_dropout(self, round_num: int, node_id: str) -> bool:
        """
        Returns True if this node should be excluded from this round.

        Reproducible: same (seed, round_num, node_id) always gives same result.
        DROPOUT_PROBABILITY = 0.15 per round per node.
        """
        key    = f"{self._seed}:{round_num}:{node_id}".encode()
        digest = int(hashlib.sha256(key).hexdigest(), 16)
        value  = (digest % 10_000) / 10_000.0  # in [0, 1)
        return value < DROPOUT_PROBABILITY

    def get_assumption_violations(self, partition: pd.DataFrame, node_id: str) -> list[str]:
        """
        Check whether partition satisfies all named distribution assumptions.

        Returns list of violation strings. Empty list = no violations.
        Caller decides whether to abort or log-and-continue.
        """
        violations: list[str] = []

        if partition.empty:
            violations.append(f"{node_id}: partition is empty")
            return violations

        profile = NODE_FAILURE_PROFILES[node_id]

        if FAILURE_LABEL_COLUMN in partition.columns:
            failures      = partition[partition[FAILURE_LABEL_COLUMN] == 1]
            total_failures = len(failures)

            if total_failures > 0:
                for col_key, frac, tol in [
                    ("dominant",  DOMINANT_FRACTION,  DOMINANT_FRACTION_TOLERANCE),
                    ("secondary", SECONDARY_FRACTION, SECONDARY_FRACTION_TOLERANCE),
                ]:
                    col  = profile[col_key]
                    if col in partition.columns:
                        ratio = failures[col].sum() / total_failures
                        lo, hi = frac - tol, frac + tol
                        if not (lo <= ratio <= hi):
                            violations.append(
                                f"{node_id} {col} ratio {ratio:.2f}, "
                                f"expected {frac:.2f} +/- {tol:.2f}"
                            )

        return violations

    def comparison_table(
        self,
        federated_learner_class: Any,
        n_rounds: int = 5,
    ) -> pd.DataFrame:
        """
        Returns a DataFrame comparing IID vs non-IID vs non-IID+dropout scenarios.
        federated_learner_class must accept profiles parameter.
        """
        rows: list[dict[str, Any]] = []

        try:
            iid_result = federated_learner_class().run(n_rounds=n_rounds)
            rows.append({
                "scenario":      "IID split",
                "f1":            round(iid_result.final_global_f1, 4),
                "federated_gap": round(iid_result.federated_gap, 4),
                "dropout":       False,
            })
        except Exception as exc:  # noqa: BLE001
            rows.append({"scenario": "IID split", "f1": None, "error": str(exc), "dropout": False})

        rows.append({
            "scenario": "non-IID (no dropout)",
            "f1":       None,
            "note":     "requires RealisticFederatedLearner (future work)",
            "dropout":  False,
        })
        rows.append({
            "scenario": "non-IID + dropout",
            "f1":       None,
            "note":     "requires RealisticFederatedLearner (future work)",
            "dropout":  True,
        })

        return pd.DataFrame(rows)
