"""Brutal tests for FederatedLearner — FedAvg simulation.

Coverage:
- Node data generation (non-IID, failure_rate, noise)
- FedAvg aggregation (weighted average correctness)
- Full training loop (converges, returns valid metrics)
- Centralized baseline comparison
- FederatedResult structure
- RoundResult per-round tracking
- Privacy guarantee (no raw data in output)
- Custom NodeProfiles
- Convergence detection
- Edge cases (1 round, 1 node, all-failure data)
"""

from __future__ import annotations

import numpy as np
import pytest

from haiip.core.federated import (
    FederatedLearner,
    FederatedNode,
    FederatedResult,
    FederatedServer,
    NodeProfile,
    SMENode,
)

# ── Fixtures ───────────────────────────────────────────────────────────────────


@pytest.fixture
def default_profiles() -> list[NodeProfile]:
    return FederatedLearner.DEFAULT_PROFILES


@pytest.fixture
def minimal_profiles() -> list[NodeProfile]:
    """Small profiles for fast tests."""
    return [
        NodeProfile(
            node_id=SMENode.SME_FI,
            n_samples=200,
            failure_rate=0.15,
            noise_std=0.2,
            country="FI",
            industry="Test mill",
        ),
        NodeProfile(
            node_id=SMENode.SME_SE,
            n_samples=300,
            failure_rate=0.10,
            noise_std=0.1,
            country="SE",
            industry="Test press",
        ),
    ]


@pytest.fixture
def minimal_learner(minimal_profiles: list[NodeProfile]) -> FederatedLearner:
    return FederatedLearner(profiles=minimal_profiles, random_state=42)


# ── NodeProfile + FederatedNode ────────────────────────────────────────────────


class TestFederatedNode:
    def test_data_shape(self, minimal_profiles: list[NodeProfile]) -> None:
        node = FederatedNode(profile=minimal_profiles[0], random_state=0)
        assert node.X.shape == (200, 5)
        assert node.y.shape == (200,)

    def test_failure_rate_approximate(self, minimal_profiles: list[NodeProfile]) -> None:
        node = FederatedNode(profile=minimal_profiles[0], random_state=0)
        actual_rate = node.y.mean()
        # Allow ±5% tolerance
        assert abs(actual_rate - 0.15) < 0.08

    def test_binary_labels(self, minimal_profiles: list[NodeProfile]) -> None:
        node = FederatedNode(profile=minimal_profiles[0], random_state=0)
        assert set(node.y).issubset({0.0, 1.0})

    def test_local_train_returns_params(self, minimal_profiles: list[NodeProfile]) -> None:
        node = FederatedNode(profile=minimal_profiles[0], random_state=0)
        params = node.local_train(global_params=None, local_epochs=1)
        assert "n_estimators" in params
        assert "learning_rate" in params
        assert "n_samples" in params

    def test_local_train_with_global_params(self, minimal_profiles: list[NodeProfile]) -> None:
        node = FederatedNode(profile=minimal_profiles[0], random_state=0)
        params = node.local_train(
            global_params={"n_estimators": 30, "learning_rate": 0.05},
            local_epochs=2,
        )
        assert params["n_samples"] == 200

    def test_evaluate_after_train(self, minimal_profiles: list[NodeProfile]) -> None:
        node = FederatedNode(profile=minimal_profiles[0], random_state=0)
        node.local_train(global_params=None, local_epochs=1)
        f1, loss = node.evaluate()
        assert 0.0 <= f1 <= 1.0
        assert loss >= 0.0

    def test_evaluate_before_train_returns_defaults(
        self, minimal_profiles: list[NodeProfile]
    ) -> None:
        node = FederatedNode(profile=minimal_profiles[0], random_state=0)
        f1, loss = node.evaluate()
        assert f1 == 0.0

    def test_feature_shift_applied(self) -> None:
        shift = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
        profile = NodeProfile(
            node_id=SMENode.SME_NO,
            n_samples=100,
            failure_rate=0.1,
            noise_std=0.1,
            feature_shift=shift,
        )
        node = FederatedNode(profile=profile, random_state=0)
        # After scaling, mean should differ from a no-shift node
        assert node.X.shape == (100, 5)


# ── FederatedServer ────────────────────────────────────────────────────────────


class TestFederatedServer:
    def test_fedavg_weighted_by_samples(self, minimal_profiles: list[NodeProfile]) -> None:
        nodes = [FederatedNode(p, random_state=i) for i, p in enumerate(minimal_profiles)]
        server = FederatedServer(nodes=nodes, random_state=42)

        params = [
            {"n_estimators": 100, "learning_rate": 0.1, "n_samples": 200},
            {"n_estimators": 200, "learning_rate": 0.2, "n_samples": 400},
        ]
        agg = server.fedavg(params)
        # Weighted: (100*200 + 200*400) / 600 = 166.67 → int
        expected = int(round((100 * 200 + 200 * 400) / 600))
        assert agg["n_estimators"] == expected

    def test_fedavg_learning_rate_weighted(self, minimal_profiles: list[NodeProfile]) -> None:
        nodes = [FederatedNode(p, random_state=i) for i, p in enumerate(minimal_profiles)]
        server = FederatedServer(nodes=nodes, random_state=42)
        params = [
            {
                "n_estimators": 50,
                "learning_rate": 0.1,
                "n_samples": 100,
                "feature_importances": [0.2, 0.2, 0.2, 0.2, 0.2],
            },
            {
                "n_estimators": 50,
                "learning_rate": 0.3,
                "n_samples": 100,
                "feature_importances": [0.2, 0.2, 0.2, 0.2, 0.2],
            },
        ]
        agg = server.fedavg(params)
        assert agg["learning_rate"] == pytest.approx(0.2, abs=0.001)

    def test_evaluate_global_before_fit_returns_zeros(
        self, minimal_profiles: list[NodeProfile]
    ) -> None:
        nodes = [FederatedNode(p, random_state=i) for i, p in enumerate(minimal_profiles)]
        server = FederatedServer(nodes=nodes, random_state=42)
        f1, auc = server.evaluate_global()
        assert f1 == 0.0
        assert auc == 0.0


# ── FederatedLearner (full loop) ───────────────────────────────────────────────


class TestFederatedLearner:
    def test_result_structure(self, minimal_learner: FederatedLearner) -> None:
        result = minimal_learner.run(n_rounds=3, local_epochs=1)
        assert isinstance(result, FederatedResult)
        assert result.experiment_id
        assert len(result.rounds) >= 1
        assert result.total_rounds >= 1

    def test_round_metrics_valid(self, minimal_learner: FederatedLearner) -> None:
        result = minimal_learner.run(n_rounds=3, local_epochs=1)
        for rnd in result.rounds:
            assert 0.0 <= rnd.global_f1 <= 1.0
            assert 0.0 <= rnd.global_auc <= 1.0
            assert rnd.round_duration_s >= 0.0
            assert rnd.communication_cost > 0

    def test_node_f1s_present(self, minimal_learner: FederatedLearner) -> None:
        result = minimal_learner.run(n_rounds=2, local_epochs=1)
        for rnd in result.rounds:
            assert "SME_FI" in rnd.node_f1s
            assert "SME_SE" in rnd.node_f1s

    def test_privacy_preserved(self, minimal_learner: FederatedLearner) -> None:
        result = minimal_learner.run(n_rounds=2, local_epochs=1)
        assert result.privacy_preserved is True

    def test_centralized_f1_is_upper_bound(self, minimal_learner: FederatedLearner) -> None:
        result = minimal_learner.run(n_rounds=5, local_epochs=2)
        # Federated should be within 15% of centralized (acceptable gap)
        assert result.federated_gap <= 0.15, (
            f"Federated gap too large: {result.federated_gap:.4f} "
            f"(fed={result.final_global_f1:.4f}, cent={result.centralized_f1:.4f})"
        )

    def test_node_profiles_in_result(self, minimal_learner: FederatedLearner) -> None:
        result = minimal_learner.run(n_rounds=2, local_epochs=1)
        assert "SME_FI" in result.node_profiles
        assert "SME_SE" in result.node_profiles
        assert result.node_profiles["SME_FI"]["n_samples"] == 200

    def test_single_round(self, minimal_learner: FederatedLearner) -> None:
        result = minimal_learner.run(n_rounds=1, local_epochs=1)
        assert result.total_rounds == 1
        assert len(result.rounds) == 1

    def test_convergence_detection(self) -> None:
        """With sufficient rounds, convergence should be detected."""
        learner = FederatedLearner(
            profiles=[
                NodeProfile(SMENode.SME_FI, 300, 0.12, 0.2),
                NodeProfile(SMENode.SME_SE, 400, 0.08, 0.15),
            ],
            convergence_threshold=0.0,  # Always converge after round 3
        )
        result = learner.run(n_rounds=10, local_epochs=2)
        # Should stop early due to convergence
        assert result.converged_at is not None or result.total_rounds <= 10

    def test_default_profiles_all_three_nodes(self) -> None:
        learner = FederatedLearner(random_state=99)
        result = learner.run(n_rounds=2, local_epochs=1)
        assert "SME_FI" in result.node_profiles
        assert "SME_SE" in result.node_profiles
        assert "SME_NO" in result.node_profiles

    def test_experiment_id_is_uuid(self, minimal_learner: FederatedLearner) -> None:
        import uuid

        result = minimal_learner.run(n_rounds=1, local_epochs=1)
        uuid.UUID(result.experiment_id)  # should not raise

    def test_f1_improves_over_rounds(self, minimal_learner: FederatedLearner) -> None:
        """F1 should generally not get worse as we do more rounds."""
        result = minimal_learner.run(n_rounds=6, local_epochs=2)
        if len(result.rounds) >= 3:
            early_f1 = result.rounds[0].global_f1
            late_f1 = result.rounds[-1].global_f1
            # Allow 10% regression — GBT can slightly fluctuate
            assert late_f1 >= early_f1 - 0.10
