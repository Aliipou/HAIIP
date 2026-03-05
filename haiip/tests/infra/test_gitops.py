"""Tests for GitOps manifests — ArgoCD application + Kustomize overlays."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

GITOPS_ROOT = Path(__file__).parents[3] / "gitops"


# ── ArgoCD application tests ──────────────────────────────────────────────────

class TestArgoCDApplication:

    def _load_app(self, name: str) -> list[dict]:
        path = GITOPS_ROOT / "argocd" / "application.yaml"
        assert path.exists(), f"Missing: {path}"
        docs = list(yaml.safe_load_all(path.read_text()))
        assert len(docs) >= 1, "application.yaml must have at least one document"
        return docs

    def test_application_yaml_is_valid(self):
        docs = self._load_app("application.yaml")
        for doc in docs:
            assert doc is not None

    def test_application_apiversion(self):
        docs = self._load_app("application.yaml")
        for doc in docs:
            assert doc["apiVersion"] == "argoproj.io/v1alpha1"

    def test_application_kind(self):
        docs = self._load_app("application.yaml")
        for doc in docs:
            assert doc["kind"] == "Application"

    def test_prod_app_has_automated_sync(self):
        docs = self._load_app("application.yaml")
        prod = next(d for d in docs if d["metadata"]["name"] == "haiip")
        sync = prod["spec"]["syncPolicy"]["automated"]
        assert sync["prune"] is True
        assert sync["selfHeal"] is True
        assert sync["allowEmpty"] is False

    def test_prod_app_destination_namespace(self):
        docs = self._load_app("application.yaml")
        prod = next(d for d in docs if d["metadata"]["name"] == "haiip")
        assert prod["spec"]["destination"]["namespace"] == "haiip-prod"

    def test_prod_app_has_retry_policy(self):
        docs = self._load_app("application.yaml")
        prod = next(d for d in docs if d["metadata"]["name"] == "haiip")
        retry = prod["spec"]["syncPolicy"]["retry"]
        assert retry["limit"] >= 3

    def test_staging_app_exists(self):
        docs = self._load_app("application.yaml")
        names = [d["metadata"]["name"] for d in docs]
        assert "haiip-staging" in names

    def test_repo_url_consistent(self):
        docs = self._load_app("application.yaml")
        for doc in docs:
            assert "HAIIP" in doc["spec"]["source"]["repoURL"]


# ── Kustomize overlay tests ───────────────────────────────────────────────────

class TestKustomizeBase:

    def _base_path(self) -> Path:
        return GITOPS_ROOT / "kustomize" / "base"

    def test_base_kustomization_exists(self):
        assert (self._base_path() / "kustomization.yaml").exists()

    def test_base_kustomization_valid(self):
        doc = yaml.safe_load((self._base_path() / "kustomization.yaml").read_text())
        assert doc["apiVersion"] == "kustomize.config.k8s.io/v1beta1"
        assert doc["kind"] == "Kustomization"

    def test_base_has_resources(self):
        doc = yaml.safe_load((self._base_path() / "kustomization.yaml").read_text())
        assert len(doc["resources"]) >= 1

    def test_base_has_common_labels(self):
        doc = yaml.safe_load((self._base_path() / "kustomization.yaml").read_text())
        labels = doc.get("commonLabels", {})
        assert "app.kubernetes.io/name" in labels

    def test_base_configmap_exists(self):
        assert (self._base_path() / "configmap.yaml").exists()

    def test_base_configmap_valid(self):
        doc = yaml.safe_load((self._base_path() / "configmap.yaml").read_text())
        assert doc["kind"] == "ConfigMap"
        assert "APP_ENV" in doc["data"]

    def test_pdb_exists(self):
        assert (self._base_path() / "pdb.yaml").exists()

    def test_pdb_valid(self):
        docs = list(yaml.safe_load_all((self._base_path() / "pdb.yaml").read_text()))
        assert len(docs) >= 2
        for doc in docs:
            assert doc["kind"] == "PodDisruptionBudget"


class TestKustomizeOverlays:

    def _overlay(self, env: str) -> Path:
        return GITOPS_ROOT / "kustomize" / "overlays" / env

    @pytest.mark.parametrize("env", ["dev", "staging", "prod"])
    def test_overlay_exists(self, env):
        assert (self._overlay(env) / "kustomization.yaml").exists()

    @pytest.mark.parametrize("env", ["dev", "staging", "prod"])
    def test_overlay_valid_yaml(self, env):
        doc = yaml.safe_load((self._overlay(env) / "kustomization.yaml").read_text())
        assert doc["apiVersion"] == "kustomize.config.k8s.io/v1beta1"

    @pytest.mark.parametrize("env", ["dev", "staging", "prod"])
    def test_overlay_has_namespace(self, env):
        doc = yaml.safe_load((self._overlay(env) / "kustomization.yaml").read_text())
        assert "namespace" in doc
        assert env in doc["namespace"]

    def test_dev_uses_debug_log_level(self):
        doc = yaml.safe_load((self._overlay("dev") / "kustomization.yaml").read_text())
        generators = doc.get("configMapGenerator", [])
        literals = generators[0]["literals"] if generators else []
        assert any("LOG_LEVEL=DEBUG" in l for l in literals)

    def test_prod_has_three_api_replicas(self):
        doc = yaml.safe_load((self._overlay("prod") / "kustomization.yaml").read_text())
        patches = doc.get("patches", [])
        # At least one patch sets replicas to 3
        patch_texts = [p["patch"] for p in patches if isinstance(p, dict) and "patch" in p]
        assert any("3" in pt for pt in patch_texts)
