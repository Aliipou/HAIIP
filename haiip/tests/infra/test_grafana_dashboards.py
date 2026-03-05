"""Tests for Grafana dashboard JSON files — structural validation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

DASHBOARDS_DIR = Path(__file__).parents[3] / "grafana" / "dashboards"
PROVISIONING_DIR = Path(__file__).parents[3] / "grafana" / "provisioning"

EXPECTED_DASHBOARDS = ["fleet_health.json", "drift_detection.json", "economic_decisions.json"]


# ── File existence ─────────────────────────────────────────────────────────────

def test_dashboard_files_exist():
    for name in EXPECTED_DASHBOARDS:
        assert (DASHBOARDS_DIR / name).exists(), f"Missing dashboard: {name}"


def test_provisioning_datasources_exist():
    assert (PROVISIONING_DIR / "datasources" / "prometheus.yml").exists()


def test_provisioning_dashboards_config_exists():
    assert (PROVISIONING_DIR / "dashboards" / "all.yml").exists()


# ── JSON validity ──────────────────────────────────────────────────────────────

@pytest.mark.parametrize("filename", EXPECTED_DASHBOARDS)
def test_dashboard_is_valid_json(filename):
    path = DASHBOARDS_DIR / filename
    content = path.read_text(encoding="utf-8")
    data = json.loads(content)  # raises if invalid
    assert isinstance(data, dict)


# ── Required top-level fields ──────────────────────────────────────────────────

@pytest.mark.parametrize("filename", EXPECTED_DASHBOARDS)
def test_dashboard_has_required_fields(filename):
    data = json.loads((DASHBOARDS_DIR / filename).read_text())
    required = {"title", "uid", "panels", "schemaVersion", "tags", "refresh", "time"}
    missing = required - set(data.keys())
    assert not missing, f"{filename} missing fields: {missing}"


@pytest.mark.parametrize("filename", EXPECTED_DASHBOARDS)
def test_dashboard_uid_is_unique(filename):
    """Each dashboard must have a unique uid."""
    uids = []
    for name in EXPECTED_DASHBOARDS:
        d = json.loads((DASHBOARDS_DIR / name).read_text())
        uids.append(d["uid"])
    assert len(uids) == len(set(uids)), f"Duplicate dashboard UIDs: {uids}"


# ── Panel structure ────────────────────────────────────────────────────────────

@pytest.mark.parametrize("filename", EXPECTED_DASHBOARDS)
def test_dashboard_has_panels(filename):
    data = json.loads((DASHBOARDS_DIR / filename).read_text())
    assert len(data["panels"]) >= 1


@pytest.mark.parametrize("filename", EXPECTED_DASHBOARDS)
def test_all_panels_have_targets(filename):
    data = json.loads((DASHBOARDS_DIR / filename).read_text())
    for panel in data["panels"]:
        assert "targets" in panel, f"Panel {panel.get('id')} missing targets in {filename}"
        assert len(panel["targets"]) >= 1


@pytest.mark.parametrize("filename", EXPECTED_DASHBOARDS)
def test_all_panel_targets_have_expr(filename):
    data = json.loads((DASHBOARDS_DIR / filename).read_text())
    for panel in data["panels"]:
        for t in panel["targets"]:
            assert "expr" in t, f"Target missing 'expr' in panel {panel.get('id')} of {filename}"
            assert t["expr"].strip(), f"Empty expr in panel {panel.get('id')} of {filename}"


@pytest.mark.parametrize("filename", EXPECTED_DASHBOARDS)
def test_panels_have_unique_ids(filename):
    data = json.loads((DASHBOARDS_DIR / filename).read_text())
    ids = [p["id"] for p in data["panels"]]
    assert len(ids) == len(set(ids)), f"Duplicate panel IDs in {filename}: {ids}"


# ── Tags ───────────────────────────────────────────────────────────────────────

def test_fleet_health_tagged_correctly():
    data = json.loads((DASHBOARDS_DIR / "fleet_health.json").read_text())
    assert "haiip" in data["tags"]


def test_drift_dashboard_tagged_correctly():
    data = json.loads((DASHBOARDS_DIR / "drift_detection.json").read_text())
    assert "drift" in data["tags"]


def test_economic_dashboard_tagged_correctly():
    data = json.loads((DASHBOARDS_DIR / "economic_decisions.json").read_text())
    assert "economic" in data["tags"]


# ── Refresh intervals ─────────────────────────────────────────────────────────

def test_fleet_health_has_fast_refresh():
    data = json.loads((DASHBOARDS_DIR / "fleet_health.json").read_text())
    # Fleet health should refresh at least every minute
    assert data["refresh"] in {"15s", "30s", "1m"}


# ── Provisioning YAML ─────────────────────────────────────────────────────────

def test_datasource_provisioning_references_prometheus():
    content = (PROVISIONING_DIR / "datasources" / "prometheus.yml").read_text()
    assert "prometheus" in content.lower()
    assert "http://prometheus:9090" in content


def test_dashboard_provisioning_references_correct_path():
    content = (PROVISIONING_DIR / "dashboards" / "all.yml").read_text()
    assert "/var/lib/grafana/dashboards" in content
