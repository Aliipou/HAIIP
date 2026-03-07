"""Tests for DataPrivacyEngine — 100% branch coverage (GDPR Arts. 5/17/20/25/32)."""

from __future__ import annotations

from datetime import datetime, timezone, timedelta

import pytest

from haiip.core.data_privacy import (
    DataPrivacyEngine,
    PIIDetectionResult,
    ErasureRequest,
    DataExportResult,
)


class TestPIIDetection:
    def setup_method(self):
        self.engine = DataPrivacyEngine(tenant_salt="test-salt-abc")

    def test_detect_email(self):
        result = self.engine.detect_pii("Contact: admin@haiip.ai for support")
        assert result.has_pii
        assert "email" in result.pii_types
        assert "[PII:email]" in result.redacted_value

    def test_detect_international_phone(self):
        result = self.engine.detect_pii("+358 40 123 4567")
        assert result.has_pii
        assert "phone_intl" in result.pii_types

    def test_detect_finnish_phone(self):
        result = self.engine.detect_pii("Call 040 123 4567 now")
        assert result.has_pii
        assert "phone_fi" in result.pii_types

    def test_detect_finnish_ssn(self):
        result = self.engine.detect_pii("HETU: 010170-123N")
        assert result.has_pii
        assert "ssn_fi" in result.pii_types

    def test_detect_credit_card(self):
        result = self.engine.detect_pii("Card: 4111 1111 1111 1111")
        assert result.has_pii
        assert "credit_card" in result.pii_types

    def test_detect_iban(self):
        result = self.engine.detect_pii("IBAN: FI2112345600000785")
        assert result.has_pii
        assert "iban" in result.pii_types

    def test_no_pii_sensor_data(self):
        result = self.engine.detect_pii("vibration=300.5 Hz, temperature=72.1 C")
        assert not result.has_pii
        assert result.pii_types == []
        assert result.redacted_value == "vibration=300.5 Hz, temperature=72.1 C"

    def test_non_string_returns_no_pii(self):
        result = self.engine.detect_pii(12345)  # type: ignore[arg-type]
        assert not result.has_pii
        assert result.redacted_value == 12345

    def test_multiple_pii_types_in_one_string(self):
        result = self.engine.detect_pii("Email: user@test.com, Phone: +358 40 111 2222")
        assert result.has_pii
        assert len(result.pii_types) >= 2

    def test_empty_string_no_pii(self):
        result = self.engine.detect_pii("")
        assert not result.has_pii


class TestScrubPII:
    def setup_method(self):
        self.engine = DataPrivacyEngine(tenant_salt="test-salt")

    def test_sensitive_field_names_redacted(self):
        data = {"email": "admin@haiip.ai", "vibration": 300.0}
        result = self.engine.scrub_pii(data)
        assert result["email"] == "[REDACTED]"
        assert result["vibration"] == 300.0

    def test_all_sensitive_fields(self):
        sensitive_fields = [
            "email", "username", "full_name", "name", "phone",
            "address", "ip_address", "user_agent", "password",
            "hashed_password", "access_token", "refresh_token",
        ]
        data = {f: f"value_{f}" for f in sensitive_fields}
        result = self.engine.scrub_pii(data)
        for f in sensitive_fields:
            assert result[f] == "[REDACTED]", f"Field {f} was not redacted"

    def test_nested_dict_scrubbed(self):
        data = {"user": {"email": "x@y.com", "machine_id": "M-001"}}
        result = self.engine.scrub_pii(data)
        assert result["user"]["email"] == "[REDACTED]"
        assert result["user"]["machine_id"] == "M-001"

    def test_list_of_dicts_scrubbed(self):
        data = {"users": [{"email": "a@b.com"}, {"email": "c@d.com"}]}
        result = self.engine.scrub_pii(data)
        assert result["users"][0]["email"] == "[REDACTED]"
        assert result["users"][1]["email"] == "[REDACTED]"

    def test_list_of_strings_scrubbed(self):
        data = {"notes": ["contact admin@test.com", "temp=300"]}
        result = self.engine.scrub_pii(data)
        assert "[PII:email]" in result["notes"][0]
        assert result["notes"][1] == "temp=300"

    def test_string_value_with_pii_scrubbed(self):
        data = {"comment": "call +358 40 123 4567 for info"}
        result = self.engine.scrub_pii(data)
        assert "+358 40 123 4567" not in result["comment"]

    def test_depth_limit_stops_recursion(self):
        # Build a deeply nested dict (depth > 5)
        deep: dict = {"val": "safe"}
        for _ in range(8):
            deep = {"nested": deep}
        # Should not crash or recurse infinitely
        result = self.engine.scrub_pii(deep)
        assert isinstance(result, dict)

    def test_detect_in_values_false_skips_regex(self):
        engine = DataPrivacyEngine(tenant_salt="salt", detect_in_values=False)
        data = {"comment": "user@test.com here"}
        result = engine.scrub_pii(data)
        # Value not scrubbed since detect_in_values=False
        assert result["comment"] == "user@test.com here"

    def test_numeric_values_pass_through(self):
        data = {"reading": 300.5, "count": 42}
        result = self.engine.scrub_pii(data)
        assert result["reading"] == 300.5
        assert result["count"] == 42

    def test_custom_redaction_placeholder(self):
        engine = DataPrivacyEngine(tenant_salt="s", redaction_placeholder="***")
        data = {"email": "x@y.com"}
        result = engine.scrub_pii(data)
        assert result["email"] == "***"


class TestPseudonymization:
    def setup_method(self):
        self.engine = DataPrivacyEngine(tenant_salt="my-secret-salt")

    def test_pseudonymize_deterministic(self):
        token1 = self.engine.pseudonymize("user-123")
        token2 = self.engine.pseudonymize("user-123")
        assert token1 == token2

    def test_pseudonymize_different_ids_different_tokens(self):
        t1 = self.engine.pseudonymize("user-001")
        t2 = self.engine.pseudonymize("user-002")
        assert t1 != t2

    def test_pseudonymize_returns_hex_string_16chars(self):
        token = self.engine.pseudonymize("test-id")
        assert len(token) == 16
        assert all(c in "0123456789abcdef" for c in token)

    def test_pseudonymize_no_salt_uses_sha256(self):
        engine = DataPrivacyEngine(tenant_salt="")
        token = engine.pseudonymize("user-123")
        assert len(token) == 16

    def test_pseudonymize_salt_changes_output(self):
        e1 = DataPrivacyEngine(tenant_salt="salt-a")
        e2 = DataPrivacyEngine(tenant_salt="salt-b")
        assert e1.pseudonymize("user-1") != e2.pseudonymize("user-1")

    def test_pseudonymize_dict_replaces_fields(self):
        data = {"user_id": "u-123", "machine_id": "M-001", "reading": 300.0}
        result = self.engine.pseudonymize_dict(data, ["user_id"])
        assert result["user_id"] != "u-123"
        assert result["machine_id"] == "M-001"
        assert result["reading"] == 300.0

    def test_pseudonymize_dict_skips_non_string(self):
        data = {"count": 42, "score": 0.9}
        result = self.engine.pseudonymize_dict(data, ["count"])
        assert result["count"] == 42  # non-string fields untouched

    def test_pseudonymize_dict_missing_field_ignored(self):
        data = {"a": "val"}
        result = self.engine.pseudonymize_dict(data, ["b"])  # 'b' not in data
        assert result == {"a": "val"}


class TestConsentManagement:
    def test_valid_consent_record(self):
        record = {
            "subject_id": "u-1",
            "tenant_id": "t-1",
            "purpose": "predictive_maintenance",
            "granted_at": "2024-01-01T00:00:00Z",
            "legal_basis": "legitimate_interest",
        }
        valid, missing = DataPrivacyEngine.validate_consent_record(record)
        assert valid
        assert missing == []

    def test_missing_fields_returned(self):
        record = {"subject_id": "u-1", "tenant_id": "t-1"}
        valid, missing = DataPrivacyEngine.validate_consent_record(record)
        assert not valid
        assert "purpose" in missing
        assert "granted_at" in missing
        assert "legal_basis" in missing

    def test_consent_not_expired(self):
        recent = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
        assert not DataPrivacyEngine.is_consent_expired(recent, expiry_days=365)

    def test_consent_expired(self):
        old = (datetime.now(timezone.utc) - timedelta(days=400)).isoformat()
        assert DataPrivacyEngine.is_consent_expired(old, expiry_days=365)

    def test_malformed_date_treated_as_expired(self):
        assert DataPrivacyEngine.is_consent_expired("not-a-date")

    def test_consent_expired_custom_window(self):
        two_months_ago = (datetime.now(timezone.utc) - timedelta(days=61)).isoformat()
        assert DataPrivacyEngine.is_consent_expired(two_months_ago, expiry_days=60)

    def test_consent_not_expired_custom_window(self):
        one_month_ago = (datetime.now(timezone.utc) - timedelta(days=29)).isoformat()
        assert not DataPrivacyEngine.is_consent_expired(one_month_ago, expiry_days=60)

    def test_z_suffix_iso_date_parsed(self):
        recent = (datetime.now(timezone.utc) - timedelta(days=10)).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
        assert not DataPrivacyEngine.is_consent_expired(recent, expiry_days=365)


class TestDataMinimization:
    def test_minimize_keeps_only_allowed(self):
        data = {"machine_id": "M1", "email": "x@y.com", "vibration": 300.0}
        result = DataPrivacyEngine.minimize(data, ["machine_id", "vibration"])
        assert "email" not in result
        assert result["machine_id"] == "M1"
        assert result["vibration"] == 300.0

    def test_minimize_empty_allowed_fields(self):
        data = {"a": 1, "b": 2}
        result = DataPrivacyEngine.minimize(data, [])
        assert result == {}

    def test_minimize_all_fields_allowed(self):
        data = {"a": 1, "b": 2}
        result = DataPrivacyEngine.minimize(data, ["a", "b"])
        assert result == {"a": 1, "b": 2}


class TestRetentionPolicy:
    def _make_record(self, days_ago: int) -> dict:
        ts = (datetime.now(timezone.utc) - timedelta(days=days_ago)).isoformat()
        return {"ts": ts, "val": 1}

    def test_keeps_recent_deletes_old(self):
        records = [
            self._make_record(5),   # within 30d → keep
            self._make_record(40),  # outside 30d → delete
        ]
        kept, deleted = DataPrivacyEngine.apply_retention_policy(records, "ts", max_age_days=30)
        assert len(kept) == 1
        assert deleted == 1

    def test_all_recent_none_deleted(self):
        records = [self._make_record(i) for i in range(1, 5)]
        kept, deleted = DataPrivacyEngine.apply_retention_policy(records, "ts", max_age_days=30)
        assert len(kept) == 4
        assert deleted == 0

    def test_all_old_all_deleted(self):
        records = [self._make_record(100), self._make_record(200)]
        kept, deleted = DataPrivacyEngine.apply_retention_policy(records, "ts", max_age_days=30)
        assert len(kept) == 0
        assert deleted == 2

    def test_missing_timestamp_field_retained(self):
        records = [{"val": 1, "no_ts": True}]
        kept, deleted = DataPrivacyEngine.apply_retention_policy(records, "ts", max_age_days=30)
        assert len(kept) == 1
        assert deleted == 0

    def test_unparseable_timestamp_retained(self):
        records = [{"ts": "not-a-date", "val": 2}]
        kept, deleted = DataPrivacyEngine.apply_retention_policy(records, "ts", max_age_days=30)
        assert len(kept) == 1

    def test_naive_datetime_treated_as_utc(self):
        # Naive datetime without timezone
        ts = (datetime.utcnow() - timedelta(days=5)).isoformat()
        records = [{"ts": ts}]
        kept, deleted = DataPrivacyEngine.apply_retention_policy(records, "ts", max_age_days=30)
        assert len(kept) == 1

    def test_empty_records_returns_empty(self):
        kept, deleted = DataPrivacyEngine.apply_retention_policy([], "ts", max_age_days=30)
        assert kept == []
        assert deleted == 0


class TestDataclasses:
    def test_erasure_request_defaults(self):
        req = ErasureRequest(
            tenant_id="t-1", subject_id="u-1", requested_at="2024-01-01T00:00:00Z"
        )
        assert req.tables_affected == []
        assert req.records_deleted == 0
        assert req.status == "pending"

    def test_data_export_result_defaults(self):
        res = DataExportResult(tenant_id="t-1", subject_id="u-1", exported_at="now")
        assert res.tables == {}
        assert res.record_count == 0

    def test_pii_detection_result_fields(self):
        r = PIIDetectionResult(has_pii=True, pii_types=["email"], redacted_value="[PII:email]")
        assert r.has_pii
        assert r.pii_types == ["email"]
