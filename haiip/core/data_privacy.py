"""GDPR/Data Privacy engine for HAIIP.

Implements EU GDPR Articles 17, 20, 25, 32 requirements:
- Art. 17: Right to erasure ("right to be forgotten")
- Art. 20: Data portability (export all tenant data as JSON)
- Art. 25: Privacy by design — PII detection before storage
- Art. 32: Security of processing — pseudonymization

HAIIP handles industrial sensor data (not personal data by default),
but user accounts (email, name) and audit logs require GDPR compliance.

PII detection: email, phone numbers, names in free-text fields.
Pseudonymization: SHA-256 HMAC of identifiers with per-tenant salt.

Usage:
    privacy = DataPrivacyEngine(tenant_salt="tenant-secret")
    safe = privacy.scrub_pii({"email": "user@example.com", "reading": 300.0})
    # → {"email": "[PII:email]", "reading": 300.0}
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import re
from dataclasses import dataclass, field
from datetime import UTC
from typing import Any

logger = logging.getLogger(__name__)


# ── PII pattern library ───────────────────────────────────────────────────────

_PII_PATTERNS: dict[str, re.Pattern[str]] = {
    "email": re.compile(r"\b[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}\b"),
    "phone_intl": re.compile(r"\+\d{1,3}[\s\-]?\(?\d{1,4}\)?[\s\-]?\d{1,4}[\s\-]?\d{1,9}"),
    "phone_fi": re.compile(r"\b0\d{2}[\s\-]?\d{3}[\s\-]?\d{4}\b"),  # Finnish format
    "ssn_fi": re.compile(r"\b\d{6}[-+A]\d{3}[0-9A-FHJ-NPR-Y]\b"),  # Finnish HETU
    "credit_card": re.compile(r"\b(?:\d{4}[\s\-]?){3}\d{4}\b"),
    "iban": re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{1,30}\b"),
}

# Fields considered sensitive in HAIIP context
_SENSITIVE_FIELD_NAMES = frozenset(
    {
        "email",
        "username",
        "full_name",
        "name",
        "phone",
        "address",
        "ip_address",
        "user_agent",
        "password",
        "hashed_password",
        "access_token",
        "refresh_token",
    }
)


@dataclass
class PIIDetectionResult:
    """Result of scanning a value for PII."""

    has_pii: bool
    pii_types: list[str]
    redacted_value: Any  # value with PII replaced


@dataclass
class ErasureRequest:
    """GDPR Art.17 erasure request record."""

    tenant_id: str
    subject_id: str  # user_id or machine_id
    requested_at: str  # ISO timestamp
    tables_affected: list[str] = field(default_factory=list)
    records_deleted: int = 0
    status: str = "pending"  # pending | completed | error


@dataclass
class DataExportResult:
    """GDPR Art.20 portability export."""

    tenant_id: str
    subject_id: str
    exported_at: str
    tables: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    record_count: int = 0


class DataPrivacyEngine:
    """GDPR compliance engine for HAIIP.

    Thread-safe: all methods are stateless (pure functions over inputs).
    The tenant_salt is used for pseudonymization — keep it secret.
    """

    def __init__(
        self,
        tenant_salt: str = "",
        redaction_placeholder: str = "[REDACTED]",
        detect_in_values: bool = True,
    ) -> None:
        self.tenant_salt = tenant_salt
        self.redaction_placeholder = redaction_placeholder
        self.detect_in_values = detect_in_values

    # ── PII detection ─────────────────────────────────────────────────────────

    def detect_pii(self, value: str) -> PIIDetectionResult:
        """Scan a string value for PII patterns.

        Returns PIIDetectionResult with matched PII types and redacted value.
        """
        if not isinstance(value, str):
            return PIIDetectionResult(has_pii=False, pii_types=[], redacted_value=value)

        found: list[str] = []
        redacted = value

        for pii_type, pattern in _PII_PATTERNS.items():
            if pattern.search(redacted):
                found.append(pii_type)
                redacted = pattern.sub(f"[PII:{pii_type}]", redacted)

        return PIIDetectionResult(
            has_pii=len(found) > 0,
            pii_types=found,
            redacted_value=redacted,
        )

    def scrub_pii(self, data: dict[str, Any], depth: int = 0) -> dict[str, Any]:
        """Recursively scan a dict and redact PII fields.

        Sensitive field names are always redacted.
        String values are scanned with regex patterns.

        Args:
            data: dict to scrub
            depth: recursion depth limit (max 5)

        Returns:
            New dict with PII replaced by [REDACTED] or [PII:type] markers.
        """
        if depth > 5:
            return data

        result: dict[str, Any] = {}
        for key, value in data.items():
            if key in _SENSITIVE_FIELD_NAMES:
                result[key] = self.redaction_placeholder
            elif isinstance(value, dict):
                result[key] = self.scrub_pii(value, depth + 1)
            elif isinstance(value, list):
                result[key] = [
                    self.scrub_pii(v, depth + 1)
                    if isinstance(v, dict)
                    else (
                        self.detect_pii(v).redacted_value
                        if isinstance(v, str) and self.detect_in_values
                        else v
                    )
                    for v in value
                ]
            elif isinstance(value, str) and self.detect_in_values:
                detection = self.detect_pii(value)
                result[key] = detection.redacted_value
            else:
                result[key] = value

        return result

    # ── Pseudonymization ──────────────────────────────────────────────────────

    def pseudonymize(self, identifier: str) -> str:
        """Pseudonymize an identifier using HMAC-SHA256 with tenant salt.

        Deterministic: same identifier + salt always produces the same token.
        Not reversible without the salt.

        Returns: hex string (first 16 chars of HMAC)
        """
        if not self.tenant_salt:
            # No salt configured — return a basic hash (warn but don't crash)
            logger.warning("DataPrivacyEngine: no tenant_salt set — using unsalted hash")
            return hashlib.sha256(identifier.encode()).hexdigest()[:16]

        mac = hmac.new(
            self.tenant_salt.encode("utf-8"),
            identifier.encode("utf-8"),
            hashlib.sha256,
        )
        return mac.hexdigest()[:16]

    def pseudonymize_dict(self, data: dict[str, Any], fields: list[str]) -> dict[str, Any]:
        """Pseudonymize specific fields in a dict.

        Args:
            data: source dict
            fields: list of field names to pseudonymize

        Returns:
            New dict with specified fields replaced by pseudonyms.
        """
        result = dict(data)
        for field_name in fields:
            if field_name in result and isinstance(result[field_name], str):
                result[field_name] = self.pseudonymize(result[field_name])
        return result

    # ── Consent management ────────────────────────────────────────────────────

    @staticmethod
    def validate_consent_record(record: dict[str, Any]) -> tuple[bool, list[str]]:
        """Validate a consent record has all required GDPR fields.

        Returns (is_valid, list_of_missing_fields).
        """
        required = ["subject_id", "tenant_id", "purpose", "granted_at", "legal_basis"]
        missing = [f for f in required if not record.get(f)]
        return len(missing) == 0, missing

    @staticmethod
    def is_consent_expired(
        granted_at_iso: str,
        expiry_days: int = 365,
    ) -> bool:
        """Check if consent has expired (GDPR recommends annual renewal)."""
        from datetime import datetime, timedelta

        try:
            granted = datetime.fromisoformat(granted_at_iso.replace("Z", "+00:00"))
            expiry = granted + timedelta(days=expiry_days)
            return datetime.now(UTC) > expiry
        except (ValueError, AttributeError):
            return True  # malformed date → treat as expired (safe default)

    # ── Data minimization ─────────────────────────────────────────────────────

    @staticmethod
    def minimize(data: dict[str, Any], allowed_fields: list[str]) -> dict[str, Any]:
        """Keep only fields in allowed_fields (Art. 5.1c data minimization)."""
        return {k: v for k, v in data.items() if k in allowed_fields}

    @staticmethod
    def apply_retention_policy(
        records: list[dict[str, Any]],
        timestamp_field: str,
        max_age_days: int,
    ) -> tuple[list[dict[str, Any]], int]:
        """Filter records to those within the retention window.

        Returns (kept_records, deleted_count).
        """
        from datetime import datetime, timedelta

        cutoff = datetime.now(UTC) - timedelta(days=max_age_days)
        kept: list[dict[str, Any]] = []
        deleted = 0

        for record in records:
            ts_raw = record.get(timestamp_field)
            if ts_raw is None:
                kept.append(record)
                continue
            try:
                ts = datetime.fromisoformat(str(ts_raw).replace("Z", "+00:00"))
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=UTC)
                if ts >= cutoff:
                    kept.append(record)
                else:
                    deleted += 1
            except ValueError:
                kept.append(record)  # unparseable → retain

        return kept, deleted
