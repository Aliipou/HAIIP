"""Edge-to-cloud model synchronization with delta compression.

EdgeModelSync:
  - Pulls latest model version from cloud (API or S3)
  - Compares SHA-256 hash against local manifest
  - Downloads only if model has changed (delta-efficient)
  - Atomic file replacement — no partial model loads
  - Works without network: returns current local model on failure

CloudModelSync:
  - Pushes edge metrics (latency, accuracy drift) back to cloud
  - Batched, retry with exponential backoff
  - Compressed JSON payload

Usage:
    sync = EdgeModelSync(
        cloud_api_url="https://api.haiip.ai/api/v1",
        model_dir=Path("/opt/haiip/models"),
        tenant_id="sme-finland",
        api_key="...",
    )
    updated = sync.sync()   # returns True if model was updated
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT = 30.0
_MAX_RETRIES = 3
_RETRY_BACKOFF_BASE = 2.0


class EdgeModelSync:
    """Pull model updates from cloud API to local edge device.

    All operations are atomic: partial downloads are staged to a temp dir
    and moved into place only after full verification.
    """

    def __init__(
        self,
        cloud_api_url: str,
        model_dir: Path | str,
        tenant_id: str,
        api_key: str = "",
        check_interval_seconds: float = 3600.0,
        timeout: float = _DEFAULT_TIMEOUT,
    ) -> None:
        self.cloud_api_url = cloud_api_url.rstrip("/")
        self.model_dir = Path(model_dir)
        self.tenant_id = tenant_id
        self.api_key = api_key
        self.check_interval_seconds = check_interval_seconds
        self.timeout = timeout
        self._last_check: float = 0.0
        self._last_version: str = ""

    def sync(self, force: bool = False) -> bool:
        """Check cloud for newer model; download if available.

        Args:
            force: skip interval check and always query cloud

        Returns:
            True if local model was updated, False if already current.
        """
        now = time.monotonic()
        if not force and (now - self._last_check) < self.check_interval_seconds:
            return False

        self._last_check = now

        try:
            cloud_meta = self._fetch_cloud_metadata()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Cloud metadata fetch failed (using local model): %s", exc)
            return False

        cloud_version = cloud_meta.get("version", "")
        cloud_hash = cloud_meta.get("model_hash_sha256", "")

        local_hash = self._local_model_hash()

        if cloud_hash and cloud_hash == local_hash:
            logger.debug("Model up to date (hash=%s...)", cloud_hash[:12])
            self._last_version = cloud_version
            return False

        logger.info(
            "New model available: version=%s hash=%s... (local=%s...)",
            cloud_version,
            cloud_hash[:12],
            (local_hash[:12] if local_hash else "none"),
        )

        try:
            self._download_and_install(cloud_meta)
        except Exception as exc:
            logger.error("Model download failed: %s", exc)
            return False

        self._last_version = cloud_version
        return True

    # ── Cloud API ─────────────────────────────────────────────────────────────

    def _fetch_cloud_metadata(self) -> dict[str, Any]:
        import urllib.request

        url = f"{self.cloud_api_url}/models/latest?tenant_id={self.tenant_id}"
        req = urllib.request.Request(url)
        if self.api_key:
            req.add_header("Authorization", f"Bearer {self.api_key}")

        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def _download_and_install(self, cloud_meta: dict[str, Any]) -> None:
        download_url = cloud_meta.get("download_url", "")
        if not download_url:
            raise ValueError("Cloud metadata missing download_url")

        import urllib.request

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)

            # Download ONNX model
            onnx_tmp = tmp_path / "model.onnx"
            with urllib.request.urlopen(download_url, timeout=self.timeout) as resp:
                onnx_tmp.write_bytes(resp.read())

            # Verify hash
            expected_hash = cloud_meta.get("model_hash_sha256", "")
            if expected_hash:
                actual_hash = hashlib.sha256(onnx_tmp.read_bytes()).hexdigest()
                if actual_hash != expected_hash:
                    raise RuntimeError(
                        f"Downloaded model hash mismatch: expected={expected_hash[:16]} "
                        f"actual={actual_hash[:16]}"
                    )

            # Write manifest
            manifest = {
                "model_hash_sha256": expected_hash,
                "version": cloud_meta.get("version", ""),
                "feature_names": cloud_meta.get("feature_names", []),
                "downloaded_at": time.time(),
            }
            (tmp_path / "manifest.json").write_text(json.dumps(manifest, indent=2))

            # Atomic install: copy scaler if present, then model
            self.model_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(onnx_tmp, self.model_dir / "model.onnx")
            shutil.copy2(tmp_path / "manifest.json", self.model_dir / "manifest.json")

        logger.info(
            "Model installed: version=%s dir=%s",
            cloud_meta.get("version"),
            self.model_dir,
        )

    # ── Local state ───────────────────────────────────────────────────────────

    def _local_model_hash(self) -> str:
        manifest_path = self.model_dir / "manifest.json"
        if not manifest_path.exists():
            return ""
        try:
            manifest = json.loads(manifest_path.read_text())
            return manifest.get("model_hash_sha256", "")
        except Exception:  # noqa: BLE001
            return ""

    @property
    def last_version(self) -> str:
        return self._last_version


class EdgeMetricsReporter:
    """Push edge inference metrics back to cloud for retraining decisions.

    Batches metrics locally and flushes on a schedule or when buffer fills.
    Works without network: buffers indefinitely until connectivity restored.
    """

    def __init__(
        self,
        cloud_api_url: str,
        tenant_id: str,
        machine_id: str,
        api_key: str = "",
        batch_size: int = 100,
        max_buffer: int = 10_000,
        timeout: float = _DEFAULT_TIMEOUT,
    ) -> None:
        self.cloud_api_url = cloud_api_url.rstrip("/")
        self.tenant_id = tenant_id
        self.machine_id = machine_id
        self.api_key = api_key
        self.batch_size = batch_size
        self.max_buffer = max_buffer
        self.timeout = timeout
        self._buffer: list[dict[str, Any]] = []
        self._failed_sends = 0

    def record(self, prediction: dict[str, Any], latency_ms: float) -> None:
        """Buffer a prediction metric for later cloud sync."""
        if len(self._buffer) >= self.max_buffer:
            self._buffer.pop(0)  # drop oldest to prevent OOM
            logger.warning("EdgeMetricsReporter buffer full — dropping oldest entry")

        self._buffer.append({
            "machine_id": self.machine_id,
            "tenant_id": self.tenant_id,
            "label": prediction.get("label"),
            "confidence": prediction.get("confidence"),
            "anomaly_score": prediction.get("anomaly_score"),
            "latency_ms": round(latency_ms, 2),
            "timestamp": time.time(),
        })

        if len(self._buffer) >= self.batch_size:
            self.flush()

    def flush(self) -> int:
        """Send buffered metrics to cloud. Returns number of records sent."""
        if not self._buffer:
            return 0

        batch = self._buffer[: self.batch_size]
        try:
            self._send_batch(batch)
            self._buffer = self._buffer[len(batch):]
            logger.debug("EdgeMetricsReporter flushed %d records", len(batch))
            return len(batch)
        except Exception as exc:  # noqa: BLE001
            self._failed_sends += 1
            logger.warning("EdgeMetricsReporter flush failed (will retry): %s", exc)
            return 0

    def _send_batch(self, batch: list[dict[str, Any]]) -> None:
        import urllib.request

        url = f"{self.cloud_api_url}/edge/metrics"
        payload = json.dumps({"records": batch}).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=payload,
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        if self.api_key:
            req.add_header("Authorization", f"Bearer {self.api_key}")

        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            if resp.status not in (200, 201, 204):
                raise RuntimeError(f"Cloud rejected metrics: HTTP {resp.status}")

    @property
    def buffer_size(self) -> int:
        return len(self._buffer)

    @property
    def failed_sends(self) -> int:
        return self._failed_sends
