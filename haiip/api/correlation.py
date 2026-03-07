"""Request correlation ID middleware.

Every HTTP request gets a unique `X-Request-ID` header (generated if absent).
The ID is bound to the structlog context so every log line in that request
automatically carries `request_id`, making distributed tracing trivial.

Usage in logs:
    {"event": "predict.start", "request_id": "3f2a...", "tenant_id": "sme-fi", ...}

Usage in responses:
    X-Request-ID: 3f2a1b8c-0e4d-4f7e-9b2a-1234567890ab
"""

from __future__ import annotations

import uuid

import structlog
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

logger = structlog.get_logger(__name__)

HEADER = "X-Request-ID"


class CorrelationIDMiddleware(BaseHTTPMiddleware):
    """Inject a unique request ID into every log event for the request lifetime."""

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        request_id = request.headers.get(HEADER) or str(uuid.uuid4())

        # Bind to structlog context — all log calls in this request carry it
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(
            request_id=request_id,
            method=request.method,
            path=request.url.path,
        )

        response = await call_next(request)
        response.headers[HEADER] = request_id
        return response
