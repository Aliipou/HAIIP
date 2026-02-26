"""FastAPI application factory.

Import order matters — models must be imported before create_all_tables()
so SQLAlchemy metadata is populated.
"""

import time
from contextlib import asynccontextmanager
from typing import Any

import structlog
from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from haiip.api.config import get_settings
from haiip.api.database import check_database_connection, create_all_tables
from haiip.api.logging_config import configure_logging
from haiip.api.models import (  # noqa: F401 — imported for side-effect (metadata)
    Alert,
    AuditLog,
    FeedbackLog,
    ModelRegistry,
    Prediction,
    Tenant,
    User,
)
from haiip.api.middleware import SecurityMiddleware
from haiip.api.routes import admin, agent, alerts, auth, documents, economic, feedback, metrics, predict

settings = get_settings()
logger = structlog.get_logger(__name__)

_start_time = time.monotonic()


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):  # type: ignore[type-arg]
    configure_logging()
    logger.info("haiip.startup", env=settings.app_env, version="0.1.0")
    await create_all_tables()
    logger.info("haiip.db.ready")
    yield
    logger.info("haiip.shutdown")


# ── App factory ───────────────────────────────────────────────────────────────

def create_app() -> FastAPI:
    application = FastAPI(
        title="HAIIP — Human-Aligned Industrial Intelligence Platform",
        description=(
            "RDI-grade AI platform for SME predictive maintenance, "
            "anomaly detection, and human-robot collaboration."
        ),
        version="0.1.0",
        docs_url="/api/docs" if not settings.is_production else None,
        redoc_url="/api/redoc" if not settings.is_production else None,
        openapi_url="/api/openapi.json" if not settings.is_production else None,
        lifespan=lifespan,
    )

    # ── Middleware ─────────────────────────────────────────────────────────────
    # SecurityMiddleware first: rate limiting + headers (outermost layer)
    application.add_middleware(SecurityMiddleware)
    application.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000", "http://localhost:8501"],
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE"],
        allow_headers=["Authorization", "Content-Type", "X-Tenant-Slug"],
    )

    # ── Prometheus instrumentation ─────────────────────────────────────────────
    if settings.prometheus_enabled:
        try:
            from prometheus_fastapi_instrumentator import Instrumentator

            Instrumentator(
                should_group_status_codes=True,
                excluded_handlers=["/health", "/metrics"],
            ).instrument(application).expose(application, endpoint="/metrics")
        except ImportError:
            logger.warning("prometheus_fastapi_instrumentator not installed — skipping")

    # ── Exception handlers ────────────────────────────────────────────────────
    @application.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={
                "success": False,
                "error": "Validation error",
                "detail": exc.errors(),
            },
        )

    @application.exception_handler(Exception)
    async def generic_exception_handler(
        request: Request, exc: Exception
    ) -> JSONResponse:
        logger.error(
            "unhandled_exception",
            path=request.url.path,
            method=request.method,
            exc=str(exc),
        )
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"success": False, "error": "Internal server error"},
        )

    # ── Routes ─────────────────────────────────────────────────────────────────
    prefix = "/api/v1"
    application.include_router(auth.router, prefix=prefix, tags=["auth"])
    application.include_router(predict.router, prefix=prefix, tags=["predictions"])
    application.include_router(alerts.router, prefix=prefix, tags=["alerts"])
    application.include_router(metrics.router, prefix=prefix, tags=["metrics"])
    application.include_router(documents.router, prefix=prefix, tags=["documents"])
    application.include_router(feedback.router, prefix=prefix, tags=["feedback"])
    application.include_router(admin.router, prefix=prefix, tags=["admin"])
    application.include_router(agent.router, prefix=prefix, tags=["agent"])
    application.include_router(economic.router, prefix=prefix, tags=["economic"])

    # ── Health check ──────────────────────────────────────────────────────────
    @application.get("/health", include_in_schema=False)
    async def health() -> dict[str, Any]:
        db_ok = await check_database_connection()
        return {
            "status": "healthy" if db_ok else "degraded",
            "database": db_ok,
            "uptime_seconds": round(time.monotonic() - _start_time, 2),
            "version": "0.1.0",
        }

    return application


app = create_app()
