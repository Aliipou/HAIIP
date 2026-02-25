"""Structured logging configuration using structlog.

Outputs JSON in production, coloured console in development.
All log events include: timestamp, level, logger name, and event.
"""

import logging
import sys

import structlog

from haiip.api.config import get_settings


def configure_logging() -> None:
    settings = get_settings()
    level = getattr(logging, settings.log_level.upper(), logging.INFO)

    shared_processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
    ]

    if settings.is_production:
        # JSON output for log aggregation (Loki, CloudWatch, etc.)
        processors: list[structlog.types.Processor] = [
            *shared_processors,
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ]
        formatter = structlog.stdlib.ProcessorFormatter(
            processors=processors,
        )
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(formatter)
    else:
        # Human-readable coloured console for development
        processors = [
            *shared_processors,
            structlog.dev.ConsoleRenderer(colors=True),
        ]
        formatter = structlog.stdlib.ProcessorFormatter(
            processors=processors,
        )
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.handlers = [handler]
    root_logger.setLevel(level)

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(level),
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
