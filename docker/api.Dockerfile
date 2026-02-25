# ── Build stage ───────────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ curl && \
    rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir hatchling && \
    pip install --no-cache-dir ".[dev]" --target /build/deps || \
    pip install --no-cache-dir . --target /build/deps

# ── Development stage ─────────────────────────────────────────────────────────
FROM python:3.11-slim AS development

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

COPY --from=builder /build/deps /usr/local/lib/python3.11/site-packages

COPY . .

EXPOSE 8000
CMD ["uvicorn", "haiip.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

# ── Production stage ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS production

WORKDIR /app

# Create non-root user for security
RUN groupadd -r haiip && useradd -r -g haiip -d /app -s /sbin/nologin haiip && \
    apt-get update && apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

COPY --from=builder /build/deps /usr/local/lib/python3.11/site-packages

COPY haiip/ ./haiip/

# Owned by non-root user
RUN chown -R haiip:haiip /app
USER haiip

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "haiip.api.main:app", "--host", "0.0.0.0", "--port", "8000", \
     "--workers", "2", "--no-access-log"]
