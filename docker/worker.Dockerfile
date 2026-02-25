FROM python:3.11-slim AS builder
WORKDIR /build
RUN apt-get update && apt-get install -y --no-install-recommends gcc g++ && \
    rm -rf /var/lib/apt/lists/*
COPY pyproject.toml .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir . --target /build/deps

FROM python:3.11-slim AS development
WORKDIR /app
COPY --from=builder /build/deps /usr/local/lib/python3.11/site-packages
COPY . .
CMD ["celery", "-A", "haiip.workers.tasks", "worker", "--loglevel=info", "--concurrency=2"]

FROM python:3.11-slim AS production
WORKDIR /app
RUN groupadd -r haiip && useradd -r -g haiip haiip && \
    rm -rf /var/lib/apt/lists/*
COPY --from=builder /build/deps /usr/local/lib/python3.11/site-packages
COPY haiip/ ./haiip/
RUN chown -R haiip:haiip /app
USER haiip
CMD ["celery", "-A", "haiip.workers.tasks", "worker", "--loglevel=info", "--concurrency=4"]
