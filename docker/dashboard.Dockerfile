FROM python:3.11-slim AS builder
WORKDIR /build
RUN apt-get update && apt-get install -y --no-install-recommends gcc && \
    rm -rf /var/lib/apt/lists/*
COPY pyproject.toml .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir . --target /build/deps

FROM python:3.11-slim AS development
WORKDIR /app
COPY --from=builder /build/deps /usr/local/lib/python3.11/site-packages
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "haiip/dashboard/app.py", "--server.port=8501", "--server.address=0.0.0.0"]

FROM python:3.11-slim AS production
WORKDIR /app
RUN groupadd -r haiip && useradd -r -g haiip haiip
COPY --from=builder /build/deps /usr/local/lib/python3.11/site-packages
COPY haiip/dashboard/ ./haiip/dashboard/
RUN chown -R haiip:haiip /app
USER haiip
EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=10s CMD curl -f http://localhost:8501/_stcore/health || exit 1
CMD ["streamlit", "run", "haiip/dashboard/app.py", "--server.port=8501", "--server.address=0.0.0.0", \
     "--server.headless=true", "--browser.gatherUsageStats=false"]
