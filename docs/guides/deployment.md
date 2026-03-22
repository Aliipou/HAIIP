# Deployment Guide

## Docker Compose (Development / Demo)

```bash
git clone https://github.com/Aliipou/HAIIP.git
cd HAIIP
cp .env.example .env  # edit with your settings
docker compose up --build -d
```

Services started:
- API: http://localhost:8000
- Dashboard: http://localhost:8501
- PostgreSQL: localhost:5432
- Redis: localhost:6379

## Production Deployment

### Environment Variables

```bash
# .env.production
POSTGRES_DSN=postgresql://haiip:${DB_PASSWORD}@db-host/haiip
REDIS_URL=redis://:${REDIS_PASSWORD}@redis-host:6379
OPCUA_ENDPOINT=opc.tcp://plc-host:4840
SECRET_KEY=${SECRET_KEY}
ENVIRONMENT=production
LOG_LEVEL=INFO
```

### Docker Compose (Production)

```bash
docker compose -f docker-compose.prod.yml up -d
```

### Health Checks

```bash
# API health
curl http://localhost:8000/health

# Model status
curl http://localhost:8000/api/models/status

# OPC UA connection
curl http://localhost:8000/api/datasources/status
```

### Backup

```bash
# Backup audit trail and model artifacts
./scripts/backup.sh --destination s3://your-bucket/haiip-backup
```

## Edge Deployment

For on-premise deployment without cloud connectivity:

```bash
# Export ONNX models for edge inference
python -m haiip.models export --format onnx --output models/edge/

# Run edge inference server
docker compose -f docker-compose.edge.yml up -d
```
