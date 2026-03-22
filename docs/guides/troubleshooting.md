# Troubleshooting Guide

## OPC UA Connection Issues

**Symptom:** `ConnectionRefusedError` in logs

**Checks:**
1. PLC firewall allows TCP port 4840
2. OPC UA server is enabled on the PLC
3. Security policy matches (None/Basic128Rsa15/Basic256)

```bash
# Test connectivity
python -c "import asyncua; asyncio.run(asyncua.Client('opc.tcp://plc:4840').connect())"
```

## Model Not Detecting Anomalies

**Symptom:** No alerts despite obvious machine issues

**Checks:**
1. Model has < 72 hours of baseline data
2. Contamination parameter set too low
3. Data drift — model trained on different operating conditions

```bash
# Check model age
python -m haiip.models status --machine motor-01

# Retrain
python -m haiip.train anomaly --data data/motor-01 --model models/motor-01
```

## High False Positive Rate

**Symptom:** Too many alerts, operator trust eroding

**Solution:** Increase contamination threshold or add operating-mode context:

```python
# During planned maintenance, suppress anomaly alerts
await client.set_mode("motor-01", mode="maintenance", duration_minutes=60)
```

## Streamlit Dashboard Not Loading

```bash
# Check service status
docker compose ps

# View logs
docker compose logs ui --tail 50

# Restart
docker compose restart ui
```

## PostgreSQL Connection Pool Exhausted

```
asyncpg.TooManyConnectionsError
```

Reduce `max_size` in connection pool or increase PostgreSQL `max_connections`.

```yaml
# docker-compose.yml
environment:
  POSTGRES_DSN: postgresql://haiip:pass@db/haiip?min_size=2&max_size=10
```
