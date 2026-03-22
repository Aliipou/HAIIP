# Configuration Reference

## Environment Variables

### Required

| Variable | Description | Example |
|---|---|---|
| `POSTGRES_DSN` | PostgreSQL connection string | `postgresql://user:pass@db/haiip` |
| `REDIS_URL` | Redis connection URL | `redis://cache:6379` |
| `SECRET_KEY` | JWT signing secret (32+ chars) | `openssl rand -hex 32` |

### Optional

| Variable | Default | Description |
|---|---|---|
| `OPCUA_ENDPOINT` | — | OPC UA server URL |
| `MQTT_BROKER` | — | MQTT broker address |
| `HAIIP_DATA_SOURCE` | `opcua` | `opcua`, `mqtt`, `csv`, `simulator` |
| `HAIIP_LOG_LEVEL` | `INFO` | `DEBUG`, `INFO`, `WARNING`, `ERROR` |
| `HAIIP_ALERT_WEBHOOK` | — | Slack webhook for alerts |
| `MODEL_DIR` | `/models` | Path to model artifacts |
| `AUDIT_RETENTION_DAYS` | `1825` | 5 years |

## config/settings.yml

```yaml
models:
  anomaly:
    contamination: 0.02     # expected anomaly rate
    retrain_threshold: 0.15 # retrain if drift score > 0.15
  maintenance:
    prediction_horizon_h: 24
    min_training_samples: 500

alerting:
  cooldown_minutes: 30      # minimum time between same alert
  channels:
    slack:
      webhook: ${HAIIP_ALERT_WEBHOOK}

ui:
  theme: dark
  demo_mode: false          # set true to bypass auth in demos
  page_title: "HAIIP Dashboard"
```
