# Dataset Card — HAIIP Training Data

## Dataset Summary

Industrial sensor data from Nordic manufacturing SMEs for training predictive maintenance models.

| Property | Value |
|---|---|
| Format | CSV, Parquet |
| Size | ~50GB raw, ~8GB processed |
| Time period | 2022-2024 |
| Geographic scope | Finland, Sweden, Norway |

## Data Sources

1. **OPC UA telemetry** — Real-time reads from PLCs
2. **MQTT streams** — Vibration, temperature, pressure sensors
3. **Maintenance logs** — Expert-annotated failure records
4. **Synthetic data** — Simulator-generated samples for rare failure modes

## Preprocessing

1. Resampling to 1-second intervals
2. Z-score normalization per sensor per machine
3. Rolling window features: mean, std, min, max over 60s/300s/3600s
4. Labels: failure = 1 if maintenance action within next 24h

## Splits

| Split | Size | Period |
|---|---|---|
| Train | 70% | 2022-01 to 2023-09 |
| Validation | 15% | 2023-10 to 2024-03 |
| Test | 15% | 2024-04 to 2024-12 |

## License

Restricted research license. Contact Centria University of Applied Sciences for access.
