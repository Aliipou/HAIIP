# MLOps Guide

## Model Versioning

HAIIP tracks all model versions:

```bash
# List model versions
python -m haiip.models list --machine motor-01

# Activate specific version
python -m haiip.models activate --machine motor-01 --version v3

# Compare versions
python -m haiip.models compare --machine motor-01 --versions v2,v3
```

## Automated Retraining

The `AutoRetrainPipeline` triggers retraining when:
1. Data drift score > 0.15 (KS-test p-value)
2. Precision drops below 0.80 on recent labeled data
3. Manual trigger via API

```bash
# Check drift
GET /api/machines/motor-01/drift

{
  "drift_score": 0.18,
  "threshold": 0.15,
  "triggered_retrain": true,
  "retrain_started_at": "2024-01-15T02:00:00Z"
}
```

## A/B Testing Models

```python
# Run two models simultaneously, compare performance
python -m haiip.deploy ab-test \
    --machine motor-01 \
    --model-a models/v2/anomaly_detector \
    --model-b models/v3/anomaly_detector \
    --split 0.2  # 20% to model B
    --duration-days 7
```

## CI/CD for Models

```yaml
# .github/workflows/model-train.yml
on:
  schedule:
    - cron: '0 1 * * 0'  # Weekly retraining
  workflow_dispatch:

jobs:
  train:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: python -m haiip.train all --auto-deploy
```
