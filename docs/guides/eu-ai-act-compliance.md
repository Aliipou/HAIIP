# EU AI Act Compliance Guide

## HAIIP Classification

HAIIP falls under **Limited Risk** (Article 52) of the EU AI Act:

- It does not make autonomous decisions without human oversight
- The operator always has an override option
- All AI decisions are explained in plain language
- Full audit trail is maintained

## Compliance Checklist

### Transparency (Article 52)
- [x] Users notified they are interacting with an AI system
- [x] Every prediction includes a plain-language explanation
- [x] Confidence scores displayed with all predictions

### Human Oversight
- [x] Override button visible on every AI decision screen
- [x] Override actions logged with timestamp and reason
- [x] No autonomous actuation without operator confirmation

### Audit Trail
- [x] Every prediction stored with: timestamp, model version, input features, output, confidence
- [x] Audit records are append-only (no modification)
- [x] Retention period: 5 years (configurable)

### Data Governance
- [x] No personal data processed
- [x] Sensor data anonymized (site ID and machine serial removed before storage)
- [x] Data sharing agreements in place with all SME partners

## Generating Compliance Reports

```bash
# Monthly compliance report
python -m haiip.compliance report --month 2024-01 --output report.pdf

# Audit trail export
python -m haiip.compliance export --start 2024-01-01 --end 2024-01-31 --format csv
```

## Model Card and Dataset Card

See [MODEL_CARD.md](../MODEL_CARD.md) and [DATASET_CARD.md](../DATASET_CARD.md) for EU AI Act technical documentation.
