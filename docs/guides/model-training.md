# Model Training Guide

## Prerequisites

- Minimum 72 hours of sensor data per machine
- At least 5 labeled failure events for supervised models
- Python 3.11, PyTorch 2.2

## Training the Anomaly Detector

```bash
# Prepare data
python -m haiip.train prepare \
    --source data/raw/motor-01 \
    --output data/processed/motor-01

# Train Isolation Forest
python -m haiip.train anomaly \
    --data data/processed/motor-01 \
    --model models/motor-01/anomaly_detector \
    --contamination 0.02

# Evaluate
python -m haiip.evaluate \
    --model models/motor-01/anomaly_detector \
    --test-data data/processed/motor-01/test
```

## Training the Maintenance Predictor

```bash
# Train Gradient Boosting (requires labeled failure events)
python -m haiip.train maintenance \
    --data data/processed/motor-01 \
    --labels data/labels/motor-01.csv \
    --model models/motor-01/maintenance_predictor \
    --prediction-horizon 24h  # predict failures 24h ahead
```

## SHAP Explainability

```bash
# Generate SHAP values for model interpretation
python -m haiip.explain \
    --model models/motor-01/maintenance_predictor \
    --data data/processed/motor-01/test \
    --output reports/motor-01-shap.html
```

## ONNX Export (Edge Deployment)

```bash
python -m haiip.export \
    --model models/motor-01/anomaly_detector \
    --format onnx \
    --output models/motor-01/anomaly_detector.onnx
```
