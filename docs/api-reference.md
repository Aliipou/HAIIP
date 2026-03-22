# API Reference

Base URL: `http://localhost:8000/api/v1`

## Endpoints

- `GET /machines` - List machines
- `GET /machines/{id}/status` - Machine health
- `GET /machines/{id}/predictions` - Prediction history
- `POST /machines/{id}/predict` - Manual prediction
- `GET /alerts?status=active` - Active alerts
- `GET /audit` - Audit trail
