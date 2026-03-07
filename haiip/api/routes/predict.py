"""Prediction routes — single reading and batch prediction."""

import json

import structlog
from fastapi import APIRouter, HTTPException, status
from sqlalchemy import func, select

from haiip.api.deps import DB, CurrentUser
from haiip.api.models import Prediction
from haiip.api.schemas import (
    APIResponse,
    BatchPredictRequest,
    Page,
    PredictionResponse,
    SensorReading,
)
from haiip.core.anomaly import AnomalyDetector
from haiip.core.maintenance import MaintenancePredictor

router = APIRouter()
logger = structlog.get_logger(__name__)

# Module-level model instances (loaded once)
_anomaly_detector: AnomalyDetector | None = None
_maintenance_predictor: MaintenancePredictor | None = None


def _get_anomaly_detector() -> AnomalyDetector:
    global _anomaly_detector
    if _anomaly_detector is None:
        _anomaly_detector = AnomalyDetector()
    return _anomaly_detector


def _get_maintenance_predictor() -> MaintenancePredictor:
    global _maintenance_predictor
    if _maintenance_predictor is None:
        _maintenance_predictor = MaintenancePredictor()
    return _maintenance_predictor


def _reading_to_features(reading: SensorReading) -> list[float]:
    features = [
        reading.air_temperature,
        reading.process_temperature,
        reading.rotational_speed,
        reading.torque,
        reading.tool_wear,
    ]
    if reading.extra_features:
        features.extend(reading.extra_features.values())
    return features


@router.post(
    "/predict",
    response_model=APIResponse[PredictionResponse],
    status_code=status.HTTP_201_CREATED,
    summary="Anomaly detection — single reading",
    description=(
        "Run anomaly detection on a single sensor reading. "
        "Returns label (normal/anomaly), confidence [0,1], anomaly_score [0,1], "
        "SHAP/z-score explanation, and the active model_version for audit traceability."
    ),
)
async def predict_single(
    body: SensorReading,
    current_user: CurrentUser,
    db: DB,
) -> APIResponse[PredictionResponse]:
    """Run anomaly detection on a single sensor reading."""
    import time

    from haiip.api.ml_metrics import record_prediction
    from haiip.core.model_registry import get_active_version

    detector = _get_anomaly_detector()
    features = _reading_to_features(body)
    model_version = get_active_version(current_user.tenant_id, "anomaly_detector")

    t0 = time.perf_counter()
    result = detector.predict(features)
    latency_s = time.perf_counter() - t0

    prediction = Prediction(
        tenant_id=current_user.tenant_id,
        machine_id=body.machine_id,
        model_type="anomaly_detection",
        prediction_label=result["label"],
        confidence=result["confidence"],
        anomaly_score=result.get("anomaly_score"),
        input_features=json.dumps(body.model_dump()),
        explanation=json.dumps(
            {
                **result.get("explanation", {}),
                "model_version": model_version,
            }
        ),
    )
    db.add(prediction)
    await db.flush()
    await db.refresh(prediction)

    record_prediction(
        model_type="anomaly_detection",
        tenant_id=current_user.tenant_id,
        anomaly_score=result.get("anomaly_score"),
        label=result["label"],
        latency_s=latency_s,
    )

    logger.info(
        "predict.single",
        machine_id=body.machine_id,
        label=result["label"],
        confidence=result["confidence"],
        model_version=model_version,
        latency_ms=round(latency_s * 1000, 2),
        tenant_id=current_user.tenant_id,
    )

    return APIResponse(data=prediction)


@router.post(
    "/predict/batch",
    response_model=APIResponse[list[PredictionResponse]],
    status_code=status.HTTP_201_CREATED,
)
async def predict_batch(
    body: BatchPredictRequest,
    current_user: CurrentUser,
    db: DB,
) -> APIResponse[list[PredictionResponse]]:
    """Run predictions on a batch of sensor readings (max 100)."""
    if body.model_type == "anomaly_detection":
        detector = _get_anomaly_detector()
        predictor_fn = detector.predict
    else:
        predictor = _get_maintenance_predictor()
        predictor_fn = predictor.predict

    predictions: list[Prediction] = []
    for reading in body.readings:
        features = _reading_to_features(reading)
        result = predictor_fn(features)

        pred = Prediction(
            tenant_id=current_user.tenant_id,
            machine_id=reading.machine_id,
            model_type=body.model_type,
            prediction_label=result["label"],
            confidence=result["confidence"],
            anomaly_score=result.get("anomaly_score"),
            rul_cycles=result.get("rul_cycles"),
            input_features=json.dumps(reading.model_dump()),
            explanation=json.dumps(result.get("explanation")),
        )
        db.add(pred)
        predictions.append(pred)

    await db.flush()
    for pred in predictions:
        await db.refresh(pred)

    logger.info(
        "predict.batch",
        count=len(predictions),
        model_type=body.model_type,
        tenant_id=current_user.tenant_id,
    )

    return APIResponse(data=predictions)


@router.get("/predictions", response_model=Page[PredictionResponse])
async def list_predictions(
    current_user: CurrentUser,
    db: DB,
    machine_id: str | None = None,
    page: int = 1,
    size: int = 20,
) -> Page[PredictionResponse]:
    """List predictions for the current tenant, optionally filtered by machine."""
    base_query = select(Prediction).where(Prediction.tenant_id == current_user.tenant_id)
    if machine_id:
        base_query = base_query.where(Prediction.machine_id == machine_id)

    count_result = await db.execute(select(func.count()).select_from(base_query.subquery()))
    total = count_result.scalar_one()

    result = await db.execute(
        base_query.order_by(Prediction.created_at.desc()).offset((page - 1) * size).limit(size)
    )
    items = list(result.scalars().all())

    return Page(items=items, total=total, page=page, size=size)


@router.get("/predictions/{prediction_id}", response_model=APIResponse[PredictionResponse])
async def get_prediction(
    prediction_id: str,
    current_user: CurrentUser,
    db: DB,
) -> APIResponse[PredictionResponse]:
    """Get a single prediction by ID."""
    result = await db.execute(
        select(Prediction).where(
            Prediction.id == prediction_id,
            Prediction.tenant_id == current_user.tenant_id,
        )
    )
    prediction = result.scalar_one_or_none()
    if prediction is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Prediction not found")

    return APIResponse(data=prediction)
