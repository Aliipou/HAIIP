"""HAIIP Bearing Fault Classifier -- FastAPI Service"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
import numpy as np
import joblib
import json
import time
from pathlib import Path


# -- Load model once at startup ----------------------------------------------
MODEL_PATH = Path("models/bearing_fault_classifier.pkl")
META_PATH  = Path("models/metadata.json")

pipeline = joblib.load(MODEL_PATH)
with open(META_PATH) as f:
    metadata = json.load(f)


# -- Pydantic schemas --------------------------------------------------------
class SensorReading(BaseModel):
    rms:          float = Field(..., ge=0.0, le=10.0,    description="RMS vibration (g)")
    peak_to_peak: float = Field(..., ge=0.0, le=50.0,    description="Peak-to-peak (g)")
    kurtosis:     float = Field(..., ge=1.0, le=50.0,    description="Kurtosis factor")
    crest_factor: float = Field(..., ge=1.0, le=20.0,    description="Crest factor")
    temperature:  float = Field(..., ge=0.0, le=150.0,   description="Temperature (deg C)")
    rpm:          float = Field(..., ge=0.0, le=10000.0, description="Rotational speed (rpm)")
    machine_id:   Optional[str] = None


class PredictionResponse(BaseModel):
    machine_id:    Optional[str]
    label:         str
    fault_prob:    float
    confidence:    float
    inference_ms:  float
    model_version: str


# -- App ---------------------------------------------------------------------
app = FastAPI(
    title="HAIIP Bearing Fault Classifier",
    version="1.0.0",
    description="Real-time bearing fault detection for industrial predictive maintenance"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)


@app.get("/health")
def health():
    return {"status": "ok", "model": metadata["model_name"], "version": metadata["version"]}


@app.get("/model/info")
def model_info():
    return metadata


@app.post("/predict", response_model=PredictionResponse)
def predict(reading: SensorReading):
    x = np.array([[reading.rms, reading.peak_to_peak, reading.kurtosis,
                   reading.crest_factor, reading.temperature, reading.rpm]])
    t0 = time.perf_counter()
    prob = float(pipeline.predict_proba(x)[0, 1])
    ms   = (time.perf_counter() - t0) * 1000
    return PredictionResponse(
        machine_id=reading.machine_id,
        label="Fault" if prob > 0.5 else "Normal",
        fault_prob=round(prob, 4),
        confidence=round(max(prob, 1 - prob), 4),
        inference_ms=round(ms, 2),
        model_version=metadata["version"]
    )


@app.post("/predict/batch")
def predict_batch(readings: list[SensorReading]):
    if len(readings) > 1000:
        raise HTTPException(status_code=400, detail="Batch size limited to 1000")
    X = np.array([[r.rms, r.peak_to_peak, r.kurtosis, r.crest_factor, r.temperature, r.rpm]
                  for r in readings])
    probs = pipeline.predict_proba(X)[:, 1]
    return [
        {"machine_id": r.machine_id, "fault_prob": round(float(p), 4),
         "label": "Fault" if p > 0.5 else "Normal"}
        for r, p in zip(readings, probs)
    ]