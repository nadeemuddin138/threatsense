"""
api/main.py
===========
ThreatSense FastAPI backend.

Routes
------
  POST /predict              — single flow prediction
  POST /analyze              — batch CSV upload + analysis
  POST /report/{incident_id} — generate LLM incident report
  GET  /incidents            — list all stored incidents
  GET  /incidents/{id}       — get one incident by ID
  GET  /health               — health check

Data persistence
----------------
  SQLite database at data/threatsense.db (auto-created on startup).
  Two tables: incidents, reports.

Run
---
  uvicorn api.main:app --reload --port 8000

Then open:
  http://localhost:8000/docs   ← Swagger UI
  http://localhost:8000/redoc  ← ReDoc
"""

from __future__ import annotations

import io
import json
import logging
import os
import sqlite3
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("threatsense.api")

DB_PATH = Path(os.getenv("DB_PATH", "data/threatsense.db"))


# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------

def get_db() -> sqlite3.Connection:
    """Return a SQLite connection with row_factory set to Row.

    Returns:
        sqlite3.Connection configured to return dict-like rows.
    """
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Create database tables if they don't exist.

    Tables
    ------
    incidents — one row per detection event.
    reports   — one row per generated LLM/fallback incident report.
    """
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = get_db()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS incidents (
            id              TEXT PRIMARY KEY,
            created_at      TEXT NOT NULL,
            predicted_class TEXT NOT NULL,
            anomaly_score   REAL NOT NULL,
            is_anomaly      INTEGER NOT NULL,
            confidence      REAL NOT NULL,
            severity        TEXT NOT NULL,
            detection_json  TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS reports (
            id              TEXT PRIMARY KEY,
            incident_id     TEXT NOT NULL,
            created_at      TEXT NOT NULL,
            threat_class    TEXT NOT NULL,
            severity        TEXT NOT NULL,
            report_json     TEXT NOT NULL,
            FOREIGN KEY (incident_id) REFERENCES incidents(id)
        );
    """)
    conn.commit()
    conn.close()
    logger.info("Database ready at %s", DB_PATH)


# ---------------------------------------------------------------------------
# Lifespan (startup / shutdown)
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialise DB and pre-load models on startup."""
    logger.info("ThreatSense API starting up ...")
    init_db()
    # Pre-load models so the first request isn't slow
    try:
        from src.inference import _load_artifacts
        _load_artifacts()
        logger.info("Models pre-loaded successfully.")
    except FileNotFoundError as exc:
        logger.warning("Models not found on startup: %s", exc)
    yield
    logger.info("ThreatSense API shutting down.")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="ThreatSense API",
    description="LLM-powered SIEM analyst — CICIDS2017 threat detection & SOC report generation.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],    # tighten in production
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class FlowFeatures(BaseModel):
    """Network flow feature vector for single-flow prediction.

    All values must be finite floats. Feature names must match the columns
    the model was trained on (see data/processed/X_train.csv header).
    """
    features: dict[str, float] = Field(
        ...,
        description="Mapping of feature name to float value.",
        example={
            "Destination Port": 80.0,
            "Flow Duration": 1234567.0,
            "Total Fwd Packets": 12.0,
            "Flow Bytes/s": 9843.21,
            "Flow Packets/s": 1.23,
        },
    )

    @field_validator("features")
    @classmethod
    def features_must_be_finite(cls, v: dict) -> dict:
        """Reject any Infinity or NaN values in the feature dict."""
        import math
        for name, val in v.items():
            if not math.isfinite(val):
                raise ValueError(f"Feature '{name}' has a non-finite value: {val}")
        return v


class SHAPFeature(BaseModel):
    """A single SHAP feature contribution."""
    feature:    str
    value:      float
    shap_value: float


class DetectionResult(BaseModel):
    """Detection result returned by /predict."""
    incident_id:        str
    predicted_class:    str
    anomaly_score:      float
    is_anomaly:         bool
    confidence:         float
    severity:           str
    class_probabilities: dict[str, float]
    top_shap_features:  list[SHAPFeature]


class BatchSummary(BaseModel):
    """Summary returned by /analyze after batch CSV upload."""
    total_flows:      int
    threat_breakdown: dict[str, int]
    anomaly_count:    int
    incident_ids:     list[str]
    errors:           int


class IncidentSummary(BaseModel):
    """One row from the incidents table."""
    id:              str
    created_at:      str
    predicted_class: str
    anomaly_score:   float
    is_anomaly:      bool
    confidence:      float
    severity:        str


class IncidentReport(BaseModel):
    """Full incident report returned by /report/{incident_id}."""
    incident_id:         str
    timestamp:           str
    threat_class:        str
    severity:            str
    anomaly_score:       float
    confidence:          float
    triage_summary:      str
    mitre_techniques:    list[dict]
    top_indicators:      list[str]
    executive_summary:   str
    threat_analysis:     str
    recommended_actions: list[str]
    investigation_notes: str
    generated_by:        str


# ---------------------------------------------------------------------------
# Helper: persist incident to DB
# ---------------------------------------------------------------------------

def _save_incident(detection: dict, incident_id: str) -> None:
    """Persist a detection result to the incidents table.

    Args:
        detection:   Detection dict from inference pipeline.
        incident_id: Unique incident ID string.
    """
    from src.mitre_mapper import get_severity
    severity   = get_severity(detection.get("predicted_class", "Benign"))
    confidence = max(detection.get("class_probabilities", {}).values(), default=0.0)

    conn = get_db()
    conn.execute(
        """
        INSERT OR REPLACE INTO incidents
            (id, created_at, predicted_class, anomaly_score,
             is_anomaly, confidence, severity, detection_json)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            incident_id,
            datetime.now(timezone.utc).isoformat(),
            detection.get("predicted_class", "Unknown"),
            detection.get("anomaly_score", 0.0),
            int(detection.get("is_anomaly", False)),
            round(confidence, 4),
            severity,
            json.dumps(detection),
        ),
    )
    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health", tags=["System"])
def health_check() -> dict:
    """Return API health status.

    Returns:
        Dict with status and version.
    """
    return {"status": "ok", "version": "1.0.0"}


@app.post("/predict", response_model=DetectionResult, tags=["Detection"])
def predict(payload: FlowFeatures) -> DetectionResult:
    """Run threat detection on a single network flow.

    Scales the input, runs Isolation Forest + XGBoost, computes SHAP
    explanations, maps severity, and persists the incident to SQLite.

    Args:
        payload: FlowFeatures with a dict of feature_name -> float.

    Returns:
        DetectionResult with class, anomaly score, probabilities, SHAP.

    Raises:
        422: If any feature value is non-finite.
        500: If model inference fails.
    """
    try:
        from src.inference import predict_single
        from src.mitre_mapper import get_severity

        detection   = predict_single(payload.features)
        incident_id = f"INC-{uuid.uuid4().hex[:8].upper()}"
        severity    = get_severity(detection["predicted_class"])
        confidence  = max(detection["class_probabilities"].values(), default=0.0)

        _save_incident(detection, incident_id)

        logger.info(
            "POST /predict  id=%s  class=%s  severity=%s",
            incident_id, detection["predicted_class"], severity,
        )
        return DetectionResult(
            incident_id=incident_id,
            predicted_class=detection["predicted_class"],
            anomaly_score=detection["anomaly_score"],
            is_anomaly=detection["is_anomaly"],
            confidence=round(confidence, 4),
            severity=severity,
            class_probabilities=detection["class_probabilities"],
            top_shap_features=[SHAPFeature(**f) for f in detection["top_shap_features"]],
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        logger.error("POST /predict error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/analyze", response_model=BatchSummary, tags=["Detection"])
async def analyze_batch(file: UploadFile = File(...)) -> BatchSummary:
    """Analyze an uploaded CSV of network flows.

    Accepts a CSV file (max 50 MB) with the same feature columns as the
    training data. Runs inference on every row, persists all incidents,
    and returns a summary breakdown.

    Args:
        file: Uploaded .csv file.

    Returns:
        BatchSummary with threat breakdown, anomaly count, incident IDs.

    Raises:
        400: If file is not a CSV or columns are missing.
        413: If file exceeds 50 MB.
        500: If batch inference fails.
    """
    # Validate file type
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only .csv files are accepted.")

    contents = await file.read()

    # Security flag: reject files over 50 MB
    if len(contents) > 50 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="File exceeds 50 MB limit.")

    try:
        df = pd.read_csv(io.StringIO(contents.decode("utf-8")))
    except UnicodeDecodeError:
        df = pd.read_csv(io.StringIO(contents.decode("latin-1")))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not parse CSV: {exc}")

    if df.empty:
        raise HTTPException(status_code=400, detail="Uploaded CSV is empty.")

    try:
        from src.inference import predict_batch

        results = predict_batch(df)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.error("POST /analyze error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))

    incident_ids    = []
    threat_breakdown: dict[str, int] = {}
    anomaly_count   = 0
    error_count     = 0

    for det in results:
        if "error" in det:
            error_count += 1
            continue
        incident_id = f"INC-{uuid.uuid4().hex[:8].upper()}"
        _save_incident(det, incident_id)
        incident_ids.append(incident_id)

        cls = det["predicted_class"]
        threat_breakdown[cls] = threat_breakdown.get(cls, 0) + 1
        if det["is_anomaly"]:
            anomaly_count += 1

    logger.info(
        "POST /analyze  rows=%d  threats=%s  anomalies=%d  errors=%d",
        len(df), threat_breakdown, anomaly_count, error_count,
    )
    return BatchSummary(
        total_flows=len(df),
        threat_breakdown=threat_breakdown,
        anomaly_count=anomaly_count,
        incident_ids=incident_ids,
        errors=error_count,
    )


@app.post("/report/{incident_id}", response_model=IncidentReport, tags=["Reports"])
def generate_report(incident_id: str) -> IncidentReport:
    """Generate a SOC incident report for an existing incident.

    Looks up the incident in SQLite, passes the detection to the LangGraph
    agent, and persists + returns the generated report.

    Args:
        incident_id: Incident ID string (e.g. "INC-A3F2B1C9").

    Returns:
        Full IncidentReport with LLM narrative and recommended actions.

    Raises:
        404: If incident_id is not found in the database.
        500: If report generation fails.
    """
    conn = get_db()
    row  = conn.execute(
        "SELECT detection_json FROM incidents WHERE id = ?", (incident_id,)
    ).fetchone()
    conn.close()

    if row is None:
        raise HTTPException(
            status_code=404,
            detail=f"Incident '{incident_id}' not found.",
        )

    detection = json.loads(row["detection_json"])

    try:
        from src.agent import generate_incident_report
        report = generate_incident_report(detection)
    except Exception as exc:
        logger.error("POST /report/%s error: %s", incident_id, exc)
        raise HTTPException(status_code=500, detail=str(exc))

    # Override incident_id with the one from the URL
    report["incident_id"] = incident_id

    # Persist the report
    conn = get_db()
    conn.execute(
        """
        INSERT OR REPLACE INTO reports
            (id, incident_id, created_at, threat_class, severity, report_json)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            f"RPT-{uuid.uuid4().hex[:8].upper()}",
            incident_id,
            datetime.now(timezone.utc).isoformat(),
            report.get("threat_class", "Unknown"),
            report.get("severity", "Unknown"),
            json.dumps(report),
        ),
    )
    conn.commit()
    conn.close()

    logger.info("POST /report/%s  class=%s", incident_id, report.get("threat_class"))
    return IncidentReport(**report)


@app.get("/incidents", response_model=list[IncidentSummary], tags=["Incidents"])
def list_incidents(
    limit: int = 50,
    threat_class: Optional[str] = None,
) -> list[IncidentSummary]:
    """List stored incidents, most recent first.

    Args:
        limit:        Max number of incidents to return (default 50, max 500).
        threat_class: Optional filter by class name (e.g. "DoS").

    Returns:
        List of IncidentSummary rows from the database.
    """
    limit = min(limit, 500)
    conn  = get_db()
    if threat_class:
        rows = conn.execute(
            "SELECT * FROM incidents WHERE predicted_class = ? ORDER BY created_at DESC LIMIT ?",
            (threat_class, limit),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM incidents ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
    conn.close()
    return [
        IncidentSummary(
            id=r["id"],
            created_at=r["created_at"],
            predicted_class=r["predicted_class"],
            anomaly_score=r["anomaly_score"],
            is_anomaly=bool(r["is_anomaly"]),
            confidence=r["confidence"],
            severity=r["severity"],
        )
        for r in rows
    ]


@app.get("/incidents/{incident_id}", response_model=IncidentSummary, tags=["Incidents"])
def get_incident(incident_id: str) -> IncidentSummary:
    """Retrieve a single incident by ID.

    Args:
        incident_id: Incident ID string.

    Returns:
        IncidentSummary for the requested incident.

    Raises:
        404: If not found.
    """
    conn = get_db()
    row  = conn.execute(
        "SELECT * FROM incidents WHERE id = ?", (incident_id,)
    ).fetchone()
    conn.close()
    if row is None:
        raise HTTPException(status_code=404, detail=f"Incident '{incident_id}' not found.")
    return IncidentSummary(
        id=row["id"],
        created_at=row["created_at"],
        predicted_class=row["predicted_class"],
        anomaly_score=row["anomaly_score"],
        is_anomaly=bool(row["is_anomaly"]),
        confidence=row["confidence"],
        severity=row["severity"],
    )
