"""
tests/test_api.py
=================
Integration tests for the ThreatSense FastAPI backend.

Uses FastAPI's TestClient — no running server needed.
ML model calls are patched at their source modules so no .pkl files required.

Run:
    pytest tests/test_api.py -v
"""

import math
from unittest.mock import patch

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

MOCK_DETECTION = {
    "predicted_class":     "DoS",
    "anomaly_score":       -0.152,
    "is_anomaly":          True,
    "class_probabilities": {
        "Benign": 0.02, "DoS": 0.94,
        "PortScan": 0.01, "Brute Force": 0.02, "Bot": 0.01,
    },
    "top_shap_features": [
        {"feature": "Flow Duration",     "value": 0.0002, "shap_value": 2.341},
        {"feature": "Total Fwd Packets", "value": 1.0,    "shap_value": 1.872},
    ],
}

MOCK_REPORT = {
    "incident_id":         "INC-TEST0001",
    "timestamp":           "2026-01-01T00:00:00+00:00",
    "threat_class":        "DoS",
    "severity":            "Critical",
    "anomaly_score":       -0.152,
    "confidence":          0.94,
    "triage_summary":      "DoS attack detected.",
    "mitre_techniques":    [],
    "top_indicators":      [],
    "executive_summary":   "A DoS attack was detected.",
    "threat_analysis":     "Volumetric attack targeting HTTP port.",
    "recommended_actions": ["Block source IP", "Enable rate limiting"],
    "investigation_notes": "Check firewall logs.",
    "generated_by":        "fallback-template",
}


@pytest.fixture
def client(tmp_path):
    """TestClient with a temp SQLite DB and mocked ML artifact loading."""
    with patch("api.main.DB_PATH", tmp_path / "test.db"), \
         patch("src.inference._load_artifacts"):
        from api.main import app, init_db
        init_db()
        with TestClient(app) as c:
            yield c


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------

class TestHealth:

    def test_health_returns_ok(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"

    def test_health_returns_version(self, client):
        r = client.get("/health")
        assert "version" in r.json()


# ---------------------------------------------------------------------------
# /predict
# ---------------------------------------------------------------------------

class TestPredict:

    def test_valid_prediction_returns_200(self, client):
        # Patch at the source module, not at api.main (local import)
        with patch("src.inference.predict_single", return_value=MOCK_DETECTION):
            r = client.post("/predict", json={
                "features": {"Flow Duration": 1234.0, "Flow Bytes/s": 9843.0}
            })
        assert r.status_code == 200

    def test_prediction_contains_required_fields(self, client):
        with patch("src.inference.predict_single", return_value=MOCK_DETECTION):
            r = client.post("/predict", json={"features": {"Flow Duration": 1.0}})
        data = r.json()
        required = {
            "incident_id", "predicted_class", "anomaly_score",
            "is_anomaly", "confidence", "severity",
            "class_probabilities", "top_shap_features",
        }
        assert required.issubset(data.keys())

    def test_predicted_class_is_string(self, client):
        with patch("src.inference.predict_single", return_value=MOCK_DETECTION):
            r = client.post("/predict", json={"features": {"Flow Duration": 1.0}})
        assert isinstance(r.json()["predicted_class"], str)

    def test_infinite_feature_rejected_by_validator(self):
        # float("inf") can't be sent as JSON — test the Pydantic validator directly
        from api.main import FlowFeatures
        with pytest.raises(ValidationError):
            FlowFeatures(features={"Flow Bytes/s": math.inf})

    def test_nan_feature_rejected_by_validator(self):
        from api.main import FlowFeatures
        with pytest.raises(ValidationError):
            FlowFeatures(features={"Flow Bytes/s": math.nan})

    def test_incident_persisted_to_db(self, client):
        with patch("src.inference.predict_single", return_value=MOCK_DETECTION):
            r = client.post("/predict", json={"features": {"Flow Duration": 1.0}})
        incident_id = r.json()["incident_id"]
        r2 = client.get(f"/incidents/{incident_id}")
        assert r2.status_code == 200
        assert r2.json()["predicted_class"] == "DoS"


# ---------------------------------------------------------------------------
# /analyze (batch CSV upload)
# ---------------------------------------------------------------------------

class TestAnalyzeBatch:

    def _make_csv(self, n: int = 5) -> bytes:
        df = pd.DataFrame({
            "Flow Duration":    [1000.0] * n,
            "Total Fwd Packets":[10.0]   * n,
            "Flow Bytes/s":     [500.0]  * n,
            "Flow Packets/s":   [1.0]    * n,
            "Destination Port": [80.0]   * n,
        })
        return df.to_csv(index=False).encode()

    def test_valid_csv_returns_200(self, client):
        mock_results = [MOCK_DETECTION.copy() for _ in range(5)]
        with patch("src.inference.predict_batch", return_value=mock_results):
            r = client.post(
                "/analyze",
                files={"file": ("test.csv", self._make_csv(), "text/csv")},
            )
        assert r.status_code == 200

    def test_summary_contains_required_fields(self, client):
        mock_results = [MOCK_DETECTION.copy() for _ in range(5)]
        with patch("src.inference.predict_batch", return_value=mock_results):
            r = client.post(
                "/analyze",
                files={"file": ("test.csv", self._make_csv(), "text/csv")},
            )
        data = r.json()
        assert "total_flows"      in data
        assert "threat_breakdown" in data
        assert "anomaly_count"    in data
        assert "incident_ids"     in data

    def test_non_csv_file_rejected(self, client):
        r = client.post(
            "/analyze",
            files={"file": ("test.txt", b"hello", "text/plain")},
        )
        assert r.status_code == 400

    def test_empty_csv_rejected(self, client):
        r = client.post(
            "/analyze",
            files={"file": ("empty.csv", b"", "text/csv")},
        )
        assert r.status_code in (400, 422)


# ---------------------------------------------------------------------------
# /incidents
# ---------------------------------------------------------------------------

class TestListIncidents:

    def _add_incident(self, client):
        with patch("src.inference.predict_single", return_value=MOCK_DETECTION):
            r = client.post("/predict", json={"features": {"Flow Duration": 1.0}})
        return r.json()["incident_id"]

    def test_empty_list_on_fresh_db(self, client):
        r = client.get("/incidents")
        assert r.status_code == 200
        assert r.json() == []

    def test_incident_appears_after_predict(self, client):
        self._add_incident(client)
        r = client.get("/incidents")
        assert r.status_code == 200
        assert len(r.json()) == 1

    def test_limit_parameter(self, client):
        for _ in range(5):
            self._add_incident(client)
        r = client.get("/incidents?limit=3")
        assert len(r.json()) == 3

    def test_threat_class_filter(self, client):
        self._add_incident(client)
        r = client.get("/incidents?threat_class=DoS")
        assert all(i["predicted_class"] == "DoS" for i in r.json())

    def test_get_single_incident(self, client):
        inc_id = self._add_incident(client)
        r = client.get(f"/incidents/{inc_id}")
        assert r.status_code == 200
        assert r.json()["id"] == inc_id

    def test_get_nonexistent_incident_returns_404(self, client):
        r = client.get("/incidents/INC-DOESNOTEXIST")
        assert r.status_code == 404


# ---------------------------------------------------------------------------
# /report
# ---------------------------------------------------------------------------

class TestGenerateReport:

    def _add_incident(self, client):
        with patch("src.inference.predict_single", return_value=MOCK_DETECTION):
            r = client.post("/predict", json={"features": {"Flow Duration": 1.0}})
        return r.json()["incident_id"]

    def test_report_returns_200(self, client):
        inc_id = self._add_incident(client)
        with patch("src.agent.generate_incident_report", return_value=MOCK_REPORT):
            r = client.post(f"/report/{inc_id}")
        assert r.status_code == 200

    def test_report_contains_required_fields(self, client):
        inc_id = self._add_incident(client)
        with patch("src.agent.generate_incident_report", return_value=MOCK_REPORT):
            r = client.post(f"/report/{inc_id}")
        required = {
            "incident_id", "threat_class", "severity",
            "executive_summary", "recommended_actions",
        }
        assert required.issubset(r.json().keys())

    def test_report_on_unknown_incident_returns_404(self, client):
        r = client.post("/report/INC-DOESNOTEXIST")
        assert r.status_code == 404
