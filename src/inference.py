"""
inference.py
============
Model inference pipeline for ThreatSense.

Loads all trained artifacts once at module level (scaler, label encoder,
Isolation Forest, XGBoost, SHAP explainer) and exposes two functions:

  predict_single(flow)  — single flow dict  → detection dict
  predict_batch(df)     — DataFrame of flows → list of detection dicts

The detection dict is the contract between the ML layer and the FastAPI
backend / LangGraph agent. Its schema is:

  {
    "predicted_class":      str,
    "anomaly_score":        float,
    "class_probabilities":  {class_name: probability, ...},
    "top_shap_features":    [
        {"feature": str, "value": float, "shap_value": float},
        ...                                 # top 5, sorted by |shap_value|
    ],
    "is_anomaly":           bool,           # True if Isolation Forest flags it
  }
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

logger = logging.getLogger("threatsense.inference")


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _load_artifacts(artifacts_dir: str = "models") -> dict[str, Any]:
    """Load all model artifacts from disk, caching them in memory.

    Using lru_cache means models are loaded once per process — not on every
    request — which keeps API latency low.

    Args:
        artifacts_dir: Directory containing the .pkl files.

    Returns:
        Dict with keys: scaler, label_encoder, iso_forest,
        xgboost_model, shap_explainer.

    Raises:
        FileNotFoundError: If any required artifact is missing.
    """
    artifacts_dir = Path(artifacts_dir)
    required = [
        "scaler.pkl",
        "label_encoder.pkl",
        "iso_forest.pkl",
        "xgboost_model.pkl",
        "shap_explainer.pkl",
    ]
    for fname in required:
        if not (artifacts_dir / fname).exists():
            raise FileNotFoundError(
                f"Missing artifact: {artifacts_dir / fname}\n"
                "Run src.train before starting the API."
            )

    logger.info("Loading model artifacts from %s ...", artifacts_dir)
    artifacts = {
        "scaler":          joblib.load(artifacts_dir / "scaler.pkl"),
        "label_encoder":   joblib.load(artifacts_dir / "label_encoder.pkl"),
        "iso_forest":      joblib.load(artifacts_dir / "iso_forest.pkl"),
        "xgboost_model":   joblib.load(artifacts_dir / "xgboost_model.pkl"),
        "shap_explainer":  joblib.load(artifacts_dir / "shap_explainer.pkl"),
    }
    logger.info("All artifacts loaded.")
    return artifacts


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_top_shap_features(
    shap_values: np.ndarray,
    feature_names: list[str],
    feature_values: np.ndarray,
    top_n: int = 5,
) -> list[dict]:
    """Extract the top N most influential features from a SHAP value array.

    For multiclass models, shap_values may be 2-D (n_features, n_classes)
    or 1-D (n_features,). We take the mean absolute value across classes
    to get a single global importance score per feature.

    Args:
        shap_values:    SHAP values for one sample.
        feature_names:  Names of all features.
        feature_values: Raw (scaled) feature values for that sample.
        top_n:          How many top features to return.

    Returns:
        List of dicts sorted by descending |shap_value|.
    """
    if shap_values.ndim == 2:
        # (n_features, n_classes) — take mean absolute across classes
        importance = np.abs(shap_values).mean(axis=1)
        # Use the shap value of the predicted class (last dim) for sign
        shap_1d = shap_values[:, np.argmax(np.abs(shap_values).sum(axis=0))]
    else:
        importance = np.abs(shap_values)
        shap_1d = shap_values

    top_idx = np.argsort(importance)[::-1][:top_n]
    return [
        {
            "feature":    feature_names[i],
            "value":      float(feature_values[i]),
            "shap_value": float(shap_1d[i]),
        }
        for i in top_idx
    ]


def _run_inference(
    X_scaled: np.ndarray,
    feature_names: list[str],
    artifacts: dict,
) -> list[dict]:
    """Core inference: run all models on a scaled feature matrix.

    Args:
        X_scaled:      Already-scaled feature matrix (n_samples, n_features).
        feature_names: Feature column names.
        artifacts:     Loaded model artifacts dict.

    Returns:
        List of detection dicts, one per row.
    """
    le      = artifacts["label_encoder"]
    iso     = artifacts["iso_forest"]
    xgb     = artifacts["xgboost_model"]
    explainer = artifacts["shap_explainer"]

    # Isolation Forest — anomaly scores (negative = more anomalous)
    anomaly_scores = iso.score_samples(X_scaled)
    iso_predictions = iso.predict(X_scaled)    # 1 = normal, -1 = anomaly

    # XGBoost — class probabilities + hard predictions
    proba = xgb.predict_proba(X_scaled)        # (n, 5)
    pred_ints = proba.argmax(axis=1)

    # SHAP — compute for each sample
    raw_shap = explainer.shap_values(X_scaled) # list[ndarray] or 3-D ndarray

    results = []
    for i in range(len(X_scaled)):
        class_name = le.inverse_transform([pred_ints[i]])[0]
        class_probs = {
            le.inverse_transform([j])[0]: float(proba[i, j])
            for j in range(proba.shape[1])
        }

        # Extract per-sample SHAP values
        if isinstance(raw_shap, list):
            # Old SHAP API: list of (n_samples, n_features) arrays
            sv_i = np.stack([sv[i] for sv in raw_shap], axis=1)  # (n_feat, n_cls)
        elif raw_shap.ndim == 3:
            # New SHAP API: (n_samples, n_features, n_classes)
            sv_i = raw_shap[i]   # (n_feat, n_cls)
        else:
            sv_i = raw_shap[i]   # binary / single output

        top_shap = _get_top_shap_features(sv_i, feature_names, X_scaled[i])

        results.append({
            "predicted_class":     class_name,
            "anomaly_score":       float(anomaly_scores[i]),
            "is_anomaly":          bool(iso_predictions[i] == -1),
            "class_probabilities": class_probs,
            "top_shap_features":   top_shap,
        })

    return results


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def predict_single(
    flow: dict[str, float],
    artifacts_dir: str = "models",
) -> dict:
    """Run inference on a single network flow.

    Args:
        flow:          Dict mapping feature names to float values.
                       Must contain the same features the model was trained on.
        artifacts_dir: Directory containing model .pkl files.

    Returns:
        Detection dict with keys: predicted_class, anomaly_score, is_anomaly,
        class_probabilities, top_shap_features.

    Raises:
        ValueError: If required features are missing from the flow dict.
    """
    artifacts     = _load_artifacts(artifacts_dir)
    scaler        = artifacts["scaler"]
    feature_names = list(scaler.feature_names_in_)

    missing = set(feature_names) - set(flow.keys())
    if missing:
        raise ValueError(f"Missing features in input flow: {missing}")

    X = pd.DataFrame([flow])[feature_names].astype("float32")
    X_scaled = scaler.transform(X)

    results = _run_inference(X_scaled, feature_names, artifacts)
    return results[0]


def predict_batch(
    df: pd.DataFrame,
    artifacts_dir: str = "models",
) -> list[dict]:
    """Run inference on a DataFrame of network flows.

    Rows with any NaN / infinite feature values are dropped before inference
    and their results are returned as error entries.

    Args:
        df:            DataFrame where each row is one network flow.
                       Column names must match the training features.
        artifacts_dir: Directory containing model .pkl files.

    Returns:
        List of detection dicts (same schema as predict_single), one per row.
        Dropped rows appear as {"error": "invalid features", "row_index": i}.
    """
    artifacts     = _load_artifacts(artifacts_dir)
    scaler        = artifacts["scaler"]
    feature_names = list(scaler.feature_names_in_)

    # Keep only the feature columns present in the model
    available = [c for c in feature_names if c in df.columns]
    missing   = set(feature_names) - set(available)
    if missing:
        raise ValueError(f"DataFrame is missing required feature columns: {missing}")

    X = df[feature_names].astype("float32")

    # Track and drop invalid rows
    bad_mask = X.isnull().any(axis=1) | np.isinf(X.to_numpy()).any(axis=1)
    good_idx = X.index[~bad_mask].tolist()
    bad_idx  = X.index[bad_mask].tolist()

    X_clean  = X.loc[good_idx]
    X_scaled = scaler.transform(X_clean)

    good_results = _run_inference(X_scaled, feature_names, artifacts)

    # Reconstruct output aligned to original row order
    good_iter = iter(good_results)
    bad_set   = set(bad_idx)
    output    = []
    for i in range(len(df)):
        if i in bad_set:
            output.append({"error": "invalid features", "row_index": i})
        else:
            output.append(next(good_iter))

    logger.info(
        "Batch inference: %d rows processed, %d dropped (invalid)",
        len(good_idx), len(bad_idx),
    )
    return output
