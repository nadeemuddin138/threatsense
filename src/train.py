"""
train.py
========
Training pipeline for the two ThreatSense ML models.

Models
------
1. Isolation Forest  (sklearn)
   Unsupervised anomaly detector. Outputs a continuous anomaly score
   per flow; negative score = more anomalous. Used as a first-pass
   filter and as an extra signal in the inference pipeline.

2. XGBoost  (xgboost)
   Multiclass classifier for the 5 ThreatSense threat categories:
       Benign | DoS | PortScan | Brute Force | Bot
   Class imbalance is handled via compute_sample_weight so every class
   contributes equally to the loss regardless of frequency.

Outputs (saved to --artifacts-dir, default: models/)
-----------------------------------------------------
   iso_forest.pkl        Trained Isolation Forest
   xgboost_model.pkl     Trained XGBoost classifier
   shap_explainer.pkl    SHAP TreeExplainer (used by inference + dashboard)
   metrics.json          F1 / accuracy / per-class breakdown
   docs/confusion_matrix.png
   docs/shap_summary.png

Run
---
   # Full dataset (may take 10-20 min on 2.7 GB training set):
   python -m src.train

   # Fast dev run on 100k rows (2-3 min):
   python -m src.train --sample 100000

   # Skip SHAP plots (saves time, still saves the explainer):
   python -m src.train --no-shap-plot
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
import xgboost as xgb
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.utils.class_weight import compute_sample_weight

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("threatsense.train")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_processed_data(
    processed_dir: str | Path,
    sample: int | None = None,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    """Load the four processed splits produced by preprocess.py.

    Args:
        processed_dir: Directory containing X_train/X_test/y_train/y_test CSVs.
        sample: If set, randomly sample this many rows from the training set.
                Useful for quick development runs on the full 2.7 GB dataset.
        random_state: Seed for reproducible sampling.

    Returns:
        Tuple of (X_train, X_test, y_train, y_test) where labels are int arrays.

    Raises:
        FileNotFoundError: If any expected split file is missing.
    """
    processed_dir = Path(processed_dir)
    expected = ["X_train.csv", "X_test.csv", "y_train.csv", "y_test.csv"]
    for f in expected:
        if not (processed_dir / f).exists():
            raise FileNotFoundError(
                f"Missing {f} in {processed_dir}. Run src.preprocess first."
            )

    logger.info("Loading processed splits from %s ...", processed_dir)
    t0 = time.time()

    # float32 halves RAM usage (train set is ~2.7 GB as float64 CSV).
    X_train = pd.read_csv(processed_dir / "X_train.csv", dtype="float32")
    X_test  = pd.read_csv(processed_dir / "X_test.csv",  dtype="float32")
    y_train = pd.read_csv(processed_dir / "y_train.csv")["Label"].to_numpy()
    y_test  = pd.read_csv(processed_dir / "y_test.csv")["Label"].to_numpy()

    logger.info("Loaded in %.1f s  |  train %s  test %s", time.time() - t0, X_train.shape, X_test.shape)

    if sample and sample < len(X_train):
        rng = np.random.default_rng(random_state)
        idx = rng.choice(len(X_train), size=sample, replace=False)
        X_train = X_train.iloc[idx].reset_index(drop=True)
        y_train = y_train[idx]
        logger.info("Sampled %d rows from training set for fast run", sample)

    return X_train, X_test, y_train, y_test


# ---------------------------------------------------------------------------
# Isolation Forest
# ---------------------------------------------------------------------------

def train_isolation_forest(
    X_train: pd.DataFrame,
    contamination: float | str = "auto",
    n_estimators: int = 100,
    random_state: int = 42,
) -> IsolationForest:
    """Train an Isolation Forest anomaly detector on the training features.

    The contamination parameter controls the expected proportion of anomalies.
    "auto" uses the IsolationForest default (0.1). For CICIDS2017 you can also
    pass the true fraction of attack flows, e.g. contamination=0.44.

    Args:
        X_train: Training feature DataFrame (already scaled).
        contamination: Expected proportion of anomalies, or "auto".
        n_estimators: Number of isolation trees.
        random_state: Seed for reproducibility.

    Returns:
        Fitted IsolationForest model.
    """
    logger.info("Training Isolation Forest  (n_estimators=%d) ...", n_estimators)
    t0 = time.time()
    iso = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1,
    )
    iso.fit(X_train)
    logger.info("Isolation Forest trained in %.1f s", time.time() - t0)
    return iso


# ---------------------------------------------------------------------------
# XGBoost
# ---------------------------------------------------------------------------

def compute_class_weights(y: np.ndarray) -> np.ndarray:
    """Compute per-sample weights so every class is treated equally.

    CICIDS2017 is heavily imbalanced (Benign dominates). Without weighting,
    XGBoost learns to predict Benign almost exclusively and achieves high
    accuracy but near-zero recall on minority attack classes.

    Args:
        y: Integer label array.

    Returns:
        Float array of per-sample weights, same length as y.
    """
    weights = compute_sample_weight(class_weight="balanced", y=y)
    unique, counts = np.unique(y, return_counts=True)
    logger.info(
        "Class distribution (train): %s",
        dict(zip(unique.tolist(), counts.tolist())),
    )
    return weights


def train_xgboost(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    n_estimators: int = 300,
    max_depth: int = 6,
    learning_rate: float = 0.1,
    early_stopping_rounds: int = 20,
    random_state: int = 42,
) -> xgb.XGBClassifier:
    """Train an XGBoost multiclass classifier with early stopping.

    Uses sample weights to handle class imbalance. Tree method is set to
    'hist' for speed on large datasets. GPU is used automatically if
    available (xgboost ≥ 2.0 handles device selection).

    Args:
        X_train: Training features.
        y_train: Training labels (int 0..4).
        X_val: Validation features for early stopping.
        y_val: Validation labels.
        n_estimators: Maximum number of boosting rounds.
        max_depth: Maximum tree depth.
        learning_rate: Step size shrinkage.
        early_stopping_rounds: Stop if validation loss doesn't improve.
        random_state: Seed for reproducibility.

    Returns:
        Fitted XGBClassifier.
    """
    n_classes = len(np.unique(y_train))
    sample_weights = compute_class_weights(y_train)

    logger.info(
        "Training XGBoost  (max_estimators=%d, max_depth=%d, lr=%.2f, classes=%d) ...",
        n_estimators, max_depth, learning_rate, n_classes,
    )
    t0 = time.time()

    model = xgb.XGBClassifier(
        objective="multi:softprob",
        num_class=n_classes,
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist",          # fast histogram method
        eval_metric="mlogloss",
        early_stopping_rounds=early_stopping_rounds,
        random_state=random_state,
        n_jobs=-1,
        verbosity=0,
    )
    model.fit(
        X_train, y_train,
        sample_weight=sample_weights,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    best = model.best_iteration
    logger.info(
        "XGBoost trained in %.1f s  |  best round: %d",
        time.time() - t0, best,
    )
    return model


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(
    model: xgb.XGBClassifier,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    label_encoder: object,
    docs_dir: Path,
) -> dict:
    """Evaluate the XGBoost model and save a confusion matrix PNG.

    Args:
        model: Trained XGBClassifier.
        X_test: Test features.
        y_test: True integer labels.
        label_encoder: Fitted LabelEncoder (for class names on the plot).
        docs_dir: Directory to save confusion_matrix.png.

    Returns:
        Dictionary of evaluation metrics (macro F1, weighted F1, per-class).
    """
    docs_dir.mkdir(parents=True, exist_ok=True)
    class_names = list(label_encoder.classes_)

    logger.info("Evaluating on test set (%d rows) ...", len(X_test))
    y_pred = model.predict(X_test)

    macro_f1    = f1_score(y_test, y_pred, average="macro")
    weighted_f1 = f1_score(y_test, y_pred, average="weighted")
    report      = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)

    logger.info("Macro F1: %.4f  |  Weighted F1: %.4f", macro_f1, weighted_f1)
    logger.info("\n%s", classification_report(y_test, y_pred, target_names=class_names))

    # --- Confusion matrix ---
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names, ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("ThreatSense — Confusion Matrix (XGBoost)")
    fig.tight_layout()
    cm_path = docs_dir / "confusion_matrix.png"
    fig.savefig(cm_path, dpi=150)
    plt.close(fig)
    logger.info("Confusion matrix saved -> %s", cm_path)

    metrics = {
        "macro_f1":    round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "per_class":   {
            cls: {
                "precision": round(report[cls]["precision"], 4),
                "recall":    round(report[cls]["recall"], 4),
                "f1":        round(report[cls]["f1-score"], 4),
                "support":   int(report[cls]["support"]),
            }
            for cls in class_names
        },
    }
    return metrics


# ---------------------------------------------------------------------------
# SHAP explainability
# ---------------------------------------------------------------------------

def compute_shap(
    model: xgb.XGBClassifier,
    X_train: pd.DataFrame,
    docs_dir: Path,
    max_samples: int = 5000,
    save_plot: bool = True,
) -> shap.TreeExplainer:
    """Build a SHAP TreeExplainer and optionally save a summary plot.

    Computes SHAP values on a random subsample of the training set (default
    5000 rows) to keep run time manageable. The explainer object is saved so
    inference.py can compute per-prediction explanations at runtime.

    Args:
        model: Trained XGBClassifier.
        X_train: Training features (used to sample background data).
        docs_dir: Directory to save shap_summary.png.
        max_samples: Max rows to use for SHAP computation.
        save_plot: Whether to generate and save the SHAP summary plot.

    Returns:
        Fitted shap.TreeExplainer.
    """
    logger.info("Building SHAP TreeExplainer ...")
    explainer = shap.TreeExplainer(model)

    if save_plot:
        docs_dir.mkdir(parents=True, exist_ok=True)
        n = min(max_samples, len(X_train))
        X_sample = X_train.sample(n=n, random_state=42)
        logger.info("Computing SHAP values on %d rows ...", n)

        shap_values = explainer.shap_values(X_sample)
        # For multiclass, shap_values is a list of arrays (one per class).
        # Use mean absolute value across classes for a global importance view.
        if isinstance(shap_values, list):
            # Old SHAP API: list of (n_samples, n_features) arrays
            global_importance = np.mean([np.abs(sv) for sv in shap_values], axis=0).mean(axis=0)
        elif hasattr(shap_values, "ndim") and shap_values.ndim == 3:
            # New SHAP API (0.40+): single (n_samples, n_features, n_classes) array
            global_importance = np.abs(shap_values).mean(axis=(0, 2))
        else:
            # Binary: (n_samples, n_features)
            global_importance = np.abs(shap_values).mean(axis=0)

        fig, ax = plt.subplots(figsize=(10, 6))
        feature_importance = pd.Series(
            global_importance, index=X_train.columns
        ).sort_values(ascending=False)

        sns.barplot(
            x=feature_importance.values[:15],
            y=feature_importance.index[:15],
            ax=ax,
            palette="viridis",
        )
        ax.set_title("ThreatSense — Top 15 Features by Mean |SHAP|")
        ax.set_xlabel("Mean |SHAP value|")
        fig.tight_layout()
        shap_path = docs_dir / "shap_summary.png"
        fig.savefig(shap_path, dpi=150)
        plt.close(fig)
        logger.info("SHAP summary saved -> %s", shap_path)

    return explainer


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def train(
    processed_dir: str | Path = "data/processed",
    artifacts_dir: str | Path = "models",
    docs_dir: str | Path = "docs",
    sample: int | None = None,
    val_fraction: float = 0.1,
    save_shap_plot: bool = True,
    random_state: int = 42,
) -> None:
    """Run the full ThreatSense training pipeline end to end.

    Loads processed splits, trains Isolation Forest + XGBoost, evaluates on
    the test set, generates SHAP explanations, and saves all artifacts.

    Args:
        processed_dir: Directory with processed train/test CSV splits.
        artifacts_dir: Where to save model .pkl files and metrics.json.
        docs_dir: Where to save confusion_matrix.png and shap_summary.png.
        sample: If set, subsample training set to this many rows (dev mode).
        val_fraction: Fraction of training set used for XGBoost early stopping.
        save_shap_plot: Whether to generate and save the SHAP summary plot.
        random_state: Seed for all random operations.
    """
    artifacts_dir = Path(artifacts_dir)
    docs_dir      = Path(docs_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load data
    X_train, X_test, y_train, y_test = load_processed_data(
        processed_dir, sample=sample, random_state=random_state
    )

    # 2. Load label encoder (needed for evaluation plot labels)
    encoder_path = artifacts_dir / "label_encoder.pkl"
    if not encoder_path.exists():
        raise FileNotFoundError(
            f"label_encoder.pkl not found in {artifacts_dir}. "
            "Run src.preprocess first."
        )
    label_encoder = joblib.load(encoder_path)

    # 3. Split a validation set from training data for XGBoost early stopping
    from sklearn.model_selection import train_test_split
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train,
        test_size=val_fraction,
        stratify=y_train,
        random_state=random_state,
    )
    logger.info("Train: %d | Val: %d | Test: %d", len(X_tr), len(X_val), len(X_test))

    # 4. Train Isolation Forest
    # Contamination = fraction of non-Benign flows in training set
    benign_class = list(label_encoder.classes_).index("Benign")
    contamination = float(np.mean(y_tr != benign_class))
    contamination = round(min(max(contamination, 0.01), 0.49), 3)
    logger.info("IF contamination set to %.3f (actual attack fraction)", contamination)

    iso = train_isolation_forest(X_tr, contamination=contamination)
    joblib.dump(iso, artifacts_dir / "iso_forest.pkl")
    logger.info("Saved iso_forest.pkl")

    # 5. Train XGBoost
    xgb_model = train_xgboost(X_tr, y_tr, X_val, y_val, random_state=random_state)
    joblib.dump(xgb_model, artifacts_dir / "xgboost_model.pkl")
    logger.info("Saved xgboost_model.pkl")

    # 6. Evaluate on held-out test set
    metrics = evaluate(xgb_model, X_test, y_test, label_encoder, docs_dir)
    metrics_path = artifacts_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))
    logger.info("Saved metrics.json  ->  macro F1: %.4f", metrics["macro_f1"])

    # 7. SHAP explainability
    explainer = compute_shap(
        xgb_model, X_tr, docs_dir,
        max_samples=5000,
        save_plot=save_shap_plot,
    )
    joblib.dump(explainer, artifacts_dir / "shap_explainer.pkl")
    logger.info("Saved shap_explainer.pkl")

    logger.info("=" * 60)
    logger.info("Training complete.")
    logger.info("  Macro F1    : %.4f", metrics["macro_f1"])
    logger.info("  Weighted F1 : %.4f", metrics["weighted_f1"])
    logger.info("  Artifacts   : %s", artifacts_dir.resolve())
    logger.info("  Plots       : %s", docs_dir.resolve())
    logger.info("=" * 60)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train ThreatSense ML models.")
    p.add_argument("--processed-dir",  default="data/processed", help="Processed data directory.")
    p.add_argument("--artifacts-dir",  default="models",          help="Where to save model artifacts.")
    p.add_argument("--docs-dir",       default="docs",            help="Where to save evaluation plots.")
    p.add_argument("--sample",         type=int, default=None,    help="Subsample N rows for fast dev run.")
    p.add_argument("--val-fraction",   type=float, default=0.1,   help="Fraction of train set for validation.")
    p.add_argument("--no-shap-plot",   action="store_true",       help="Skip SHAP summary plot (saves time).")
    p.add_argument("--random-state",   type=int, default=42,      help="Random seed.")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    train(
        processed_dir=args.processed_dir,
        artifacts_dir=args.artifacts_dir,
        docs_dir=args.docs_dir,
        sample=args.sample,
        val_fraction=args.val_fraction,
        save_shap_plot=not args.no_shap_plot,
        random_state=args.random_state,
    )
