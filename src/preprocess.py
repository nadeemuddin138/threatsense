"""
preprocess.py
=============
Preprocessing pipeline for the CICIDS2017 network-flow dataset used by ThreatSense.

CICIDS2017 ships as several per-day CSV files and has a number of well-known
data-quality issues that this module handles explicitly:

  1. Column names contain leading/trailing whitespace (e.g. " Label").
  2. Rate columns ("Flow Bytes/s", "Flow Packets/s") contain Infinity / NaN
     (division by a zero-duration flow).
  3. A few label strings use non-UTF-8 characters (the "Web Attack" labels),
     which break a naive utf-8 read -> we read with a latin-1 fallback.
  4. The raw labels are fine-grained -> we collapse them into the 5 target
     classes used by ThreatSense: Benign, DoS, PortScan, Brute Force, Bot.
  5. Severe class imbalance (Benign dominates) -> reported here, handled at
     model-training time (class weights / sampling in inference/training).
  6. Identifier columns (Flow ID, IPs, source port, timestamp) leak host
     identity -> dropped so the model learns flow *behaviour*, not machines.

Run:
    python -m src.preprocess --raw-dir data/raw --out-dir data/processed \
        --artifacts-dir models --test-size 0.2

Outputs:
    data/processed/{X_train,X_test,y_train,y_test}.csv
    models/scaler.pkl, models/label_encoder.pkl
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("threatsense.preprocess")


# --- Configuration ----------------------------------------------------------

# The five classes ThreatSense classifies into.
TARGET_CLASSES = ["Benign", "DoS", "PortScan", "Brute Force", "Bot"]

# Identifier / leakage-prone columns to drop if present (matched AFTER
# whitespace stripping). "Destination Port" and "Protocol" are intentionally
# kept as legitimate behavioural features.
DROP_COLUMNS = [
    "Flow ID",
    "Source IP", "Src IP",
    "Destination IP", "Dst IP",
    "Source Port", "Src Port",
    "Timestamp",
]


def _map_label(raw_label: object) -> str | None:
    """Collapse a fine-grained CICIDS2017 label into one of the 5 target classes.

    Uses keyword matching so it is robust to the non-UTF-8 dash characters in
    the original "Web Attack" labels and to minor spacing/casing differences.

    Args:
        raw_label: A single label value from the dataset.

    Returns:
        One of TARGET_CLASSES, or None if the label is outside the 5-class
        problem (e.g. Infiltration, Heartbleed, Web XSS, SQL Injection), in
        which case the row should be dropped.
    """
    label = str(raw_label).strip().lower()
    if label == "benign":
        return "Benign"
    if "portscan" in label or "port scan" in label:
        return "PortScan"
    if label == "bot":
        return "Bot"
    if "patator" in label or "brute force" in label or "bruteforce" in label:
        return "Brute Force"
    if label.startswith("dos") or "ddos" in label:
        return "DoS"
    return None  # unmapped attack type -> dropped


def load_raw_data(raw_dir: str | Path) -> pd.DataFrame:
    """Load and concatenate every CSV or Parquet file in the raw data directory.

    Args:
        raw_dir: Directory containing CICIDS2017 data files (.csv or .parquet).

    Returns:
        A single concatenated DataFrame of all flows.
    """
    raw_dir = Path(raw_dir)
    paths = sorted(
        p for p in raw_dir.iterdir()
        if p.suffix.lower() in {".csv", ".parquet"}
    )
    if not paths:
        raise FileNotFoundError(
            f"No .csv or .parquet files found in {raw_dir.resolve()}"
        )

    frames = []
    for path in paths:
        if path.suffix.lower() == ".parquet":
            df = pd.read_parquet(path)
        else:
            try:
                df = pd.read_csv(path, encoding="utf-8", low_memory=False)
            except UnicodeDecodeError:
                logger.warning("utf-8 decode failed for %s, retrying with latin-1", path.name)
                df = pd.read_csv(path, encoding="latin-1", low_memory=False)
        logger.info("Loaded %s (%d rows)", path.name, len(df))
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    logger.info("Combined dataset: %d rows x %d columns", *combined.shape)
    return combined


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Strip leading/trailing whitespace from all column names.

    CICIDS2017 columns commonly arrive as " Flow Duration", " Label", etc.

    Args:
        df: Raw DataFrame.

    Returns:
        DataFrame with cleaned column names (operates on a copy).
    """
    df = df.copy()
    df.columns = df.columns.str.strip()
    return df


def find_label_column(df: pd.DataFrame) -> str:
    """Locate the label column regardless of casing.

    Args:
        df: DataFrame with already-cleaned column names.

    Returns:
        The name of the label column.

    Raises:
        KeyError: If no label-like column is found.
    """
    for col in df.columns:
        if col.strip().lower() == "label":
            return col
    raise KeyError("No 'Label' column found in dataset.")


def drop_identifier_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop identifier / leakage-prone columns if present.

    Args:
        df: DataFrame with cleaned column names.

    Returns:
        DataFrame without identifier columns (operates on a copy).
    """
    df = df.copy()
    to_drop = [c for c in DROP_COLUMNS if c in df.columns]
    if to_drop:
        logger.info("Dropping identifier columns: %s", to_drop)
        df = df.drop(columns=to_drop)
    return df


def handle_infinite_and_missing(df: pd.DataFrame, label_col: str) -> pd.DataFrame:
    """Coerce features to numeric, replace Infinity with NaN, drop NaN rows.

    "Flow Bytes/s" and "Flow Packets/s" can contain Infinity. Feature columns
    are coerced to numeric first so any stray string values also become NaN.
    Rows with missing feature values are dropped (they are a small fraction of
    CICIDS2017); switch to imputation here if you prefer to keep every row.

    Args:
        df: DataFrame with cleaned column names.
        label_col: Name of the label column (excluded from numeric coercion).

    Returns:
        Cleaned DataFrame with no infinite/NaN feature values.
    """
    df = df.copy()
    feature_cols = [c for c in df.columns if c != label_col]

    # Coerce features to numeric; non-numeric junk -> NaN.
    df[feature_cols] = df[feature_cols].apply(pd.to_numeric, errors="coerce")

    # Replace +/- infinity with NaN, then drop affected rows.
    df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan)

    before = len(df)
    df = df.dropna(subset=feature_cols)
    dropped = before - len(df)
    if dropped:
        logger.info(
            "Dropped %d rows with infinite/NaN feature values (%.2f%%)",
            dropped, 100 * dropped / before if before else 0,
        )
    return df


def map_labels(df: pd.DataFrame, label_col: str) -> pd.DataFrame:
    """Collapse raw labels into the 5 target classes; drop unmapped rows.

    Args:
        df: Cleaned DataFrame.
        label_col: Name of the label column.

    Returns:
        DataFrame whose label column contains only TARGET_CLASSES, renamed to
        the canonical name 'Label'.
    """
    df = df.copy()
    df[label_col] = df[label_col].map(_map_label)

    before = len(df)
    df = df.dropna(subset=[label_col])
    dropped = before - len(df)
    if dropped:
        logger.info("Dropped %d rows whose labels are outside the 5-class problem", dropped)

    if label_col != "Label":
        df = df.rename(columns={label_col: "Label"})

    logger.info("Class distribution after mapping:\n%s", df["Label"].value_counts().to_string())
    return df


def downcast_numeric(df: pd.DataFrame, exclude: list[str] | None = None) -> pd.DataFrame:
    """Downcast float64 columns to float32 to roughly halve memory use.

    CICIDS2017 is large (~2.8M rows); float32 keeps it comfortably in RAM.

    Args:
        df: DataFrame to downcast.
        exclude: Column names to leave untouched (e.g. the label).

    Returns:
        Memory-optimised DataFrame (operates on a copy).
    """
    df = df.copy()
    exclude = exclude or []
    float_cols = [c for c in df.select_dtypes(include="float64").columns if c not in exclude]
    if float_cols:
        df[float_cols] = df[float_cols].astype("float32")
    return df


def encode_labels(y: pd.Series) -> tuple[np.ndarray, LabelEncoder]:
    """Encode string class labels into integers 0..4.

    Args:
        y: Series of target-class strings.

    Returns:
        Tuple of (encoded label array, fitted LabelEncoder). The encoder is
        saved so predictions can be mapped back to class names downstream.
    """
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)
    mapping = dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))
    logger.info("Label encoding: %s", mapping)
    return y_encoded, encoder


def scale_features(
    X_train: pd.DataFrame, X_test: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray, StandardScaler]:
    """Standard-scale features, fitting on train only to avoid data leakage.

    Scaling is not strictly required by the tree-based models (XGBoost,
    Isolation Forest) but keeps the pipeline reproducible and friendly to any
    distance-based additions later.

    Args:
        X_train: Training feature frame.
        X_test: Test feature frame.

    Returns:
        Tuple of (scaled train array, scaled test array, fitted StandardScaler).
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler


def preprocess(
    raw_dir: str | Path,
    out_dir: str | Path,
    artifacts_dir: str | Path,
    test_size: float = 0.2,
    random_state: int = 42,
    scale: bool = True,
) -> None:
    """Run the full CICIDS2017 preprocessing pipeline end to end.

    Loads raw CSVs, cleans them, maps labels to the 5 target classes, performs a
    stratified train/test split, optionally scales, and writes the processed
    splits plus the fitted scaler and label encoder to disk.

    Args:
        raw_dir: Directory of raw CICIDS2017 CSVs.
        out_dir: Directory to write processed train/test CSVs.
        artifacts_dir: Directory to write scaler.pkl and label_encoder.pkl.
        test_size: Fraction of data held out for testing.
        random_state: Seed for reproducible splits.
        scale: Whether to standard-scale the features.
    """
    out_dir = Path(out_dir)
    artifacts_dir = Path(artifacts_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    df = load_raw_data(raw_dir)
    df = clean_column_names(df)
    label_col = find_label_column(df)
    df = drop_identifier_columns(df)
    df = handle_infinite_and_missing(df, label_col)
    df = map_labels(df, label_col)
    df = downcast_numeric(df, exclude=["Label"])

    X = df.drop(columns=["Label"])
    y = df["Label"]
    feature_names = list(X.columns)

    y_encoded, label_encoder = encode_labels(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded,
        test_size=test_size,
        random_state=random_state,
        stratify=y_encoded,  # preserve class ratios in both splits
    )
    logger.info("Train: %d rows | Test: %d rows", len(X_train), len(X_test))

    if scale:
        X_train_arr, X_test_arr, scaler = scale_features(X_train, X_test)
        X_train = pd.DataFrame(X_train_arr, columns=feature_names)
        X_test = pd.DataFrame(X_test_arr, columns=feature_names)
        joblib.dump(scaler, artifacts_dir / "scaler.pkl")
        logger.info("Saved scaler -> %s", artifacts_dir / "scaler.pkl")

    joblib.dump(label_encoder, artifacts_dir / "label_encoder.pkl")
    logger.info("Saved label encoder -> %s", artifacts_dir / "label_encoder.pkl")

    # Persist processed splits.
    X_train.to_csv(out_dir / "X_train.csv", index=False)
    X_test.to_csv(out_dir / "X_test.csv", index=False)
    pd.Series(y_train, name="Label").to_csv(out_dir / "y_train.csv", index=False)
    pd.Series(y_test, name="Label").to_csv(out_dir / "y_test.csv", index=False)
    logger.info("Wrote processed splits to %s", out_dir.resolve())


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments for standalone execution."""
    parser = argparse.ArgumentParser(description="Preprocess CICIDS2017 for ThreatSense.")
    parser.add_argument("--raw-dir", default="data/raw", help="Directory of raw CICIDS2017 CSVs.")
    parser.add_argument("--out-dir", default="data/processed", help="Output dir for processed splits.")
    parser.add_argument("--artifacts-dir", default="models", help="Where to save scaler/label encoder.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split fraction.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed.")
    parser.add_argument("--no-scale", action="store_true", help="Disable feature scaling.")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    preprocess(
        raw_dir=args.raw_dir,
        out_dir=args.out_dir,
        artifacts_dir=args.artifacts_dir,
        test_size=args.test_size,
        random_state=args.random_state,
        scale=not args.no_scale,
    )
