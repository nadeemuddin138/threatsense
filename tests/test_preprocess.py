"""
tests/test_preprocess.py
========================
Unit tests for src/preprocess.py preprocessing functions.

These tests are pure unit tests — no file I/O, no model loading.
They run in CI without any dataset present.

Run:
    pytest tests/test_preprocess.py -v
"""

import numpy as np
import pandas as pd
import pytest

from src.preprocess import (
    _map_label,
    clean_column_names,
    drop_identifier_columns,
    find_label_column,
    handle_infinite_and_missing,
    map_labels,
)


# ---------------------------------------------------------------------------
# _map_label
# ---------------------------------------------------------------------------

class TestMapLabel:

    def test_benign_maps_correctly(self):
        assert _map_label("BENIGN") == "Benign"
        assert _map_label("benign") == "Benign"

    def test_dos_variants(self):
        for label in ["DoS Hulk", "DoS GoldenEye", "DoS Slowloris", "DDoS"]:
            assert _map_label(label) == "DoS", f"Failed for {label}"

    def test_portscan_maps_correctly(self):
        assert _map_label("PortScan") == "PortScan"
        assert _map_label("port scan") == "PortScan"

    def test_brute_force_variants(self):
        for label in ["FTP-Patator", "SSH-Patator", "Brute Force", "BruteForce"]:
            assert _map_label(label) == "Brute Force", f"Failed for {label}"

    def test_bot_maps_correctly(self):
        assert _map_label("Bot") == "Bot"
        assert _map_label("BOT") == "Bot"

    def test_out_of_scope_returns_none(self):
        for label in ["Infiltration", "Heartbleed", "Web Attack XSS", "SQL Injection"]:
            assert _map_label(label) is None, f"Expected None for {label}"

    def test_web_attack_brute_force_maps_correctly(self):
        # The cp1252 en-dash variant
        assert _map_label("Web Attack \u2013 Brute Force") == "Brute Force"


# ---------------------------------------------------------------------------
# clean_column_names
# ---------------------------------------------------------------------------

class TestCleanColumnNames:

    def test_strips_leading_spaces(self):
        df = pd.DataFrame({" Flow Duration": [1], " Label": ["BENIGN"]})
        cleaned = clean_column_names(df)
        assert "Flow Duration" in cleaned.columns
        assert "Label" in cleaned.columns

    def test_strips_trailing_spaces(self):
        df = pd.DataFrame({"Flow Duration ": [1]})
        cleaned = clean_column_names(df)
        assert "Flow Duration" in cleaned.columns

    def test_does_not_modify_original(self):
        df = pd.DataFrame({" Col": [1]})
        _ = clean_column_names(df)
        assert " Col" in df.columns   # original unchanged


# ---------------------------------------------------------------------------
# find_label_column
# ---------------------------------------------------------------------------

class TestFindLabelColumn:

    def test_finds_label(self):
        df = pd.DataFrame({"Flow Duration": [1], "Label": ["BENIGN"]})
        assert find_label_column(df) == "Label"

    def test_case_insensitive(self):
        df = pd.DataFrame({"label": ["BENIGN"]})
        assert find_label_column(df) == "label"

    def test_raises_if_missing(self):
        df = pd.DataFrame({"Flow Duration": [1]})
        with pytest.raises(KeyError, match="No 'Label' column"):
            find_label_column(df)


# ---------------------------------------------------------------------------
# drop_identifier_columns
# ---------------------------------------------------------------------------

class TestDropIdentifierColumns:

    def test_drops_flow_id(self):
        df = pd.DataFrame({"Flow ID": ["x"], "Flow Duration": [1]})
        result = drop_identifier_columns(df)
        assert "Flow ID" not in result.columns
        assert "Flow Duration" in result.columns

    def test_drops_source_ip(self):
        df = pd.DataFrame({"Source IP": ["1.2.3.4"], "Flow Duration": [1]})
        result = drop_identifier_columns(df)
        assert "Source IP" not in result.columns

    def test_keeps_destination_port(self):
        df = pd.DataFrame({"Source IP": ["1.2.3.4"], "Destination Port": [80]})
        result = drop_identifier_columns(df)
        assert "Destination Port" in result.columns

    def test_no_error_if_columns_absent(self):
        df = pd.DataFrame({"Flow Duration": [1], "Label": ["BENIGN"]})
        result = drop_identifier_columns(df)
        assert list(result.columns) == ["Flow Duration", "Label"]


# ---------------------------------------------------------------------------
# handle_infinite_and_missing
# ---------------------------------------------------------------------------

class TestHandleInfiniteAndMissing:

    def _make_df(self):
        return pd.DataFrame({
            "Flow Duration":    [1.0, 2.0, 3.0, 4.0],
            "Flow Bytes/s":     [100.0, np.inf, -np.inf, 500.0],
            "Flow Packets/s":   [1.0, 2.0, 3.0, np.nan],
            "Label":            ["BENIGN", "DoS Hulk", "PortScan", "Bot"],
        })

    def test_removes_inf_rows(self):
        df = self._make_df()
        result = handle_infinite_and_missing(df, "Label")
        assert not np.isinf(result.drop(columns=["Label"]).to_numpy()).any()

    def test_removes_nan_rows(self):
        df = self._make_df()
        result = handle_infinite_and_missing(df, "Label")
        assert not result.drop(columns=["Label"]).isna().any().any()

    def test_keeps_clean_rows(self):
        df = self._make_df()
        result = handle_infinite_and_missing(df, "Label")
        # Only rows 0 and 3 are fully clean (indices 1, 2, 3 have inf/nan)
        assert len(result) == 1   # only row 0 is fully clean
        assert result.iloc[0]["Label"] == "BENIGN"

    def test_label_column_not_coerced(self):
        df = self._make_df()
        result = handle_infinite_and_missing(df, "Label")
        # pandas 3.0 uses StringDtype instead of object — check values, not dtype
        assert result["Label"].iloc[0] == "BENIGN"


# ---------------------------------------------------------------------------
# map_labels
# ---------------------------------------------------------------------------

class TestMapLabels:

    def _make_df(self):
        return pd.DataFrame({
            "Flow Duration": range(7),
            "Label": [
                "BENIGN", "DoS Hulk", "DDoS", "PortScan",
                "FTP-Patator", "Bot", "Infiltration",
            ],
        })

    def test_all_five_classes_mapped(self):
        df = self._make_df()
        result = map_labels(df, "Label")
        assert set(result["Label"].unique()) == {"Benign", "DoS", "PortScan", "Brute Force", "Bot"}

    def test_out_of_scope_rows_dropped(self):
        df = self._make_df()
        result = map_labels(df, "Label")
        # Infiltration should be dropped → 6 rows become 6, check count
        assert len(result) == 6

    def test_label_column_renamed_to_label(self):
        df = pd.DataFrame({
            "Flow Duration": [1],
            "label": ["BENIGN"],
        })
        result = map_labels(df, "label")
        assert "Label" in result.columns
