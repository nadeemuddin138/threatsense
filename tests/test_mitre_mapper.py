"""
tests/test_mitre_mapper.py
==========================
Unit tests for src/mitre_mapper.py.

Run:
    pytest tests/test_mitre_mapper.py -v
"""

import pytest
from src.mitre_mapper import (
    get_mitre_info,
    get_primary_technique,
    get_all_techniques,
    get_severity,
    list_threat_classes,
)

VALID_CLASSES = ["Benign", "DoS", "PortScan", "Brute Force", "Bot"]


# ---------------------------------------------------------------------------
# get_mitre_info
# ---------------------------------------------------------------------------

class TestGetMitreInfo:

    def test_dos_returns_two_techniques(self):
        result = get_mitre_info("DoS")
        assert len(result) == 2

    def test_dos_technique_ids(self):
        ids = {t["technique_id"] for t in get_mitre_info("DoS")}
        assert ids == {"T1498", "T1499"}

    def test_portscan_returns_t1046(self):
        result = get_mitre_info("PortScan")
        assert len(result) == 1
        assert result[0]["technique_id"] == "T1046"

    def test_brute_force_returns_two_techniques(self):
        result = get_mitre_info("Brute Force")
        assert len(result) == 2

    def test_bot_has_c2_technique(self):
        ids = {t["technique_id"] for t in get_mitre_info("Bot")}
        assert "T1071" in ids

    def test_benign_returns_empty_list(self):
        assert get_mitre_info("Benign") == []

    def test_invalid_class_raises_value_error(self):
        with pytest.raises(ValueError, match="Unknown threat class"):
            get_mitre_info("NotAClass")

    def test_each_technique_has_required_keys(self):
        required = {"tactic", "technique_id", "technique_name", "url", "description", "severity"}
        for cls in VALID_CLASSES:
            for technique in get_mitre_info(cls):
                assert required.issubset(technique.keys()), \
                    f"Missing keys in {cls} technique: {required - technique.keys()}"

    def test_all_urls_point_to_attack_mitre(self):
        for cls in VALID_CLASSES:
            for t in get_mitre_info(cls):
                assert t["url"].startswith("https://attack.mitre.org/"), \
                    f"Bad URL for {cls}: {t['url']}"

    def test_severity_values_are_valid(self):
        valid_severities = {"Critical", "High", "Medium", "Low"}
        for cls in VALID_CLASSES:
            for t in get_mitre_info(cls):
                assert t["severity"] in valid_severities, \
                    f"Invalid severity '{t['severity']}' for {cls}"


# ---------------------------------------------------------------------------
# get_primary_technique
# ---------------------------------------------------------------------------

class TestGetPrimaryTechnique:

    def test_benign_returns_none(self):
        assert get_primary_technique("Benign") is None

    def test_dos_primary_is_critical(self):
        primary = get_primary_technique("DoS")
        assert primary is not None
        assert primary["severity"] == "Critical"

    def test_portscan_primary_is_t1046(self):
        primary = get_primary_technique("PortScan")
        assert primary["technique_id"] == "T1046"

    def test_bot_primary_is_c2(self):
        primary = get_primary_technique("Bot")
        assert primary["technique_id"] == "T1071"

    def test_returns_dict(self):
        primary = get_primary_technique("DoS")
        assert isinstance(primary, dict)


# ---------------------------------------------------------------------------
# get_all_techniques
# ---------------------------------------------------------------------------

class TestGetAllTechniques:

    def test_returns_all_five_classes(self):
        all_t = get_all_techniques()
        assert set(all_t.keys()) == set(VALID_CLASSES)

    def test_benign_is_empty_in_full_map(self):
        assert get_all_techniques()["Benign"] == []

    def test_values_are_lists(self):
        for cls, techniques in get_all_techniques().items():
            assert isinstance(techniques, list), f"{cls} should map to a list"


# ---------------------------------------------------------------------------
# get_severity
# ---------------------------------------------------------------------------

class TestGetSeverity:

    def test_dos_is_critical(self):
        assert get_severity("DoS") == "Critical"

    def test_portscan_is_medium(self):
        assert get_severity("PortScan") == "Medium"

    def test_bot_is_critical(self):
        assert get_severity("Bot") == "Critical"

    def test_brute_force_is_high(self):
        assert get_severity("Brute Force") == "High"

    def test_benign_is_none(self):
        assert get_severity("Benign") == "None"


# ---------------------------------------------------------------------------
# list_threat_classes
# ---------------------------------------------------------------------------

class TestListThreatClasses:

    def test_returns_five_classes(self):
        assert len(list_threat_classes()) == 5

    def test_all_expected_classes_present(self):
        assert set(list_threat_classes()) == set(VALID_CLASSES)

    def test_returns_sorted_list(self):
        classes = list_threat_classes()
        assert classes == sorted(classes)
