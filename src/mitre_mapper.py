"""
mitre_mapper.py
===============
Maps ThreatSense threat classifications to MITRE ATT&CK techniques.

Each of the 5 ThreatSense classes is mapped to one or more ATT&CK
technique entries.  The mapping is stored as a plain dict so it is
trivial to extend as new techniques are added to ATT&CK.

Usage
-----
    from src.mitre_mapper import get_mitre_info, get_all_techniques

    info = get_mitre_info("DoS")
    # Returns a list of MitreTechnique dicts, one per technique.

    all_techniques = get_all_techniques()
    # Returns the full mapping for dashboard / report rendering.

MITRE ATT&CK reference: https://attack.mitre.org/
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class MitreTechnique:
    """A single MITRE ATT&CK technique entry.

    Attributes:
        tactic:         The ATT&CK tactic this technique belongs to.
        technique_id:   ATT&CK technique identifier (e.g. "T1498").
        technique_name: Human-readable technique name.
        url:            Direct link to the ATT&CK technique page.
        description:    Brief description of how this technique relates
                        to the ThreatSense threat class.
        severity:       Estimated severity level: "Critical", "High",
                        "Medium", or "Low".
    """
    tactic: str
    technique_id: str
    technique_name: str
    url: str
    description: str
    severity: str

    def to_dict(self) -> dict:
        """Return the technique as a plain dictionary."""
        return asdict(self)


# ---------------------------------------------------------------------------
# Mapping table
# ---------------------------------------------------------------------------

# Each key is one of the 5 ThreatSense class names.
# Each value is a list of MitreTechnique entries — a class can map to
# multiple techniques (e.g. DoS maps to both T1498 and T1499).
_MITRE_MAP: dict[str, list[MitreTechnique]] = {

    "DoS": [
        MitreTechnique(
            tactic="Impact",
            technique_id="T1498",
            technique_name="Network Denial of Service",
            url="https://attack.mitre.org/techniques/T1498/",
            description=(
                "Adversary floods the network layer to exhaust bandwidth "
                "or overwhelm network devices, causing service disruption."
            ),
            severity="Critical",
        ),
        MitreTechnique(
            tactic="Impact",
            technique_id="T1499",
            technique_name="Endpoint Denial of Service",
            url="https://attack.mitre.org/techniques/T1499/",
            description=(
                "Adversary exhausts system resources (CPU, memory, connections) "
                "on a specific endpoint to degrade or stop service availability."
            ),
            severity="High",
        ),
    ],

    "PortScan": [
        MitreTechnique(
            tactic="Discovery",
            technique_id="T1046",
            technique_name="Network Service Discovery",
            url="https://attack.mitre.org/techniques/T1046/",
            description=(
                "Adversary scans the network to enumerate open ports and "
                "running services, typically as a reconnaissance step before "
                "exploitation."
            ),
            severity="Medium",
        ),
    ],

    "Brute Force": [
        MitreTechnique(
            tactic="Credential Access",
            technique_id="T1110",
            technique_name="Brute Force",
            url="https://attack.mitre.org/techniques/T1110/",
            description=(
                "Adversary attempts to gain access by systematically trying "
                "username/password combinations against SSH, FTP, or other "
                "authentication services (e.g. Patator)."
            ),
            severity="High",
        ),
        MitreTechnique(
            tactic="Credential Access",
            technique_id="T1110.001",
            technique_name="Password Guessing",
            url="https://attack.mitre.org/techniques/T1110/001/",
            description=(
                "Sub-technique: adversary guesses common or default passwords "
                "without a complete credential list."
            ),
            severity="High",
        ),
    ],

    "Bot": [
        MitreTechnique(
            tactic="Command and Control",
            technique_id="T1071",
            technique_name="Application Layer Protocol",
            url="https://attack.mitre.org/techniques/T1071/",
            description=(
                "Bot malware communicates with its C2 server using standard "
                "application-layer protocols (HTTP/S, DNS) to blend in with "
                "legitimate traffic and evade detection."
            ),
            severity="Critical",
        ),
        MitreTechnique(
            tactic="Execution",
            technique_id="T1059",
            technique_name="Command and Scripting Interpreter",
            url="https://attack.mitre.org/techniques/T1059/",
            description=(
                "Bot agent executes commands received from the C2 server, "
                "potentially running scripts or shell commands on the victim host."
            ),
            severity="High",
        ),
    ],

    "Benign": [],  # No ATT&CK techniques for benign traffic.
}

# Severity ordering for sorting / prioritisation in the dashboard.
_SEVERITY_ORDER = {"Critical": 0, "High": 1, "Medium": 2, "Low": 3}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_mitre_info(threat_class: str) -> list[dict]:
    """Return MITRE ATT&CK technique entries for a given threat class.

    Args:
        threat_class: One of the 5 ThreatSense class names
                      ("Benign", "DoS", "PortScan", "Brute Force", "Bot").

    Returns:
        List of technique dicts (tactic, technique_id, technique_name, url,
        description, severity).  Empty list for "Benign".

    Raises:
        ValueError: If threat_class is not a recognised ThreatSense class.
    """
    if threat_class not in _MITRE_MAP:
        valid = list(_MITRE_MAP.keys())
        raise ValueError(
            f"Unknown threat class '{threat_class}'. "
            f"Valid classes: {valid}"
        )
    return [t.to_dict() for t in _MITRE_MAP[threat_class]]


def get_primary_technique(threat_class: str) -> Optional[dict]:
    """Return the single highest-severity technique for a threat class.

    Useful when you need one concise technique entry (e.g. for a summary
    card in the dashboard or the header of an incident report).

    Args:
        threat_class: One of the 5 ThreatSense class names.

    Returns:
        The highest-severity technique dict, or None for "Benign".
    """
    techniques = get_mitre_info(threat_class)
    if not techniques:
        return None
    return sorted(techniques, key=lambda t: _SEVERITY_ORDER.get(t["severity"], 99))[0]


def get_all_techniques() -> dict[str, list[dict]]:
    """Return the complete mapping of all threat classes to their techniques.

    Returns:
        Dict mapping each threat class name to a list of technique dicts.
        Useful for rendering the full ATT&CK heatmap in the dashboard.
    """
    return {cls: get_mitre_info(cls) for cls in _MITRE_MAP}


def get_severity(threat_class: str) -> str:
    """Return the highest severity level for a given threat class.

    Args:
        threat_class: One of the 5 ThreatSense class names.

    Returns:
        Severity string: "Critical", "High", "Medium", "Low", or "None".
    """
    primary = get_primary_technique(threat_class)
    if primary is None:
        return "None"
    return primary["severity"]


def list_threat_classes() -> list[str]:
    """Return all supported threat class names.

    Returns:
        List of class name strings in alphabetical order.
    """
    return sorted(_MITRE_MAP.keys())
