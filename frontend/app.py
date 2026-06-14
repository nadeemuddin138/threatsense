"""
frontend/app.py
===============
ThreatSense Streamlit dashboard.

Sections
--------
  1. Sidebar     — API connection status + configuration
  2. Upload       — drag-and-drop CSV log upload -> batch analysis
  3. Detections   — table of all incidents with severity colour coding
  4. Threat Chart — bar chart of threat class breakdown
  5. SHAP Panel   — top feature contributions for a selected incident
  6. Report       — LLM-generated SOC incident report viewer

All data comes from the FastAPI backend (default: http://localhost:8000).
No ML code runs inside this file.

Run
---
  streamlit run frontend/app.py
"""

from __future__ import annotations

import json
from datetime import datetime
from io import StringIO

import pandas as pd
import requests
import streamlit as st

# ---------------------------------------------------------------------------
# Page config (must be first Streamlit call)
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="ThreatSense — SOC Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SEVERITY_COLOURS = {
    "Critical": "🔴",
    "High":     "🟠",
    "Medium":   "🟡",
    "Low":      "🟢",
    "None":     "⚪",
}

CLASS_COLOURS = {
    "DoS":         "#e74c3c",
    "Bot":         "#8e44ad",
    "Brute Force": "#e67e22",
    "PortScan":    "#f1c40f",
    "Benign":      "#2ecc71",
}

# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------

def api_url(path: str, base: str) -> str:
    """Build a full API URL from a path.

    Args:
        path: API endpoint path (e.g. "/health").
        base: Base URL (e.g. "http://localhost:8000").

    Returns:
        Full URL string.
    """
    return f"{base.rstrip('/')}{path}"


def check_health(base_url: str) -> bool:
    """Ping the API health endpoint.

    Args:
        base_url: Base URL of the ThreatSense API.

    Returns:
        True if the API is reachable and healthy.
    """
    try:
        r = requests.get(api_url("/health", base_url), timeout=3)
        return r.status_code == 200
    except Exception:
        return False


def fetch_incidents(base_url: str, limit: int = 200, threat_class: str | None = None) -> list[dict]:
    """Fetch incident list from the API.

    Args:
        base_url:     API base URL.
        limit:        Max incidents to fetch.
        threat_class: Optional filter string.

    Returns:
        List of incident dicts.
    """
    params: dict = {"limit": limit}
    if threat_class and threat_class != "All":
        params["threat_class"] = threat_class
    try:
        r = requests.get(api_url("/incidents", base_url), params=params, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as exc:
        st.error(f"Failed to fetch incidents: {exc}")
        return []


def upload_csv(base_url: str, file_bytes: bytes, filename: str) -> dict | None:
    """Upload a CSV file to the /analyze endpoint.

    Args:
        base_url:   API base URL.
        file_bytes: Raw CSV bytes.
        filename:   Original filename (used for MIME type detection).

    Returns:
        BatchSummary dict from the API, or None on error.
    """
    try:
        r = requests.post(
            api_url("/analyze", base_url),
            files={"file": (filename, file_bytes, "text/csv")},
            timeout=120,
        )
        r.raise_for_status()
        return r.json()
    except requests.HTTPError as exc:
        st.error(f"Upload failed ({exc.response.status_code}): {exc.response.text}")
    except Exception as exc:
        st.error(f"Upload error: {exc}")
    return None


def fetch_report(base_url: str, incident_id: str) -> dict | None:
    """Generate and fetch a SOC incident report from the API.

    Args:
        base_url:    API base URL.
        incident_id: Incident ID string.

    Returns:
        IncidentReport dict, or None on error.
    """
    try:
        r = requests.post(
            api_url(f"/report/{incident_id}", base_url),
            timeout=60,
        )
        r.raise_for_status()
        return r.json()
    except requests.HTTPError as exc:
        st.error(f"Report generation failed: {exc.response.text}")
    except Exception as exc:
        st.error(f"Report error: {exc}")
    return None


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.image("https://img.icons8.com/color/96/shield.png", width=60)
    st.title("ThreatSense")
    st.caption("LLM-Powered SIEM Analyst")
    st.divider()

    base_url = st.text_input(
        "API Base URL",
        value="http://localhost:8000",
        help="URL of the running ThreatSense FastAPI backend.",
    )

    healthy = check_health(base_url)
    if healthy:
        st.success("✅ API connected")
    else:
        st.error("❌ API unreachable — run: uvicorn api.main:app --port 8000")

    st.divider()
    st.markdown("**Filter incidents**")
    class_filter = st.selectbox(
        "Threat class",
        ["All", "DoS", "PortScan", "Brute Force", "Bot", "Benign"],
    )
    limit = st.slider("Max incidents", 10, 500, 100)

    st.divider()
    st.markdown(
        "**CICIDS2017 dataset**  \n"
        "XGBoost + Isolation Forest  \n"
        "Macro F1: **0.9348**  \n"
        "LLM: Groq Llama 3.1 70B"
    )

# ---------------------------------------------------------------------------
# Main header
# ---------------------------------------------------------------------------

st.title("🛡️ ThreatSense — SOC Dashboard")
st.caption(f"Connected to: `{base_url}` | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if not healthy:
    st.warning(
        "⚠️ Cannot reach the API. "
        "Start it with: `uvicorn api.main:app --reload --port 8000`"
    )
    st.stop()

# ---------------------------------------------------------------------------
# Section 1 — Upload CSV
# ---------------------------------------------------------------------------

st.header("📁 Upload Network Flow Logs")

uploaded = st.file_uploader(
    "Drop a CSV of network flows here (same columns as CICIDS2017)",
    type=["csv"],
    help="Max 50 MB. The API will run Isolation Forest + XGBoost on every row.",
)

if uploaded is not None:
    with st.spinner(f"Analysing {uploaded.name} ..."):
        summary = upload_csv(base_url, uploaded.getvalue(), uploaded.name)

    if summary:
        st.success(f"✅ Processed **{summary['total_flows']:,}** flows")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Flows",  f"{summary['total_flows']:,}")
        col2.metric("Anomalies",    f"{summary['anomaly_count']:,}")
        col3.metric("Incidents",    f"{len(summary['incident_ids']):,}")
        col4.metric("Errors",       f"{summary['errors']:,}")

        st.markdown("**Threat breakdown:**")
        breakdown_df = pd.DataFrame(
            summary["threat_breakdown"].items(),
            columns=["Threat Class", "Count"],
        ).sort_values("Count", ascending=False)
        st.dataframe(breakdown_df, use_container_width=True, hide_index=True)

st.divider()

# ---------------------------------------------------------------------------
# Section 2 — Detections table
# ---------------------------------------------------------------------------

st.header("🔍 Detections")

incidents = fetch_incidents(base_url, limit=limit, threat_class=class_filter if class_filter != "All" else None)

if not incidents:
    st.info("No incidents found. Upload a CSV above to get started.")
else:
    df_inc = pd.DataFrame(incidents)
    df_inc["severity_icon"] = df_inc["severity"].map(SEVERITY_COLOURS)
    df_inc["display"] = df_inc["severity_icon"] + "  " + df_inc["predicted_class"]

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Incidents",  len(df_inc))
    col2.metric("Anomalies",        int(df_inc["is_anomaly"].sum()))
    col3.metric("Critical / High",  int((df_inc["severity"].isin(["Critical", "High"])).sum()))
    col4.metric("Avg Confidence",   f"{df_inc['confidence'].mean():.1%}")

    # Styled table
    display_df = df_inc[[
        "id", "predicted_class", "severity", "anomaly_score",
        "confidence", "is_anomaly", "created_at",
    ]].copy()
    display_df.columns = [
        "Incident ID", "Threat Class", "Severity", "Anomaly Score",
        "Confidence", "Anomaly?", "Timestamp",
    ]
    display_df["Confidence"] = display_df["Confidence"].apply(lambda x: f"{x:.1%}")
    display_df["Anomaly Score"] = display_df["Anomaly Score"].apply(lambda x: f"{x:.4f}")

    st.dataframe(display_df, use_container_width=True, hide_index=True)

st.divider()

# ---------------------------------------------------------------------------
# Section 3 — Threat breakdown chart
# ---------------------------------------------------------------------------

st.header("📊 Threat Breakdown")

if incidents:
    class_counts = (
        pd.Series([i["predicted_class"] for i in incidents])
        .value_counts()
        .reset_index()
    )
    class_counts.columns = ["Threat Class", "Count"]

    col1, col2 = st.columns([2, 1])
    with col1:
        st.bar_chart(
            class_counts.set_index("Threat Class"),
            color="#e74c3c",
            height=300,
        )
    with col2:
        st.markdown("**Class distribution**")
        for _, row in class_counts.iterrows():
            icon = SEVERITY_COLOURS.get(
                {"DoS": "Critical", "Bot": "Critical",
                 "Brute Force": "High", "PortScan": "Medium",
                 "Benign": "None"}.get(row["Threat Class"], "Low"),
                "⚪",
            )
            pct = row["Count"] / class_counts["Count"].sum() * 100
            st.write(f"{icon} **{row['Threat Class']}**: {row['Count']:,} ({pct:.1f}%)")

st.divider()

# ---------------------------------------------------------------------------
# Section 4 — Incident report generator
# ---------------------------------------------------------------------------

st.header("📋 SOC Incident Report")

if incidents:
    incident_ids = [i["id"] for i in incidents]
    selected_id  = st.selectbox("Select an incident to generate report", incident_ids)

    selected_incident = next((i for i in incidents if i["id"] == selected_id), None)
    if selected_incident:
        col1, col2, col3 = st.columns(3)
        col1.metric("Threat Class", selected_incident["predicted_class"])
        col2.metric("Severity",     SEVERITY_COLOURS.get(selected_incident["severity"], "⚪") + " " + selected_incident["severity"])
        col3.metric("Confidence",   f"{selected_incident['confidence']:.1%}")

    if st.button("🤖 Generate SOC Report", type="primary"):
        with st.spinner("Generating report with Groq Llama 3.1 ..."):
            report = fetch_report(base_url, selected_id)

        if report:
            st.success("Report generated!")

            # Header
            st.subheader(f"Incident: {report['incident_id']}")
            st.caption(f"Generated: {report.get('timestamp', '')} | Model: {report.get('generated_by', '')}")

            # Executive summary
            st.markdown("### 📌 Executive Summary")
            st.info(report.get("executive_summary", "N/A"))

            # Threat analysis
            st.markdown("### 🔬 Threat Analysis")
            st.write(report.get("threat_analysis", "N/A"))

            # MITRE ATT&CK
            st.markdown("### 🗺️ MITRE ATT&CK Techniques")
            techniques = report.get("mitre_techniques", [])
            if techniques:
                for t in techniques:
                    with st.expander(f"**{t['technique_id']}** — {t['technique_name']} ({t['tactic']})"):
                        st.write(t.get("description", ""))
                        st.markdown(f"[View on ATT&CK ↗]({t['url']})")
                        st.badge(t.get("severity", ""), color="red" if t.get("severity") == "Critical" else "orange")
            else:
                st.write("No techniques mapped (Benign traffic).")

            # Recommended actions
            st.markdown("### ✅ Recommended Actions")
            actions = report.get("recommended_actions", [])
            for i, action in enumerate(actions, 1):
                st.write(f"{i}. {action}")

            # Top indicators
            st.markdown("### 📡 Top Network Indicators")
            indicators = report.get("top_indicators", [])
            if indicators:
                for ind in indicators:
                    st.code(ind)
            else:
                st.write("No indicators available.")

            # Investigation notes
            st.markdown("### 🔎 Investigation Notes")
            st.warning(report.get("investigation_notes", "N/A"))

            # Raw JSON expander
            with st.expander("View raw report JSON"):
                st.json(report)
else:
    st.info("No incidents to report on yet. Upload a CSV above first.")
