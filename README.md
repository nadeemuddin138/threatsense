# 🛡️ ThreatSense — LLM-Powered SIEM Analyst

![CI](https://github.com/<your-username>/threatsense/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/Python-3.12+-blue)
![XGBoost](https://img.shields.io/badge/XGBoost-3.2-orange)
![LangGraph](https://img.shields.io/badge/LangGraph-1.2-green)
![FastAPI](https://img.shields.io/badge/FastAPI-0.136-teal)

> An end-to-end AI/ML system that ingests network flow logs, detects threats,
> maps them to MITRE ATT\&CK, and auto-generates SOC incident reports using a
> LangGraph multi-agent pipeline powered by Groq Llama 3.1 70B.

---

## 📸 Demo

<!-- Add a GIF of the dashboard here -->
![Dashboard Demo](docs/demo.gif)

---

## 🏗️ Architecture

```
Network Flow Logs (CICIDS2017)
        │
        ▼
┌─────────────────┐     ┌──────────────────┐
│  Isolation      │     │    XGBoost       │
│  Forest         │────▶│    Classifier    │  5 classes:
│  (Anomaly Score)│     │    (Threat Class)│  Benign / DoS / PortScan
└─────────────────┘     └────────┬─────────┘  Brute Force / Bot
                                 │
                         ┌───────▼────────┐
                         │  MITRE ATT&CK  │
                         │  Mapper        │
                         └───────┬────────┘
                                 │
                         ┌───────▼────────┐
                         │  LangGraph     │  triage → enrich → report
                         │  Agent         │  (Groq Llama 3.1 70B)
                         └───────┬────────┘
                                 │
              ┌──────────────────┼──────────────────┐
              ▼                  ▼                   ▼
        FastAPI Backend     SQLite DB          Streamlit
        (REST API)          (Incidents)        Dashboard
```

---

## 📊 Results

| Metric | Score |
|---|---|
| Macro F1 | **0.9348** |
| Weighted F1 | **0.9991** |
| Benign F1 | 1.00 |
| DoS F1 | 1.00 |
| Brute Force F1 | 1.00 |
| PortScan F1 | 0.95 |
| Bot F1 | 0.73* |

*Bot has only 1,035 training samples vs 1.4M Benign — class imbalance handled
via `compute_sample_weight`. All Bot flows are caught (recall = 1.00).

---

## 🚀 Quick Start

### Prerequisites
- Python 3.12+
- A free [Groq API key](https://console.groq.com) for LLM report generation

### 1. Clone and install
```bash
git clone https://github.com/<your-username>/threatsense.git
cd threatsense
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Configure secrets
```bash
cp .env.example .env
# Edit .env and add your GROQ_API_KEY
```

### 3. Get the dataset
Download the CICIDS2017 Parquet files from
[Kaggle](https://www.kaggle.com/datasets/dhoogla/cicids2017) and place
all `.parquet` files in `data/raw/`.

### 4. Preprocess
```bash
python -m src.preprocess --raw-dir data/raw --out-dir data/processed --artifacts-dir models
```

### 5. Train models
```bash
python -m src.train                     # full dataset (~15 min)
python -m src.train --sample 100000     # fast dev run (~3 min)
```

### 6. Start the API
```bash
uvicorn api.main:app --reload --port 8000
# Swagger UI → http://localhost:8000/docs
```

### 7. Start the dashboard
```bash
streamlit run frontend/app.py
# Dashboard → http://localhost:8501
```

---

## 📁 Repo Structure

```
threatsense/
├── data/
│   ├── raw/            ← CICIDS2017 .parquet files (gitignored)
│   └── processed/      ← train/test splits (gitignored)
├── models/             ← trained .pkl files (gitignored)
├── src/
│   ├── preprocess.py   ← CICIDS2017 cleaning pipeline
│   ├── train.py        ← Isolation Forest + XGBoost training
│   ├── inference.py    ← prediction engine
│   ├── mitre_mapper.py ← ATT&CK technique mapping
│   └── agent.py        ← LangGraph SOC report agent
├── api/
│   └── main.py         ← FastAPI backend
├── frontend/
│   └── app.py          ← Streamlit dashboard
├── tests/              ← pytest test suite
├── docs/               ← screenshots, confusion matrix, SHAP plots
├── docker/             ← Dockerfiles
└── docker-compose.yml
```

---

## 🔌 API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Health check |
| `POST` | `/predict` | Single flow threat detection |
| `POST` | `/analyze` | Batch CSV upload + analysis |
| `POST` | `/report/{id}` | Generate LLM incident report |
| `GET` | `/incidents` | List all incidents |
| `GET` | `/incidents/{id}` | Get incident by ID |

Full interactive docs at `http://localhost:8000/docs`

---

## 🧠 Tech Stack

| Layer | Technology |
|---|---|
| ML Models | XGBoost, Isolation Forest, scikit-learn |
| Explainability | SHAP |
| LLM Agent | LangGraph, Groq (Llama 3.1 70B) |
| Backend | FastAPI, SQLite |
| Frontend | Streamlit |
| DevOps | Docker Compose, GitHub Actions |
| Dataset | CICIDS2017 (2.3M network flows) |

---

## 🗺️ MITRE ATT&CK Mapping

| Threat Class | Technique | Tactic | Severity |
|---|---|---|---|
| DoS | T1498, T1499 | Impact | 🔴 Critical |
| Bot | T1071, T1059 | Command & Control | 🔴 Critical |
| Brute Force | T1110, T1110.001 | Credential Access | 🟠 High |
| PortScan | T1046 | Discovery | 🟡 Medium |
| Benign | — | — | ⚪ None |

---

## 🧪 Running Tests

```bash
pytest tests/ -v
```

---

## 👤 Author

**Nadeemuddin** — Final-year B.E. AI & Data Science, CBIT Hyderabad

- GitHub: [@your-username](https://github.com/your-username)
- LinkedIn: [your-linkedin](https://linkedin.com/in/your-linkedin)
