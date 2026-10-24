<!-- Terna Grid Outage Analytics & Predictive Maintenance · README -->
<div align="center">

<img width="100%" alt="Terna Grid AI header" src="https://capsule-render.vercel.app/api?type=waving&color=0:0F172A,45:1D4ED8,75:2563EB,100:06B6D4&height=230&section=header&text=Terna%20Grid%20AI&fontSize=54&fontColor=FFFFFF&fontAlignY=40&desc=Outage%20Analytics%20%E2%80%A2%20Predictive%20Maintenance%20%E2%80%A2%20Italian%20Transmission%20Grid&descAlignY=68&descSize=17&animation=fadeIn"/>

<br/>

<img alt="Python" src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
<img alt="ML" src="https://img.shields.io/badge/ML-XGBoost%20%C2%B7%20scikit--learn-189FDD?style=for-the-badge"/>
<img alt="Tracking" src="https://img.shields.io/badge/Tracking-MLflow-0194E2?style=for-the-badge&logo=mlflow&logoColor=white"/>
<img alt="API" src="https://img.shields.io/badge/API-FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white"/>
<img alt="Dashboard" src="https://img.shields.io/badge/Dashboard-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white"/>

<br/><br/>

<a href="#-overview"><b>Overview</b></a> •
<a href="#-architecture"><b>Architecture</b></a> •
<a href="#-engineering-highlights"><b>Highlights</b></a> •
<a href="#-quick-start"><b>Quick Start</b></a> •
<a href="#-recruiter-snapshot"><b>Recruiter Snapshot</b></a> •
<a href="#-contact"><b>Contact</b></a>

</div>

---

<div align="center">

<a href="https://terna-ai-platform-msfn6wl5ifyrq6nulp6t4d.streamlit.app/"><img src="https://img.shields.io/badge/%F0%9F%9A%80%20Live%20Demo-Open%20Dashboard-16A34A?style=for-the-badge" alt="Live demo"/></a>

</div>

# 🧭 Overview

An **end-to-end industrial ML platform** built on public operational data from **Terna**, Italy's electricity transmission system operator. It cleans raw weekly outage logs, engineers **leakage-free rolling features and composite risk scores**, trains **four models**, serves predictions through a **FastAPI** service and presents everything in a **multipage Streamlit dashboard**.

---

# 🏗️ Architecture

<details open>
<summary><b>🧠 Data → features → models → product (click to collapse)</b></summary>

<br/>

```mermaid
flowchart LR
  A["📥 Raw outage &amp;<br/>operational Excels"] --> B["🧹 ETL pipeline"]
  B --> C["🧬 Feature engineering<br/>temporal · holiday · rolling stats<br/>composite risk score"]
  C --> D1["🌲 Random Forest<br/>outage duration"]
  C --> D2["⚡ XGBoost<br/>long-outage risk"]
  C --> D3["🧭 K-Means<br/>asset segments"]
  C --> D4["🚨 Isolation Forest<br/>anomalies"]
  D1 & D2 & D3 & D4 --> E["⚡ FastAPI<br/>/predict · /risk · /assets"]
  E --> F["📊 Streamlit dashboard<br/>Outages · Grid Status · Prediction<br/>ML Performance · Asset Explorer"]
```

</details>


---

# 📈 Model Results

<details open>
<summary><b>🏁 Model cockpit (click to collapse)</b></summary>

<br/>

<table>
<tr>
<td width="25%" valign="top">

### ⚡ Risk Classifier
**XGBoost** — outage > 24 h?

Accuracy **78.0%** · F1 **0.767** · ROC-AUC **0.865**

</td>
<td width="25%" valign="top">

### 🌲 Duration Regressor
**Random Forest** — outage hours

MAE 427.8 h · R² 0.195 *(hard, heavy-tailed target)*

</td>
<td width="25%" valign="top">

### 🧭 Segmentation
**K-Means (k=3)** maintenance behaviour

Silhouette **0.536**

</td>
<td width="25%" valign="top">

### 🚨 Anomalies
**Isolation Forest** (5% contamination)

**893** abnormal records flagged

</td>
</tr>
</table>

</details>


---

# ⚙️ Engineering Highlights

- **Leakage-free features** — rolling statistics computed strictly on past data, verified by tests
- **Single preprocessing pipeline** (`ColumnTransformer`) shared by training and serving
- **MLflow** experiment logging and versioned `joblib` model artifacts
- **Pydantic-validated REST API** with Swagger docs; test suite covering ETL, features, models and API

---

# 🚀 Quick Start

```bash
pip install -r requirements.txt
python src/etl/etl_pipeline.py
python src/feature_engineering/feature_engineering.py
python src/models/train_models.py
uvicorn src.api.main:app --port 8000          # API  → /docs
streamlit run src/dashboard/app.py            # UI   → :8501
```

---

# 🎯 Recruiter Snapshot

<details open>
<summary><b>📌 What to remember in 15 seconds (click to collapse)</b></summary>

<br/>

<table>
<tr>
<td width="50%" valign="top">

### ✅ Demonstrated
- Real-world industrial data, end to end
- Supervised + unsupervised ML
- API serving & deployed dashboard
- Testing and experiment tracking

</td>
<td width="50%" valign="top">

### 🧠 Why it matters
- Prioritises maintenance on high-risk assets
- Honest metric reporting
- Live and explorable today

</td>
</tr>
</table>

</details>


---

# 📚 Full Technical Documentation

<details>
<summary><b>🛠️ Setup, configuration, API reference and implementation details (click to expand)</b></summary>

<br/>

An enterprise-grade, end-to-end industrial machine learning and analytics platform built on public operational datasets from **Terna**, the Italian electricity transmission system operator (TSO). 

The platform cleanses raw grid maintenance logs, engineers leakage-free rolling features and risk scores, trains four machine learning models, exposes REST API prediction services via FastAPI, and presents an interactive six-page dashboard in Streamlit.

---
Deployed at https://terna-ai-platform-msfn6wl5ifyrq6nulp6t4d.streamlit.app/Asset_Explorer
## 🏗️ System Architecture

```mermaid
flowchart TD
    subgraph Data Layer
        A1[Raw Weekly Outage Excels] --> B[ETL Pipeline]
        A2[Raw Operational Excels] --> B
        B --> C[Cleaned Merged CSVs]
    end

    subgraph Feature Engineering Layer
        C --> D[Temporal, Holiday, and Rolling Stats Generation]
        D --> E[Composite Grid Risk Scoring]
        E --> F[Feature Matrix: engineered_outages.csv]
    end

    subgraph Machine Learning Layer
        F --> G[Pipeline Preprocessing: ColumnTransformer]
        G --> H1[Random Forest Regressor: Duration Prediction]
        G --> H2[XGBoost Classifier: Long Outage Risk]
        G --> H3[K-Means Clustering: Asset Segmentation]
        G --> H4[Isolation Forest: Anomaly Detection]
    end

    subgraph serving [Serving & Presentation Layer]
        H1 & H2 & H3 & H4 --> I[FastAPI REST API Service]
        I --> J[Streamlit Analytics Dashboard]
    end
```

---

## 📁 Repository Structure

```text
terna-ai-platform/
├── data/
│   ├── raw/                        # Raw weekly Excel logs (untracked)
│   └── processed/                  # Cleaned merged CSV datasets
├── models/                         # Fitted preprocessing and ML binaries
│   ├── preprocessor.joblib         # Categorical/numerical preprocessor
│   ├── duration_regressor.joblib   # Random Forest outage duration regressor
│   ├── risk_classifier.joblib      # XGBoost risk length classifier
│   ├── asset_clusterer.joblib      # K-Means asset grouping model
│   └── anomaly_detector.joblib     # Isolation Forest anomaly detector
├── notebooks/
│   └── 01_exploratory_data_analysis.ipynb
├── reports/
│   ├── eda_report.md               # Compiler report for statistical insights
│   └── images/                     # Compiled visualization plots (PNGs)
├── src/
│   ├── api/
│   │   ├── main.py                 # FastAPI application and endpoints
│   │   └── schemas.py              # Pydantic validation schemas
│   ├── dashboard/
│   │   ├── app.py                  # Streamlit frontend Overview entry
│   │   └── pages/                  # Multipage dashboard tabs
│   ├── etl/
│   │   └── etl_pipeline.py         # Outage and load files ETL script
│   ├── feature_engineering/
│   │   └── feature_engineering.py  # Mappings, rolling stats, and risk scores
│   └── models/
│   │   └── train_models.py         # Multi-model training and MLflow logging
├── tests/
│   ├── test_api.py                 # API endpoints validation
│   ├── test_etl.py                 # ETL pipeline validation
│   ├── test_features.py            # Feature engineering validation
│   └── test_models.py              # ML model inference validation
├── requirements.txt                # Python package dependencies
└── README.md                       # Documentation overview
```

---

## ⚡ Setup & Installation

### 1. Initialize Virtual Environment
Ensure you have Python 3.10+ installed. Run:
```bash
python -m venv .venv
.venv\Scripts\activate      # On Windows
source .venv/bin/activate    # On Unix/macOS
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Pipeline Processes
Execute the pipelines sequentially to stage data, extract features, and train the model weights:
```bash
# Run ETL Pipeline
python src/etl/etl_pipeline.py

# Run Feature Engineering
python src/feature_engineering/feature_engineering.py

# Run Model Training & MLflow Registration
python src/models/train_models.py
```

---

## 🚀 Running the Services

The platform runs as a decoupled frontend-backend microservice setup.

### 1. Start FastAPI REST Backend
Start the prediction and query server:
```bash
uvicorn src.api.main:app --host 127.0.0.1 --port 8000
```
*   **API Documentation**: Browse interactive Swagger docs at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).
*   **Health Check**: [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health).

### 2. Start Streamlit Interactive Dashboard
In a separate terminal window:
```bash
streamlit run src/dashboard/app.py --server.port 8501
```
Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 🔌 API Endpoints Summary

| Endpoint | Method | Description |
|---|---|---|
| `/health` | `GET` | Server service check. |
| `/predict` | `POST` | Runs a payload through preprocessing and returns duration, risk class, cluster, and anomaly predictions. |
| `/assets` | `GET` | Returns unique grid assets sorted by outages volume or cumulative downtime. |
| `/outages` | `GET` | Paginated list of individual outages filterable by asset type or voltage. |
| `/risk` | `GET` | Top high-risk assets sorted by composite risk scores. |
| `/dashboard` | `GET` | Aggregate KPIs and distributions for the Streamlit cockpit. |
| `/model` | `GET` | ML validation performance metrics. |

---

## 🤖 Machine Learning Model Cockpit

1. **Duration Prediction (Regression)**: Trains a `Random Forest Regressor` to predict continuous outage hours (`duration_hours`).
   * *MAE*: 427.80 hours
   * *R² Score*: 0.195
2. **Risk Classification (Classification)**: Trains an `XGBoost Classifier` to predict if an outage will exceed 24 hours (`is_long`).
   * *Accuracy*: 78.0%
   * *F1-Score*: 0.767
   * *ROC-AUC*: 0.865
3. **Asset Clustering**: Fits `K-Means` ($k=3$) to segment maintenance behavioral patterns.
   * *Silhouette Score*: 0.536
4. **Anomaly Detection**: Fits `Isolation Forest` with a 5% expected contamination rate to flag abnormal operational signatures.
   * *Detected Anomalies*: 893 records

---

## 🧪 Running Tests

A comprehensive unit and integration test suite is located in `/tests`. Execute all tests with:
```bash
# Set PYTHONPATH to root directory
$env:PYTHONPATH="."
pytest
```
All tests assert data validation ranges, ensure no data leakage on rolling features, and check API request schemas.

</details>


---

# 📬 Contact

<div align="center">

**Krishan Kumar** — AI Engineer & Data Scientist · Rome, Italy

<a href="https://www.linkedin.com/in/krishan-kumar-ml/"><img src="https://img.shields.io/badge/LinkedIn-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white" alt="LinkedIn"/></a>
<a href="mailto:krishbijarniya@gmail.com"><img src="https://img.shields.io/badge/Email-EA4335?style=for-the-badge&logo=gmail&logoColor=white" alt="Email"/></a>
<a href="https://krishan-kumar-portfolio.vercel.app/"><img src="https://img.shields.io/badge/Portfolio-0F172A?style=for-the-badge&logo=vercel&logoColor=white" alt="Portfolio"/></a>
<a href="https://github.com/krishanbijarniya"><img src="https://img.shields.io/badge/More%20projects-181717?style=for-the-badge&logo=github&logoColor=white" alt="GitHub"/></a>

</div>

<img width="100%" src="https://capsule-render.vercel.app/api?type=waving&color=0:06B6D4,50:1D4ED8,100:0F172A&height=120&section=footer" alt=""/>
