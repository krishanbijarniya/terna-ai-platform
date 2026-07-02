import os
import logging
import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, Optional

from src.api.schemas import PredictionRequest, PredictionResponse, AssetItem, OutageItem, DashboardKPIs, ModelMetadata

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Dynamic paths setup
API_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(os.path.dirname(API_DIR))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "engineered_outages.csv")

app = FastAPI(
    title="Terna Grid Outage Analytics API",
    description="REST API service exposing grid planned outage predictions, anomaly detection, clustering, and risk assessments.",
    version="1.0.0"
)

# Enable CORS for dashboard web frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables to store loaded models and datasets
models = {}
db_data = None
assets_db = None
dashboard_kpis = None

@app.on_event("startup")
def startup_event():
    global db_data, assets_db, dashboard_kpis
    logger.info("Initializing REST API backend service...")
    
    # 1. Load pipeline artifacts
    required_artifacts = {
        "preprocessor": "preprocessor.joblib",
        "regressor": "duration_regressor.joblib",
        "classifier": "risk_classifier.joblib",
        "cluster_scaler": "cluster_scaler.joblib",
        "clusterer": "asset_clusterer.joblib",
        "anomaly_detector": "anomaly_detector.joblib"
    }
    
    for key, filename in required_artifacts.items():
        path = os.path.join(MODELS_DIR, filename)
        if not os.path.exists(path):
            logger.error(f"Critical model binary missing: {path}")
            raise RuntimeError(f"Artifact {filename} missing!")
        models[key] = joblib.load(path)
        logger.info(f"Successfully loaded: {filename}")
        
    # 2. Ingest engineered dataset
    if not os.path.exists(DATA_PATH):
        logger.error(f"Engineered dataset missing: {DATA_PATH}")
        raise RuntimeError("Dataset engineered_outages.csv missing!")
        
    db_data = pd.read_csv(DATA_PATH)
    logger.info(f"Loaded master dataset containing {len(db_data)} rows.")
    
    # Score anomalies on all dataset rows using Isolation Forest
    logger.info("Scoring dataset anomalies dynamically...")
    anomaly_cols = ['duration_hours', 'voltage_kv', 'risk_score', 'frequency_index']
    X_anomaly = db_data[anomaly_cols].fillna(0.0)
    db_data['is_anomaly'] = models['anomaly_detector'].predict(X_anomaly) == -1
    
    # Cache grouped assets database
    logger.info("Caching asset aggregation table...")
    # Group by asset concerned
    assets_grouped = db_data.groupby('assets_concerned').agg(
        asset_type=('asset_type', 'first'),
        voltage_kv=('voltage_kv', 'first'),
        total_outages=('assets_concerned', 'count'),
        total_downtime_hours=('duration_hours', 'sum'),
        avg_duration_hours=('duration_hours', 'mean'),
        avg_risk_score=('risk_score', 'mean')
    ).reset_index()
    assets_db = assets_grouped
    
    # Cache dashboard KPIs
    logger.info("Calculating dashboard KPI metrics...")
    total_out = len(db_data)
    avg_dur = float(db_data['duration_hours'].mean())
    anomaly_cnt = int(db_data['is_anomaly'].sum())
    anomaly_rt = float(anomaly_cnt / total_out) if total_out > 0 else 0.0
    distinct_assets = int(db_data['assets_concerned'].nunique())
    avg_risk = float(db_data['risk_score'].mean())
    
    # Distributions
    # Map months to strings
    db_data['month_str'] = pd.to_datetime(db_data['start_datetime']).dt.strftime('%B %Y')
    outages_by_month = db_data['month_str'].value_counts().to_dict()
    outages_by_asset_type = db_data['asset_type'].value_counts().to_dict()
    outages_by_voltage_category = db_data['voltage_category'].value_counts().to_dict()
    
    dashboard_kpis = {
        "total_outages": total_out,
        "avg_duration_hours": avg_dur,
        "anomaly_rate": anomaly_rt,
        "affected_assets_count": distinct_assets,
        "avg_risk_score": avg_risk,
        "outages_by_month": outages_by_month,
        "outages_by_asset_type": outages_by_asset_type,
        "outages_by_voltage_category": outages_by_voltage_category
    }
    
    logger.info("API Startup sequence complete!")

@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "Terna Grid Outage Analytics Backend"}

@app.post("/predict", response_model=PredictionResponse)
def predict_outage(request: PredictionRequest):
    """
    Run predictive inference on a new outage request:
    - Outage duration (regression - Random Forest)
    - Long/short outage class (classification - XGBoost)
    - Asset behavioral cluster assignment (K-Means)
    - Anomaly status (Isolation Forest)
    """
    try:
        # Convert request to pandas DataFrame
        input_dict = request.dict()
        df_input = pd.DataFrame([input_dict])
        
        # 1. Preprocess features
        X_processed = models['preprocessor'].transform(df_input)
        
        # 2. Predict duration (regression)
        pred_duration = float(models['regressor'].predict(X_processed)[0])
        
        # 3. Predict long/short class (classification)
        # Cast to float for XGBoost compatibility
        X_processed_float = X_processed.astype(float)
        pred_class = int(models['classifier'].predict(X_processed_float)[0])
        pred_proba = float(models['classifier'].predict_proba(X_processed_float)[0][1])
        
        # 4. Predict behavioral cluster (K-Means)
        cluster_features = [[
            request.voltage_kv,
            request.prev_outages_count,
            request.rolling_mean_duration_3,
            request.frequency_index,
            request.risk_score
        ]]
        cluster_scaled = models['cluster_scaler'].transform(cluster_features)
        pred_cluster = int(models['clusterer'].predict(cluster_scaled)[0])
        
        # 5. Predict anomaly status (Isolation Forest)
        # Isolation Forest is fitted on: duration_hours, voltage_kv, risk_score, frequency_index
        anomaly_features = [[
            pred_duration,  # Use predicted duration for the anomaly check
            request.voltage_kv,
            request.risk_score,
            request.frequency_index
        ]]
        anomaly_label = models['anomaly_detector'].predict(anomaly_features)[0]
        is_anom = bool(anomaly_label == -1)
        anom_score = float(models['anomaly_detector'].decision_function(anomaly_features)[0])
        
        return PredictionResponse(
            predicted_duration_hours=pred_duration,
            is_long_outage=bool(pred_class == 1),
            long_outage_probability=pred_proba,
            cluster_assignment=pred_cluster,
            is_anomaly=is_anom,
            anomaly_score=anom_score
        )
        
    except Exception as e:
        logger.error(f"Inference error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Inference pipeline execution failed: {str(e)}")

@app.get("/assets", response_model=List[AssetItem])
def get_assets(
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    sort_by: str = Query("outages", regex="^(outages|downtime)$")
):
    """
    Returns a paginated list of grid assets grouped by total outages or cumulative downtime.
    """
    if assets_db is None:
        raise HTTPException(status_code=503, detail="Dataset not ready")
        
    sort_col = "total_outages" if sort_by == "outages" else "total_downtime_hours"
    sorted_df = assets_db.sort_values(by=sort_col, ascending=False)
    
    paginated = sorted_df.iloc[offset : offset + limit]
    
    return paginated.to_dict(orient="records")

@app.get("/outages", response_model=List[OutageItem])
def get_outages(
    asset_type: Optional[str] = None,
    voltage_category: Optional[str] = None,
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0)
):
    """
    Returns a paginated list of individual outages with filters.
    """
    if db_data is None:
        raise HTTPException(status_code=503, detail="Dataset not ready")
        
    filtered_df = db_data.copy()
    
    if asset_type:
        filtered_df = filtered_df[filtered_df['asset_type'] == asset_type.upper().strip()]
    if voltage_category:
        filtered_df = filtered_df[filtered_df['voltage_category'] == voltage_category.strip()]
        
    # Sort chronologically (latest first)
    filtered_df = filtered_df.sort_values(by='start_datetime', ascending=False)
    paginated = filtered_df.iloc[offset : offset + limit]
    
    # Cast date to string for pydantic serialization
    records = paginated.to_dict(orient="records")
    for r in records:
        r['start_datetime'] = str(r['start_datetime'])
        r['stop_datetime'] = str(r['stop_datetime'])
        
    return records

@app.get("/risk", response_model=List[AssetItem])
def get_high_risk_assets(
    limit: int = Query(50, ge=1, le=200)
):
    """
    Returns top assets sorted by their average risk score.
    """
    if assets_db is None:
        raise HTTPException(status_code=503, detail="Dataset not ready")
        
    sorted_df = assets_db.sort_values(by="avg_risk_score", ascending=False)
    paginated = sorted_df.iloc[:limit]
    
    return paginated.to_dict(orient="records")

@app.get("/dashboard", response_model=DashboardKPIs)
def get_dashboard_kpis():
    """
    Returns cached KPI card summaries and distributions for dashboard charts.
    """
    if dashboard_kpis is None:
        raise HTTPException(status_code=503, detail="KPI calculations not ready")
    return dashboard_kpis

@app.get("/model", response_model=ModelMetadata)
def get_model_metadata():
    """
    Returns validation performance metrics logged during model training.
    """
    return ModelMetadata(
        duration_regressor={
            "MAE_hours": 427.80,
            "RMSE_hours": 3828.10,
            "R2_score": 0.195
        },
        risk_classifier={
            "Accuracy": 0.780,
            "F1_score": 0.767,
            "ROC_AUC": 0.865
        },
        asset_clusterer={
            "Silhouette_score": 0.536
        },
        anomaly_detector={
            "Contamination_rate_percent": 5,
            "Detected_anomalies_count": 893
        }
    )
