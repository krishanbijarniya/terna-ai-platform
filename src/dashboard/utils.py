import os
import joblib
import pandas as pd
import numpy as np
import requests
import streamlit as st

API_BASE_URL = "http://127.0.0.1:8000"

# Compute local file paths relative to this file
DASHBOARD_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(os.path.dirname(DASHBOARD_DIR))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "engineered_outages.csv")

# 1. Cached Local Model Loader (only loads once to conserve memory)
@st.cache_resource
def load_local_models():
    """
    Loads all saved joblib model components.
    """
    required = {
        "preprocessor": "preprocessor.joblib",
        "regressor": "duration_regressor.joblib",
        "classifier": "risk_classifier.joblib",
        "cluster_scaler": "cluster_scaler.joblib",
        "clusterer": "asset_clusterer.joblib",
        "anomaly_detector": "anomaly_detector.joblib"
    }
    loaded = {}
    for key, filename in required.items():
        path = os.path.join(MODELS_DIR, filename)
        if os.path.exists(path):
            loaded[key] = joblib.load(path)
        else:
            raise FileNotFoundError(f"Model file missing: {path}")
    return loaded

# 2. Cached Local Dataset Loader
@st.cache_data
def load_local_dataset():
    """
    Loads processed engineered outages dataset.
    """
    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH)
        return df
    return None

def check_api_online():
    """
    Checks if the FastAPI backend is running and reachable.
    """
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=1.5)
        return response.status_code == 200
    except Exception:
        return False

# 3. Hybrid Prediction Handler
def predict_outage(payload):
    """
    Attempts to predict using backend API, falls back to local models if API is offline.
    """
    if check_api_online():
        try:
            response = requests.post(f"{API_BASE_URL}/predict", json=payload, timeout=3)
            if response.status_code == 200:
                return response.json(), "API Server"
        except Exception:
            pass
            
    # Local Fallback Execution
    local_models = load_local_models()
    df_input = pd.DataFrame([payload])
    
    # 1. Preprocess
    X_processed = local_models['preprocessor'].transform(df_input)
    
    # 2. Predict regressor
    pred_duration = float(local_models['regressor'].predict(X_processed)[0])
    
    # 3. Predict classifier
    X_processed_float = X_processed.astype(float)
    pred_class = int(local_models['classifier'].predict(X_processed_float)[0])
    pred_proba = float(local_models['classifier'].predict_proba(X_processed_float)[0][1])
    
    # 4. Predict K-Means
    cluster_features = [[
        payload["voltage_kv"],
        payload["prev_outages_count"],
        payload["rolling_mean_duration_3"],
        payload["frequency_index"],
        payload["risk_score"]
    ]]
    cluster_scaled = local_models['cluster_scaler'].transform(cluster_features)
    pred_cluster = int(local_models['clusterer'].predict(cluster_scaled)[0])
    
    # 5. Predict anomaly
    anomaly_features = [[
        pred_duration,
        payload["voltage_kv"],
        payload["risk_score"],
        payload["frequency_index"]
    ]]
    anomaly_label = local_models['anomaly_detector'].predict(anomaly_features)[0]
    is_anom = bool(anomaly_label == -1)
    anom_score = float(local_models['anomaly_detector'].decision_function(anomaly_features)[0])
    
    local_res = {
        "predicted_duration_hours": pred_duration,
        "is_long_outage": bool(pred_class == 1),
        "long_outage_probability": pred_proba,
        "cluster_assignment": pred_cluster,
        "is_anomaly": is_anom,
        "anomaly_score": anom_score
    }
    return local_res, "Local Container (Cloud Fallback)"

# 4. Hybrid Dashboard KPI Handler
def get_dashboard_kpis():
    """
    Fetches dashboard aggregates from API, falls back to local data processing if API is offline.
    """
    if check_api_online():
        try:
            response = requests.get(f"{API_BASE_URL}/dashboard", timeout=3)
            if response.status_code == 200:
                return response.json(), "API Server"
        except Exception:
            pass
            
    # Local Fallback Processing
    df = load_local_dataset()
    if df is None:
        raise FileNotFoundError("Local database file not found.")
        
    # Anomaly scoring on data
    local_models = load_local_models()
    anomaly_cols = ['duration_hours', 'voltage_kv', 'risk_score', 'frequency_index']
    X_anomaly = df[anomaly_cols].fillna(0.0)
    df['is_anomaly'] = local_models['anomaly_detector'].predict(X_anomaly) == -1
    
    total_out = len(df)
    avg_dur = float(df['duration_hours'].mean())
    anomaly_cnt = int(df['is_anomaly'].sum())
    anomaly_rt = float(anomaly_cnt / total_out) if total_out > 0 else 0.0
    distinct_assets = int(df['assets_concerned'].nunique())
    avg_risk = float(df['risk_score'].mean())
    
    df['month_str'] = pd.to_datetime(df['start_datetime']).dt.strftime('%B %Y')
    outages_by_month = df['month_str'].value_counts().to_dict()
    outages_by_asset_type = df['asset_type'].value_counts().to_dict()
    outages_by_voltage_category = df['voltage_category'].value_counts().to_dict()
    
    local_kpis = {
        "total_outages": total_out,
        "avg_duration_hours": avg_dur,
        "anomaly_rate": anomaly_rt,
        "affected_assets_count": distinct_assets,
        "avg_risk_score": avg_risk,
        "outages_by_month": outages_by_month,
        "outages_by_asset_type": outages_by_asset_type,
        "outages_by_voltage_category": outages_by_voltage_category
    }
    return local_kpis, "Local Container (Cloud Fallback)"

# 5. Hybrid Model Metadata Handler
def get_model_metadata():
    """
    Fetches ML performance metrics from API, returns local constants if offline.
    """
    if check_api_online():
        try:
            response = requests.get(f"{API_BASE_URL}/model", timeout=2)
            if response.status_code == 200:
                return response.json(), "API Server"
        except Exception:
            pass
            
    local_meta = {
        "duration_regressor": {"MAE_hours": 427.80, "RMSE_hours": 3828.10, "R2_score": 0.195},
        "risk_classifier": {"Accuracy": 0.780, "F1_score": 0.767, "ROC_AUC": 0.865},
        "asset_clusterer": {"Silhouette_score": 0.536},
        "anomaly_detector": {"Contamination_rate_percent": 5, "Detected_anomalies_count": 893}
    }
    return local_meta, "Local Container (Cloud Fallback)"

# 6. Hybrid Assets Handler
def get_assets_list(limit=500, sort_by="outages"):
    """
    Fetches asset listings from API, processes locally if offline.
    """
    if check_api_online():
        try:
            response = requests.get(f"{API_BASE_URL}/assets?limit={limit}&sort_by={sort_by}", timeout=3)
            if response.status_code == 200:
                return response.json(), "API Server"
        except Exception:
            pass
            
    # Local Fallback Processing
    df = load_local_dataset()
    if df is None:
        raise FileNotFoundError("Local database file not found.")
        
    assets_grouped = df.groupby('assets_concerned').agg(
        asset_type=('asset_type', 'first'),
        voltage_kv=('voltage_kv', 'first'),
        total_outages=('assets_concerned', 'count'),
        total_downtime_hours=('duration_hours', 'sum'),
        avg_duration_hours=('duration_hours', 'mean'),
        avg_risk_score=('risk_score', 'mean')
    ).reset_index()
    
    sort_col = "total_outages" if sort_by == "outages" else "total_downtime_hours"
    sorted_df = assets_grouped.sort_values(by=sort_col, ascending=False)
    paginated = sorted_df.iloc[:limit]
    
    return paginated.to_dict(orient="records"), "Local Container (Cloud Fallback)"

# 7. Hybrid Risk Rankings Handler
def get_risk_rankings(limit=15):
    """
    Fetches risk rankings from API, processes locally if offline.
    """
    if check_api_online():
        try:
            response = requests.get(f"{API_BASE_URL}/risk?limit={limit}", timeout=3)
            if response.status_code == 200:
                return response.json(), "API Server"
        except Exception:
            pass
            
    # Local Fallback Processing
    assets, source = get_assets_list(limit=limit, sort_by="outages")
    df_assets = pd.DataFrame(assets)
    if df_assets.empty:
        return [], "Local Container (Cloud Fallback)"
    sorted_df = df_assets.sort_values(by="avg_risk_score", ascending=False).iloc[:limit]
    return sorted_df.to_dict(orient="records"), "Local Container (Cloud Fallback)"
