import os
import joblib
import pytest
import pandas as pd
import numpy as np

BASE_DIR = r"C:\Users\kk928\.gemini\antigravity\scratch\terna-ai-platform"
MODELS_DIR = os.path.join(BASE_DIR, "models")

@pytest.fixture
def dummy_input():
    """
    Creates a single dummy row matching the original feature schema
    to verify pipeline preprocessing and inference.
    """
    row = {
        'voltage_kv': 132.0,
        'prev_outages_count': 1,
        'rolling_mean_duration_3': 24.0,
        'rolling_downtime_3': 72.0,
        'frequency_index': 0.5,
        'risk_score': 0.2,
        'start_month': 5,
        'start_quarter': 2,
        'start_week': 20,
        'start_day': 15,
        'start_dayofweek': 4,
        'asset_type': 'LIN',
        'voltage_category': '132-150 kV',
        'maintenance_category': 'Technical Controls',
        'season': 'Spring',
        'is_weekend': False,
        'is_holiday': False,
        'daily_restoring': False
    }
    return pd.DataFrame([row])

def test_model_binaries_exist():
    """
    Verify all required pipeline components are saved.
    """
    expected_files = [
        "preprocessor.joblib",
        "feature_names.joblib",
        "duration_regressor.joblib",
        "risk_classifier.joblib",
        "cluster_scaler.joblib",
        "asset_clusterer.joblib",
        "anomaly_detector.joblib"
    ]
    for filename in expected_files:
        path = os.path.join(MODELS_DIR, filename)
        assert os.path.exists(path), f"Required model file '{filename}' is missing!"

def test_inference_pipeline(dummy_input):
    """
    Test loading components and executing end-to-end inference.
    """
    # 1. Load Preprocessor
    prep_path = os.path.join(MODELS_DIR, "preprocessor.joblib")
    preprocessor = joblib.load(prep_path)
    
    # 2. Transform dummy input
    processed_X = preprocessor.transform(dummy_input)
    assert processed_X.shape[0] == 1, "Transformation row count mismatch!"
    
    # 3. Test Model 1: Regressor
    reg_path = os.path.join(MODELS_DIR, "duration_regressor.joblib")
    regressor = joblib.load(reg_path)
    reg_pred = regressor.predict(processed_X)
    assert isinstance(reg_pred[0], float) or isinstance(reg_pred[0], np.float64)
    assert reg_pred[0] >= 0.0, "Regressor predicted negative duration!"

    # 4. Test Model 2: Classifier
    clf_path = os.path.join(MODELS_DIR, "risk_classifier.joblib")
    classifier = joblib.load(clf_path)
    
    # XGBoost requires float or double type array
    processed_X_float = processed_X.astype(float)
    clf_pred = classifier.predict(processed_X_float)
    clf_proba = classifier.predict_proba(processed_X_float)[:, 1]
    
    assert clf_pred[0] in [0, 1], "Classifier returned invalid class prediction!"
    assert 0.0 <= clf_proba[0] <= 1.0, "Classifier probability out of bounds!"

def test_asset_clustering():
    """
    Verify K-Means cluster scaler and model load and execute.
    """
    scaler_path = os.path.join(MODELS_DIR, "cluster_scaler.joblib")
    clusterer_path = os.path.join(MODELS_DIR, "asset_clusterer.joblib")
    
    scaler = joblib.load(scaler_path)
    clusterer = joblib.load(clusterer_path)
    
    # Dummy cluster input: voltage_kv, prev_outages_count, rolling_mean_duration_3, frequency_index, risk_score
    dummy_cluster_data = np.array([[132.0, 2.0, 48.0, 1.5, 0.35]])
    scaled_data = scaler.transform(dummy_cluster_data)
    
    cluster_pred = clusterer.predict(scaled_data)
    assert cluster_pred[0] in [0, 1, 2], "Cluster assignment out of bounds!"

def test_anomaly_detection():
    """
    Verify Isolation Forest anomaly detector loads and predicts.
    """
    detector_path = os.path.join(MODELS_DIR, "anomaly_detector.joblib")
    detector = joblib.load(detector_path)
    
    # Dummy anomaly input: duration_hours, voltage_kv, risk_score, frequency_index
    dummy_anomaly_data = pd.DataFrame([{
        'duration_hours': 120.0,
        'voltage_kv': 380.0,
        'risk_score': 0.85,
        'frequency_index': 4.0
    }])
    
    anomaly_pred = detector.predict(dummy_anomaly_data)
    # -1 represents an anomaly, 1 represents normal data
    assert anomaly_pred[0] in [-1, 1], "Anomaly detector returned invalid prediction class!"
