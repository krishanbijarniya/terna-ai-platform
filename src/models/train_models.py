import os
import logging
import joblib
import pandas as pd
import numpy as np
import mlflow

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, silhouette_score
from xgboost import XGBClassifier

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

BASE_DIR = r"C:\Users\kk928\.gemini\antigravity\scratch\terna-ai-platform"
DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "engineered_outages.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
MLFLOW_URI = f"sqlite:///{BASE_DIR.replace('\\', '/')}/mlflow.db"

os.makedirs(MODELS_DIR, exist_ok=True)

# Set MLflow tracking URI
mlflow.set_tracking_uri(MLFLOW_URI)
mlflow.set_experiment("Grid Outage Analytics Platform")

def train_pipeline():
    logger.info("Loading engineered outages dataset...")
    if not os.path.exists(DATA_PATH):
        logger.error(f"Dataset not found at: {DATA_PATH}")
        return
        
    df = pd.read_csv(DATA_PATH)
    
    # 1. Feature Columns Definition
    numerical_features = [
        'voltage_kv', 'prev_outages_count', 'rolling_mean_duration_3', 
        'rolling_downtime_3', 'frequency_index', 'risk_score',
        'start_month', 'start_quarter', 'start_week', 'start_day', 'start_dayofweek'
    ]
    categorical_features = ['asset_type', 'voltage_category', 'maintenance_category', 'season']
    boolean_features = ['is_weekend', 'is_holiday', 'daily_restoring']
    
    features = numerical_features + categorical_features + boolean_features
    target_reg = 'duration_hours'
    
    # Drop rows where target is missing (if any)
    df = df.dropna(subset=[target_reg])
    
    X = df[features]
    y_reg = df[target_reg]
    
    # Target for classification (Long outage: duration > 24 hours)
    y_clf = (y_reg > 24.0).astype(int)
    
    # Split into train/test (80% train, 20% test)
    X_train, X_test, y_train_reg, y_test_reg, y_train_clf, y_test_clf = train_test_split(
        X, y_reg, y_clf, test_size=0.2, random_state=42
    )
    
    # 2. Fit Preprocessor (ColumnTransformer)
    logger.info("Initializing and fitting preprocessor...")
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features),
            ('bool', 'passthrough', boolean_features)
        ]
    )
    
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    # Save the preprocessor
    preprocessor_path = os.path.join(MODELS_DIR, "preprocessor.joblib")
    joblib.dump(preprocessor, preprocessor_path)
    logger.info(f"Preprocessor saved to: {preprocessor_path}")
    
    # Get processed feature names
    cat_encoder = preprocessor.named_transformers_['cat']
    encoded_cat_cols = cat_encoder.get_feature_names_out(categorical_features).tolist()
    processed_feature_names = numerical_features + encoded_cat_cols + boolean_features
    logger.info(f"Total processed features: {len(processed_feature_names)}")
    
    # Save feature names list for API use
    feature_names_path = os.path.join(MODELS_DIR, "feature_names.joblib")
    joblib.dump(processed_feature_names, feature_names_path)

    # 3. Model 1: Regression (Random Forest)
    logger.info("Training Model 1: Outage Duration Regressor (Random Forest)...")
    reg_model = RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1)
    
    with mlflow.start_run(run_name="Duration_Regression_RandomForest"):
        reg_model.fit(X_train_processed, y_train_reg)
        
        # Predictions and evaluation
        preds = reg_model.predict(X_test_processed)
        mae = mean_absolute_error(y_test_reg, preds)
        rmse = np.sqrt(mean_squared_error(y_test_reg, preds))
        r2 = r2_score(y_test_reg, preds)
        
        logger.info(f"Regression Results - MAE: {mae:.2f}, RMSE: {rmse:.2f}, R2: {r2:.3f}")
        
        # Log to MLflow
        mlflow.log_params({"n_estimators": 100, "max_depth": 12, "model_type": "RandomForestRegressor"})
        mlflow.log_metrics({"MAE": mae, "RMSE": rmse, "R2": r2})
        mlflow.log_artifact(DATA_PATH)
        
        # Save model file
        model_path = os.path.join(MODELS_DIR, "duration_regressor.joblib")
        joblib.dump(reg_model, model_path)
        mlflow.log_artifact(model_path)
        logger.info(f"Regressor saved to: {model_path}")
        
    # 4. Model 2: Classification (XGBoost)
    logger.info("Training Model 2: Risk Classifier (XGBoost)...")
    # map boolean types to int/float for XGBoost safety
    X_train_proc_xgb = X_train_processed.astype(float)
    X_test_proc_xgb = X_test_processed.astype(float)
    
    clf_model = XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, eval_metric="logloss", n_jobs=-1)
    
    with mlflow.start_run(run_name="Risk_Classification_XGBoost"):
        clf_model.fit(X_train_proc_xgb, y_train_clf)
        
        # Predictions and evaluation
        preds_class = clf_model.predict(X_test_proc_xgb)
        preds_prob = clf_model.predict_proba(X_test_proc_xgb)[:, 1]
        
        acc = accuracy_score(y_test_clf, preds_class)
        prec = precision_score(y_test_clf, preds_class, zero_division=0)
        rec = recall_score(y_test_clf, preds_class, zero_division=0)
        f1 = f1_score(y_test_clf, preds_class, zero_division=0)
        auc = roc_auc_score(y_test_clf, preds_prob)
        
        logger.info(f"Classification Results - Acc: {acc:.3f}, F1: {f1:.3f}, AUC: {auc:.3f}")
        
        # Log to MLflow
        mlflow.log_params({"n_estimators": 100, "max_depth": 6, "learning_rate": 0.1, "model_type": "XGBClassifier"})
        mlflow.log_metrics({"Accuracy": acc, "Precision": prec, "Recall": rec, "F1-Score": f1, "ROC-AUC": auc})
        
        # Save model file
        model_path = os.path.join(MODELS_DIR, "risk_classifier.joblib")
        joblib.dump(clf_model, model_path)
        mlflow.log_artifact(model_path)
        logger.info(f"Classifier saved to: {model_path}")

    # 5. Model 3: Asset Clustering (K-Means)
    logger.info("Training Model 3: Asset Clusterer (K-Means)...")
    # Features for clustering (standardized asset behavioral features)
    cluster_cols = ['voltage_kv', 'prev_outages_count', 'rolling_mean_duration_3', 'frequency_index', 'risk_score']
    
    cluster_scaler = StandardScaler()
    X_cluster = cluster_scaler.fit_transform(df[cluster_cols])
    
    # Save cluster scaler
    scaler_path = os.path.join(MODELS_DIR, "cluster_scaler.joblib")
    joblib.dump(cluster_scaler, scaler_path)
    
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    
    with mlflow.start_run(run_name="Asset_Clustering_KMeans"):
        kmeans.fit(X_cluster)
        
        # Silhouette score on a sample of 2000 points to keep execution fast
        sample_indices = np.random.choice(X_cluster.shape[0], min(2000, X_cluster.shape[0]), replace=False)
        sil = silhouette_score(X_cluster[sample_indices], kmeans.labels_[sample_indices])
        
        logger.info(f"Clustering Results - Silhouette Score (sample): {sil:.3f}")
        
        # Log to MLflow
        mlflow.log_params({"n_clusters": 3, "model_type": "KMeans"})
        mlflow.log_metrics({"Silhouette": sil})
        
        # Save model file
        model_path = os.path.join(MODELS_DIR, "asset_clusterer.joblib")
        joblib.dump(kmeans, model_path)
        mlflow.log_artifact(model_path)
        mlflow.log_artifact(scaler_path)
        logger.info(f"Clusterer saved to: {model_path}")
        
    # 6. Model 4: Anomaly Detection (Isolation Forest)
    logger.info("Training Model 4: Anomaly Detector (Isolation Forest)...")
    anomaly_cols = ['duration_hours', 'voltage_kv', 'risk_score', 'frequency_index']
    
    X_anomaly = df[anomaly_cols].fillna(0.0)
    
    detector = IsolationForest(contamination=0.05, random_state=42, n_jobs=-1)
    
    with mlflow.start_run(run_name="Anomaly_Detection_IsolationForest"):
        detector.fit(X_anomaly)
        
        # Predictions (-1 = Anomaly, 1 = Normal)
        preds_anomaly = detector.predict(X_anomaly)
        anomaly_count = int((preds_anomaly == -1).sum())
        logger.info(f"Anomaly Detection Results - Found {anomaly_count} anomalies out of {X_anomaly.shape[0]} records.")
        
        # Log to MLflow
        mlflow.log_params({"contamination": 0.05, "model_type": "IsolationForest"})
        mlflow.log_metrics({"Anomalies_Count": anomaly_count})
        
        # Save model file
        model_path = os.path.join(MODELS_DIR, "anomaly_detector.joblib")
        joblib.dump(detector, model_path)
        mlflow.log_artifact(model_path)
        logger.info(f"Anomaly detector saved to: {model_path}")
        
    logger.info("All four models successfully trained, evaluated, and saved!")

if __name__ == "__main__":
    train_pipeline()
