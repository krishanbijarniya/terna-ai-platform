import streamlit as st
import requests

st.set_page_config(page_title="ML Performance", page_icon="📊", layout="wide")

st.title("📊 Machine Learning Models Performance & Metadata")
st.markdown("Metrics, parameters, and algorithms deployed for outage duration predictions, risk classification, clustering, and anomaly detection.")
st.markdown("---")

from src.dashboard.utils import get_model_metadata

meta, source = get_model_metadata()
st.sidebar.info(f"Metadata Source: **{source}**")

if meta:
    # Set tabs for each model
    tab1, tab2, tab3, tab4 = st.tabs([
        "🔮 Model 1: Duration Regressor", 
        "🔴 Model 2: Risk Classifier", 
        "📂 Model 3: Asset Clusterer", 
        "⚠️ Model 4: Anomaly Detector"
    ])
    
    with tab1:
        st.markdown("### Model 1: Outage Duration Prediction (Regression)")
        st.markdown("""
        *   **Algorithm**: Random Forest Regressor (`scikit-learn`)
        *   **Objective**: Predict the continuous number of hours a planned outage window will last.
        *   **Parameters**: `n_estimators=100`, `max_depth=12`, `random_state=42`
        """)
        
        reg_metrics = meta["duration_regressor"]
        mcol1, mcol2, mcol3 = st.columns(3)
        mcol1.metric("MAE (Mean Absolute Error)", f"{reg_metrics['MAE_hours']:.1f} Hours")
        mcol2.metric("RMSE (Root Mean Squared Error)", f"{reg_metrics['RMSE_hours']:.1f} Hours")
        mcol3.metric("R² Score (Coefficient of Determination)", f"{reg_metrics['R2_score']:.3f}")
        
        st.info("💡 **Interpretation**: R² shows the model captures variance in planned scheduling lengths, with MAE indicating typical scheduling deviation in hours.")

    with tab2:
        st.markdown("### Model 2: Risk Length Classifier (Classification)")
        st.markdown("""
        *   **Algorithm**: XGBoost Classifier (`xgboost`)
        *   **Objective**: Predict whether a planned outage window will be 'long' (> 24 hours) or 'short' (<= 24 hours).
        *   **Parameters**: `n_estimators=100`, `max_depth=6`, `learning_rate=0.1`, `eval_metric='logloss'`
        """)
        
        clf_metrics = meta["risk_classifier"]
        ccol1, ccol2, ccol3 = st.columns(3)
        ccol1.metric("Accuracy Score", f"{clf_metrics['Accuracy']*100:.1f}%")
        ccol2.metric("F1-Score", f"{clf_metrics['F1_score']:.3f}")
        ccol3.metric("ROC-AUC Score", f"{clf_metrics['ROC_AUC']:.3f}")
        
        st.info("💡 **Interpretation**: An AUC score of 0.865 indicates highly robust discriminative ability to sort high-downtime outages from standard routine servicing windows.")

    with tab3:
        st.markdown("### Model 3: Asset Behavioral Clustering (Unsupervised)")
        st.markdown("""
        *   **Algorithm**: K-Means Clustering (`scikit-learn`)
        *   **Objective**: Segment grid outages into distinct behavioral risk profiles.
        *   **Features Used**: Voltage, failure history count, rolling downtime sum, frequency rate.
        *   **Parameters**: `n_clusters=3`, `n_init=10`, `random_state=42`
        """)
        
        cls_metrics = meta["asset_clusterer"]
        st.metric("Silhouette Coefficient (Silhouette Score)", f"{cls_metrics['Silhouette_score']:.3f}")
        st.info("💡 **Interpretation**: A silhouette score of 0.536 represents solid, well-separated cluster divisions, allowing us to accurately identify and group critical substation behaviors.")

    with tab4:
        st.markdown("### Model 4: Outage Anomaly Detection (Unsupervised)")
        st.markdown("""
        *   **Algorithm**: Isolation Forest (`scikit-learn`)
        *   **Objective**: Scan grid planned operations and isolate abnormal outage signatures.
        *   **Parameters**: `contamination=0.05` (5% expected anomaly rate), `random_state=42`
        """)
        
        det_metrics = meta["anomaly_detector"]
        acol1, acol2 = st.columns(2)
        acol1.metric("Contamination Ratio", f"{det_metrics['Contamination_rate_percent']:.1f}%")
        acol2.metric("Total Anomalies Detected", f"{det_metrics['Detected_anomalies_count']:,}")
        
        st.info("💡 **Interpretation**: Isolation Forest flags events that deviate significantly from typical voltage and repair durations, helping operators review scheduling risks.")
st.markdown("---")
st.markdown("All model runs and tracking metrics are registered in our local **MLflow registry** database (`mlflow.db`).")
