import streamlit as st
import pandas as pd
import requests
import datetime

st.set_page_config(page_title="Outage Predictor", page_icon="🤖", layout="wide")

st.title("🤖 Grid Outage Duration & Risk Predictor")
st.markdown("Submit grid asset details to predict planned outage duration, risk category, behavioral cluster, and detect anomaly patterns.")
st.markdown("---")

API_URL = "http://127.0.0.1:8000/predict"

def get_easter_date(year):
    easter_dates = {
        2024: datetime.date(2024, 3, 31),
        2025: datetime.date(2025, 4, 20),
        2026: datetime.date(2026, 4, 5)
    }
    return easter_dates.get(year, None)

def is_italian_holiday(date):
    year = date.year
    fixed_holidays = {
        (1, 1), (1, 6), (4, 25), (5, 1), (6, 2),
        (8, 15), (11, 1), (12, 8), (12, 25), (12, 26)
    }
    if (date.month, date.day) in fixed_holidays:
        return True
    easter = get_easter_date(year)
    if easter:
        pasquetta = easter + datetime.timedelta(days=1)
        if date == easter or date == pasquetta:
            return True
    return False

# Form columns split
col1, col2 = st.columns(2)

with col1:
    st.markdown("#### ⚙️ Asset Configuration")
    asset_type = st.selectbox("Asset Type Code", ["LIN", "STL", "SBA", "ATR", "STK", "TRF", "RIF", "TRC", "STT"])
    voltage_kv = st.slider("Voltage Level (kV)", min_value=10.0, max_value=400.0, value=132.0, step=1.0)
    
    # Auto voltage category mapping
    if voltage_kv < 100:
        vol_cat = "LV/MV"
    elif voltage_kv <= 150:
        vol_cat = "132-150 kV"
    elif voltage_kv <= 220:
        vol_cat = "220 kV"
    else:
        vol_cat = "400 kV"
    st.info(f"Automatically mapped Voltage Category: **{vol_cat}**")
    
    maintenance_category = st.selectbox("Maintenance Reason Category", 
                                        ["Technical Controls", "Replacement", "Interference", "Development/Rinnovo", "Other"])
    
    daily_restoring = st.checkbox("Daily Restoring Flag (Outage restored at the end of each day)")

with col2:
    st.markdown("#### 🕒 Asset History & Schedule")
    prev_outages_count = st.number_input("Cumulative Previous Outages Count", min_value=0, max_value=100, value=2)
    rolling_mean_duration_3 = st.number_input("Rolling Mean Duration of Last 3 Outages (hours)", min_value=0.0, max_value=2000.0, value=24.0)
    rolling_downtime_3 = st.number_input("Rolling Downtime Sum of Last 3 Outages (hours)", min_value=0.0, max_value=6000.0, value=72.0)
    frequency_index = st.slider("Monthly Outage Frequency Rate", min_value=0.0, max_value=10.0, value=0.5, step=0.1)
    
    # Grid Risk Score calculations (matched to formula)
    vol_norm = voltage_kv / 400.0
    cnt_norm = min(prev_outages_count / 10.0, 1.0)
    dur_norm = min(rolling_mean_duration_3 / 168.0, 1.0)
    risk_score = 0.4 * vol_norm + 0.3 * cnt_norm + 0.3 * dur_norm
    st.info(f"Automatically calculated local Risk Score: **{risk_score:.2f}**")
    
    # Time Pickers
    start_date = st.date_input("Start Date", value=datetime.date(2026, 6, 15))
    
    # Extract time fields
    start_month = start_date.month
    start_quarter = (start_month - 1) // 3 + 1
    start_week = start_date.isocalendar()[1]
    start_day = start_date.day
    start_dayofweek = start_date.weekday()
    is_weekend = bool(start_dayofweek in [5, 6])
    is_holiday = bool(is_italian_holiday(start_date))
    
    season_map = {12: "Winter", 1: "Winter", 2: "Winter", 3: "Spring", 4: "Spring", 5: "Spring", 6: "Summer", 7: "Summer", 8: "Summer", 9: "Autumn", 10: "Autumn", 11: "Autumn"}
    season = season_map.get(start_month, "Spring")

st.markdown("<br>", unsafe_allow_html=True)
if st.button("🔮 Run Predictive Ingest", use_container_width=True):
    # Prepare payload
    payload = {
        "voltage_kv": float(voltage_kv),
        "prev_outages_count": int(prev_outages_count),
        "rolling_mean_duration_3": float(rolling_mean_duration_3),
        "rolling_downtime_3": float(rolling_downtime_3),
        "frequency_index": float(frequency_index),
        "risk_score": float(risk_score),
        "start_month": int(start_month),
        "start_quarter": int(start_quarter),
        "start_week": int(start_week),
        "start_day": int(start_day),
        "start_dayofweek": int(start_dayofweek),
        "asset_type": asset_type,
        "voltage_category": vol_cat,
        "maintenance_category": maintenance_category,
        "season": season,
        "is_weekend": is_weekend,
        "is_holiday": is_holiday,
        "daily_restoring": daily_restoring
    }
    
    with st.spinner("Processing features and executing model models..."):
        try:
            response = requests.post(API_URL, json=payload, timeout=5)
            if response.status_code == 200:
                res = response.json()
                
                st.markdown("### 📊 Prediction Outputs")
                st.markdown("---")
                
                # Render results columns
                res_col1, res_col2 = st.columns(2)
                
                with res_col1:
                    st.metric("⏳ Predicted Duration", f"{res['predicted_duration_hours']:.1f} Hours")
                    
                    # Risk Classification Alert Box
                    if res['is_long_outage']:
                        st.error(f"🔴 **High Outage Risk Warning**: Predicted to exceed 24 hours (Probability: {res['long_outage_probability']*100:.1f}%)")
                    else:
                        st.success(f"🟢 **Low Outage Risk**: Predicted to stay under 24 hours (Probability: {(1 - res['long_outage_probability'])*100:.1f}%)")
                        
                with res_col2:
                    # Cluster Assign
                    cluster_id = res['cluster_assignment']
                    cluster_desc = {
                        0: "Cluster 0: Low Maintenance Footprint (typically short, routine operations).",
                        1: "Cluster 1: Moderate Maintenance Load (standard substation check windows).",
                        2: "Cluster 2: Critical Outage Signature (high voltage levels, high repeat rates)."
                    }
                    st.info(f"📂 **Maintenance Cluster**: Cluster {cluster_id}\n\n{cluster_desc.get(cluster_id)}")
                    
                    # Anomaly Warning
                    if res['is_anomaly']:
                        st.warning(f"⚠️ **Outage Pattern Anomaly Detected**: Outage characteristics are abnormal compared to the historical training base (Decision Score: {res['anomaly_score']:.2f})")
                    else:
                        st.success(f"✓ **Outage Pattern Normal**: Outage matches typical grid scheduling standards (Decision Score: {res['anomaly_score']:.2f})")
                        
            else:
                st.error(f"API Error (HTTP {response.status_code}): {response.text}")
        except Exception as e:
            st.error(f"Could not connect to the FastAPI prediction endpoint: {e}")
