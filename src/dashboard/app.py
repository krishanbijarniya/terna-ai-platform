import os
import sys

# Add project root to sys.path so we can import from `src`
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

import streamlit as st
import requests
import pandas as pd
import plotly.express as px

# Set page config
st.set_page_config(
    page_title="Terna Grid Outage Analytics Platform",
    page_icon="🏠",
    layout="wide"
)

# Import hybrid utils
from src.dashboard.utils import get_dashboard_kpis

# Inject custom premium CSS styling
st.markdown("""
<style>
    .kpi-container {
        display: flex;
        justify-content: space-between;
        margin-bottom: 20px;
    }
    .kpi-card {
        background-color: #f8f9fa;
        border-left: 5px solid #1f77b4;
        border-radius: 5px;
        padding: 15px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.05);
        flex: 1;
        margin-right: 15px;
        text-align: center;
    }
    .kpi-card:last-child {
        margin-right: 0;
    }
    .kpi-title {
        font-size: 14px;
        color: #6c757d;
        font-weight: 500;
        margin-bottom: 5px;
    }
    .kpi-value {
        font-size: 26px;
        color: #212529;
        font-weight: 700;
    }
</style>
""", unsafe_allow_html=True)

st.title("🔌 AI-Driven Transmission Grid Outage Analytics Platform")
st.markdown("### Italian Electricity Network Operational Decision Support System")
st.markdown("---")

# Fetch dashboard KPIs using hybrid engine
kpis, source = get_dashboard_kpis()
st.sidebar.info(f"Data Connection: **{source}**")

if kpis:
    # 1. Render KPI Metric Cards
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">📅 Total Outages</div>
            <div class="kpi-value">{kpis['total_outages']:,}</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown(f"""
        <div class="kpi-card" style="border-left-color: #ff7f0e;">
            <div class="kpi-title">⏳ Avg Duration</div>
            <div class="kpi-value">{kpis['avg_duration_hours']:.1f} hrs</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col3:
        st.markdown(f"""
        <div class="kpi-card" style="border-left-color: #d62728;">
            <div class="kpi-title">⚠️ Anomaly Rate</div>
            <div class="kpi-value">{kpis['anomaly_rate']*100:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col4:
        st.markdown(f"""
        <div class="kpi-card" style="border-left-color: #2ca02c;">
            <div class="kpi-title">🏗️ Affected Assets</div>
            <div class="kpi-value">{kpis['affected_assets_count']:,}</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col5:
        st.markdown(f"""
        <div class="kpi-card" style="border-left-color: #9467bd;">
            <div class="kpi-title">⚡ Avg Risk Score</div>
            <div class="kpi-value">{kpis['avg_risk_score']:.2f}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # 2. Render Charts
    chart_col1, chart_col2 = st.columns(2)
    
    with chart_col1:
        st.markdown("#### Outages Trend by Month (March - June 2026)")
        month_data = pd.DataFrame(list(kpis['outages_by_month'].items()), columns=['Month', 'Outage Count'])
        # Sort chronologically
        month_order = ['March 2026', 'April 2026', 'May 2026', 'June 2026']
        month_data['Month'] = pd.Categorical(month_data['Month'], categories=month_order, ordered=True)
        month_data = month_data.sort_values('Month')
        
        fig_month = px.bar(month_data, x='Month', y='Outage Count', color='Outage Count', color_continuous_scale='viridis')
        fig_month.update_layout(showlegend=False, height=400, margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig_month, use_container_width=True)
        
    with chart_col2:
        st.markdown("#### Outages Distribution by Asset Type")
        type_data = pd.DataFrame(list(kpis['outages_by_asset_type'].items()), columns=['Asset Type', 'Count']).head(8)
        fig_type = px.pie(type_data, values='Count', names='Asset Type', hole=0.4, color_discrete_sequence=px.colors.qualitative.Pastel)
        fig_type.update_layout(height=400, margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig_type, use_container_width=True)

st.markdown("---")
st.markdown("""
### 🏗️ Platform System Architecture
This platform acts as an **AI-driven decision support system** built on Terna's open dataset, designed to analyze planned grid outages and predict maintenance constraints.

*   **Phase 1 & 2 (Data Engineering)**: Automatically reads, cleans, deduplicates, and merges weekly outage sheets.
*   **Phase 3 (Exploratory Data Analysis)**: Evaluates weak points in transmission lines, voltage grids, and substation devices.
*   **Phase 4 (Feature Engineering)**: Generates calendar, holiday, rolling asset failure logs, and composite risk levels.
*   **Phase 5 (Machine Learning)**: Predicts outage duration (Random Forest Regressor), long/short downtime probability (XGBoost Classifier), maintenance clusters (K-Means), and anomalous grid signatures (Isolation Forest).
*   **Phase 6 (REST API)**: Serving predictions and queries via FastAPI at `http://localhost:8000`.

*You can navigate through other deep-dive dashboard pages (Analytics, Grid Status, Prediction, ML Performance, and Asset Explorer) using the sidebar.*
""")
