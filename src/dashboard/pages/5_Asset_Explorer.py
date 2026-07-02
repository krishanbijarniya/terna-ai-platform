import streamlit as st
import pandas as pd
import requests
import plotly.express as px

st.set_page_config(page_title="Asset Explorer", page_icon="📍", layout="wide")

st.title("📍 Grid Asset Explorer & Risk Rankings")
st.markdown("Query, filter, and inspect specific grid transmission assets, their historical downtime profiles, and composite risk metrics.")
st.markdown("---")

API_BASE_URL = "http://127.0.0.1:8000"

col1, col2 = st.columns(2)

# Load assets database from API
try:
    response_assets = requests.get(f"{API_BASE_URL}/assets?limit=500", timeout=5)
    response_risk = requests.get(f"{API_BASE_URL}/risk?limit=15", timeout=5)
    
    if response_assets.status_code == 200 and response_risk.status_code == 200:
        assets_data = response_assets.json()
        risk_data = response_risk.json()
    else:
        st.error("Failed to retrieve asset data from API backend.")
        assets_data = None
        risk_data = None
except Exception as e:
    st.error(f"Could not connect to FastAPI server asset endpoints: {e}")
    assets_data = None
    risk_data = None

if assets_data and risk_data:
    df_assets = pd.DataFrame(assets_data)
    df_risk = pd.DataFrame(risk_data)
    
    with col1:
        st.markdown("#### 🔍 Asset Query Database")
        search_query = st.text_input("Search Asset Name (regex / substring):", value="")
        
        # Filter table
        if search_query:
            df_filtered = df_assets[df_assets['assets_concerned'].str.contains(search_query, case=False, na=False)]
        else:
            df_filtered = df_assets
            
        st.markdown(f"Found **{len(df_filtered)}** matching asset records.")
        
        # Display clean dataframe
        st.dataframe(
            df_filtered.rename(columns={
                'assets_concerned': 'Asset Name',
                'asset_type': 'Type',
                'voltage_kv': 'Voltage [kV]',
                'total_outages': 'Total Outages',
                'total_downtime_hours': 'Total Downtime [hrs]',
                'avg_duration_hours': 'Avg Duration [hrs]',
                'avg_risk_score': 'Avg Risk Score'
            }),
            use_container_width=True,
            hide_index=True
        )
        
    with col2:
        st.markdown("#### ⚡ Top 15 Highest Risk Grid Assets")
        
        # Plotly bar chart of risk scores
        fig_risk = px.bar(
            df_risk.head(15), 
            x='avg_risk_score', 
            y='assets_concerned', 
            orientation='h', 
            color='avg_risk_score',
            color_continuous_scale='sunset',
            labels={'avg_risk_score': 'Average Risk Score', 'assets_concerned': 'Asset Name'}
        )
        fig_risk.update_layout(height=450, showlegend=False, margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig_risk, use_container_width=True)
        
        st.markdown("""
        **Risk Score Drivers**:
        *   **Voltage Level**: Assets operating at `400 kV` receive higher base risk weights.
        *   **Failures Frequency**: Assets with repeat outages (`prev_outages_count`) scale up the risk.
        *   **Repair Complexity**: Long rolling mean durations increase the score.
        """)
