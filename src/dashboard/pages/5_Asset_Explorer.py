import streamlit as st
import pandas as pd
import requests
import plotly.express as px

st.set_page_config(page_title="Asset Explorer", page_icon="📍", layout="wide")

st.title("📍 Grid Asset Explorer & Risk Rankings")
st.markdown("Query, filter, and inspect specific grid transmission assets, their historical downtime profiles, and composite risk metrics.")
st.markdown("---")

from src.dashboard.utils import get_assets_list, get_risk_rankings

col1, col2 = st.columns(2)

# Load assets database from API
assets_data, source_assets = get_assets_list(limit=500, sort_by="outages")
risk_data, source_risk = get_risk_rankings(limit=15)
st.sidebar.info(f"Asset Data Source: **{source_assets}**")

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
