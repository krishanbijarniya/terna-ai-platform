import streamlit as st
import pandas as pd
import plotly.express as px
import os

st.set_page_config(page_title="Outage Analytics", page_icon="📈", layout="wide")

st.title("📈 Transmission Grid Outage Analytics")
st.markdown("Detailed breakdown of Terna grid planned outages, asset types, and repair cycles.")
st.markdown("---")

# Locate data
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
data_path = os.path.join(base_dir, "data", "processed", "merged_outages.csv")

if not os.path.exists(data_path):
    st.error(f"Processed outages dataset missing: {data_path}")
else:
    df = pd.read_csv(data_path)
    df['start_datetime'] = pd.to_datetime(df['start_datetime'])
    df['stop_datetime'] = pd.to_datetime(df['stop_datetime'])
    
    # Extract month and week categories
    df['Month'] = df['start_datetime'].dt.strftime('%B %Y')
    df['Week'] = df['start_datetime'].dt.to_period('W').astype(str)
    
    # Extract numeric kV levels for filter
    voltages = sorted(df['voltage_kv'].dropna().unique().tolist())
    voltages_str = [f"{int(v)} kV" for v in voltages]
    
    # 1. Sidebar Filters
    st.sidebar.header("Filter Outages")
    selected_asset = st.sidebar.multiselect("Select Asset Type", options=df['asset_type'].unique(), default=df['asset_type'].unique()[:3])
    selected_voltage = st.sidebar.multiselect("Select Voltage Level", options=voltages_str, default=voltages_str[:4])
    
    # Filter logic
    selected_voltage_floats = [float(v.split(' ')[0]) for v in selected_voltage]
    
    filtered_df = df[
        df['asset_type'].isin(selected_asset) & 
        df['voltage_kv'].isin(selected_voltage_floats)
    ]
    
    if filtered_df.empty:
        st.warning("No data matches the selected filters. Please select different options.")
    else:
        st.markdown(f"Showing **{len(filtered_df):,}** outages out of **{len(df):,}** total records.")
        
        # Row 1: Time Series & Durations
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Weekly Outage Ingestion Trend")
            weekly_counts = filtered_df.groupby('Week').size().reset_index(name='Outages Count')
            fig_week = px.line(weekly_counts, x='Week', y='Outages Count', markers=True, color_discrete_sequence=['#ff7f0e'])
            fig_week.update_layout(height=350, margin=dict(l=20, r=20, t=20, b=20))
            st.plotly_chart(fig_week, use_container_width=True)
            
        with col2:
            st.markdown("#### Distribution of Outage Durations (Hours)")
            fig_dist = px.histogram(filtered_df, x='duration_hours', nbins=50, color_discrete_sequence=['#1f77b4'], log_y=True)
            fig_dist.update_layout(height=350, margin=dict(l=20, r=20, t=20, b=20), xaxis_title="Duration (Hours)", yaxis_title="Count (Log Scale)")
            st.plotly_chart(fig_dist, use_container_width=True)
            
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Row 2: Reasons & Voltage Splits
        col3, col4 = st.columns(2)
        
        with col3:
            st.markdown("#### Top 10 Maintenance Reasons")
            reasons = filtered_df['reason'].value_counts().head(10).reset_index(name='Count')
            reasons.rename(columns={'reason': 'Reason'}, inplace=True)
            fig_reason = px.bar(reasons, y='Reason', x='Count', orientation='h', color='Count', color_continuous_scale='plasma')
            fig_reason.update_layout(height=350, showlegend=False, margin=dict(l=20, r=20, t=20, b=20))
            st.plotly_chart(fig_reason, use_container_width=True)
            
        with col4:
            st.markdown("#### Average Outage Duration by Voltage Level")
            volt_dur = filtered_df.groupby('voltage_kv')['duration_hours'].mean().reset_index()
            volt_dur['voltage_kv'] = volt_dur['voltage_kv'].astype(int).astype(str) + " kV"
            fig_volt = px.bar(volt_dur, x='voltage_kv', y='duration_hours', color='duration_hours', color_continuous_scale='magma')
            fig_volt.update_layout(height=350, showlegend=False, margin=dict(l=20, r=20, t=20, b=20), xaxis_title="Voltage Level", yaxis_title="Average Duration (Hours)")
            st.plotly_chart(fig_volt, use_container_width=True)
