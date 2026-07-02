import streamlit as st
import pandas as pd
import plotly.express as px
import os

st.set_page_config(page_title="Grid Status", page_icon="⚡", layout="wide")

st.title("⚡ Grid Operation Status & Renewable Connections")
st.markdown("Overview of the wider Italian grid system metrics, fuel generation mixes, and renewable connection requests.")
st.markdown("---")

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
processed_dir = os.path.join(base_dir, "data", "processed")

# Helper function to load CSV
def load_csv(filename):
    path = os.path.join(processed_dir, filename)
    if os.path.exists(path):
        return pd.read_csv(path)
    return None

# Load datasets
demand = load_csv("demand_and_forecast.csv")
generation = load_csv("generation_by_source.csv")
imports = load_csv("imports_exports.csv")
requests_df = load_csv("renewable_connection_requests.csv")

if demand is None or generation is None or imports is None or requests_df is None:
    st.error("Failed to load processed operational datasets. Please ensure Phase 2 ETL ran successfully.")
else:
    # Convert dates
    demand["datetime"] = pd.to_datetime(demand["datetime"])
    generation["datetime"] = pd.to_datetime(generation["datetime"])
    imports["datetime"] = pd.to_datetime(imports["datetime"])

    # Row 1: Demand & Generation
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Electricity Demand vs Forecast by Bidding Zone")
        # Average load by bidding zone
        bz_load = demand.groupby("bidding_zone")[["total_load_mw", "forecast_total_load_mw"]].mean().reset_index()
        bz_load = bz_load.sort_values(by="total_load_mw", ascending=False)
        melted_demand = bz_load.melt(id_vars="bidding_zone", value_vars=["total_load_mw", "forecast_total_load_mw"], 
                                     var_name="Load Type", value_name="Avg Load [MW]")
        
        fig_demand = px.bar(melted_demand, x="bidding_zone", y="Avg Load [MW]", color="Load Type", barmode="group",
                            color_discrete_map={"total_load_mw": "#2ca02c", "forecast_total_load_mw": "#aec7e8"})
        fig_demand.update_layout(height=350, margin=dict(l=20, r=20, t=20, b=20), xaxis_title="Bidding Zone")
        st.plotly_chart(fig_demand, use_container_width=True)
        
    with col2:
        st.markdown("#### Average Generation Contribution by Primary Source")
        gen_grouped = generation.groupby("primary_source")["actual_generation_mw"].mean().reset_index()
        gen_grouped = gen_grouped.sort_values("actual_generation_mw", ascending=False)
        
        fig_gen = px.bar(gen_grouped, x="primary_source", y="actual_generation_mw", color="actual_generation_mw",
                         color_continuous_scale="algae")
        fig_gen.update_layout(height=350, showlegend=False, margin=dict(l=20, r=20, t=20, b=20), 
                              xaxis_title="Primary Source", yaxis_title="Average Generation [MW]")
        st.plotly_chart(fig_gen, use_container_width=True)
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Row 2: Imports & Renewable connection requests
    col3, col4 = st.columns(2)
    
    with col3:
        st.markdown("#### Net Cross-Border Exchanges (Positive = Imports, Negative = Exports)")
        border = imports.groupby("country")["scheduled_foreign_exchange_mw"].mean().reset_index()
        border = border.sort_values("scheduled_foreign_exchange_mw", ascending=False)
        
        fig_border = px.bar(border, x="country", y="scheduled_foreign_exchange_mw", color="scheduled_foreign_exchange_mw",
                            color_continuous_scale="RdBu_r")
        fig_border.add_hline(y=0, line_dash="dash", line_color="black")
        fig_border.update_layout(height=350, showlegend=False, margin=dict(l=20, r=20, t=20, b=20),
                                 xaxis_title="Border Country", yaxis_title="Net Scheduled Flow [MW]")
        st.plotly_chart(fig_border, use_container_width=True)
        
    with col4:
        st.markdown("#### Top 10 Regions by Requested Renewable Connection Power (MW)")
        req_grouped = requests_df.groupby("region")["power_mw"].sum().reset_index()
        req_grouped = req_grouped.sort_values("power_mw", ascending=False).head(10)
        
        fig_req = px.bar(req_grouped, x="power_mw", y="region", orientation="h", color="power_mw",
                         color_continuous_scale="teal")
        fig_req.update_layout(height=350, showlegend=False, margin=dict(l=20, r=20, t=20, b=20),
                              xaxis_title="Requested Power [MW]", yaxis_title="Region")
        st.plotly_chart(fig_req, use_container_width=True)
