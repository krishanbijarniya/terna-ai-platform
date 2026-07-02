from pydantic import BaseModel, Field
from typing import List, Dict, Optional

class PredictionRequest(BaseModel):
    voltage_kv: float = Field(..., description="Voltage level in kV (e.g. 132.0, 380.0)", example=132.0)
    prev_outages_count: int = Field(..., description="Cumulative previous outages of this asset", example=1)
    rolling_mean_duration_3: float = Field(..., description="Average duration of the last 3 outages of this asset (hours)", example=24.0)
    rolling_downtime_3: float = Field(..., description="Cumulative downtime of the last 3 outages of this asset (hours)", example=72.0)
    frequency_index: float = Field(..., description="Monthly failure frequency index", example=0.5)
    risk_score: float = Field(..., description="Grid Risk Score (0.0 to 1.0)", example=0.35)
    start_month: int = Field(..., description="Month of year (1-12)", example=5)
    start_quarter: int = Field(..., description="Quarter (1-4)", example=2)
    start_week: int = Field(..., description="Week of year (1-53)", example=20)
    start_day: int = Field(..., description="Day of month (1-31)", example=15)
    start_dayofweek: int = Field(..., description="Day of week (0=Mon, 6=Sun)", example=4)
    asset_type: str = Field(..., description="Asset type code (e.g. LIN, STL, SBA)", example="LIN")
    voltage_category: str = Field(..., description="Voltage category (e.g. 132-150 kV, 400 kV)", example="132-150 kV")
    maintenance_category: str = Field(..., description="Maintenance category (e.g. Technical Controls, Interference)", example="Technical Controls")
    season: str = Field(..., description="Season name (Spring, Summer, Autumn, Winter)", example="Spring")
    is_weekend: bool = Field(..., description="Is the start day a weekend?", example=False)
    is_holiday: bool = Field(..., description="Is the start day a holiday?", example=False)
    daily_restoring: bool = Field(..., description="Is the outage daily restored?", example=False)

class PredictionResponse(BaseModel):
    predicted_duration_hours: float = Field(..., description="Predicted continuous outage duration (hours)")
    is_long_outage: bool = Field(..., description="Predicted if duration exceeds 24 hours")
    long_outage_probability: float = Field(..., description="Probability of a long outage (> 24 hours)")
    cluster_assignment: int = Field(..., description="Behavioral cluster assignment (0, 1, or 2)")
    is_anomaly: bool = Field(..., description="Is the outage pattern anomalous?")
    anomaly_score: float = Field(..., description="Anomaly detector decision score")

class AssetItem(BaseModel):
    assets_concerned: str
    asset_type: str
    voltage_kv: float
    total_outages: int
    total_downtime_hours: float
    avg_duration_hours: float
    avg_risk_score: float

class OutageItem(BaseModel):
    assets_concerned: str
    asset_type: str
    voltage_kv: float
    start_datetime: str
    stop_datetime: str
    duration_hours: float
    reason: str
    daily_restoring: bool
    risk_score: float
    is_anomaly: Optional[bool] = None

class DashboardKPIs(BaseModel):
    total_outages: int
    avg_duration_hours: float
    anomaly_rate: float
    affected_assets_count: int
    avg_risk_score: float
    outages_by_month: Dict[str, int]
    outages_by_asset_type: Dict[str, int]
    outages_by_voltage_category: Dict[str, int]

class ModelMetadata(BaseModel):
    duration_regressor: Dict[str, float] = Field(..., description="Regression evaluation metrics (MAE, RMSE, R2)")
    risk_classifier: Dict[str, float] = Field(..., description="Classification evaluation metrics (Accuracy, F1, AUC)")
    asset_clusterer: Dict[str, float] = Field(..., description="Clustering silhouette score")
    anomaly_detector: Dict[str, int] = Field(..., description="Anomaly counts detected")
