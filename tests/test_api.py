import pytest
from fastapi.testclient import TestClient
from src.api.main import app

@pytest.fixture(scope="module")
def client():
    """
    Fixture that runs the FastAPI startup events and yields a test client.
    """
    with TestClient(app) as c:
        yield c

def test_health_check(client):
    """
    Verify the health check endpoint returns 200 OK and healthy status.
    """
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_dashboard_kpis(client):
    """
    Verify the dashboard KPI endpoint returns the cached aggregates.
    """
    response = client.get("/dashboard")
    assert response.status_code == 200
    data = response.json()
    
    assert "total_outages" in data
    assert "avg_duration_hours" in data
    assert "anomaly_rate" in data
    assert "affected_assets_count" in data
    assert "avg_risk_score" in data
    
    assert data["total_outages"] == 17846
    assert data["affected_assets_count"] > 0
    assert 0.0 <= data["anomaly_rate"] <= 1.0

def test_assets_list(client):
    """
    Verify the grouped assets endpoint returns data and respects sorting/limits.
    """
    # Test default limit and outages sorting
    response = client.get("/assets?limit=10&sort_by=outages")
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 10
    
    # Assert sorted descending by outages
    outages_counts = [x["total_outages"] for x in data]
    assert outages_counts == sorted(outages_counts, reverse=True)
    
    # Test downtime sorting
    response = client.get("/assets?limit=5&sort_by=downtime")
    assert response.status_code == 200
    data_dt = response.json()
    assert len(data_dt) == 5
    downtimes = [x["total_downtime_hours"] for x in data_dt]
    assert downtimes == sorted(downtimes, reverse=True)

def test_outages_list(client):
    """
    Verify the individual outages list filter and pagination.
    """
    response = client.get("/outages?limit=5&asset_type=LIN&voltage_category=132-150%20kV")
    assert response.status_code == 200
    data = response.json()
    assert len(data) <= 5
    for item in data:
        assert item["asset_type"] == "LIN"
        assert "voltage_kv" in item
        assert "start_datetime" in item

def test_risk_rankings(client):
    """
    Verify the risk score ranking query returns 200 and is sorted correctly.
    """
    response = client.get("/risk?limit=10")
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 10
    
    risk_scores = [x["avg_risk_score"] for x in data]
    assert risk_scores == sorted(risk_scores, reverse=True)

def test_model_metadata(client):
    """
    Verify model metadata metrics endpoint.
    """
    response = client.get("/model")
    assert response.status_code == 200
    data = response.json()
    assert "duration_regressor" in data
    assert "risk_classifier" in data
    assert "asset_clusterer" in data

def test_prediction_endpoint(client):
    """
    Test end-to-end inference prediction with a valid payload.
    """
    payload = {
        "voltage_kv": 132.0,
        "prev_outages_count": 2,
        "rolling_mean_duration_3": 12.5,
        "rolling_downtime_3": 37.5,
        "frequency_index": 0.33,
        "risk_score": 0.45,
        "start_month": 6,
        "start_quarter": 2,
        "start_week": 24,
        "start_day": 20,
        "start_dayofweek": 5,
        "asset_type": "LIN",
        "voltage_category": "132-150 kV",
        "maintenance_category": "Technical Controls",
        "season": "Summer",
        "is_weekend": True,
        "is_holiday": False,
        "daily_restoring": False
    }
    
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    
    assert "predicted_duration_hours" in data
    assert "is_long_outage" in data
    assert "long_outage_probability" in data
    assert "cluster_assignment" in data
    assert "is_anomaly" in data
    assert "anomaly_score" in data
    
    assert data["predicted_duration_hours"] >= 0.0
    assert data["cluster_assignment"] in [0, 1, 2]
    assert isinstance(data["is_anomaly"], bool)
