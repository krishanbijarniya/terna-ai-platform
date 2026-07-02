import os
import pytest
import pandas as pd
import numpy as np

BASE_DIR = r"C:\Users\kk928\.gemini\antigravity\scratch\terna-ai-platform"
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")

@pytest.fixture
def outages_df():
    out_file = os.path.join(PROCESSED_DIR, "merged_outages.csv")
    assert os.path.exists(out_file), "merged_outages.csv does not exist!"
    df = pd.read_csv(out_file)
    return df

def test_outages_schema(outages_df):
    """
    Verify that the merged outages dataset contains all expected columns and has rows.
    """
    expected_cols = [
        'assets_concerned', 'asset_type', 'voltage_kv_raw', 'reason', 
        'source_file', 'start_datetime', 'stop_datetime', 'voltage_kv', 
        'daily_restoring', 'duration_hours'
    ]
    for col in expected_cols:
        assert col in outages_df.columns, f"Column '{col}' is missing from merged_outages.csv"
        
    assert len(outages_df) > 0, "merged_outages.csv is empty!"

def test_outages_data_integrity(outages_df):
    """
    Verify datetimes are parsed correctly, durations are non-negative, and daily_restoring is boolean.
    """
    # Check start and stop datetimes are valid
    starts = pd.to_datetime(outages_df['start_datetime'], errors='coerce')
    stops = pd.to_datetime(outages_df['stop_datetime'], errors='coerce')
    
    assert starts.isna().sum() == 0, "Some start_datetime values could not be parsed!"
    assert stops.isna().sum() == 0, "Some stop_datetime values could not be parsed!"
    
    # Check duration is non-negative
    assert (outages_df['duration_hours'] >= 0).all(), "Found outages with negative duration!"
    
    # Check durations match calculated duration
    calculated_durations = (stops - starts) / pd.Timedelta(hours=1)
    # Check difference is negligible (within rounding/epsilon)
    assert np.allclose(outages_df['duration_hours'], calculated_durations, atol=1e-3), "Outage durations do not match start/stop times!"

def test_outages_deduplication(outages_df):
    """
    Verify that there are no duplicate outages in the final dataset.
    Duplicates are defined by the same asset name, start time, and stop time.
    """
    duplicates = outages_df.duplicated(subset=['assets_concerned', 'start_datetime', 'stop_datetime'])
    assert duplicates.sum() == 0, f"Found {duplicates.sum()} duplicate outage records!"

def test_outages_types(outages_df):
    """
    Verify data types of specific columns.
    """
    assert outages_df['daily_restoring'].dtype == bool or outages_df['daily_restoring'].isin([True, False]).all()
    assert pd.api.types.is_numeric_dtype(outages_df['voltage_kv'])
    assert pd.api.types.is_numeric_dtype(outages_df['duration_hours'])

@pytest.mark.parametrize("filename, expected_cols", [
    ("demand_and_forecast.csv", ["datetime", "total_load_mw", "forecast_total_load_mw", "bidding_zone"]),
    ("generation_by_source.csv", ["datetime", "actual_generation_mw", "primary_source"]),
    ("generation_coverage.csv", ["year", "region", "source", "coverage_gwh"]),
    ("imports_exports.csv", ["datetime", "country", "import_mw", "export_mw", "scheduled_foreign_exchange_mw"]),
    ("available_capacity.csv", ["datetime", "macroarea", "plant_type", "prevailing_fuel", "available_capacity_mw"]),
    ("forecast_capacity.csv", ["datetime", "macroarea", "macrouser", "plant_type", "prevailing_fuel", "forecast_capacity_mw"]),
    ("regional_consumption_by_sector.csv", ["year", "region", "province", "sector", "consumption_gwh"]),
    ("renewable_connection_requests.csv", ["region", "plant_type", "source", "connection_status", "power_mw", "num_requests"])
])
def test_supplementary_datasets(filename, expected_cols):
    """
    Verify schema and basic properties of supplementary datasets.
    """
    path = os.path.join(PROCESSED_DIR, filename)
    assert os.path.exists(path), f"Supplementary file {filename} does not exist!"
    df = pd.read_csv(path)
    
    assert len(df) > 0, f"Dataset {filename} is empty!"
    for col in expected_cols:
        assert col in df.columns, f"Column '{col}' is missing in {filename}"
        
    # Check that Date/year columns do not contain footer metadata strings
    if "datetime" in df.columns:
        times = pd.to_datetime(df["datetime"], errors="coerce")
        assert times.isna().sum() == 0, f"Found unparseable dates in {filename}"
    elif "year" in df.columns:
        years = pd.to_numeric(df["year"], errors="coerce")
        assert years.isna().sum() == 0, f"Found unparseable years in {filename}"
