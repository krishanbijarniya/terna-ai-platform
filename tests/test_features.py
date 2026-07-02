import os
import pytest
import pandas as pd
import numpy as np

BASE_DIR = r"C:\Users\kk928\.gemini\antigravity\scratch\terna-ai-platform"
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")

@pytest.fixture
def engineered_df():
    path = os.path.join(PROCESSED_DIR, "engineered_outages.csv")
    assert os.path.exists(path), "engineered_outages.csv does not exist!"
    df = pd.read_csv(path)
    df['start_datetime'] = pd.to_datetime(df['start_datetime'])
    return df

def test_feature_columns(engineered_df):
    """
    Verify that all expected engineered feature columns exist and the dataset has rows.
    """
    expected_features = [
        'start_month', 'start_quarter', 'start_week', 'start_day', 
        'start_dayofweek', 'is_weekend', 'season', 'is_holiday',
        'voltage_category', 'maintenance_category', 'prev_outages_count', 
        'rolling_mean_duration_3', 'rolling_downtime_3', 'frequency_index', 
        'risk_score'
    ]
    for col in expected_features:
        assert col in engineered_df.columns, f"Feature column '{col}' is missing!"
        
    assert len(engineered_df) > 0, "engineered_outages.csv is empty!"

def test_no_data_leakage(engineered_df):
    """
    Ensure no data leakage is present in the rolling features.
    Rolling features for any event must only rely on past events (shifted by 1).
    """
    # Sort chronologically to trace progression
    df = engineered_df.sort_values(by='start_datetime').reset_index(drop=True)
    
    # 1. For the first outage of any asset (prev_outages_count == 0),
    # rolling stats must be 0.0 since there is no history.
    first_events = df[df['prev_outages_count'] == 0]
    assert (first_events['rolling_mean_duration_3'] == 0.0).all(), "First events have non-zero rolling mean!"
    assert (first_events['rolling_downtime_3'] == 0.0).all(), "First events have non-zero rolling downtime!"
    
    # 2. For the second outage of any asset (prev_outages_count == 1),
    # rolling_mean_duration_3 must exactly equal the duration of the first outage of that asset.
    second_events = df[df['prev_outages_count'] == 1]
    
    for idx, row in second_events.iterrows():
        asset = row['assets_concerned']
        # Find the first event for this asset
        first_event = df[(df['assets_concerned'] == asset) & (df['prev_outages_count'] == 0)].iloc[0]
        
        # The rolling mean of second event must equal the duration of the first event
        assert np.allclose(row['rolling_mean_duration_3'], first_event['duration_hours'], atol=1e-3), \
            f"Data leakage detected! Asset {asset} second event rolling mean {row['rolling_mean_duration_3']} does not match first event duration {first_event['duration_hours']}"

def test_risk_score_bounds(engineered_df):
    """
    Verify that the grid risk score is strictly between 0.0 and 1.0.
    """
    assert (engineered_df['risk_score'] >= 0.0).all(), "Risk score contains negative values!"
    assert (engineered_df['risk_score'] <= 1.0).all(), "Risk score exceeds maximum bound of 1.0!"

def test_categorical_mapping(engineered_df):
    """
    Verify correct value categories for mapped fields.
    """
    # Check seasons
    assert engineered_df['season'].isin(['Winter', 'Spring', 'Summer', 'Autumn']).all()
    
    # Check voltage category
    assert engineered_df['voltage_category'].isin(['LV/MV', '132-150 kV', '220 kV', '400 kV', 'Unknown']).all()
    
    # Check maintenance category
    assert engineered_df['maintenance_category'].isin(['Technical Controls', 'Replacement', 'Interference', 'Development/Rinnovo', 'Other']).all()

def test_calendar_logic(engineered_df):
    """
    Validate weekend and day of week alignment.
    """
    # check that is_weekend is True iff day of week is 5 or 6
    expected_weekend = engineered_df['start_dayofweek'].isin([5, 6])
    assert (engineered_df['is_weekend'] == expected_weekend).all()
    
    # check months
    assert (engineered_df['start_month'] >= 1).all() and (engineered_df['start_month'] <= 12).all()
