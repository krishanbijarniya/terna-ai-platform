import os
import logging
import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

BASE_DIR = r"C:\Users\kk928\.gemini\antigravity\scratch\terna-ai-platform"
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")

def get_easter_date(year):
    """
    Returns Easter Sunday as a datetime.date object for years 2024 to 2026.
    Simple lookup for project scope.
    """
    easter_dates = {
        2024: pd.Timestamp("2024-03-31").date(),
        2025: pd.Timestamp("2025-04-20").date(),
        2026: pd.Timestamp("2026-04-05").date()
    }
    return easter_dates.get(year, None)

def is_italian_holiday(dt):
    """
    Check if a given datetime is a national holiday in Italy.
    """
    date = dt.date()
    year = date.year
    
    # Fixed Italian holidays
    fixed_holidays = {
        (1, 1),   # Capodanno
        (1, 6),   # Epifania
        (4, 25),  # Liberazione
        (5, 1),   # Festa del Lavoro
        (6, 2),   # Festa della Repubblica
        (8, 15),  # Assunzione (Ferragosto)
        (11, 1),  # Tutti i Santi
        (12, 8),  # Immacolata Concezione
        (12, 25), # Natale
        (12, 26)  # Santo Stefano
    }
    
    if (date.month, date.day) in fixed_holidays:
        return True
        
    # Easter and Easter Monday (Pasquetta)
    easter = get_easter_date(year)
    if easter:
        pasquetta = easter + pd.Timedelta(days=1)
        if date == easter or date == pasquetta:
            return True
            
    return False

def map_voltage_category(kv):
    if pd.isna(kv):
        return "Unknown"
    elif kv < 100:
        return "LV/MV"
    elif kv <= 150:
        return "132-150 kV"
    elif kv <= 220:
        return "220 kV"
    else:
        return "400 kV"

def map_maintenance_category(reason):
    reason_str = str(reason).lower()
    if "controlli tecnici" in reason_str or "controllo" in reason_str:
        return "Technical Controls"
    elif "sostituzione" in reason_str or "sostituzioni" in reason_str:
        return "Replacement"
    elif "interferenza" in reason_str:
        return "Interference"
    elif "sviluppo" in reason_str or "rinnovo" in reason_str:
        return "Development/Rinnovo"
    else:
        return "Other"

def run_feature_engineering():
    logger.info("Starting Phase 4: Feature Engineering...")
    
    outages_path = os.path.join(PROCESSED_DIR, "merged_outages.csv")
    if not os.path.exists(outages_path):
        logger.error(f"Processed outages file not found: {outages_path}")
        return
        
    df = pd.read_csv(outages_path)
    df['start_datetime'] = pd.to_datetime(df['start_datetime'])
    df['stop_datetime'] = pd.to_datetime(df['stop_datetime'])
    
    # IMPORTANT: Chronological sorting is required for correct historical/rolling feature calculation
    df = df.sort_values(by='start_datetime').reset_index(drop=True)
    
    # 1. Temporal & Calendar Features
    logger.info("Engineering calendar and holiday features...")
    df['start_month'] = df['start_datetime'].dt.month
    df['start_quarter'] = df['start_datetime'].dt.quarter
    df['start_week'] = df['start_datetime'].dt.isocalendar().week.astype(int)
    df['start_day'] = df['start_datetime'].dt.day
    df['start_dayofweek'] = df['start_datetime'].dt.dayofweek
    df['is_weekend'] = df['start_dayofweek'].isin([5, 6])
    
    # Seasons mapping
    season_map = {
        12: 'Winter', 1: 'Winter', 2: 'Winter',
        3: 'Spring', 4: 'Spring', 5: 'Spring',
        6: 'Summer', 7: 'Summer', 8: 'Summer',
        9: 'Autumn', 10: 'Autumn', 11: 'Autumn'
    }
    df['season'] = df['start_month'].map(season_map)
    
    # Holiday mapping
    df['is_holiday'] = df['start_datetime'].apply(is_italian_holiday)
    
    # 2. Categorical Encodings & Groupings
    logger.info("Grouping categorical variables...")
    df['voltage_category'] = df['voltage_kv'].apply(map_voltage_category)
    df['maintenance_category'] = df['reason'].apply(map_maintenance_category)
    
    # 3. Historical Asset Maintenance Features
    logger.info("Calculating asset historical maintenance and rolling statistics...")
    
    # Cumulative previous outages count per asset (shifted/exclusive of current event)
    # cumcount() gives 0-based index of occurrence, which exactly equals the count of previous outages!
    df['prev_outages_count'] = df.groupby('assets_concerned').cumcount()
    
    # Shift duration to avoid target leakage (crucial: we must not include the current duration in the rolling features)
    df['shifted_duration'] = df.groupby('assets_concerned')['duration_hours'].shift(1)
    
    # Rolling mean duration of last 3 outages
    # min_periods=1 ensures that if an asset has only 1 previous outage, we use it rather than returning NaN
    df['rolling_mean_duration_3'] = df.groupby('assets_concerned')['shifted_duration'].transform(
        lambda x: x.rolling(window=3, min_periods=1).mean()
    ).fillna(0.0)
    
    # Rolling sum of downtime of last 3 outages
    df['rolling_downtime_3'] = df.groupby('assets_concerned')['shifted_duration'].transform(
        lambda x: x.rolling(window=3, min_periods=1).sum()
    ).fillna(0.0)
    
    # Drop the temporary shifted column
    df = df.drop(columns=['shifted_duration'])
    
    # Outage frequency index (cumulative count / months since first outage)
    first_outage_time = df.groupby('assets_concerned')['start_datetime'].transform('min')
    months_elapsed = (df['start_datetime'] - first_outage_time) / pd.Timedelta(days=30)
    months_elapsed = np.clip(months_elapsed, 1.0, None)  # min 1 month to avoid division by zero
    
    # Count of outages up to this event (prev_outages_count + 1) divided by elapsed months
    df['frequency_index'] = (df['prev_outages_count'] + 1) / months_elapsed
    
    # 4. Grid Risk Score
    logger.info("Computing composite grid risk scores...")
    
    # Normalize voltage (relative to max backbone voltage 400kV)
    vol_norm = df['voltage_kv'].fillna(132.0) / 400.0
    
    # Normalize previous outage count (capped at 10 outages)
    cnt_norm = np.clip(df['prev_outages_count'] / 10.0, 0.0, 1.0)
    
    # Normalize rolling duration (capped at 168 hours = 1 week)
    dur_norm = np.clip(df['rolling_mean_duration_3'] / 168.0, 0.0, 1.0)
    
    # Composite risk score calculation (0.0 to 1.0)
    df['risk_score'] = 0.4 * vol_norm + 0.3 * cnt_norm + 0.3 * dur_norm
    
    # Save the engineered dataset
    out_path = os.path.join(PROCESSED_DIR, "engineered_outages.csv")
    df.to_csv(out_path, index=False)
    
    logger.info(f"Feature engineering complete. Saved dataset to: {out_path}")
    logger.info(f"Engineered dataset shape: {df.shape}")

if __name__ == "__main__":
    run_feature_engineering()
