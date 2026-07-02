import os
import glob
import re
import logging
import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Base directory setup
BASE_DIR = r"C:\Users\kk928\.gemini\antigravity\scratch\terna-ai-platform"
RAW_OUTAGES_DIR = os.path.join(BASE_DIR, "data", "raw", "outages")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")

def convert_excel_date(date_series, hour_series):
    """
    Convert Excel serial date and decimal/string hour columns into datetime series.
    Supports Excel serial floats, pandas datetime objects, strings, and integer days.
    """
    datetimes = []
    excel_origin = pd.Timestamp("1899-12-30")
    
    # Fill NA to prevent type errors
    date_series = date_series.fillna(np.nan)
    hour_series = hour_series.fillna(0.0)
    
    for d_val, h_val in zip(date_series, hour_series):
        if pd.isna(d_val):
            datetimes.append(pd.NaT)
            continue
            
        # Parse date component
        dt_part = None
        if isinstance(d_val, (int, float)) or (isinstance(d_val, str) and d_val.replace('.', '', 1).isdigit()):
            # Excel serial date
            dt_part = excel_origin + pd.to_timedelta(float(d_val), unit='D')
        else:
            # Try parsing as standard datetime
            try:
                dt_part = pd.to_datetime(d_val)
            except Exception:
                datetimes.append(pd.NaT)
                continue
        
        # Parse hour component
        td_part = pd.to_timedelta(0)
        if isinstance(h_val, (int, float)):
            # Excel decimal hour (fraction of a day)
            td_part = pd.to_timedelta(float(h_val), unit='D')
        elif isinstance(h_val, str):
            h_val = h_val.strip()
            # If formatted like HH:MM:SS or HH:MM
            if ':' in h_val:
                try:
                    parts = h_val.split(':')
                    if len(parts) == 2:
                        td_part = pd.to_timedelta(int(parts[0]), unit='h') + pd.to_timedelta(int(parts[1]), unit='m')
                    elif len(parts) == 3:
                        td_part = pd.to_timedelta(int(parts[0]), unit='h') + pd.to_timedelta(int(parts[1]), unit='m') + pd.to_timedelta(float(parts[2]), unit='s')
                except Exception:
                    pass
            else:
                # Try reading as numeric float if possible
                try:
                    td_part = pd.to_timedelta(float(h_val), unit='D')
                except Exception:
                    pass
        elif isinstance(h_val, pd.Timedelta):
            td_part = h_val
            
        # Combine date and hour
        combined = dt_part + td_part
        # Round to nearest minute for clean data
        combined = combined.round('min')
        datetimes.append(combined)
        
    return pd.Series(datetimes)

def parse_voltage_kv(voltage_series):
    """
    Extract numeric kV value from strings like '132 kV', '380kV', etc.
    """
    numeric_kv = []
    for val in voltage_series:
        if pd.isna(val):
            numeric_kv.append(np.nan)
            continue
        val_str = str(val).strip()
        match = re.search(r'(\d+)', val_str)
        if match:
            numeric_kv.append(float(match.group(1)))
        else:
            numeric_kv.append(np.nan)
    return pd.Series(numeric_kv)

def process_outages():
    logger.info("Starting processing of weekly outages...")
    files = glob.glob(os.path.join(RAW_OUTAGES_DIR, "*.xlsx"))
    logger.info(f"Found {len(files)} weekly outage Excel files.")
    
    all_dfs = []
    
    for f in sorted(files):
        filename = os.path.basename(f)
        logger.info(f"Reading file: {filename}")
        try:
            # Header is on Row 4 (0-indexed)
            df = pd.read_excel(f, sheet_name="Report_ind_pub", header=4)
            logger.info(f"Loaded {len(df)} rows from {filename}")
            
            # Map columns
            col_mapping = {
                'assets concerned': 'assets_concerned',
                'type of asset': 'asset_type',
                'kV': 'voltage_kv_raw',
                'start date': 'start_date_raw',
                'start hour': 'start_hour_raw',
                'stop date': 'stop_date_raw',
                'stop hour': 'stop_hour_raw',
                'daily restoring Y/N': 'daily_restoring_raw',
                'reasons': 'reason'
            }
            
            # Filter actual columns present
            df = df.rename(columns=col_mapping)
            
            # Retain only our mapped columns (remove any extra columns)
            cols_to_keep = [c for c in col_mapping.values() if c in df.columns]
            df = df[cols_to_keep]
            
            # Record source file name for traceability
            df['source_file'] = filename
            all_dfs.append(df)
            
        except Exception as e:
            logger.error(f"Error processing file {filename}: {e}", exc_info=True)
            
    if not all_dfs:
        logger.error("No outage files were successfully parsed!")
        return
        
    # Merge all outages
    merged_df = pd.concat(all_dfs, ignore_index=True)
    logger.info(f"Total merged raw rows: {len(merged_df)}")
    
    # 1. Parse timestamps
    logger.info("Parsing start and stop datetimes...")
    merged_df['start_datetime'] = convert_excel_date(merged_df['start_date_raw'], merged_df['start_hour_raw'])
    merged_df['stop_datetime'] = convert_excel_date(merged_df['stop_date_raw'], merged_df['stop_hour_raw'])
    
    # 2. Parse numeric voltage kV
    logger.info("Standardizing kV values...")
    merged_df['voltage_kv'] = parse_voltage_kv(merged_df['voltage_kv_raw'])
    
    # 3. Standardize daily restoring flag to boolean
    logger.info("Cleaning daily restoring flags...")
    merged_df['daily_restoring'] = merged_df['daily_restoring_raw'].astype(str).str.strip().str.upper().map({'Y': True, 'N': False}).fillna(False)
    
    # 4. Clean strings
    merged_df['assets_concerned'] = merged_df['assets_concerned'].astype(str).str.strip()
    merged_df['asset_type'] = merged_df['asset_type'].astype(str).str.strip().str.upper()
    merged_df['reason'] = merged_df['reason'].astype(str).str.strip()
    
    # 5. Drop raw columns we no longer need for modeling
    merged_df = merged_df.drop(columns=['start_date_raw', 'start_hour_raw', 'stop_date_raw', 'stop_hour_raw'])
    
    # Remove records that failed timestamp conversion (if any)
    initial_cnt = len(merged_df)
    merged_df = merged_df.dropna(subset=['start_datetime', 'stop_datetime'])
    drop_cnt = initial_cnt - len(merged_df)
    if drop_cnt > 0:
        logger.warning(f"Dropped {drop_cnt} rows due to invalid/missing datetimes.")
        
    # 6. Deduplicate
    # Over weeks, identical outages are reported in multiple files.
    # Deduplicate by assets concerned, start time, and stop time. We keep the first report.
    logger.info("Deduplicating outages...")
    before_dedup = len(merged_df)
    
    # Sort by start_datetime so we keep the earliest reporting file
    merged_df = merged_df.sort_values(by=['start_datetime', 'assets_concerned'])
    merged_df = merged_df.drop_duplicates(subset=['assets_concerned', 'start_datetime', 'stop_datetime'], keep='first')
    
    after_dedup = len(merged_df)
    logger.info(f"Deduplicated from {before_dedup} to {after_dedup} rows (removed {before_dedup - after_dedup} duplicates).")
    
    # 7. Calculate outage duration in hours
    logger.info("Calculating outage duration in hours...")
    merged_df['duration_hours'] = (merged_df['stop_datetime'] - merged_df['start_datetime']) / pd.Timedelta(hours=1)
    
    # Remove outages with negative durations
    neg_durations = merged_df[merged_df['duration_hours'] < 0]
    if len(neg_durations) > 0:
        logger.warning(f"Found {len(neg_durations)} outages with negative durations. Dropping them.")
        merged_df = merged_df[merged_df['duration_hours'] >= 0]
        
    # Clean output path and save
    out_file = os.path.join(PROCESSED_DIR, "merged_outages.csv")
    merged_df.to_csv(out_file, index=False)
    logger.info(f"Saved merged outage dataset to: {out_file}")
    logger.info(f"Final outage dataset shape: {merged_df.shape}")

def process_supplementary_datasets():
    logger.info("Processing supplementary grid operation datasets...")
    
    # 1. Demand and Forecast
    demand_file = os.path.join(BASE_DIR, "data", "raw", "demand", "demand_and_forecast.xlsx")
    if os.path.exists(demand_file):
        logger.info("Processing demand_and_forecast.xlsx...")
        df = pd.read_excel(demand_file)
        df.columns = df.columns.str.strip()
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df = df.dropna(subset=['Date'])
        df = df.rename(columns={
            'Date': 'datetime',
            'Total Load [MW]': 'total_load_mw',
            'Forecast Total Load [MW]': 'forecast_total_load_mw',
            'Bidding Zone': 'bidding_zone'
        })
        df['bidding_zone'] = df['bidding_zone'].str.strip()
        out_path = os.path.join(PROCESSED_DIR, "demand_and_forecast.csv")
        df.to_csv(out_path, index=False)
        logger.info(f"Saved: {out_path} (Shape: {df.shape})")
        
    # 2. Generation by Source
    gen_file = os.path.join(BASE_DIR, "data", "raw", "generation", "generation_by_source.xlsx")
    if os.path.exists(gen_file):
        logger.info("Processing generation_by_source.xlsx...")
        df = pd.read_excel(gen_file)
        df.columns = df.columns.str.strip()
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df = df.dropna(subset=['Date'])
        df = df.rename(columns={
            'Date': 'datetime',
            'Actual Generation': 'actual_generation_mw',
            'Primary Source': 'primary_source'
        })
        df['primary_source'] = df['primary_source'].str.strip()
        out_path = os.path.join(PROCESSED_DIR, "generation_by_source.csv")
        df.to_csv(out_path, index=False)
        logger.info(f"Saved: {out_path} (Shape: {df.shape})")

    # 3. Generation Coverage
    cov_file = os.path.join(BASE_DIR, "data", "raw", "generation", "generation_coverage.xlsx")
    if os.path.exists(cov_file):
        logger.info("Processing generation_coverage.xlsx...")
        df = pd.read_excel(cov_file)
        df.columns = df.columns.str.strip()
        df = df.rename(columns={
            'Anno': 'year',
            'Regione': 'region',
            'Fonte': 'source',
            'Copertura (GWh)': 'coverage_gwh'
        })
        df['year'] = pd.to_numeric(df['year'], errors='coerce')
        df = df.dropna(subset=['year'])
        df['year'] = df['year'].astype(int)
        df['region'] = df['region'].str.strip()
        df['source'] = df['source'].str.strip()
        out_path = os.path.join(PROCESSED_DIR, "generation_coverage.csv")
        df.to_csv(out_path, index=False)
        logger.info(f"Saved: {out_path} (Shape: {df.shape})")

    # 4. Cross-border Imports/Exports
    imports_file = os.path.join(BASE_DIR, "data", "raw", "imports", "imports_exports.xlsx")
    if os.path.exists(imports_file):
        logger.info("Processing imports_exports.xlsx...")
        df = pd.read_excel(imports_file)
        df.columns = df.columns.str.strip()
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df = df.dropna(subset=['Date'])
        df = df.rename(columns={
            'Date': 'datetime',
            'Country': 'country',
            'Import': 'import_mw',
            'Export': 'export_mw',
            'Scheduled Foreign Exchange': 'scheduled_foreign_exchange_mw'
        })
        df['country'] = df['country'].str.strip()
        out_path = os.path.join(PROCESSED_DIR, "imports_exports.csv")
        df.to_csv(out_path, index=False)
        logger.info(f"Saved: {out_path} (Shape: {df.shape})")

    # 5. Available Capacity
    avail_file = os.path.join(BASE_DIR, "data", "raw", "capacity", "available_capacity.xlsx")
    if os.path.exists(avail_file):
        logger.info("Processing available_capacity.xlsx...")
        df = pd.read_excel(avail_file)
        df.columns = df.columns.str.strip()
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df = df.dropna(subset=['Date'])
        df = df.rename(columns={
            'Date': 'datetime',
            'Macroarea': 'macroarea',
            'Tipo Impianto': 'plant_type',
            'Combustibile Prev.': 'prevailing_fuel',
            'Available Capacity [MW]': 'available_capacity_mw'
        })
        df['macroarea'] = df['macroarea'].str.strip()
        df['plant_type'] = df['plant_type'].str.strip()
        df['prevailing_fuel'] = df['prevailing_fuel'].str.strip()
        out_path = os.path.join(PROCESSED_DIR, "available_capacity.csv")
        df.to_csv(out_path, index=False)
        logger.info(f"Saved: {out_path} (Shape: {df.shape})")

    # 6. Forecast Capacity
    fore_file = os.path.join(BASE_DIR, "data", "raw", "capacity", "forecast_capacity.xlsx")
    if os.path.exists(fore_file):
        logger.info("Processing forecast_capacity.xlsx...")
        df = pd.read_excel(fore_file)
        df.columns = df.columns.str.strip()
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df = df.dropna(subset=['Date'])
        df = df.rename(columns={
            'Date': 'datetime',
            'Macroarea': 'macroarea',
            'Macroutente': 'macrouser',
            'Tipo Impianto': 'plant_type',
            'Combustibile Prev.': 'prevailing_fuel',
            'Forecast Capacity [MW]': 'forecast_capacity_mw'
        })
        df['macroarea'] = df['macroarea'].str.strip()
        df['macrouser'] = df['macrouser'].str.strip()
        df['plant_type'] = df['plant_type'].str.strip()
        df['prevailing_fuel'] = df['prevailing_fuel'].str.strip()
        out_path = os.path.join(PROCESSED_DIR, "forecast_capacity.csv")
        df.to_csv(out_path, index=False)
        logger.info(f"Saved: {out_path} (Shape: {df.shape})")

    # 7. Regional Consumption
    cons_file = os.path.join(BASE_DIR, "data", "raw", "consumption", "regional_consumption_by_sector.xlsx")
    if os.path.exists(cons_file):
        logger.info("Processing regional_consumption_by_sector.xlsx...")
        df = pd.read_excel(cons_file)
        df.columns = df.columns.str.strip()
        df = df.rename(columns={
            'Anno': 'year',
            'Regione': 'region',
            'Provincia': 'province',
            'Settore': 'sector',
            'Consumo (GWh)': 'consumption_gwh'
        })
        df['year'] = pd.to_numeric(df['year'], errors='coerce')
        df = df.dropna(subset=['year'])
        df['year'] = df['year'].astype(int)
        df['region'] = df['region'].str.strip()
        df['province'] = df['province'].str.strip()
        df['sector'] = df['sector'].str.strip()
        out_path = os.path.join(PROCESSED_DIR, "regional_consumption_by_sector.csv")
        df.to_csv(out_path, index=False)
        logger.info(f"Saved: {out_path} (Shape: {df.shape})")

    # 8. Renewable Connection Requests
    req_file = os.path.join(BASE_DIR, "data", "raw", "requests", "renewable_connection_requests.xlsx")
    if os.path.exists(req_file):
        logger.info("Processing renewable_connection_requests.xlsx...")
        df = pd.read_excel(req_file)
        df.columns = df.columns.str.strip()
        df = df.rename(columns={
            'Regione': 'region',
            'Tipo Impianto': 'plant_type',
            'Fonte': 'source',
            'Stato Connessione': 'connection_status',
            'Potenza (MW)': 'power_mw',
            'Numero Pratiche': 'num_requests'
        })
        df['power_mw'] = pd.to_numeric(df['power_mw'], errors='coerce')
        df = df.dropna(subset=['power_mw'])
        df['num_requests'] = pd.to_numeric(df['num_requests'], errors='coerce').fillna(0).astype(int)
        df['region'] = df['region'].str.strip()
        df['plant_type'] = df['plant_type'].str.strip()
        df['source'] = df['source'].str.strip()
        df['connection_status'] = df['connection_status'].str.strip()
        out_path = os.path.join(PROCESSED_DIR, "renewable_connection_requests.csv")
        df.to_csv(out_path, index=False)
        logger.info(f"Saved: {out_path} (Shape: {df.shape})")

if __name__ == "__main__":
    # Create processed dir if not exists
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    process_outages()
    process_supplementary_datasets()
    logger.info("ETL pipeline execution complete!")
