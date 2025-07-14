#!/usr/bin/env python3
"""
Master Report Automation Script
Converts the Master report.ipynb to a Python script for automated execution.
Stops at the BreakBreakBreak cell (cell 11).
"""

import os
import sys
import logging
from datetime import datetime, timedelta
import warnings

# Data manipulation and analysis
import pandas as pd
import numpy as np

# API connections
from simple_salesforce import Salesforce
import gspread
import requests
from io import StringIO

# Visualization
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px

# Google services
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

warnings.filterwarnings('ignore')

def setup_logging():
    """Setup logging for the master report."""
    log_filename = f'master_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_filename)
        ]
    )
    return logging.getLogger(__name__)

def get_quarter_info(current_date):
    """Get quarter information based on fiscal calendar."""
    fiscal_year = current_date.year if current_date.month >= 2 else current_date.year - 1

    if datetime(current_date.year, 2, 1) <= current_date < datetime(current_date.year, 5, 1):
        quarter = f"{fiscal_year + 1}-Q1"
        quarter_start_date = datetime(current_date.year, 2, 1)
        quarter_end_date = datetime(current_date.year, 4, 30)
    elif datetime(current_date.year, 5, 1) <= current_date < datetime(current_date.year, 8, 1):
        quarter = f"{fiscal_year + 1}-Q2"
        quarter_start_date = datetime(current_date.year, 5, 1)
        quarter_end_date = datetime(current_date.year, 7, 31)
    elif datetime(current_date.year, 8, 1) <= current_date < datetime(current_date.year, 11, 1):
        quarter = f"{fiscal_year + 1}-Q3"
        quarter_start_date = datetime(current_date.year, 8, 1)
        quarter_end_date = datetime(current_date.year, 10, 31)
    else:
        # Q4: Ends on Jan 31 of the following year
        if current_date.month == 1:
            quarter = f"{fiscal_year + 1}-Q4"
            quarter_start_date = datetime(current_date.year - 1, 11, 1)
            quarter_end_date = datetime(current_date.year, 1, 31)
        else:
            quarter = f"{fiscal_year + 1}-Q4"
            quarter_start_date = datetime(current_date.year, 11, 1)
            quarter_end_date = datetime(current_date.year + 1, 1, 31)
    return quarter, quarter_start_date, quarter_end_date

def compute_week_boundaries_for_quarter(quarter_start_date, quarter_end_date):
    """Compute week boundaries for a quarter."""
    total_days = (quarter_end_date - quarter_start_date).days + 1
    start_wd = quarter_start_date.weekday()  # Monday=0,...,Sunday=6

    # Fixed 11 weeks in the middle: 11*7 = 77 days
    leftover = total_days - 77
    solutions = []
    half = leftover / 2.0

    for W1 in range(2, 13):
        W13 = leftover - W1
        if 4 <= W13 <= 12 and ((start_wd + W1) % 7 == 4):
            solutions.append((W1, W13))

    if not solutions:
        raise ValueError(f"Could not find a valid week distribution for quarter starting {quarter_start_date}.")

    solutions.sort(key=lambda x: abs(x[0] - half))
    best_W1, best_W13 = solutions[0]

    week_lengths = [best_W1] + [7]*11 + [best_W13]
    week_boundaries = []
    start_of_week = quarter_start_date
    for length in week_lengths:
        end_of_week = start_of_week + timedelta(days=length - 1)
        week_boundaries.append((start_of_week, end_of_week))
        start_of_week = end_of_week + timedelta(days=1)
    return week_boundaries

def breakdown_date(date_val):
    """Break down a single date into quarter, week, etc."""
    if pd.isnull(date_val):
        return pd.Series({
            'Quarter': None,
            'Week_of_Quarter': None,
            'Month': None,
            'Day_of_Week': None,
            'Day_Name': None
        })
    # Use custom logic for quarter and week
    quarter, q_start, q_end = get_quarter_info(date_val)
    week_boundaries = compute_week_boundaries_for_quarter(q_start, q_end)
    week_of_quarter = None
    for i, (wstart, wend) in enumerate(week_boundaries):
        if wstart <= date_val <= wend:
            week_of_quarter = i + 1
            break
    return pd.Series({
        'Quarter': quarter,
        'Week_of_Quarter': week_of_quarter,
        'Month': date_val.month,
        'Day_of_Week': date_val.weekday(),       # Monday=0, Sunday=6
        'Day_Name': date_val.strftime("%A")
    })

def get_last_completed_quarters(n, today=None):
    """Returns a list of the last n completed quarters."""
    if today is None:
        today = datetime.today()
    current_quarter, q_start, _ = get_quarter_info(today)
    last_date = q_start - timedelta(days=1)
    quarters = []
    for _ in range(n):
        q, q_start, _ = get_quarter_info(last_date)
        quarters.append(q)
        last_date = q_start - timedelta(days=1)
    return quarters[::-1]

def compute_day_of_quarter(date_val):
    """Compute day of quarter information."""
    if pd.isnull(date_val):
        return pd.Series({'Day_of_Quarter': np.nan, 
                          'Total_Days_in_Quarter': np.nan, 
                          'Pct_Day': np.nan})
    quarter, q_start, q_end = get_quarter_info(date_val)
    day_of_quarter = (date_val - q_start).days + 1
    total_days = (q_end - q_start).days + 1
    pct_day = day_of_quarter / total_days * 100
    return pd.Series({'Day_of_Quarter': day_of_quarter, 
                      'Total_Days_in_Quarter': total_days, 
                      'Pct_Day': pct_day})

def compute_cumulative_raw(df_in):
    """Compute cumulative raw data."""
    grouped = df_in.groupby(['Quarter', 'Pct_Day_Bin'])['ARR Change'].sum().reset_index()
    pivot = grouped.pivot(index='Pct_Day_Bin', columns='Quarter', values='ARR Change').fillna(0)
    full_index = pd.Index(range(0, 101), name='Pct_Day_Bin')
    pivot = pivot.reindex(full_index, fill_value=0)
    cum = pivot.cumsum()
    return cum

def compute_cumulative_raw_pipegen(df_in):
    """Compute cumulative raw data for pipegen."""
    grouped = df_in.groupby(['Pipegen_Quarter', 'Pipegen_Pct_Day_Bin'])['ARR Change'].sum().reset_index()
    pivot = grouped.pivot(index='Pipegen_Pct_Day_Bin', columns='Pipegen_Quarter', values='ARR Change').fillna(0)
    full_index = pd.Index(range(0, 101), name='Pipegen_Pct_Day_Bin')
    pivot = pivot.reindex(full_index, fill_value=0)
    cum = pivot.cumsum()
    return cum

def connect_to_salesforce():
    """Connect to Salesforce using environment variables."""
    logger = logging.getLogger(__name__)
    
    # Get credentials from environment variables
    sf_username = os.getenv('SF_USERNAME')
    sf_password = os.getenv('SF_PASSWORD')
    sf_security_token = os.getenv('SF_SECURITY_TOKEN')
    
    if not all([sf_username, sf_password, sf_security_token]):
        raise ValueError("Salesforce credentials not found in environment variables")
    
    logger.info("Connecting to Salesforce...")
    sf = Salesforce(username=sf_username, password=sf_password, security_token=sf_security_token)
    return sf

def connect_to_google_sheets():
    """Connect to Google Sheets using service account."""
    logger = logging.getLogger(__name__)
    
    # Get credentials path from environment variable
    credentials_path = os.getenv('GOOGLE_CREDENTIALS_PATH')
    if not credentials_path:
        raise ValueError("Google credentials path not found in environment variables")
    
    logger.info("Connecting to Google Sheets...")
    gc = gspread.service_account(filename=credentials_path)
    spreadsheet_key = os.getenv('GOOGLE_SHEET_KEY', '1A6Q8dvoWwLi26tnQQoEo6ZnekvZBQbpgCyP20iQV6Ug')
    sht1 = gc.open_by_key(spreadsheet_key)
    return gc, sht1

def fetch_salesforce_report(sf):
    """Fetch Salesforce report data."""
    logger = logging.getLogger(__name__)
    
    logger.info("Fetching Salesforce report...")
    
    sf_instance = 'https://envoy.my.salesforce.com/'
    reportId = '00OUO000009IZVD2A4'
    export = '?isdtp=p1&export=1&enc=UTF-8&xf=csv'
    sfUrl = sf_instance + reportId + export
    
    response = requests.get(sfUrl, headers=sf.headers, cookies={'sid': sf.session_id})
    download_report = response.content.decode('utf-8')
    df = pd.read_csv(StringIO(download_report))
    
    logger.info(f"Downloaded {len(df)} records from Salesforce")
    return df

def process_date_columns(df):
    """Process date columns and add breakdown information."""
    logger = logging.getLogger(__name__)
    
    logger.info("Processing date columns...")
    
    # List of date columns to enhance
    date_columns = ['Created Date', 'SQO Date', 'SAO Date', 'Timestamp: Solution Validation', 'Close Date']
    
    # Convert each to datetime
    for col in date_columns:
        df[col] = pd.to_datetime(df[col], errors='coerce')
    
    # Process each date column and merge the breakdown back into df
    for col in date_columns:
        # Apply the breakdown function to the column
        breakdown_df = df[col].apply(breakdown_date)
        # Rename the resulting columns to include the original column name as prefix
        breakdown_df = breakdown_df.rename(columns={
            'Quarter': f'{col}_Quarter',
            'Week_of_Quarter': f'{col}_Week_of_Quarter',
            'Month': f'{col}_Month',
            'Day_of_Week': f'{col}_Day_of_Week',
            'Day_Name': f'{col}_Day_Name'
        })
        # Concatenate the breakdown info back into the main DataFrame
        df = pd.concat([df, breakdown_df], axis=1)
    
    return df

def update_google_sheet(sht1, df, worksheet_name="Data"):
    """Update Google Sheet with data."""
    logger = logging.getLogger(__name__)
    
    logger.info(f"Updating Google Sheet worksheet: {worksheet_name}")
    
    # Get worksheet
    worksheet = sht1.worksheet(worksheet_name)
    
    # Clear existing data
    worksheet.clear()
    
    # Copy DataFrame
    data_to_update = df.copy()
    
    # Ensure headers are strings
    data_to_update.columns = data_to_update.columns.astype(str)
    
    # Trim whitespace from strings
    data_to_update = data_to_update.map(lambda x: x.strip() if isinstance(x, str) else x)
    
    # Turn any "only-spaces" cell into empty string
    data_to_update = data_to_update.replace(r'^\s*$', '', regex=True)
    
    # Convert Timestamps to YYYY-MM-DD
    data_to_update = data_to_update.map(
        lambda x: x.strftime('%Y-%m-%d') if isinstance(x, pd.Timestamp) else x
    )
    
    # Replace NaN, ±inf with empty string
    data_to_update = data_to_update.fillna('').replace([np.inf, -np.inf], '')
    
    # Build the list of lists, mapping '' → None so Google Sheets sees true blanks
    headers = data_to_update.columns.tolist()
    rows = data_to_update.values.tolist()
    values = [headers]
    for row in rows:
        clean_row = [None if (isinstance(cell, str) and cell == '') else cell for cell in row]
        values.append(clean_row)
    
    # Update the sheet
    result = worksheet.update(values, 'A1')
    logger.info(f"Updated {result['updatedCells']} cells in {worksheet_name}")
    
    return result

def create_sql_pivot(df):
    """Create SQL pivot table."""
    logger = logging.getLogger(__name__)
    
    logger.info("Creating SQL pivot table...")
    
    # Get the last 5 quarters (4 completed + active)
    today = datetime.today()
    active_quarter = get_quarter_info(today)[0]
    last_4 = get_last_completed_quarters(4, today)
    quarters_5 = last_4 + [active_quarter]
    
    # Slim down DataFrame
    df_pivot = df[[
        'Created Date_Quarter',
        'Source',
        'Segment - historical',
        'Bookings Type'
    ]].copy()
    
    # Map any non-standard source into "Other"
    allowed_src = ['AE','BDR','Channel','Marketing','Success']
    df_pivot['Source'] = df_pivot['Source'].where(df_pivot['Source'].isin(allowed_src), 'Other')
    
    # Filter to exactly the segments & booking types you want
    allowed_segs = ['Enterprise','Mid Market','SMB']
    allowed_types = ['Expansion','New Business']
    df_pivot = df_pivot[
        df_pivot['Segment - historical'].isin(allowed_segs) &
        df_pivot['Bookings Type'].isin(allowed_types)
    ]
    
    # Build static row index
    rows = []
    for src in ['AE','BDR','Channel','Marketing']:
        for seg in allowed_segs:
            for bt in allowed_types:
                rows.append((src, seg, bt))
            rows.append((src, seg, f'{seg} Total'))
        rows.append((src, '', f'{src} Total'))
    
    rows.append(('Other','','Other Total'))
    
    for seg in allowed_segs:
        for bt in allowed_types:
            rows.append(('Success', seg, bt))
        rows.append(('Success', seg, f'{seg} Total'))
    rows.append(('Success','','Success Total'))
    
    rows.append(('','','Grand Total'))
    
    index = pd.MultiIndex.from_tuples(
        rows,
        names=['Source','Segment - historical','Bookings Type']
    )
    
    # Initialize pivot shell
    pivot = pd.DataFrame(
        0,
        index=index,
        columns=quarters_5 + ['Grand Total']
    )
    
    # Fill the detail counts
    grp = df_pivot.groupby(
        ['Source','Segment - historical','Bookings Type','Created Date_Quarter']
    ).size()
    
    for (src, seg, bt, q), cnt in grp.items():
        if (src,seg,bt) in pivot.index and q in quarters_5:
            pivot.at[(src,seg,bt), q] = cnt
    
    # Segment subtotals
    grp_seg = df_pivot.groupby(
        ['Source','Segment - historical','Created Date_Quarter']
    ).size()
    
    for (src, seg, q), cnt in grp_seg.items():
        row = (src, seg, f'{seg} Total')
        if row in pivot.index and q in quarters_5:
            pivot.at[row, q] = cnt
    
    # Source subtotals
    grp_src = df_pivot.groupby(['Source','Created Date_Quarter']).size()
    for (src, q), cnt in grp_src.items():
        row = (src, '', f'{src} Total')
        if row in pivot.index and q in quarters_5:
            pivot.at[row, q] = cnt
    
    # Grand total per quarter
    grp_all = df_pivot.groupby('Created Date_Quarter').size()
    for q, cnt in grp_all.items():
        if q in quarters_5:
            pivot.at[('', '', 'Grand Total'), q] = cnt
    
    # Final grand total column
    pivot['Grand Total'] = pivot[quarters_5].sum(axis=1)
    
    # Reset index for export/display
    pivot_display = pivot.reset_index()
    
    return pivot_display

def create_sao_pivot(df):
    """Create SAO pivot table."""
    logger = logging.getLogger(__name__)
    
    logger.info("Creating SAO pivot table...")
    
    # Get quarters
    today = datetime.today()
    active_quarter = get_quarter_info(today)[0]
    last_4 = get_last_completed_quarters(4, today)
    quarters_5 = last_4 + [active_quarter]
    
    # Slim DataFrame for SAO pivot
    df_sao = df[[
        'SAO Date',
        'SAO Date_Quarter',
        'Source',
        'Segment - historical',
        'Bookings Type'
    ]].copy()
    
    # Only keep rows that actually have an SAO date
    df_sao = df_sao[df_sao['SAO Date'].notna()]
    
    # Map "Other" sources
    allowed_src = ['AE','BDR','Channel','Marketing','Success']
    df_sao['Source'] = df_sao['Source'].where(df_sao['Source'].isin(allowed_src), 'Other')
    
    # Filter segments & types
    allowed_segs = ['Enterprise','Mid Market','SMB']
    allowed_types = ['Expansion','New Business']
    df_sao = df_sao[
        df_sao['Segment - historical'].isin(allowed_segs) &
        df_sao['Bookings Type'].isin(allowed_types)
    ]
    
    # Static row layout (same as before)
    rows = []
    for src in ['AE','BDR','Channel','Marketing']:
        for seg in allowed_segs:
            for bt in allowed_types:
                rows.append((src, seg, bt))
            rows.append((src, seg, f'{seg} Total'))
        rows.append((src, '', f'{src} Total'))
    
    rows.append(('Other','','Other Total'))
    
    for seg in allowed_segs:
        for bt in allowed_types:
            rows.append(('Success', seg, bt))
        rows.append(('Success', seg, f'{seg} Total'))
    rows.append(('Success','','Success Total'))
    
    rows.append(('','','Grand Total'))
    
    index = pd.MultiIndex.from_tuples(
        rows,
        names=['Source','Segment - historical','Bookings Type']
    )
    
    # Initialize pivot shell
    pivot_sao = pd.DataFrame(
        0,
        index=index,
        columns=quarters_5 + ['Grand Total']
    )
    
    # Detail counts by SAO Date Quarter
    grp = df_sao.groupby(
        ['Source','Segment - historical','Bookings Type','SAO Date_Quarter']
    ).size()
    
    for (src, seg, bt, q), cnt in grp.items():
        if (src,seg,bt) in pivot_sao.index and q in quarters_5:
            pivot_sao.at[(src,seg,bt), q] = cnt
    
    # Segment subtotals
    grp_seg = df_sao.groupby(
        ['Source','Segment - historical','SAO Date_Quarter']
    ).size()
    
    for (src, seg, q), cnt in grp_seg.items():
        row = (src, seg, f'{seg} Total')
        if row in pivot_sao.index and q in quarters_5:
            pivot_sao.at[row, q] = cnt
    
    # Source subtotals
    grp_src = df_sao.groupby(['Source','SAO Date_Quarter']).size()
    
    for (src, q), cnt in grp_src.items():
        row = (src, '', f'{src} Total')
        if row in pivot_sao.index and q in quarters_5:
            pivot_sao.at[row, q] = cnt
    
    # Grand total per quarter
    grp_all = df_sao.groupby('SAO Date_Quarter').size()
    for q, cnt in grp_all.items():
        if q in quarters_5:
            pivot_sao.at[('', '', 'Grand Total'), q] = cnt
    
    # Final grand total column
    pivot_sao['Grand Total'] = pivot_sao[quarters_5].sum(axis=1)
    
    # Reset for export
    pivot_sao_display = pivot_sao.reset_index()
    
    return pivot_sao_display

def create_pipegen_pivot(df):
    """Create Pipegen pivot table."""
    logger = logging.getLogger(__name__)
    
    logger.info("Creating Pipegen pivot table...")
    
    # Get quarters
    today = datetime.today()
    active_quarter = get_quarter_info(today)[0]
    last_4 = get_last_completed_quarters(4, today)
    quarters_5 = last_4 + [active_quarter]
    
    # Slim DataFrame for SAO pipegen
    df_pip = df[[
        'SAO Date_Quarter',
        'Source',
        'Segment - historical',
        'Bookings Type',
        'ARR Change'
    ]].copy()
    
    # Only keep rows with an SAO
    df_pip = df_pip[df_pip['SAO Date_Quarter'].notna()]
    
    # Map non-standard sources into "Other"
    allowed_src = ['AE','BDR','Channel','Marketing','Success']
    df_pip['Source'] = df_pip['Source'].where(
        df_pip['Source'].isin(allowed_src), 'Other'
    )
    
    # Filter segments & booking types
    allowed_segs = ['Enterprise','Mid Market','SMB']
    allowed_types = ['Expansion','New Business']
    df_pip = df_pip[
        df_pip['Segment - historical'].isin(allowed_segs) &
        df_pip['Bookings Type'].isin(allowed_types)
    ]
    
    # Static row index (same as before)
    rows = []
    for src in ['AE','BDR','Channel','Marketing']:
        for seg in allowed_segs:
            for bt in allowed_types:
                rows.append((src, seg, bt))
            rows.append((src, seg, f'{seg} Total'))
        rows.append((src, '', f'{src} Total'))
    
    rows.append(('Other','','Other Total'))
    
    for seg in allowed_segs:
        for bt in allowed_types:
            rows.append(('Success', seg, bt))
        rows.append(('Success', seg, f'{seg} Total'))
    rows.append(('Success','','Success Total'))
    
    rows.append(('','','Grand Total'))
    
    index = pd.MultiIndex.from_tuples(
        rows,
        names=['Source','Segment - historical','Bookings Type']
    )
    
    # Initialize pivot shell
    pivot_pip = pd.DataFrame(
        0.0,
        index=index,
        columns=quarters_5 + ['Grand Total']
    )
    
    # Detail: sum ARR Change by SAO Date_Quarter
    grp = df_pip.groupby(
        ['Source','Segment - historical','Bookings Type','SAO Date_Quarter']
    )['ARR Change'].sum()
    
    for (src, seg, bt, q), total in grp.items():
        if (src,seg,bt) in pivot_pip.index and q in quarters_5:
            pivot_pip.at[(src,seg,bt), q] = total
    
    # Segment subtotals (sum ARR Change)
    grp_seg = df_pip.groupby(
        ['Source','Segment - historical','SAO Date_Quarter']
    )['ARR Change'].sum()
    
    for (src, seg, q), total in grp_seg.items():
        row = (src, seg, f'{seg} Total')
        if row in pivot_pip.index and q in quarters_5:
            pivot_pip.at[row, q] = total
    
    # Source subtotals
    grp_src = df_pip.groupby(['Source','SAO Date_Quarter'])['ARR Change'].sum()
    for (src, q), total in grp_src.items():
        row = (src, '', f'{src} Total')
        if row in pivot_pip.index and q in quarters_5:
            pivot_pip.at[row, q] = total
    
    # Grand total per quarter
    grp_all = df_pip.groupby('SAO Date_Quarter')['ARR Change'].sum()
    for q, total in grp_all.items():
        if q in quarters_5:
            pivot_pip.at[('', '', 'Grand Total'), q] = total
    
    # Grand total column
    pivot_pip['Grand Total'] = pivot_pip[quarters_5].sum(axis=1)
    
    # Reset index for export
    pivot_pip_display = pivot_pip.reset_index()
    
    return pivot_pip_display

def update_pivot_sheets(sht1, sql_pivot, sao_pivot, pipegen_pivot):
    """Update pivot sheets in Google Sheets."""
    logger = logging.getLogger(__name__)
    
    logger.info("Updating pivot sheets...")
    
    # Helper function to prepare data for Google Sheets
    def prepare_sheet_data(data):
        data_to_update = data.copy()
        data_to_update.columns = data_to_update.columns.astype(str)
        data_to_update = data_to_update.map(lambda x: x.strip() if isinstance(x, str) else x)
        data_to_update = data_to_update.replace(r'^\s*$', '', regex=True)
        data_to_update = data_to_update.map(
            lambda x: x.strftime('%Y-%m-%d') if hasattr(x, 'strftime') else x
        )
        data_to_update = data_to_update.fillna('').replace([np.inf, -np.inf], '')
        headers = data_to_update.columns.tolist()
        rows = data_to_update.values.tolist()
        values = [headers]
        for row in rows:
            clean = [None if (isinstance(c, str) and c == '') else c for c in row]
            values.append(clean)
        return values
    
    # Update SQLs sheet
    worksheet_sql = sht1.worksheet("SQLs")
    sql_values = prepare_sheet_data(sql_pivot)
    worksheet_sql.update(sql_values, 'B5')
    
    # Update SAOs sheet
    worksheet_sao = sht1.worksheet("SAOs")
    sao_values = prepare_sheet_data(sao_pivot)
    worksheet_sao.update(sao_values, 'B5')
    
    # Update Pipegen sheet
    worksheet_pip = sht1.worksheet("Pipegen")
    pip_values = prepare_sheet_data(pipegen_pivot)
    worksheet_pip.update(pip_values, 'B5')
    
    logger.info("All pivot sheets updated successfully")

def main():
    """Main function to run the master report."""
    logger = setup_logging()
    
    try:
        logger.info("Starting Master Report automation...")
        
        # Connect to services
        sf = connect_to_salesforce()
        gc, sht1 = connect_to_google_sheets()
        
        # Fetch data
        df = fetch_salesforce_report(sf)
        
        # Process data
        df = process_date_columns(df)
        
        # Save master report
        df.to_csv("master_report.csv", index=False)
        logger.info("Master report saved to master_report.csv")
        
        # Update main data sheet
        update_google_sheet(sht1, df, "Data")
        
        # Create pivot tables
        sql_pivot = create_sql_pivot(df)
        sao_pivot = create_sao_pivot(df)
        pipegen_pivot = create_pipegen_pivot(df)
        
        # Update pivot sheets
        update_pivot_sheets(sht1, sql_pivot, sao_pivot, pipegen_pivot)
        
        logger.info("Master report automation completed successfully!")
        
    except Exception as e:
        logger.error(f"Master report automation failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise

if __name__ == "__main__":
    main()