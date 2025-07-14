"""
Date Enhancement Utilities
Production date logic from Master report.ipynb
"""

from datetime import datetime, timedelta
import numpy as np
import pandas as pd


def get_quarter_info(current_date):
    """Get fiscal quarter, start date, and end date for a given date."""
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
    """Compute week boundaries for a fiscal quarter with optimized week distribution."""
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
    """Break down a single date into quarter, week, month, day info."""
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
    """
    Returns a list of the last n completed quarters (as strings) based on the custom fiscal calendar.
    """
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
    """Compute day of quarter, total days in quarter, and percentage."""
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


def get_fiscal_quarter_string(date_val):
    """Convert date to fiscal quarter string format (FY2024Q1)."""
    if pd.isnull(date_val):
        return None
    quarter, _, _ = get_quarter_info(date_val)
    # Convert from "2024-Q1" format to "FY2024Q1" format for compatibility
    year_part, q_part = quarter.split('-')
    return f"FY{year_part}{q_part}"


def enhance_dataframe_dates(df, date_columns=None):
    """
    Enhance a DataFrame by adding comprehensive date breakdowns for specified date columns.
    
    Parameters:
    - df: DataFrame to enhance
    - date_columns: List of date column names to process. If None, will auto-detect common date columns.
    
    Returns:
    - Enhanced DataFrame with additional date breakdown columns
    """
    if df.empty:
        print("Warning: DataFrame is empty, skipping date enhancement")
        return df
    
    df_enhanced = df.copy()
    
    # Auto-detect date columns if not specified
    if date_columns is None:
        potential_date_columns = [
            'Create Date', 'Created Date', 'Close Date', 'SQO Date', 'SAO Date', 
            'Timestamp: Solution Validation', 'Last Activity Date', 'Next Activity Date'
        ]
        date_columns = [col for col in potential_date_columns if col in df_enhanced.columns]
    
    if not date_columns:
        print("No date columns found for enhancement")
        return df_enhanced
    
    print(f"Enhancing date columns: {date_columns}")
    
    # Convert each to datetime
    for col in date_columns:
        df_enhanced[col] = pd.to_datetime(df_enhanced[col], errors='coerce')
        print(f"Converted {col} to datetime")

    # Process each date column and merge the breakdown back into DataFrame
    for col in date_columns:
        print(f"Processing date breakdowns for: {col}")
        # Apply the breakdown function to the column
        breakdown_df = df_enhanced[col].apply(breakdown_date)
        # Rename the resulting columns to include the original column name as prefix
        breakdown_df = breakdown_df.rename(columns={
            'Quarter': f'{col}_Quarter',
            'Week_of_Quarter': f'{col}_Week_of_Quarter',
            'Month': f'{col}_Month',
            'Day_of_Week': f'{col}_Day_of_Week',
            'Day_Name': f'{col}_Day_Name'
        })
        # Concatenate the breakdown info back into the main DataFrame
        df_enhanced = pd.concat([df_enhanced, breakdown_df], axis=1)
        print(f"Added breakdown columns for {col}: {list(breakdown_df.columns)}")
    
    # Update Fiscal Period - Corrected to use production logic if Close Date exists
    if 'Close Date' in date_columns:
        print("Updating Fiscal Period - Corrected using production quarter logic")
        df_enhanced['Fiscal Period - Corrected'] = df_enhanced['Close Date'].apply(get_fiscal_quarter_string)
        print("Updated Fiscal Period - Corrected column")
    
    # Add day of quarter information for Close Date if it exists
    if 'Close Date' in date_columns:
        print("Adding day of quarter information for Close Date")
        day_quarter_info = df_enhanced['Close Date'].apply(compute_day_of_quarter)
        df_enhanced = pd.concat([df_enhanced, day_quarter_info], axis=1)
        df_enhanced['Pct_Day_Bin'] = df_enhanced['Pct_Day'].round(0)
        df_enhanced['Quarter'] = df_enhanced['Close Date'].apply(lambda x: get_quarter_info(x)[0] if pd.notnull(x) else None)
        print("Added: Day_of_Quarter, Total_Days_in_Quarter, Pct_Day, Pct_Day_Bin, Quarter")
    
    print(f"Enhanced DataFrame now has {len(df_enhanced.columns)} columns")
    return df_enhanced


def compute_cumulative_raw(df_in):
    """Compute cumulative raw data for bookings analysis."""
    grouped = df_in.groupby(['Quarter', 'Pct_Day_Bin'])['ARR Change'].sum().reset_index()
    pivot = grouped.pivot(index='Pct_Day_Bin', columns='Quarter', values='ARR Change').fillna(0)
    full_index = pd.Index(range(0, 101), name='Pct_Day_Bin')
    pivot = pivot.reindex(full_index, fill_value=0)
    cum = pivot.cumsum()
    return cum


def compute_cumulative_raw_pipegen(df_in):
    """Compute cumulative raw data for pipegen analysis."""
    grouped = df_in.groupby(['Pipegen_Quarter', 'Pipegen_Pct_Day_Bin'])['ARR Change'].sum().reset_index()
    pivot = grouped.pivot(index='Pipegen_Pct_Day_Bin', columns='Pipegen_Quarter', values='ARR Change').fillna(0)
    full_index = pd.Index(range(0, 101), name='Pipegen_Pct_Day_Bin')
    pivot = pivot.reindex(full_index, fill_value=0)
    cum = pivot.cumsum()
    return cum