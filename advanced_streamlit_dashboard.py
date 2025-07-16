#!/usr/bin/env python3
"""
Advanced Master Report Streamlit Dashboard
Comprehensive analytics dashboard matching the notebook analytical framework.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import os
import sys
from io import StringIO

# Import functions from master_report.py
from master_report import (
    connect_to_salesforce, 
    fetch_salesforce_report,
    process_date_columns,
    get_quarter_info,
    get_last_completed_quarters,
    compute_day_of_quarter,
    create_sql_pivot,
    create_sao_pivot,
    create_pipegen_pivot
)

# Configuration
STAGE_WON = 'Closed Won'
STAGE_LOST = 'Closed Lost'
VALID_SEGMENTS = ['Enterprise', 'Mid Market', 'SMB']
VALID_SOURCES = ['AE', 'BDR', 'Channel', 'Marketing', 'Success']
START_FISCAL_YEAR = 2023

# Envoy Brand Colors
ENVOY_COLORS = {
    'primary': '#FA4338',      # Envoy Red
    'secondary': '#3F4450',    # Carbon
    'background': '#F6F6F9',   # Arctic
    'white': '#FFFFFF',        # White
    'gem': '#4141A2',          # Gem
    'cilantro': '#21944E',     # Cilantro
    'powder': '#B5DFEB',       # Powder
    'pistachio': '#D3D327',    # Pistachio
    'garnet': '#A00C1F',       # Garnet
    'smoothie': '#FFB0C5'      # Smoothie
}

# Color palette for charts
CHART_COLORS = [
    ENVOY_COLORS['gem'],       # Blue for Expansion
    ENVOY_COLORS['cilantro'],  # Green for New Business
    ENVOY_COLORS['primary'],   # Red for targets/lines
    ENVOY_COLORS['garnet'],    # Dark red for secondary
    ENVOY_COLORS['powder'],    # Light blue for tertiary
    ENVOY_COLORS['pistachio'], # Yellow-green for quaternary
    ENVOY_COLORS['smoothie']   # Pink for quinary
]

# Page configuration
st.set_page_config(
    page_title="Advanced Master Report Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
.main > div {
    padding-top: 1rem;
}
.metric-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 1.5rem;
    border-radius: 0.5rem;
    color: white;
    margin: 0.5rem 0;
    text-align: center;
}
.comparison-positive {
    color: #28a745;
    font-weight: bold;
}
.comparison-negative {
    color: #dc3545;
    font-weight: bold;
}
.stTabs [data-baseweb="tab-list"] {
    gap: 24px;
}
.stTabs [data-baseweb="tab"] {
    height: 50px;
    padding-left: 20px;
    padding-right: 20px;
}
</style>
""", unsafe_allow_html=True)

@st.cache_data(ttl=3600)
def load_and_process_data():
    """Load and process Salesforce data with comprehensive filtering and enhancement."""
    try:
        # Load data
        if os.path.exists("master_report.csv"):
            df = pd.read_csv("master_report.csv")
        else:
            sf = connect_to_salesforce()
            df = fetch_salesforce_report(sf)
            df = process_date_columns(df)
            df.to_csv("master_report.csv", index=False)
        
        # Handle duplicate columns and index issues
        if df.columns.duplicated().any():
            df = df.loc[:, ~df.columns.duplicated()]
        df = df.reset_index(drop=True)
        
        # Convert date columns
        date_columns = ['Created Date', 'SQO Date', 'SAO Date', 'Timestamp: Solution Validation', 'Close Date']
        for col in date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
        
        # Process date columns if not already done
        if 'Created Date_Quarter' not in df.columns:
            df = process_date_columns(df)
        
        # Filter to valid segments and sources
        df = df[df['Segment - historical'].isin(VALID_SEGMENTS)].reset_index(drop=True)
        df = df[~df['Source'].isin(['Other', 'Connect'])].reset_index(drop=True)
        
        # Create in-quarter booking flag
        df['Inquarter Booking Flag'] = (df['Created Date_Quarter'] == df['Close Date_Quarter']).fillna(False)
        
        # Ensure ARR Change is numeric
        df['ARR Change'] = pd.to_numeric(df['ARR Change'], errors='coerce').fillna(0)
        
        # Calculate sales cycle for closed won deals
        closed_won_mask = df['Stage'] == STAGE_WON
        df.loc[closed_won_mask, 'Sales Cycle Days'] = (
            df.loc[closed_won_mask, 'Close Date'] - df.loc[closed_won_mask, 'Created Date']
        ).dt.days
        
        return df
        
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

def add_comparison_columns(df):
    """Add QoQ, YoY, and vs Last 4Q Avg comparison columns matching notebook logic."""
    if df.empty:
        return df
    
    # Get quarter columns (format: YYYY-QX)
    quarter_cols = [col for col in df.columns 
                   if isinstance(col, str) and '-Q' in col and col[0].isdigit()]
    quarter_cols.sort()
    
    if len(quarter_cols) < 2:
        return df
    
    # Convert to numeric
    for col in quarter_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Use last completed quarter (second to last) for comparisons - same as notebook
    last_completed_quarter = quarter_cols[-2]  # This is the last completed quarter
    
    # QoQ Change: last completed quarter vs previous quarter
    if len(quarter_cols) >= 3:
        prev_quarter = quarter_cols[-3]
        df['QoQ Change'] = ((df[last_completed_quarter] - df[prev_quarter]) / 
                           df[prev_quarter].replace(0, np.nan) * 100).round(2)
    
    # vs Last 4Q Avg: last completed quarter vs average of 4 quarters before that
    if len(quarter_cols) >= 6:
        # Get 4 quarters before the last completed quarter
        last_4_quarters = quarter_cols[-6:-2]  # 4 quarters before last completed
        avg_last_4 = df[last_4_quarters].mean(axis=1)
        df['4Q Avg'] = ((df[last_completed_quarter] - avg_last_4) / 
                               avg_last_4.replace(0, np.nan) * 100).round(2)
    
    # YoY Change: last completed quarter vs same quarter last year
    try:
        last_completed_year = int(last_completed_quarter.split('-')[0])
        last_completed_q = last_completed_quarter.split('-')[1]
        prev_year_quarter = f"{last_completed_year-1}-{last_completed_q}"
        
        if prev_year_quarter in quarter_cols:
            df['YoY Change'] = ((df[last_completed_quarter] - df[prev_year_quarter]) / 
                               df[prev_year_quarter].replace(0, np.nan) * 100).round(2)
    except (ValueError, IndexError):
        pass  # Skip YoY if quarter format parsing fails
    
    return df

def calculate_bookings_analysis(df):
    """Calculate comprehensive bookings analysis."""
    # Filter for closed won deals
    bookings_df = df[df['Stage'] == STAGE_WON].copy()
    
    # Get last 8 completed quarters + current quarter (9 total)
    today = datetime.today()
    active_quarter = get_quarter_info(today)[0]
    last_8_completed = get_last_completed_quarters(8, today)
    all_quarters = last_8_completed + [active_quarter]
    
    # Filter for relevant quarters
    bookings_df = bookings_df[bookings_df['Close Date_Quarter'].isin(all_quarters)]
    
    # In-quarter bookings
    inq_bookings = bookings_df[bookings_df['Inquarter Booking Flag'] == True]
    
    # Analysis by Bookings Type
    bookings_by_type = pd.pivot_table(
        inq_bookings,
        index='Bookings Type',
        columns='Close Date_Quarter',
        values='ARR Change',
        aggfunc='sum',
        fill_value=0
    )
    bookings_by_type = bookings_by_type.reindex(columns=all_quarters, fill_value=0)
    
    # Total bookings
    total_bookings = bookings_df.groupby('Close Date_Quarter')['ARR Change'].sum()
    total_bookings = total_bookings.reindex(all_quarters, fill_value=0)
    
    # Only include rows with actual data (remove effectively blank booking types)
    bookings_by_type = bookings_by_type[bookings_by_type.sum(axis=1) > 0]
    
    # Remove any rows that are just index names or empty rows
    bookings_by_type = bookings_by_type[~bookings_by_type.index.isin(['Bookings Type', ''])]
    
    # Combine results
    bookings_by_type.loc['Total Inquarter'] = bookings_by_type.sum(axis=0)
    bookings_by_type.loc['Total bookings'] = total_bookings
    
    # Calculate percentage
    pct_inquarter = (bookings_by_type.loc['Total Inquarter'] / 
                    bookings_by_type.loc['Total bookings'].replace(0, np.nan) * 100).fillna(0)
    bookings_by_type.loc['Percent inquarter'] = pct_inquarter.apply(lambda x: f"{x:.1f}%")
    
    # Add comparison columns for numeric rows only
    numeric_rows = bookings_by_type.index[bookings_by_type.index != 'Percent inquarter']
    bookings_numeric = bookings_by_type.loc[numeric_rows].copy()
    bookings_numeric = add_comparison_columns(bookings_numeric)
    
    # Add back percentage row
    percent_row = bookings_by_type.loc[['Percent inquarter']].copy()
    bookings_final = pd.concat([bookings_numeric, percent_row])
    
    # Final cleanup - remove any remaining empty or header rows
    bookings_final = bookings_final[~bookings_final.index.isin(['Bookings Type', '', 'NaN'])]
    
    # Reset index to remove the index name and convert to regular column
    bookings_final = bookings_final.reset_index()
    
    # Rename the index column if it exists
    if 'index' in bookings_final.columns:
        bookings_final = bookings_final.rename(columns={'index': 'Metric'})
    elif 'Bookings Type' in bookings_final.columns:
        bookings_final = bookings_final.rename(columns={'Bookings Type': 'Metric'})
    
    # Set the first column as index but without a name
    if 'Metric' in bookings_final.columns:
        bookings_final = bookings_final.set_index('Metric')
        bookings_final.index.name = None  # Remove the index name
    
    # Remove "Close Date_Quarter" from column names if it exists
    bookings_final.columns.name = None
    
    return bookings_final

def calculate_pipegen_analysis(df):
    """Calculate pipeline generation analysis."""
    # Filter for opportunities with SAO Date
    pipegen_df = df[df['SAO Date'].notna()].copy()
    
    # Get last 8 quarters
    today = datetime.today()
    active_quarter = get_quarter_info(today)[0]
    last_7 = get_last_completed_quarters(7, today)
    last_8_quarters = last_7 + [active_quarter]
    
    # Filter for relevant quarters  
    pipegen_df = pipegen_df[pipegen_df['SAO Date_Quarter'].isin(last_8_quarters)]
    
    # Analysis by Bookings Type
    pipegen_by_type = pd.pivot_table(
        pipegen_df,
        index='Bookings Type',
        columns='SAO Date_Quarter',
        values='ARR Change',
        aggfunc='sum',
        fill_value=0
    )
    pipegen_by_type = pipegen_by_type.reindex(columns=last_8_quarters, fill_value=0)
    
    # Add total row
    pipegen_by_type.loc['Total Pipeline Generated'] = pipegen_by_type.sum(axis=0)
    
    # Add comparison columns
    pipegen_by_type = add_comparison_columns(pipegen_by_type)
    
    return pipegen_by_type

def calculate_win_rate_analysis(df):
    """Calculate win rate analysis."""
    # Filter for closed deals (won or lost)
    closed_deals = df[df['Stage'].isin([STAGE_WON, STAGE_LOST])].copy()
    
    # Get last 8 quarters
    today = datetime.today()
    active_quarter = get_quarter_info(today)[0]
    last_7 = get_last_completed_quarters(7, today)
    last_8_quarters = last_7 + [active_quarter]
    
    # Filter for relevant quarters
    closed_deals = closed_deals[closed_deals['Close Date_Quarter'].isin(last_8_quarters)]
    
    # Calculate win rates by Bookings Type
    win_rates = []
    for quarter in last_8_quarters:
        quarter_data = closed_deals[closed_deals['Close Date_Quarter'] == quarter]
        
        for booking_type in ['Expansion', 'New Business']:
            type_data = quarter_data[quarter_data['Bookings Type'] == booking_type]
            
            if len(type_data) > 0:
                won_count = len(type_data[type_data['Stage'] == STAGE_WON])
                total_count = len(type_data)
                win_rate = (won_count / total_count * 100) if total_count > 0 else 0
            else:
                win_rate = 0
            
            win_rates.append({
                'Quarter': quarter,
                'Bookings Type': booking_type,
                'Win Rate': win_rate
            })
    
    # Convert to pivot table
    win_rate_df = pd.DataFrame(win_rates)
    win_rate_pivot = pd.pivot_table(
        win_rate_df,
        index='Bookings Type',
        columns='Quarter',
        values='Win Rate',
        fill_value=0
    )
    win_rate_pivot = win_rate_pivot.reindex(columns=last_8_quarters, fill_value=0)
    
    # Add comparison columns
    win_rate_pivot = add_comparison_columns(win_rate_pivot)
    
    return win_rate_pivot

def calculate_asp_analysis(df):
    """Calculate Average Sales Price analysis."""
    # Filter for closed won deals
    asp_df = df[df['Stage'] == STAGE_WON].copy()
    
    # Get last 8 quarters
    today = datetime.today()
    active_quarter = get_quarter_info(today)[0]
    last_7 = get_last_completed_quarters(7, today)
    last_8_quarters = last_7 + [active_quarter]
    
    # Filter for relevant quarters
    asp_df = asp_df[asp_df['Close Date_Quarter'].isin(last_8_quarters)]
    
    # Calculate ASP by Bookings Type
    asp_by_type = pd.pivot_table(
        asp_df,
        index='Bookings Type',
        columns='Close Date_Quarter',
        values='ARR Change',
        aggfunc='mean',
        fill_value=0
    )
    asp_by_type = asp_by_type.reindex(columns=last_8_quarters, fill_value=0)
    
    # Add comparison columns
    asp_by_type = add_comparison_columns(asp_by_type)
    
    return asp_by_type

def calculate_asc_analysis(df):
    """Calculate Average Sales Cycle analysis."""
    # Filter for closed won deals with sales cycle data
    asc_df = df[(df['Stage'] == STAGE_WON) & (df['Sales Cycle Days'].notna())].copy()
    
    # Get last 8 quarters
    today = datetime.today()
    active_quarter = get_quarter_info(today)[0]
    last_7 = get_last_completed_quarters(7, today)
    last_8_quarters = last_7 + [active_quarter]
    
    # Filter for relevant quarters
    asc_df = asc_df[asc_df['Close Date_Quarter'].isin(last_8_quarters)]
    
    # Calculate ASC by Bookings Type
    asc_by_type = pd.pivot_table(
        asc_df,
        index='Bookings Type',
        columns='Close Date_Quarter',
        values='Sales Cycle Days',
        aggfunc='mean',
        fill_value=0
    )
    asc_by_type = asc_by_type.reindex(columns=last_8_quarters, fill_value=0)
    
    # Add comparison columns
    asc_by_type = add_comparison_columns(asc_by_type)
    
    return asc_by_type

def create_metric_chart(df, metric_name, title):
    """Create a chart for a specific metric."""
    fig = go.Figure()
    
    # Get quarter columns (format: YYYY-QX)
    quarter_cols = [col for col in df.columns 
                   if isinstance(col, str) and '-Q' in col and col[0].isdigit()]
    quarter_cols.sort()
    
    # Skip if no quarter columns found
    if not quarter_cols:
        fig.add_annotation(
            text="No quarter data available",
            x=0.5, y=0.5,
            xref="paper", yref="paper",
            showarrow=False,
            font=dict(size=16)
        )
        return fig
    
    color_index = 0
    for index in df.index:
        if index in ['Total Inquarter', 'Total bookings', 'Total Pipeline Generated', 'Percent inquarter']:
            continue  # Skip total rows for individual series
            
        y_values = [df.loc[index, col] for col in quarter_cols]
        
        fig.add_trace(go.Scatter(
            x=quarter_cols,
            y=y_values,
            mode='lines+markers',
            name=str(index),
            line=dict(width=3, color=CHART_COLORS[color_index % len(CHART_COLORS)]),
            marker=dict(size=8)
        ))
        color_index += 1
    
    fig.update_layout(
        title=title,
        xaxis_title="Quarter",
        yaxis_title=metric_name,
        height=500,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig

def format_currency_3_sig_figs(value):
    """Format currency values with 3 significant figures."""
    # Handle string values (already formatted)
    if isinstance(value, str):
        return value
        
    if pd.isna(value) or value == 0:
        return "$0"
    
    try:
        abs_value = abs(float(value))
        sign = "-" if float(value) < 0 else ""
    except (ValueError, TypeError):
        return str(value)  # Return as-is if can't convert
    
    if abs_value >= 1000000:
        # Millions
        millions = abs_value / 1000000
        if millions >= 100:
            return f"{sign}${millions:.0f}M"
        elif millions >= 10:
            return f"{sign}${millions:.1f}M"
        else:
            return f"{sign}${millions:.2f}M"
    elif abs_value >= 1000:
        # Thousands
        thousands = abs_value / 1000
        if thousands >= 100:
            return f"{sign}${thousands:.0f}K"
        elif thousands >= 10:
            return f"{sign}${thousands:.1f}K"
        else:
            return f"{sign}${thousands:.2f}K"
    elif abs_value >= 1:
        # 4 digits (1K-9.999K range shown as whole numbers)
        if abs_value >= 1000:
            return f"{sign}${abs_value:,.0f}"
        else:
            return f"{sign}${abs_value:.0f}"
    else:
        # Under 1
        return f"{sign}${abs_value:.2f}"

def format_percentage_2_sig_figs(value):
    """Format percentage values with 2 significant figures."""
    # Handle string values (already formatted)
    if isinstance(value, str):
        return value
        
    if pd.isna(value):
        return ""
    
    try:
        abs_value = abs(float(value))
        sign = "-" if float(value) < 0 else ""
    except (ValueError, TypeError):
        return str(value)  # Return as-is if can't convert
    
    if abs_value >= 10:
        return f"{sign}{abs_value:.0f}%"
    elif abs_value >= 1:
        return f"{sign}{abs_value:.1f}%"
    else:
        return f"{sign}{abs_value:.2f}%"

def format_plain_number(value):
    """Format plain numbers (for ASC - Days)."""
    # Handle string values (already formatted)
    if isinstance(value, str):
        return value
        
    if pd.isna(value) or value == 0:
        return "0"
    
    try:
        abs_value = abs(float(value))
        sign = "-" if float(value) < 0 else ""
    except (ValueError, TypeError):
        return str(value)  # Return as-is if can't convert
    
    if abs_value >= 10:
        return f"{sign}{abs_value:,.0f}"
    else:
        return f"{sign}{abs_value:.1f}"

def format_comparison_value(value):
    """Format comparison values with color coding."""
    if pd.isna(value):
        return ""
    
    color_class = "comparison-positive" if value > 0 else "comparison-negative"
    formatted_value = format_percentage_2_sig_figs(value)
    return f'<span class="{color_class}">{formatted_value}</span>'

def display_metric_table(df, metric_name):
    """Display a formatted table for a metric with custom formatting."""
    if df.empty:
        st.warning(f"No data available for {metric_name}")
        return
    
    # Format the dataframe for display
    display_df = df.copy()
    
    # Get quarter columns for formatting
    quarter_cols = [col for col in display_df.columns 
                   if isinstance(col, str) and '-Q' in col and col[0].isdigit()]
    
    # Apply metric-specific formatting
    if metric_name in ['Bookings', 'Pipeline Generation']:
        # Currency formatting for ARR values
        for col in quarter_cols:
            if col in display_df.columns:
                display_df[col] = display_df[col].apply(format_currency_3_sig_figs)
                
    elif metric_name == 'Win Rate':
        # Percentage formatting for win rates
        for col in quarter_cols:
            if col in display_df.columns:
                display_df[col] = display_df[col].apply(format_percentage_2_sig_figs)
                
    elif metric_name == 'Average Sales Price':
        # Currency formatting for ASP
        for col in quarter_cols:
            if col in display_df.columns:
                display_df[col] = display_df[col].apply(format_currency_3_sig_figs)
                
    elif metric_name == 'Average Sales Cycle':
        # Plain number formatting for ASC (days)
        for col in quarter_cols:
            if col in display_df.columns:
                display_df[col] = display_df[col].apply(format_plain_number)
    
    # Format comparison columns (always percentages)
    comparison_cols = ['QoQ Change', '4Q Avg', 'YoY Change']
    for col in comparison_cols:
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(format_comparison_value)
    
    # Format percentage rows specially
    if 'Percent inquarter' in display_df.index:
        for col in quarter_cols:
            if col in display_df.columns:
                # Convert percentage row to numeric then format
                try:
                    value = float(display_df.loc['Percent inquarter', col].replace('%', ''))
                    display_df.loc['Percent inquarter', col] = format_percentage_2_sig_figs(value)
                except:
                    pass  # Keep original if conversion fails
    
    # Display the table
    st.write(f"### {metric_name} Analysis")
    st.write(display_df.to_html(escape=False), unsafe_allow_html=True)

def create_sql_pivot_enhanced(df):
    """Create enhanced SQL pivot table with last 8 quarters."""
    # Get the last 8 quarters (7 completed + active)
    today = datetime.today()
    active_quarter = get_quarter_info(today)[0]
    last_7 = get_last_completed_quarters(7, today)
    quarters_8 = last_7 + [active_quarter]
    
    # Slim down DataFrame
    df_pivot = df[['Created Date_Quarter', 'Source', 'Segment - historical', 'Bookings Type']].copy()
    
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
    
    index = pd.MultiIndex.from_tuples(rows, names=['Source','Segment - historical','Bookings Type'])
    
    # Initialize pivot shell
    pivot = pd.DataFrame(0, index=index, columns=quarters_8 + ['Grand Total'])
    
    # Fill the detail counts
    grp = df_pivot.groupby(['Source','Segment - historical','Bookings Type','Created Date_Quarter']).size()
    
    for (src, seg, bt, q), cnt in grp.items():
        if (src,seg,bt) in pivot.index and q in quarters_8:
            pivot.at[(src,seg,bt), q] = cnt
    
    # Segment subtotals
    grp_seg = df_pivot.groupby(['Source','Segment - historical','Created Date_Quarter']).size()
    
    for (src, seg, q), cnt in grp_seg.items():
        row = (src, seg, f'{seg} Total')
        if row in pivot.index and q in quarters_8:
            pivot.at[row, q] = cnt
    
    # Source subtotals
    grp_src = df_pivot.groupby(['Source','Created Date_Quarter']).size()
    for (src, q), cnt in grp_src.items():
        row = (src, '', f'{src} Total')
        if row in pivot.index and q in quarters_8:
            pivot.at[row, q] = cnt
    
    # Grand total per quarter
    grp_all = df_pivot.groupby('Created Date_Quarter').size()
    for q, cnt in grp_all.items():
        if q in quarters_8:
            pivot.at[('', '', 'Grand Total'), q] = cnt
    
    # Final grand total column
    pivot['Grand Total'] = pivot[quarters_8].sum(axis=1)
    
    return pivot.reset_index()

def create_sao_pivot_enhanced(df):
    """Create enhanced SAO pivot table with last 8 quarters."""
    # Get quarters
    today = datetime.today()
    active_quarter = get_quarter_info(today)[0]
    last_7 = get_last_completed_quarters(7, today)
    quarters_8 = last_7 + [active_quarter]
    
    # Slim DataFrame for SAO pivot
    df_sao = df[['SAO Date', 'SAO Date_Quarter', 'Source', 'Segment - historical', 'Bookings Type']].copy()
    
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
    
    index = pd.MultiIndex.from_tuples(rows, names=['Source','Segment - historical','Bookings Type'])
    
    # Initialize pivot shell
    pivot_sao = pd.DataFrame(0, index=index, columns=quarters_8 + ['Grand Total'])
    
    # Detail counts by SAO Date Quarter
    grp = df_sao.groupby(['Source','Segment - historical','Bookings Type','SAO Date_Quarter']).size()
    
    for (src, seg, bt, q), cnt in grp.items():
        if (src,seg,bt) in pivot_sao.index and q in quarters_8:
            pivot_sao.at[(src,seg,bt), q] = cnt
    
    # Segment subtotals
    grp_seg = df_sao.groupby(['Source','Segment - historical','SAO Date_Quarter']).size()
    
    for (src, seg, q), cnt in grp_seg.items():
        row = (src, seg, f'{seg} Total')
        if row in pivot_sao.index and q in quarters_8:
            pivot_sao.at[row, q] = cnt
    
    # Source subtotals
    grp_src = df_sao.groupby(['Source','SAO Date_Quarter']).size()
    
    for (src, q), cnt in grp_src.items():
        row = (src, '', f'{src} Total')
        if row in pivot_sao.index and q in quarters_8:
            pivot_sao.at[row, q] = cnt
    
    # Grand total per quarter
    grp_all = df_sao.groupby('SAO Date_Quarter').size()
    for q, cnt in grp_all.items():
        if q in quarters_8:
            pivot_sao.at[('', '', 'Grand Total'), q] = cnt
    
    # Final grand total column
    pivot_sao['Grand Total'] = pivot_sao[quarters_8].sum(axis=1)
    
    return pivot_sao.reset_index()

def create_pipegen_pivot_enhanced(df):
    """Create enhanced Pipegen pivot table with last 8 quarters."""
    # Get quarters
    today = datetime.today()
    active_quarter = get_quarter_info(today)[0]
    last_7 = get_last_completed_quarters(7, today)
    quarters_8 = last_7 + [active_quarter]
    
    # Slim DataFrame for SAO pipegen
    df_pip = df[['SAO Date_Quarter', 'Source', 'Segment - historical', 'Bookings Type', 'ARR Change']].copy()
    
    # Only keep rows with an SAO
    df_pip = df_pip[df_pip['SAO Date_Quarter'].notna()]
    
    # Map non-standard sources into "Other"
    allowed_src = ['AE','BDR','Channel','Marketing','Success']
    df_pip['Source'] = df_pip['Source'].where(df_pip['Source'].isin(allowed_src), 'Other')
    
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
    
    index = pd.MultiIndex.from_tuples(rows, names=['Source','Segment - historical','Bookings Type'])
    
    # Initialize pivot shell
    pivot_pip = pd.DataFrame(0.0, index=index, columns=quarters_8 + ['Grand Total'])
    
    # Detail: sum ARR Change by SAO Date_Quarter
    grp = df_pip.groupby(['Source','Segment - historical','Bookings Type','SAO Date_Quarter'])['ARR Change'].sum()
    
    for (src, seg, bt, q), total in grp.items():
        if (src,seg,bt) in pivot_pip.index and q in quarters_8:
            pivot_pip.at[(src,seg,bt), q] = total
    
    # Segment subtotals (sum ARR Change)
    grp_seg = df_pip.groupby(['Source','Segment - historical','SAO Date_Quarter'])['ARR Change'].sum()
    
    for (src, seg, q), total in grp_seg.items():
        row = (src, seg, f'{seg} Total')
        if row in pivot_pip.index and q in quarters_8:
            pivot_pip.at[row, q] = total
    
    # Source subtotals
    grp_src = df_pip.groupby(['Source','SAO Date_Quarter'])['ARR Change'].sum()
    for (src, q), total in grp_src.items():
        row = (src, '', f'{src} Total')
        if row in pivot_pip.index and q in quarters_8:
            pivot_pip.at[row, q] = total
    
    # Grand total per quarter
    grp_all = df_pip.groupby('SAO Date_Quarter')['ARR Change'].sum()
    for q, total in grp_all.items():
        if q in quarters_8:
            pivot_pip.at[('', '', 'Grand Total'), q] = total
    
    # Grand total column
    pivot_pip['Grand Total'] = pivot_pip[quarters_8].sum(axis=1)
    
    return pivot_pip.reset_index()

def create_bookings_combo_chart(bookings_analysis):
    """Create a combo chart showing bookings by type with proportions."""
    # Get quarter columns
    quarter_cols = [col for col in bookings_analysis.columns 
                   if isinstance(col, str) and '-Q' in col and col[0].isdigit()]
    quarter_cols.sort()
    
    if not quarter_cols:
        fig = go.Figure()
        fig.add_annotation(
            text="No quarter data available",
            x=0.5, y=0.5,
            xref="paper", yref="paper",
            showarrow=False,
            font=dict(size=16)
        )
        return fig
    
    # Create subplot with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Get data for each booking type
    expansion_values = [bookings_analysis.loc['Expansion', col] for col in quarter_cols] if 'Expansion' in bookings_analysis.index else [0] * len(quarter_cols)
    new_business_values = [bookings_analysis.loc['New Business', col] for col in quarter_cols] if 'New Business' in bookings_analysis.index else [0] * len(quarter_cols)
    
    # Add separate bars for each bookings type (grouped, not stacked)
    fig.add_trace(
        go.Bar(
            x=quarter_cols,
            y=expansion_values,
            name='Expansion',
            marker_color=ENVOY_COLORS['gem'],
            yaxis='y'
        ),
        secondary_y=False
    )
    
    fig.add_trace(
        go.Bar(
            x=quarter_cols,
            y=new_business_values,
            name='New Business',
            marker_color=ENVOY_COLORS['cilantro'],
            yaxis='y'
        ),
        secondary_y=False
    )
    
    # Calculate proportions and add line
    total_values = [exp + nb for exp, nb in zip(expansion_values, new_business_values)]
    expansion_pct = [(exp / total * 100) if total > 0 else 0 for exp, total in zip(expansion_values, total_values)]
    
    fig.add_trace(
        go.Scatter(
            x=quarter_cols,
            y=expansion_pct,
            mode='lines+markers',
            name='Expansion %',
            line=dict(color=ENVOY_COLORS['primary'], width=3),
            marker=dict(size=8),
            yaxis='y2'
        ),
        secondary_y=True
    )
    
    # Update layout
    fig.update_layout(
        title="Bookings Trends by Type",
        xaxis_title="Quarter",
        height=500,
        showlegend=True,
        barmode='group'  # Changed from 'stack' to 'group'
    )
    
    # Update y-axes
    fig.update_yaxes(title_text="Bookings ($)", secondary_y=False, rangemode='tozero')
    fig.update_yaxes(title_text="Expansion %", secondary_y=True, range=[0, 100])
    
    return fig

def create_bookings_pacing_chart(df):
    """Create bookings pacing visualization with target line and current position."""
    today = datetime.today()
    current_quarter = get_quarter_info(today)[0]
    
    # Get current position in quarter
    current_pct = compute_day_of_quarter(today)['Pct_Day']
    
    # Filter for closed won deals in current quarter
    bookings_df = df[(df['Stage'] == STAGE_WON) & (df['Close Date_Quarter'] == current_quarter)].copy()
    
    # Get quarterly target (average of last 4 quarters)
    last_quarters = get_last_completed_quarters(4, today)
    if last_quarters:
        target_data = df[(df['Stage'] == STAGE_WON) & (df['Close Date_Quarter'].isin(last_quarters))]
        quarterly_target = target_data.groupby('Close Date_Quarter')['ARR Change'].sum().mean() if not target_data.empty else 1000000
    else:
        quarterly_target = 1000000  # Default target
    
    fig = go.Figure()
    
    if not bookings_df.empty:
        # Calculate day of quarter for each booking
        bookings_df['Pct_Day'] = bookings_df['Close Date'].apply(
            lambda x: compute_day_of_quarter(x)['Pct_Day'] if pd.notnull(x) else None
        )
        bookings_df = bookings_df.dropna(subset=['Pct_Day'])
        
        # Calculate cumulative bookings by day
        daily_bookings = bookings_df.groupby('Pct_Day')['ARR Change'].sum().sort_index().cumsum()
        
        # Add actual bookings line
        fig.add_trace(go.Scatter(
            x=daily_bookings.index,
            y=daily_bookings.values,
            mode='lines+markers',
            name='Actual Bookings',
            line=dict(color=ENVOY_COLORS['gem'], width=3),
            marker=dict(size=6)
        ))
        
        # Current performance
        current_bookings = daily_bookings.iloc[-1] if not daily_bookings.empty else 0
    else:
        current_bookings = 0
    
    # Add target line (linear progression)
    target_x = [0, 100]
    target_y = [0, quarterly_target]
    fig.add_trace(go.Scatter(
        x=target_x,
        y=target_y,
        mode='lines',
        name='Target Pace',
        line=dict(color='gray', width=2, dash='dash'),
        opacity=0.7
    ))
    
    # Add current position to actual bookings line if we have data
    if current_bookings > 0:
        # Extend the actual bookings line to current position
        extended_x = list(daily_bookings.index) + [current_pct]
        extended_y = list(daily_bookings.values) + [current_bookings]
        
        # Replace the actual bookings trace with extended version
        fig.data = []  # Clear existing traces
        fig.add_trace(go.Scatter(
            x=extended_x,
            y=extended_y,
            mode='lines+markers',
            name='Actual Bookings',
            line=dict(color=ENVOY_COLORS['gem'], width=3),
            marker=dict(size=6)
        ))
    
    # Add target line (linear progression)
    target_x = [0, 100]
    target_y = [0, quarterly_target]
    fig.add_trace(go.Scatter(
        x=target_x,
        y=target_y,
        mode='lines',
        name='Target Pace',
        line=dict(color=ENVOY_COLORS['secondary'], width=2, dash='dash'),
        opacity=0.7
    ))
    
    # Add current position indicator on target line
    current_target = (current_pct / 100) * quarterly_target
    fig.add_trace(go.Scatter(
        x=[current_pct],
        y=[current_target],
        mode='markers',
        name='Target Position',
        marker=dict(color=ENVOY_COLORS['primary'], size=10, symbol='x')
    ))
    
    # Status indicator
    if current_bookings > current_target:
        status = "🟢 AHEAD"
        status_color = "green"
    elif current_bookings < current_target * 0.9:
        status = "🔴 BEHIND"
        status_color = "red"
    else:
        status = "🟡 ON TRACK"
        status_color = "orange"
    
    fig.update_layout(
        title=f'Bookings Pacing - {current_quarter} | {status}',
        xaxis_title='% of Quarter Completed',
        yaxis_title='Cumulative Bookings ($)',
        height=500,  # Increased from 400 to 500 for better visibility
        showlegend=True,
        xaxis=dict(range=[0, 100]),  # Always show 0-100%
        annotations=[
            dict(
                x=current_pct,
                y=max(current_bookings, current_target) * 1.1,
                text=f"{current_pct:.1f}% through quarter",
                showarrow=True,
                arrowhead=2,
                arrowcolor=status_color
            )
        ]
    )
    
    return fig

def create_pipegen_pacing_chart(df):
    """Create pipeline generation pacing visualization with target line and current position."""
    today = datetime.today()
    current_quarter = get_quarter_info(today)[0]
    
    # Get current position in quarter
    current_pct = compute_day_of_quarter(today)['Pct_Day']
    
    # Filter for SAOs in current quarter
    pipegen_df = df[(df['SAO Date_Quarter'] == current_quarter) & (df['SAO Date'].notna())].copy()
    
    # Get quarterly target (average of last 4 quarters)
    last_quarters = get_last_completed_quarters(4, today)
    if last_quarters:
        target_data = df[(df['SAO Date_Quarter'].isin(last_quarters)) & (df['SAO Date'].notna())]
        quarterly_target = target_data.groupby('SAO Date_Quarter')['ARR Change'].sum().mean() if not target_data.empty else 2000000
    else:
        quarterly_target = 2000000  # Default target
    
    fig = go.Figure()
    
    if not pipegen_df.empty:
        # Calculate day of quarter for each SAO
        pipegen_df['Pct_Day'] = pipegen_df['SAO Date'].apply(
            lambda x: compute_day_of_quarter(x)['Pct_Day'] if pd.notnull(x) else None
        )
        pipegen_df = pipegen_df.dropna(subset=['Pct_Day'])
        
        # Calculate cumulative pipeline generation by day
        daily_pipegen = pipegen_df.groupby('Pct_Day')['ARR Change'].sum().sort_index().cumsum()
        
        # Add actual pipeline generation line
        fig.add_trace(go.Scatter(
            x=daily_pipegen.index,
            y=daily_pipegen.values,
            mode='lines+markers',
            name='Actual Pipeline',
            line=dict(color=ENVOY_COLORS['cilantro'], width=3),
            marker=dict(size=6)
        ))
        
        # Current performance
        current_pipegen = daily_pipegen.iloc[-1] if not daily_pipegen.empty else 0
    else:
        current_pipegen = 0
    
    # Add target line (linear progression)
    target_x = [0, 100]
    target_y = [0, quarterly_target]
    fig.add_trace(go.Scatter(
        x=target_x,
        y=target_y,
        mode='lines',
        name='Target Pace',
        line=dict(color='gray', width=2, dash='dash'),
        opacity=0.7
    ))
    
    # Add current position to actual pipeline line if we have data
    if current_pipegen > 0:
        # Extend the actual pipeline line to current position
        extended_x = list(daily_pipegen.index) + [current_pct]
        extended_y = list(daily_pipegen.values) + [current_pipegen]
        
        # Replace the actual pipeline trace with extended version
        fig.data = []  # Clear existing traces
        fig.add_trace(go.Scatter(
            x=extended_x,
            y=extended_y,
            mode='lines+markers',
            name='Actual Pipeline',
            line=dict(color=ENVOY_COLORS['cilantro'], width=3),
            marker=dict(size=6)
        ))
    
    # Add target line (linear progression)
    target_x = [0, 100]
    target_y = [0, quarterly_target]
    fig.add_trace(go.Scatter(
        x=target_x,
        y=target_y,
        mode='lines',
        name='Target Pace',
        line=dict(color=ENVOY_COLORS['secondary'], width=2, dash='dash'),
        opacity=0.7
    ))
    
    # Add current position indicator on target line
    current_target = (current_pct / 100) * quarterly_target
    fig.add_trace(go.Scatter(
        x=[current_pct],
        y=[current_target],
        mode='markers',
        name='Target Position',
        marker=dict(color=ENVOY_COLORS['primary'], size=10, symbol='x')
    ))
    
    # Status indicator
    if current_pipegen > current_target:
        status = "🟢 AHEAD"
        status_color = "green"
    elif current_pipegen < current_target * 0.9:
        status = "🔴 BEHIND"
        status_color = "red"
    else:
        status = "🟡 ON TRACK"
        status_color = "orange"
    
    fig.update_layout(
        title=f'Pipeline Generation Pacing - {current_quarter} | {status}',
        xaxis_title='% of Quarter Completed',
        yaxis_title='Cumulative Pipeline Generated ($)',
        height=400,
        showlegend=True,
        xaxis=dict(range=[0, 100]),  # Always show 0-100%
        annotations=[
            dict(
                x=current_pct,
                y=max(current_pipegen, current_target) * 1.1,
                text=f"{current_pct:.1f}% through quarter",
                showarrow=True,
                arrowhead=2,
                arrowcolor=status_color
            )
        ]
    )
    
    return fig

def main():
    """Main dashboard function."""
    st.title("📊 Advanced Master Report Dashboard")
    st.markdown("*Comprehensive analytics matching notebook framework*")
    
    # Load data
    with st.spinner("Loading and processing data..."):
        df = load_and_process_data()
    
    if df is None:
        st.error("Failed to load data. Please check your connection.")
        return
    
    # Remove global filters - will be added per tab
    
    # Sidebar info and filters
    st.sidebar.header("📋 Data Summary")
    st.sidebar.write(f"**Total Records**: {len(df):,}")
    st.sidebar.write(f"**Date Range**: {df['Created Date'].min().date()} to {df['Created Date'].max().date()}")
    st.sidebar.write(f"**Segments**: {', '.join(VALID_SEGMENTS)}")
    st.sidebar.write(f"**Sources**: {', '.join(VALID_SOURCES)}")
    
    # Global segment filter
    st.sidebar.header("🔍 Filters")
    selected_segments = st.sidebar.multiselect(
        "Filter by Segment",
        options=VALID_SEGMENTS,
        default=VALID_SEGMENTS,
        key="global_segment_filter"
    )
    
    # Apply global segment filter
    filtered_df = df.copy()
    if selected_segments:
        filtered_df = filtered_df[filtered_df['Segment - historical'].isin(selected_segments)]
    
    # Show filtered records count
    if len(selected_segments) != len(VALID_SEGMENTS):
        st.sidebar.write(f"**Filtered Records**: {len(filtered_df):,}")
    else:
        st.sidebar.write(f"**Filtered Records**: {len(filtered_df):,} (All segments)")
    
    # Current quarter metrics
    today = datetime.today()
    current_quarter = get_quarter_info(today)[0]
    
    st.subheader(f"📊 Current Quarter Overview ({current_quarter})")
    
    # Calculate current quarter metrics using filtered data
    current_q_data = filtered_df[filtered_df['Close Date_Quarter'] == current_quarter]
    current_q_bookings = current_q_data[current_q_data['Stage'] == STAGE_WON]['ARR Change'].sum()
    current_q_deals = len(current_q_data[current_q_data['Stage'] == STAGE_WON])
    
    # Standard SAO pipeline generation: SAOs created in current quarter
    current_q_created = filtered_df[filtered_df['Created Date_Quarter'] == current_quarter]
    current_q_pipeline = current_q_created[current_q_created['SAO Date'].notna()]['ARR Change'].sum()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Pipeline Generated", f"${current_q_pipeline:,.0f}", delta="Standard SAO")
    
    with col2:
        st.metric("Bookings", f"${current_q_bookings:,.0f}", delta="ARR Change")
    
    with col3:
        st.metric("Deals Closed", f"{current_q_deals:,}", delta="Won Deals")
    
    # Main analysis tabs
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "📈 Bookings", "🔄 Pipeline Gen", "🎯 Win Rate", "💰 ASP", "⏱️ ASC", "📊 Pivot Tables", "📉 Pacing"
    ])
    
    with tab1:
        st.header("Bookings Analysis")
        
        # Pacing chart on top
        st.subheader("Current Quarter Pacing")
        with st.spinner("Calculating bookings pacing..."):
            bookings_pacing_chart = create_bookings_pacing_chart(filtered_df)
            st.plotly_chart(bookings_pacing_chart, use_container_width=True, key="bookings_pacing")
        
        # Trends chart below
        st.subheader("Trends by Bookings Type")
        with st.spinner("Calculating bookings metrics..."):
            bookings_analysis = calculate_bookings_analysis(filtered_df)
            chart = create_bookings_combo_chart(bookings_analysis)
            st.plotly_chart(chart, use_container_width=True, key="bookings_trends")
        
        # Analysis table at bottom
        st.subheader("Bookings Analysis Table")
        display_metric_table(bookings_analysis, "Bookings")
    
    with tab2:
        st.header("Pipeline Generation Analysis")
        
        # Pacing chart on top
        st.subheader("Current Quarter Pacing")
        with st.spinner("Calculating pipeline generation pacing..."):
            pipegen_pacing_chart = create_pipegen_pacing_chart(filtered_df)
            st.plotly_chart(pipegen_pacing_chart, use_container_width=True, key="pipegen_pacing")
        
        # Trends chart below
        st.subheader("Trends by Type")
        with st.spinner("Calculating pipeline metrics..."):
            pipegen_analysis = calculate_pipegen_analysis(filtered_df)
            chart = create_metric_chart(pipegen_analysis, "Pipeline Generated ($)", "Pipeline Generation Trends")
            st.plotly_chart(chart, use_container_width=True, key="pipegen_trends")
        
        # Analysis table at bottom
        st.subheader("Pipeline Generation Analysis Table")
        display_metric_table(pipegen_analysis, "Pipeline Generation")
    
    with tab3:
        st.header("Win Rate Analysis")
        
        with st.spinner("Calculating win rate metrics..."):
            win_rate_analysis = calculate_win_rate_analysis(filtered_df)
            
        display_metric_table(win_rate_analysis, "Win Rate")
        
        # Create chart
        chart = create_metric_chart(win_rate_analysis, "Win Rate (%)", "Win Rate Trends by Type")
        st.plotly_chart(chart, use_container_width=True, key="win_rate_trends")
    
    with tab4:
        st.header("Average Sales Price Analysis")
        
        with st.spinner("Calculating ASP metrics..."):
            asp_analysis = calculate_asp_analysis(filtered_df)
            
        display_metric_table(asp_analysis, "Average Sales Price")
        
        # Create chart
        chart = create_metric_chart(asp_analysis, "ASP ($)", "Average Sales Price Trends")
        st.plotly_chart(chart, use_container_width=True, key="asp_trends")
    
    with tab5:
        st.header("Average Sales Cycle Analysis")
        
        with st.spinner("Calculating ASC metrics..."):
            asc_analysis = calculate_asc_analysis(filtered_df)
            
        display_metric_table(asc_analysis, "Average Sales Cycle")
        
        # Create chart
        chart = create_metric_chart(asc_analysis, "ASC (Days)", "Average Sales Cycle Trends")
        st.plotly_chart(chart, use_container_width=True, key="asc_trends")
    
    with tab6:
        st.header("Pivot Tables")
        st.markdown("*Last 8 quarters of data*")
        
        # Create pivot tables
        clean_df = filtered_df.copy()
        if clean_df.columns.duplicated().any():
            clean_df = clean_df.loc[:, ~clean_df.columns.duplicated()]
        clean_df = clean_df.reset_index(drop=True)
        
        sql_pivot = create_sql_pivot_enhanced(clean_df)
        sao_pivot = create_sao_pivot_enhanced(clean_df)
        pipegen_pivot = create_pipegen_pivot_enhanced(clean_df)
        
        pivot_tab1, pivot_tab2, pivot_tab3 = st.tabs(["SQLs", "SAOs", "Pipegen"])
        
        with pivot_tab1:
            st.subheader("SQL Pivot Table - Last 8 Quarters")
            st.dataframe(sql_pivot, use_container_width=True, height=600)
        
        with pivot_tab2:
            st.subheader("SAO Pivot Table - Last 8 Quarters") 
            st.dataframe(sao_pivot, use_container_width=True, height=600)
        
        with pivot_tab3:
            st.subheader("Pipeline Generation Pivot Table - Last 8 Quarters")
            st.dataframe(pipegen_pivot, use_container_width=True, height=600)
    
    with tab7:
        st.header("Quarter Pacing Analysis")
        st.markdown("*Current quarter progress tracking*")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Bookings Pacing")
            with st.spinner("Calculating bookings pacing..."):
                bookings_pacing_chart = create_bookings_pacing_chart(filtered_df)
                st.plotly_chart(bookings_pacing_chart, use_container_width=True, key="pacing_bookings")
        
        with col2:
            st.subheader("Pipeline Generation Pacing")
            with st.spinner("Calculating pipeline generation pacing..."):
                pipegen_pacing_chart = create_pipegen_pacing_chart(filtered_df)
                st.plotly_chart(pipegen_pacing_chart, use_container_width=True, key="pacing_pipegen")
        
        # Additional pacing insights
        st.markdown("---")
        st.markdown("**Pacing Insights:**")
        st.markdown("• **Bookings Pacing**: Shows cumulative bookings (closed won deals) by Close Date throughout the current quarter")
        st.markdown("• **Pipeline Generation Pacing**: Shows cumulative pipeline generated (SAO ARR) by SAO Date throughout the current quarter")
        st.markdown("• Both charts show percentage of quarter completed on X-axis and cumulative value on Y-axis")
    
    # Footer
    st.markdown("---")
    st.markdown(
        f"🕐 Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | "
        f"📊 Records: {len(df):,} | "
        f"🤖 Advanced Master Report Dashboard"
    )

if __name__ == "__main__":
    main()