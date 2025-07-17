#!/usr/bin/env python3
"""
Gears Dashboard - Pipeline Attainment Analysis
Displays the exact table from the screenshot
Loads plan data from CSV files for improved performance
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import os
import pytz
import re

# Configuration
VALID_SEGMENTS = ['Enterprise', 'Mid Market', 'SMB']
VALID_SOURCES = ['AE', 'BDR', 'Channel', 'Marketing', 'Success']
VALID_BOOKING_TYPES = ['New Business', 'Expansion']

# Segment mapping for display
SEGMENT_MAPPING = {
    'Enterprise': 'ENT',
    'Mid Market': 'MM',
    'SMB': 'SMB'
}

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

# Page configuration
st.set_page_config(
    page_title="Gears - Pipeline Attainment Dashboard",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for attainment color coding
st.markdown("""
<style>
.main > div {
    padding-top: 1rem;
}
.attainment-high {
    background-color: #d4edda;
    color: #155724;
    font-weight: bold;
}
.attainment-medium {
    background-color: #fff3cd;
    color: #856404;
    font-weight: bold;
}
.attainment-low {
    background-color: #f8d7da;
    color: #721c24;
    font-weight: bold;
}
.metric-table {
    font-size: 14px;
}
.metric-table th {
    background-color: #f8f9fa;
    font-weight: bold;
    text-align: center;
    padding: 8px;
    color: #000 !important;
}
@media (prefers-color-scheme: dark) {
    .metric-table th {
        background-color: #2d3748;
        color: #fff !important;
    }
    .metric-table td {
        border-color: #4a5568;
    }
}
.metric-table td {
    text-align: center;
    padding: 6px;
}
.gap-positive {
    color: #155724;
    font-weight: bold;
}
.gap-negative {
    color: #721c24;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

def get_current_quarter():
    """Determine the current fiscal quarter."""
    today = datetime.now()
    
    # Assuming fiscal year starts in February
    if today.month >= 2 and today.month <= 4:
        quarter = "Q1"
        fiscal_year = today.year + 1
    elif today.month >= 5 and today.month <= 7:
        quarter = "Q2"
        fiscal_year = today.year + 1
    elif today.month >= 8 and today.month <= 10:
        quarter = "Q3"
        fiscal_year = today.year + 1
    else:
        quarter = "Q4"
        fiscal_year = today.year + 1 if today.month >= 11 else today.year
    
    return f"FY{fiscal_year}{quarter}"

def get_quarter_progress():
    """Calculate how much of the current quarter has elapsed."""
    # Get current time in PST
    pst = pytz.timezone('America/Los_Angeles')
    today = datetime.now(pst)
    
    # Define quarter boundaries (assuming fiscal year starts in February)
    if today.month >= 2 and today.month <= 4:
        # Q1: Feb 1 - Apr 30
        quarter_start = pst.localize(datetime(today.year, 2, 1))
        quarter_end = pst.localize(datetime(today.year, 4, 30))
    elif today.month >= 5 and today.month <= 7:
        # Q2: May 1 - Jul 31
        quarter_start = pst.localize(datetime(today.year, 5, 1))
        quarter_end = pst.localize(datetime(today.year, 7, 31))
    elif today.month >= 8 and today.month <= 10:
        # Q3: Aug 1 - Oct 31
        quarter_start = pst.localize(datetime(today.year, 8, 1))
        quarter_end = pst.localize(datetime(today.year, 10, 31))
    else:
        # Q4: Nov 1 - Jan 31
        if today.month >= 11:
            quarter_start = pst.localize(datetime(today.year, 11, 1))
            quarter_end = pst.localize(datetime(today.year + 1, 1, 31))
        else:
            quarter_start = pst.localize(datetime(today.year - 1, 11, 1))
            quarter_end = pst.localize(datetime(today.year, 1, 31))
    
    # Calculate progress
    total_days = (quarter_end - quarter_start).days + 1
    days_elapsed = (today - quarter_start).days + 1
    
    # Ensure we don't go over 100% or under 0%
    progress = max(0, min(1, days_elapsed / total_days))
    
    return progress, days_elapsed, total_days

@st.cache_data(ttl=3600)
def load_master_report_data():
    """Load the master_report.csv file and create the three tables from raw data."""
    try:
        # Load the CSV file
        df = pd.read_csv("../data_sources/master_report.csv")
        
        # Load plan data to merge with calculations
        plan_data = load_plan_data_from_csv()
        
        # Get quarter progress for Plan to Date calculation
        progress, days_elapsed, total_days = get_quarter_progress()
        
        # Filter for SQLs (where SQO Date is not null)
        sql_data = df[df['SQO Date'].notna()].copy()
        
        # Filter for current fiscal quarter (FY26Q2) 
        # Use SQO Date Quarter for SQLs data
        current_quarter = "2026-Q2"  # Hard-coded for now, should match the actual data
        sql_data = sql_data[sql_data['SQO Date_Quarter'] == current_quarter]
        
        
        # Create aggregated data for Table 1: Source × Segment × Booking Type
        table1_data = []
        
        # Group by Source, Segment, and Booking Type
        grouped = sql_data.groupby(['Source', 'Segment - historical', 'Bookings Type']).size().reset_index(name='Actuals')
        
        for _, row in grouped.iterrows():
            source = row['Source']
            segment = row['Segment - historical']
            booking_type = row['Bookings Type']
            actuals = row['Actuals']
            
            # Filter for valid combinations and source-specific booking types
            valid_source_booking_types = {
                'AE': ['New Business', 'Expansion'],
                'BDR': ['New Business'],  # BDR only has New Business
                'Channel': ['New Business', 'Expansion'],
                'Marketing': ['New Business', 'Expansion'],
                'Success': ['Expansion']  # Success only has Expansion
            }
            
            if (source in VALID_SOURCES and 
                segment in ['SMB', 'Mid Market', 'Enterprise'] and 
                booking_type in valid_source_booking_types.get(source, [])):
                
                # Get plan data for this combination
                plan_total = 0
                if plan_data is not None:
                    matching_plan = plan_data[
                        (plan_data['Source'] == source) & 
                        (plan_data['Segment'] == segment) & 
                        (plan_data['Booking Type'] == booking_type)
                    ]
                    if not matching_plan.empty:
                        plan_total = matching_plan.iloc[0]['SQL Plan']
                    else:
                        # Fallback: try to find any plan for this source/booking type combo
                        fallback_plan = plan_data[
                            (plan_data['Source'] == source) & 
                            (plan_data['Booking Type'] == booking_type)
                        ]
                        if not fallback_plan.empty:
                            plan_total = fallback_plan.iloc[0]['SQL Plan']
                
                # Calculate Plan to Date based on quarter progress
                plan_to_date = plan_total * progress
                
                # Calculate attainment and gap (both vs plan to date)
                attainment = (float(actuals) / float(plan_to_date) * 100) if plan_to_date > 0 else 0
                gap = float(actuals) - float(plan_to_date)
                
                table1_data.append({
                    'Source': source,
                    'Segment': segment,
                    'Booking Type': booking_type,
                    'Actuals': actuals,
                    'Plan to Date': plan_to_date,
                    'Attainment to Date': f"{attainment:.0f}%" if attainment == int(attainment) else f"{attainment:.1f}%",
                    'Gap to Date': gap,
                    'Q2 Plan Total': plan_total
                })
        
        # Create Table 2: Reorganize by Segment first
        table2_data = []
        segment_order = ['SMB', 'Mid Market', 'Enterprise']
        
        for segment in segment_order:
            segment_items = [item for item in table1_data if item['Segment'] == segment]
            segment_items.sort(key=lambda x: (x['Source'], x['Booking Type']))
            
            for item in segment_items:
                table2_data.append({
                    'Segment': item['Segment'],
                    'Source': item['Source'],
                    'Booking Type': item['Booking Type'],
                    'Actuals': item['Actuals'],
                    'Plan to Date': item['Plan to Date'],
                    'Attainment to Date': item['Attainment to Date'],
                    'Gap to Date': item['Gap to Date'],
                    'Q2 Plan Total': item['Q2 Plan Total']
                })
        
        # Create Table 3: Source Summary
        table3_data = []
        source_grouped = sql_data.groupby('Source').size().reset_index(name='Actuals')
        
        for _, row in source_grouped.iterrows():
            source = row['Source']
            actuals = row['Actuals']
            
            if source in VALID_SOURCES:
                # Get total plan for this source (use "Total" segment entries)
                plan_total = 0
                if plan_data is not None:
                    total_plans = plan_data[
                        (plan_data['Source'] == source) & 
                        (plan_data['Segment'] == 'Total')
                    ]
                    if not total_plans.empty:
                        plan_total = total_plans['SQL Plan'].sum()
                    else:
                        # Fallback: sum all segment-specific plans for this source
                        source_plans = plan_data[
                            (plan_data['Source'] == source) & 
                            (plan_data['Segment'] != 'Total')
                        ]
                        if not source_plans.empty:
                            plan_total = source_plans['SQL Plan'].sum()
                
                # Calculate Plan to Date based on quarter progress
                plan_to_date = plan_total * progress
                
                # Calculate attainment and gap (both vs plan to date)
                attainment = (float(actuals) / float(plan_to_date) * 100) if plan_to_date > 0 else 0
                gap = float(actuals) - float(plan_to_date)
                
                table3_data.append({
                    'Source': source,
                    'Actuals': actuals,
                    'Plan to Date': plan_to_date,
                    'Attainment to Date': f"{attainment:.0f}%" if attainment == int(attainment) else f"{attainment:.1f}%",
                    'Gap to Date': gap,
                    'Q2 Plan Total': plan_total
                })
        
        # Add subtotals and totals to tables
        table1_data_with_totals = add_hierarchical_totals(table1_data, 'Source')
        table2_data_with_totals = add_hierarchical_totals(table2_data, 'Segment')
        table3_data_with_totals = add_grand_total(table3_data)
        
        # Sort tables appropriately
        table1_data.sort(key=lambda x: (x['Source'], segment_order.index(x['Segment']), x['Booking Type']))
        table3_data.sort(key=lambda x: x['Source'])
        
        return {
            'table1': table1_data_with_totals,
            'table2': table2_data_with_totals,
            'table3': table3_data_with_totals
        }
        
    except Exception as e:
        st.error(f"Error loading master_report.csv: {e}")
        return None

@st.cache_data(ttl=3600)
def load_plan_data_from_csv():
    """Load plan data from CSV files."""
    try:
        current_quarter = get_current_quarter()
        plan_dir = "../data_sources/plan_data"
        
        # Load SQL plan data
        sql_plan_file = os.path.join(plan_dir, f"sql_plan_{current_quarter.lower()}.csv")
        if os.path.exists(sql_plan_file):
            sql_plan_df = pd.read_csv(sql_plan_file)
            return sql_plan_df
        else:
            st.warning(f"Plan data file not found: {sql_plan_file}")
            st.info("Please run 'python extract_plan_data.py' to generate plan data files.")
            return None
            
    except Exception as e:
        st.error(f"Error loading plan data: {e}")
        return None

def format_number(value):
    """Format numbers without $ signs for SQLs and SAOs."""
    if pd.isna(value) or value == '' or value == 0:
        return "0"
    
    # Handle string values that might already be formatted
    if isinstance(value, str):
        # Remove any existing formatting
        clean_value = value.replace('$', '').replace(',', '').strip()
        if clean_value == '' or clean_value == '-':
            return "0"
        try:
            numeric_value = float(clean_value)
            return f"{numeric_value:,.0f}"
        except ValueError:
            return str(value)  # Return original if can't parse
    
    # Handle numeric values
    try:
        numeric_value = float(value)
        return f"{numeric_value:,.0f}"
    except (ValueError, TypeError):
        return str(value)

def format_attainment_percentage(value):
    """Format attainment percentage with smart decimal places."""
    if pd.isna(value) or value == '':
        return ""
    
    # Parse the numeric value
    numeric_value = parse_attainment_value(value)
    
    # Determine color class
    if numeric_value >= 100:
        color_class = "attainment-high"
    elif numeric_value >= 80:
        color_class = "attainment-medium"
    else:
        color_class = "attainment-low"
    
    # Smart decimal formatting
    if numeric_value == int(numeric_value):
        # No decimals for whole numbers
        formatted_value = f"{int(numeric_value)}%"
    else:
        # Show one decimal place for non-whole numbers
        formatted_value = f"{numeric_value:.1f}%"
    
    return f'<span class="{color_class}">{formatted_value}</span>'

def parse_attainment_value(value):
    """Parse attainment value to numeric."""
    if pd.isna(value) or value == '':
        return 0
    
    # Handle string values
    if isinstance(value, str):
        # Remove % sign and other formatting
        clean_value = value.replace('%', '').strip()
        try:
            return float(clean_value)
        except ValueError:
            return 0
    
    # Handle numeric values
    try:
        return float(value)
    except (ValueError, TypeError):
        return 0

def format_gap(value):
    """Format gap values with color coding."""
    if pd.isna(value) or value == '':
        return ""
    
    # Handle string values
    if isinstance(value, str):
        clean_value = value.replace(',', '').strip()
        try:
            numeric_value = float(clean_value)
        except ValueError:
            return str(value)
    else:
        try:
            numeric_value = float(value)
        except (ValueError, TypeError):
            return str(value)
    
    # Format with color coding
    if numeric_value > 0:
        color_class = "gap-positive"
        formatted_value = f"{numeric_value:,.0f}"
    elif numeric_value < 0:
        color_class = "gap-negative"
        formatted_value = f"{numeric_value:,.0f}"
    else:
        color_class = ""
        formatted_value = "0"
    
    return f'<span class="{color_class}">{formatted_value}</span>'

def create_table_html(data, title):
    """Create HTML table with proper formatting and merged cells."""
    if not data:
        return f"<h3>{title}</h3><p>No data available</p>"
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Start HTML table
    html = f"""
    <h3>{title}</h3>
    <table class="metric-table" style="width: 100%; border-collapse: collapse; margin: 20px 0; font-size: 14px;">
        <thead>
            <tr style="background-color: #f8f9fa;">
    """
    
    # Add headers
    for col in df.columns:
        html += f'<th style="border: 1px solid #ddd; padding: 8px; text-align: center; font-weight: bold; color: #000;">{col}</th>'
    
    html += """
            </tr>
        </thead>
        <tbody>
    """
    
    # Group data for merged cells
    if title == "Source Summary":
        # For Source Summary, no grouping needed
        for _, row in df.iterrows():
            html += '<tr>'
            for col in df.columns:
                value = row[col]
                formatted_value = format_table_value(value, col)
                html += f'<td style="border: 1px solid #ddd; padding: 6px; text-align: center;">{formatted_value}</td>'
            html += '</tr>'
    else:
        # Determine grouping based on table structure
        if 'Segment' in df.columns and df.columns[0] == 'Segment':
            # Table 2: Segment × Source × Booking Type
            # Group by Segment first, then Source within each segment
            segment_groups = df.groupby('Segment', sort=False)
            
            for segment, segment_df in segment_groups:
                source_groups = segment_df.groupby('Source', sort=False)
                segment_row_count = len(segment_df)
                segment_first_row = True
                
                for source, source_df in source_groups:
                    source_rows = source_df.reset_index(drop=True)
                    source_row_count = len(source_rows)
                    
                    for i, (_, row) in enumerate(source_rows.iterrows()):
                        html += '<tr>'
                        
                        for col in df.columns:
                            value = row[col]
                            formatted_value = format_table_value(value, col)
                            
                            # Merge Segment cell across all rows in segment
                            if col == 'Segment' and segment_first_row:
                                html += f'<td style="border: 1px solid #ddd; padding: 6px; text-align: center; vertical-align: middle; font-weight: bold; font-size: 16px;" rowspan="{segment_row_count}">{formatted_value}</td>'
                                segment_first_row = False
                            elif col == 'Segment':
                                continue
                            # Merge Source cell within segment
                            elif col == 'Source' and i == 0:
                                html += f'<td style="border: 1px solid #ddd; padding: 6px; text-align: center; vertical-align: middle; font-weight: bold;" rowspan="{source_row_count}">{formatted_value}</td>'
                            elif col == 'Source' and i > 0:
                                continue
                            else:
                                html += f'<td style="border: 1px solid #ddd; padding: 6px; text-align: center;">{formatted_value}</td>'
                        
                        html += '</tr>'
        else:
            # Table 1: Source × Segment × Booking Type
            # Group by Source first, then Segment within each source
            source_groups = df.groupby('Source', sort=False)
            
            for source, source_df in source_groups:
                if 'Segment' in df.columns:
                    segment_groups = source_df.groupby('Segment', sort=False)
                    source_row_count = len(source_df)
                    source_first_row = True
                    
                    for segment, segment_df in segment_groups:
                        segment_rows = segment_df.reset_index(drop=True)
                        segment_row_count = len(segment_rows)
                        
                        for i, (_, row) in enumerate(segment_rows.iterrows()):
                            html += '<tr>'
                            
                            for col in df.columns:
                                value = row[col]
                                formatted_value = format_table_value(value, col)
                                
                                # Check if this is a total row
                                is_total_row = 'Total' in str(row.get('Source', ''))
                                cell_style = 'border: 1px solid #ddd; padding: 6px; text-align: center;'
                                
                                if is_total_row:
                                    cell_style += ' font-weight: bold; background-color: #f0f0f0;'
                                
                                # Merge Source cell across all rows in source (unless it's a total row)
                                if col == 'Source' and source_first_row and not is_total_row:
                                    html += f'<td style="{cell_style} vertical-align: middle; font-weight: bold; font-size: 16px;" rowspan="{source_row_count}">{formatted_value}</td>'
                                    source_first_row = False
                                elif col == 'Source' and not is_total_row:
                                    continue
                                # Merge Segment cell within source (unless it's a total row)
                                elif col == 'Segment' and i == 0 and not is_total_row:
                                    html += f'<td style="{cell_style} vertical-align: middle; font-weight: bold;" rowspan="{segment_row_count}">{formatted_value}</td>'
                                elif col == 'Segment' and i > 0 and not is_total_row:
                                    continue
                                else:
                                    html += f'<td style="{cell_style}">{formatted_value}</td>'
                            
                            html += '</tr>'
                else:
                    # No segment column, just merge source
                    source_rows = source_df.reset_index(drop=True)
                    source_row_count = len(source_rows)
                    
                    for i, (_, row) in enumerate(source_rows.iterrows()):
                        html += '<tr>'
                        
                        for col in df.columns:
                            value = row[col]
                            formatted_value = format_table_value(value, col)
                            
                            if col == 'Source' and i == 0:
                                html += f'<td style="border: 1px solid #ddd; padding: 6px; text-align: center; vertical-align: middle; font-weight: bold; font-size: 16px;" rowspan="{source_row_count}">{formatted_value}</td>'
                            elif col == 'Source' and i > 0:
                                continue
                            else:
                                html += f'<td style="border: 1px solid #ddd; padding: 6px; text-align: center;">{formatted_value}</td>'
                        
                        html += '</tr>'
    
    html += """
        </tbody>
    </table>
    """
    
    return html

def format_table_value(value, col):
    """Format table values based on column type."""
    if col == 'Attainment to Date':
        return format_attainment_percentage(value)
    elif col == 'Gap to Date':
        return format_gap(value)
    elif col in ['Actuals', 'Plan to Date', 'Q2 Plan Total']:
        return format_number(value)
    else:
        return str(value)

def add_hierarchical_totals(data, primary_group):
    """Add hierarchical totals (segment subtotals, source totals, grand total)."""
    if not data:
        return data
    
    result = []
    df = pd.DataFrame(data)
    
    # Sort data appropriately
    segment_order = ['SMB', 'Mid Market', 'Enterprise']
    if primary_group == 'Source':
        df = df.sort_values(['Source', 'Segment', 'Booking Type'])
        df['Segment'] = pd.Categorical(df['Segment'], categories=segment_order, ordered=True)
        df = df.sort_values(['Source', 'Segment', 'Booking Type'])
    else:
        df = df.sort_values(['Segment', 'Source', 'Booking Type'])
        df['Segment'] = pd.Categorical(df['Segment'], categories=segment_order, ordered=True)
        df = df.sort_values(['Segment', 'Source', 'Booking Type'])
    
    # Group by primary group (Source or Segment)
    primary_groups = df.groupby(primary_group, sort=False)
    
    for primary_key, primary_df in primary_groups:
        # Group by secondary group within primary
        if primary_group == 'Source':
            secondary_groups = primary_df.groupby('Segment', sort=False)
        else:
            secondary_groups = primary_df.groupby('Source', sort=False)
        
        for secondary_key, secondary_df in secondary_groups:
            # Add individual rows
            for _, row in secondary_df.iterrows():
                result.append(row.to_dict())
            
            # Add segment subtotal if more than one booking type
            if len(secondary_df) > 1:
                if primary_group == 'Source':
                    subtotal_label = f"{primary_key} {secondary_key} Total"
                else:
                    subtotal_label = f"{secondary_key} {primary_key} Total"
                
                subtotal = calculate_subtotal(secondary_df, subtotal_label, "segment")
                result.append(subtotal)
        
        # Add primary group total
        primary_total = calculate_subtotal(primary_df, f"{primary_key} Total", primary_group.lower())
        result.append(primary_total)
    
    # Add grand total
    grand_total = calculate_subtotal(df, "Grand Total", "grand")
    result.append(grand_total)
    
    return result

def add_grand_total(data):
    """Add grand total row to table data."""
    if not data:
        return data
    
    result = list(data)  # Copy existing data
    df = pd.DataFrame(data)
    
    # Add grand total
    grand_total = calculate_subtotal(df, "Total", "grand")
    result.append(grand_total)
    
    return result

def calculate_subtotal(df, label, level_type):
    """Calculate subtotal row from DataFrame."""
    subtotal = {
        'Actuals': df['Actuals'].sum(),
        'Plan to Date': df['Plan to Date'].sum(),
        'Gap to Date': df['Gap to Date'].sum(),
        'Q2 Plan Total': df['Q2 Plan Total'].sum()
    }
    
    # Set the appropriate columns for the label based on level
    if level_type == 'source':
        subtotal['Source'] = label
        subtotal['Segment'] = ''
        subtotal['Booking Type'] = ''
    elif level_type == 'segment':
        # For segment subtotals, put label in Source column, clear others
        subtotal['Source'] = label
        subtotal['Segment'] = ''
        subtotal['Booking Type'] = ''
    elif level_type == 'grand':
        subtotal['Source'] = label
        subtotal['Segment'] = ''
        subtotal['Booking Type'] = ''
    else:
        subtotal['Source'] = label
        subtotal['Segment'] = ''
        subtotal['Booking Type'] = ''
    
    # Calculate attainment based on totals
    if subtotal['Plan to Date'] > 0:
        attainment = (subtotal['Actuals'] / subtotal['Plan to Date'] * 100)
        subtotal['Attainment to Date'] = f"{attainment:.0f}%" if attainment == int(attainment) else f"{attainment:.1f}%"
    else:
        subtotal['Attainment to Date'] = "0%"
    
    return subtotal

def main():
    """Main dashboard function."""
    st.title("⚙️ Gears - Pipeline Attainment Dashboard")
    
    # Load the master report data
    sqls_data = load_master_report_data()
    
    if sqls_data is None:
        st.error("❌ Could not load master report data")
        st.stop()
    
    # Load plan data from CSV files
    plan_data = load_plan_data_from_csv()
    
    if plan_data is None:
        st.warning("⚠️ Plan data not available. Using data from Master - SQLs.csv only.")
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["📊 SQLs", "🎯 SAOs", "🚀 Pipegen"])
    
    with tab1:
        st.header("📊 SQLs - Pipeline Attainment")
        
        # Add table selection
        table_options = {
            "Source Summary": "table3",
            "Source × Segment × Booking Type": "table1",
            "Segment × Source × Booking Type": "table2"
        }
        
        selected_table = st.selectbox(
            "Select table to display:",
            options=list(table_options.keys()),
            index=0
        )
        
        # Display the selected table
        if sqls_data:
            table_key = table_options[selected_table]
            st.markdown(create_table_html(sqls_data[table_key], selected_table), 
                       unsafe_allow_html=True)
        else:
            st.error("No SQLs data available")
    
    with tab2:
        st.header("🎯 SAOs - Pipeline Attainment")
        st.info("SAOs data will be implemented in the next phase")
    
    with tab3:
        st.header("🚀 Pipegen - Pipeline Attainment")
        st.info("Pipegen data will be implemented in the next phase")

if __name__ == "__main__":
    main()