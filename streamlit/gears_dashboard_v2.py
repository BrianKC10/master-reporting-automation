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

@st.cache_data(ttl=3600)
def load_master_sqls_csv():
    """Load the Master - SQLs.csv file and parse the three tables."""
    try:
        # Load the CSV file
        df = pd.read_csv("../data_sources/Master - SQLs.csv")
        
        # Load plan data to merge with Table 1
        plan_data = load_plan_data_from_csv()
        
        # Parse Table 1: Source × Segment × Booking Type (columns 1-3 are source, segment, booking type)
        table1_data = []
        for idx, row in df.iterrows():
            if len(row) > 8 and pd.notna(row.iloc[1]) and str(row.iloc[1]).strip() != '':
                source = str(row.iloc[1]).strip()
                segment = str(row.iloc[2]).strip() if pd.notna(row.iloc[2]) else ''
                booking_type = str(row.iloc[3]).strip() if pd.notna(row.iloc[3]) else ''
                
                if (source in VALID_SOURCES and 
                    segment in ['SMB', 'MM', 'ENT', 'Enterprise', 'Mid Market'] and 
                    booking_type in VALID_BOOKING_TYPES):
                    
                    # Map segment names
                    if segment == 'MM':
                        segment = 'Mid Market'
                    elif segment == 'ENT':
                        segment = 'Enterprise'
                    
                    # Get 2026-Q2 actuals from column 8 
                    actuals = row.iloc[8] if pd.notna(row.iloc[8]) else 0
                    
                    # Get plan data for this combination
                    plan_value = 0
                    if plan_data is not None:
                        matching_plan = plan_data[
                            (plan_data['Source'] == source) & 
                            (plan_data['Segment'] == segment) & 
                            (plan_data['Booking Type'] == booking_type)
                        ]
                        if not matching_plan.empty:
                            plan_value = matching_plan.iloc[0]['SQL Plan']
                    
                    # Calculate attainment and gap
                    attainment = (float(actuals) / float(plan_value) * 100) if plan_value > 0 else 0
                    gap = float(actuals) - float(plan_value)
                    
                    table1_data.append({
                        'Source': source,
                        'Segment': segment,
                        'Booking Type': booking_type,
                        'Actuals': actuals,
                        'Plan to Date': plan_value,
                        'Attainment to Date': f"{attainment:.0f}%" if attainment == int(attainment) else f"{attainment:.1f}%",
                        'Gap to Date': gap,
                        'Q2 Plan Total': plan_value
                    })
        
        # For Table 2, we need to reorganize Table 1 data by Segment first
        table2_data = []
        segment_order = ['SMB', 'Mid Market', 'Enterprise']
        
        # Reorganize the data by segment
        for segment in segment_order:
            segment_items = [item for item in table1_data if item['Segment'] == segment]
            # Sort by source and booking type
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
        
        # Parse Table 3: Source Summary (columns 29-34)
        table3_data = []
        for idx, row in df.iterrows():
            if len(row) > 34 and pd.notna(row.iloc[29]) and str(row.iloc[29]).strip() != '':
                source = str(row.iloc[29]).strip()
                
                if source in VALID_SOURCES:
                    table3_data.append({
                        'Source': source,
                        'Actuals': row.iloc[30] if pd.notna(row.iloc[30]) else 0,
                        'Plan to Date': row.iloc[31] if pd.notna(row.iloc[31]) else 0,
                        'Attainment to Date': row.iloc[32] if pd.notna(row.iloc[32]) else 0,
                        'Gap to Date': row.iloc[33] if pd.notna(row.iloc[33]) else 0,
                        'Q2 Plan Total': row.iloc[34] if pd.notna(row.iloc[34]) else 0
                    })
        
        # Sort tables appropriately
        # Table 1: Sort by source, then segment (SMB, MM, ENT), then booking type
        table1_data.sort(key=lambda x: (x['Source'], segment_order.index(x['Segment']), x['Booking Type']))
        
        # Table 3: Sort by source
        table3_data.sort(key=lambda x: x['Source'])
        
        return {
            'table1': table1_data,
            'table2': table2_data,
            'table3': table3_data
        }
        
    except Exception as e:
        st.error(f"Error loading Master - SQLs.csv: {e}")
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
    """Create HTML table with proper formatting."""
    if not data:
        return f"<h3>{title}</h3><p>No data available</p>"
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Start HTML table
    html = f"""
    <h3>{title}</h3>
    <table class="metric-table" style="width: 100%; border-collapse: collapse; margin: 20px 0;">
        <thead>
            <tr style="background-color: #f8f9fa;">
    """
    
    # Add headers
    for col in df.columns:
        html += f'<th style="border: 1px solid #ddd; padding: 8px; text-align: center;">{col}</th>'
    
    html += """
            </tr>
        </thead>
        <tbody>
    """
    
    # Add data rows
    for _, row in df.iterrows():
        html += '<tr>'
        for col in df.columns:
            value = row[col]
            
            # Apply appropriate formatting based on column type
            if col == 'Attainment to Date':
                formatted_value = format_attainment_percentage(value)
            elif col == 'Gap to Date':
                formatted_value = format_gap(value)
            elif col in ['Actuals', 'Plan to Date', 'Q2 Plan Total']:
                formatted_value = format_number(value)
            else:
                formatted_value = str(value)
            
            html += f'<td style="border: 1px solid #ddd; padding: 6px; text-align: center;">{formatted_value}</td>'
        
        html += '</tr>'
    
    html += """
        </tbody>
    </table>
    """
    
    return html

def main():
    """Main dashboard function."""
    st.title("⚙️ Gears - Pipeline Attainment Dashboard")
    
    # Load the Master - SQLs.csv data
    sqls_data = load_master_sqls_csv()
    
    if sqls_data is None:
        st.error("❌ Could not load Master - SQLs.csv data")
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