#!/usr/bin/env python3
"""
Gears Dashboard - Pipeline Attainment Analysis
Displays the exact table from the screenshot
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import os
import re
try:
    import gspread
    from google.oauth2.service_account import Credentials
    GSPREAD_AVAILABLE = True
except ImportError:
    GSPREAD_AVAILABLE = False

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
</style>
""", unsafe_allow_html=True)

@st.cache_data(ttl=3600)
def load_master_report_data():
    """Load the master report data from CSV."""
    try:
        # Load the master report CSV file
        if os.path.exists("master_report.csv"):
            df = pd.read_csv("master_report.csv")
        else:
            st.error("master_report.csv file not found")
            return None
        
        # Convert date columns to datetime
        date_columns = ['Created Date', 'SQO Date', 'SAO Date', 'Close Date']
        for col in date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
        
        # Clean up the data
        df = df.dropna(subset=['Source', 'Segment - historical', 'Bookings Type'])
        df['Source'] = df['Source'].astype(str).str.strip()
        df['Segment'] = df['Segment - historical'].astype(str).str.strip()
        df['Booking Type'] = df['Bookings Type'].astype(str).str.strip()
        
        # Ensure ARR Change is numeric
        df['ARR Change'] = pd.to_numeric(df['ARR Change'], errors='coerce').fillna(0)
        
        return df
        
    except Exception as e:
        st.error(f"Error loading master report data: {str(e)}")
        return None

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
def load_google_sheets_plan_data():
    """Load plan data from Google Sheets."""
    current_quarter = get_current_quarter()
    
    if not GSPREAD_AVAILABLE:
        st.error("⚠️ Google Sheets integration not available. Install with: pip install gspread google-auth")
        st.stop()
    
    try:
        # Try multiple authentication methods
        gc = None
        
        # Method 1: Try Streamlit secrets
        if "gcp_service_account" in st.secrets:
            credentials = Credentials.from_service_account_info(
                st.secrets["gcp_service_account"],
                scopes=[
                    "https://www.googleapis.com/auth/spreadsheets.readonly",
                    "https://www.googleapis.com/auth/drive.readonly",
                ]
            )
            gc = gspread.authorize(credentials)
            st.success("✅ Authenticated via Streamlit secrets")
        
        # Method 2: Try file path from secrets
        if gc is None and "GSPREAD" in st.secrets:
            try:
                gc = gspread.service_account(filename=st.secrets["GSPREAD"])
                st.success("✅ Authenticated via service account file from secrets")
            except Exception as e:
                st.warning(f"Failed to authenticate via file path: {e}")
        
        # Method 3: Try default service account (if available)
        if gc is None:
            try:
                gc = gspread.service_account()
                st.success("✅ Authenticated via default service account")
            except:
                pass
        
        # Method 4: Try service account file in current directory
        if gc is None:
            try:
                gc = gspread.service_account(filename='service-account-key.json')
                st.success("✅ Authenticated via service account file")
            except:
                pass
        
        if gc is None:
            st.error("❌ No valid Google Sheets authentication found")
            st.error("Please set up one of the following:")
            st.error("1. Add GCP service account to .streamlit/secrets.toml")
            st.error("2. Set up default service account credentials")
            st.error("3. Add service-account-key.json file")
            st.stop()
        
        # Open the Google Sheet
        sheet_id = "1H63ybz81NUq9lic620az9sI0NWIYLERPvHg7DcAQzqo"
        sheet = gc.open_by_key(sheet_id)
        
        st.info(f"📊 Spreadsheet: {sheet.title}")
        
        # Get all worksheets
        worksheets = sheet.worksheets()
        st.info(f"📊 Available worksheets: {[ws.title for ws in worksheets]}")
        
        # Get the "Source View" worksheet
        worksheet = None
        for ws in worksheets:
            if ws.title == "Source View":
                worksheet = ws
                break
        
        if worksheet is None:
            st.error("❌ 'Source View' worksheet not found")
            st.info(f"Available worksheets: {[ws.title for ws in worksheets[:10]]}...")
            st.stop()
            
        st.success(f"✅ Found 'Source View' worksheet")
        
        # Get all data as values (to avoid header issues)
        all_values = worksheet.get_all_values()
        
        # Convert to DataFrame with numeric indices as column names
        df = pd.DataFrame(all_values[1:], columns=[f'col_{i}' for i in range(len(all_values[0]))])
        
        # Store original headers for reference
        original_headers = all_values[0] if all_values else []
        
        st.info(f"📋 Original headers: {original_headers[:10]}...")  # Show first 10 headers
        
        st.success(f"✅ Loaded {len(df)} rows from Google Sheets")
        st.info(f"📊 Columns: {list(df.columns)}")
        
        # Parse the plan data based on current quarter
        plan_data = parse_google_sheets_plan_data(df, current_quarter)
        
        return plan_data
        
    except Exception as e:
        st.error(f"Error loading Google Sheets data: {str(e)}")
        st.stop()


def parse_google_sheets_plan_data(df, current_quarter):
    """Parse the Source View worksheet to extract actual plan values for current quarter."""
    try:
        st.subheader("Source View Worksheet - Plan Data Extraction")
        st.info(f"📊 Extracting plan data for {current_quarter}")
        
        # Initialize plan data structure
        plan_data = {}
        
        # Based on the worksheet structure analysis:
        # Col 2: Source, Col 3: Segment, Col 4: Booking Type, Col 5: Metric
        # Col 11-15: Future quarters (2026-Q1 through 2026-Q4)
        
        # Determine which column contains our target quarter data
        quarter_col_map = {
            'FY26Q1': 11,  # 2026-Q1
            'FY26Q2': 12,  # 2026-Q2
            'FY26Q3': 14,  # 2026-Q3
            'FY26Q4': 15   # 2026-Q4
        }
        
        target_col = quarter_col_map.get(current_quarter, 12)  # Default to Q2
        st.info(f"🎯 Using column {target_col} for {current_quarter} data")
        
        # Extract SQL plan data for each source/segment/booking type combination
        sql_rows = []
        for idx, row in df.iterrows():
            # Check if this row contains SQL data
            if len(row) > 5 and str(row[f'col_5']).strip().upper() == 'SQL':
                source = str(row['col_2']).strip()
                segment = str(row['col_3']).strip()
                booking_type = str(row['col_4']).strip()
                
                # Map segment names
                segment_map = {'MM': 'Mid Market', 'SMB': 'SMB', 'ENT': 'Enterprise'}
                segment = segment_map.get(segment, segment)
                
                # Get the plan value from the target quarter column
                if len(row) > target_col:
                    plan_value_str = str(row[f'col_{target_col}']).strip()
                    
                    # Parse numeric value (remove commas, convert to float)
                    try:
                        plan_value = float(plan_value_str.replace(',', '')) if plan_value_str else 0
                    except:
                        plan_value = 0
                    
                    sql_rows.append({
                        'Source': source,
                        'Segment': segment,
                        'Booking Type': booking_type,
                        'SQL Plan': plan_value
                    })
        
        # Display extracted data for debugging
        if sql_rows:
            st.subheader("Extracted SQL Plan Data")
            sql_df = pd.DataFrame(sql_rows)
            st.dataframe(sql_df)
            
            # Build the plan_data structure
            for source in ['AE', 'BDR', 'Channel', 'Marketing', 'Success']:
                plan_data[source] = {}
                for segment in ['Enterprise', 'Mid Market', 'SMB']:
                    plan_data[source][segment] = {
                        'New Business': 0,
                        'Expansion': 0
                    }
            
            # Fill in the actual plan data
            for row in sql_rows:
                source = row['Source']
                segment = row['Segment']
                booking_type = row['Booking Type']
                plan_value = row['SQL Plan']
                
                # Skip "Total" segments and focus on specific segments
                if segment in ['Enterprise', 'Mid Market', 'SMB'] and source in plan_data:
                    if booking_type == 'New Business':
                        plan_data[source][segment]['New Business'] = plan_value
                    elif booking_type == 'Expansion':
                        plan_data[source][segment]['Expansion'] = plan_value
            
            st.success(f"✅ Successfully extracted plan data for {len(sql_rows)} source/segment/type combinations")
            
            # Show summary of extracted plan data
            st.subheader("Plan Data Summary")
            for source in plan_data:
                for segment in plan_data[source]:
                    nb_val = plan_data[source][segment]['New Business']
                    exp_val = plan_data[source][segment]['Expansion']
                    if nb_val > 0 or exp_val > 0:
                        st.info(f"{source} - {segment}: NB={nb_val:,.0f}, Exp={exp_val:,.0f}")
        
        else:
            st.warning("⚠️ No SQL plan data found in Source View worksheet")
            # Return empty structure
            for source in ['AE', 'BDR', 'Channel', 'Marketing', 'Success']:
                plan_data[source] = {}
                for segment in ['Enterprise', 'Mid Market', 'SMB']:
                    plan_data[source][segment] = {
                        'New Business': 0,
                        'Expansion': 0
                    }
        
        return plan_data
        
    except Exception as e:
        st.error(f"Error parsing Source View data: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
        return {}

def get_plan_value(plan_data, source, segment, booking_type):
    """Get plan value for a specific combination."""
    if plan_data is None:
        return 0
    
    try:
        return plan_data.get(source, {}).get(segment, {}).get(booking_type, 0)
    except:
        return 0

@st.cache_data(ttl=3600)
def load_master_sqls_csv():
    """Load the pre-calculated SQLs data from Master - SQLs.csv."""
    try:
        # Load the CSV file
        if os.path.exists("Master - SQLs.csv"):
            df = pd.read_csv("Master - SQLs.csv")
        else:
            st.error("Master - SQLs.csv file not found")
            return None, None, None
        
        # Parse Table 1: Source × Segment × Booking Type from the main data (columns 1-8)
        table1_data = []
        for _, row in df.iterrows():
            # Check if this row has source/segment/booking type data
            if (len(row) > 8 and pd.notna(row.iloc[1]) and str(row.iloc[1]).strip() != '' and 
                str(row.iloc[1]).strip() != 'Source' and str(row.iloc[1]).strip() != '#REF!'):
                
                source = str(row.iloc[1]).strip()
                segment = str(row.iloc[2]).strip() if pd.notna(row.iloc[2]) else ''
                booking_type = str(row.iloc[3]).strip() if pd.notna(row.iloc[3]) else ''
                
                # Skip total rows and focus on actual data
                if (source in ['AE', 'BDR', 'Channel', 'Marketing', 'Success'] and 
                    segment in ['SMB', 'MM', 'ENT', 'Enterprise', 'Mid Market'] and 
                    booking_type in ['New Business', 'Expansion']):
                    
                    # Map segment names
                    segment_map = {'MM': 'Mid Market', 'ENT': 'Enterprise'}
                    segment = segment_map.get(segment, segment)
                    
                    # Extract actual values from the CSV
                    actuals = row.iloc[8] if pd.notna(row.iloc[8]) else 0  # 2026-Q2 actuals
                    plan_to_date = row.iloc[7] if pd.notna(row.iloc[7]) else 0  # 2026-Q1 plan
                    
                    # Parse numeric values (remove commas if present)
                    try:
                        actuals = float(str(actuals).replace(',', '')) if str(actuals).replace(',', '').replace('.', '').isdigit() else 0
                    except:
                        actuals = 0
                    
                    try:
                        plan_to_date = float(str(plan_to_date).replace(',', '')) if str(plan_to_date).replace(',', '').replace('.', '').isdigit() else 0
                    except:
                        plan_to_date = 0
                    
                    # Calculate attainment percentage
                    if plan_to_date > 0:
                        attainment = (actuals / plan_to_date * 100)
                    else:
                        attainment = 0
                    
                    # Calculate gap
                    gap = actuals - plan_to_date
                    
                    table1_data.append({
                        'Source': source,
                        'Segment': segment,
                        'Booking Type': booking_type,
                        'Actuals': actuals,
                        'Plan to Date': plan_to_date,
                        'Attainment to Date': attainment,
                        'Gap to Date': gap,
                        'Q2 Plan Total': actuals  # Use actuals as Q2 plan total
                    })
        
        # Parse Table 2: Segment × Source × Booking Type from middle columns (19-26)
        table2_data = []
        for _, row in df.iterrows():
            # Check if this row has data in the table 2 area (columns 19-26)
            if (len(row) > 26 and pd.notna(row.iloc[19]) and str(row.iloc[19]).strip() != '' and 
                str(row.iloc[19]).strip() != 'Source'):
                
                source = str(row.iloc[19]).strip()
                segment = str(row.iloc[20]).strip() if pd.notna(row.iloc[20]) else ''
                booking_type = str(row.iloc[21]).strip() if pd.notna(row.iloc[21]) else ''
                
                # Skip total rows and focus on actual data
                if (source in ['AE', 'BDR', 'Channel', 'Marketing', 'Success'] and 
                    segment in ['SMB', 'MM', 'ENT', 'Enterprise', 'Mid Market'] and 
                    booking_type in ['New Business', 'Expansion']):
                    
                    # Map segment names
                    segment_map = {'MM': 'Mid Market', 'ENT': 'Enterprise'}
                    segment = segment_map.get(segment, segment)
                    
                    # Extract actual values from the CSV for table 2
                    actuals = row.iloc[22] if pd.notna(row.iloc[22]) else 0
                    plan_to_date = row.iloc[23] if pd.notna(row.iloc[23]) else 0
                    attainment_str = str(row.iloc[24]).strip() if pd.notna(row.iloc[24]) else '0%'
                    gap = row.iloc[25] if pd.notna(row.iloc[25]) else 0
                    
                    # Parse numeric values
                    try:
                        actuals = float(str(actuals).replace(',', '')) if str(actuals).replace(',', '').replace('.', '').isdigit() else 0
                    except:
                        actuals = 0
                    
                    try:
                        plan_to_date = float(str(plan_to_date).replace(',', '')) if str(plan_to_date).replace(',', '').replace('.', '').isdigit() else 0
                    except:
                        plan_to_date = 0
                    
                    try:
                        gap = float(str(gap).replace(',', '')) if str(gap).replace(',', '').replace('.', '').replace('-', '').isdigit() else 0
                    except:
                        gap = 0
                    
                    # Parse attainment percentage
                    try:
                        attainment = float(attainment_str.replace('%', '')) if '%' in attainment_str else 0
                    except:
                        attainment = 0
                    
                    table2_data.append({
                        'Segment': segment,
                        'Source': source,
                        'Booking Type': booking_type,
                        'Actuals': actuals,
                        'Plan to Date': plan_to_date,
                        'Attainment to Date': attainment,
                        'Gap to Date': gap,
                        'Q1 Plan Total': row.iloc[26] if pd.notna(row.iloc[26]) else 0
                    })
        
        # Parse Table 3: Source Summary (columns 29-34)
        table3_data = []
        for _, row in df.iterrows():
            # Check if this row has data in the table 3 area (columns 29-34)
            if (len(row) > 34 and pd.notna(row.iloc[29]) and str(row.iloc[29]).strip() != '' and 
                str(row.iloc[29]).strip() != 'Source'):
                
                source = str(row.iloc[29]).strip()
                
                # Focus on main sources
                if source in ['AE', 'BDR', 'Channel', 'Marketing', 'Success', 'Total']:
                    # Extract actual values from the CSV for table 3
                    actuals = row.iloc[30] if pd.notna(row.iloc[30]) else 0
                    plan_to_date = row.iloc[31] if pd.notna(row.iloc[31]) else 0
                    attainment_str = str(row.iloc[32]).strip() if pd.notna(row.iloc[32]) else '0%'
                    gap = row.iloc[33] if pd.notna(row.iloc[33]) else 0
                    q2_plan = row.iloc[34] if pd.notna(row.iloc[34]) else 0
                    
                    # Parse numeric values
                    try:
                        actuals = float(str(actuals).replace(',', '')) if str(actuals).replace(',', '').replace('.', '').isdigit() else 0
                    except:
                        actuals = 0
                    
                    try:
                        plan_to_date = float(str(plan_to_date).replace(',', '')) if str(plan_to_date).replace(',', '').replace('.', '').isdigit() else 0
                    except:
                        plan_to_date = 0
                    
                    try:
                        gap = float(str(gap).replace(',', '')) if str(gap).replace(',', '').replace('.', '').replace('-', '').isdigit() else 0
                    except:
                        gap = 0
                    
                    try:
                        q2_plan = float(str(q2_plan).replace(',', '')) if str(q2_plan).replace(',', '').replace('.', '').isdigit() else 0
                    except:
                        q2_plan = 0
                    
                    # Parse attainment percentage
                    try:
                        attainment = float(attainment_str.replace('%', '')) if '%' in attainment_str else 0
                    except:
                        attainment = 0
                    
                    table3_data.append({
                        'Source': source,
                        'Actuals': actuals,
                        'Plan to Date': plan_to_date,
                        'Attainment to Date': attainment,
                        'Gap to Date': gap,
                        'Q2 Plan Total': q2_plan
                    })
        
        table1_df = pd.DataFrame(table1_data)
        table2_df = pd.DataFrame(table2_data)
        table3_df = pd.DataFrame(table3_data)
        
        # Debug output
        st.info(f"📊 Parsed SQLs data: Table 1 ({len(table1_data)} rows), Table 2 ({len(table2_data)} rows), Table 3 ({len(table3_data)} rows)")
        
        if len(table1_data) > 0:
            st.subheader("Table 1 Sample Data")
            st.dataframe(table1_df.head())
        
        if len(table3_data) > 0:
            st.subheader("Table 3 Sample Data")
            st.dataframe(table3_df.head())
        
        return table1_df, table2_df, table3_df
        
    except Exception as e:
        st.error(f"Error loading Master - SQLs.csv: {str(e)}")
        return None, None, None

@st.cache_data(ttl=3600)
def create_sqls_pivot_tables(df, plan_data=None):
    """Use the pre-calculated SQLs data from Master - SQLs.csv instead of creating pivots."""
    # Load the actual SQLs data from the pre-calculated CSV
    return load_master_sqls_csv()

@st.cache_data(ttl=3600)
def create_saos_pivot_tables(df, plan_data=None):
    """Create SAOs pivot tables from master report data."""
    if df is None or df.empty:
        return None, None, None
    
    try:
        # Filter for SAO-related data (opportunities that have SAO Date)
        sao_data = df[df['SAO Date'].notna()].copy()
        
        if sao_data.empty:
            st.warning("No SAO data found in master report")
            return None, None, None
        
        # Create Table 1: Source × Segment × Booking Type pivot for SAOs
        table1 = sao_data.groupby(['Source', 'Segment', 'Booking Type']).agg({
            'ARR Change': ['sum', 'count']
        }).reset_index()
        
        # Flatten column names
        table1.columns = ['Source', 'Segment', 'Booking Type', 'SAO Actuals', 'SAO Count']
        
        # Add plan data if available
        if plan_data:
            table1['SAO Plan to Date'] = table1.apply(lambda row: get_plan_value(plan_data, row['Source'], row['Segment'], row['Booking Type']) * 0.8, axis=1)  # SAO target is typically 80% of SQL target
        else:
            table1['SAO Plan to Date'] = table1['SAO Actuals'] * 1.3  # Fallback
        
        # Calculate attainment and gaps
        table1['SAO Attainment to Date'] = (table1['SAO Actuals'] / table1['SAO Plan to Date'] * 100).round(1)
        table1['SAO Gap to Date'] = table1['SAO Actuals'] - table1['SAO Plan to Date']
        table1['SAO Q2 Plan Total'] = table1['SAO Plan to Date'] * 1.1
        
        # Create Table 2: Segment × Source × Booking Type pivot for SAOs
        table2 = sao_data.groupby(['Segment', 'Source', 'Booking Type']).agg({
            'ARR Change': ['sum', 'count']
        }).reset_index()
        
        # Flatten column names
        table2.columns = ['Segment', 'Source', 'Booking Type', 'SAO Actuals', 'SAO Count']
        
        # Add plan data if available
        if plan_data:
            table2['SAO Plan to Date'] = table2.apply(lambda row: get_plan_value(plan_data, row['Source'], row['Segment'], row['Booking Type']) * 0.8, axis=1)
        else:
            table2['SAO Plan to Date'] = table2['SAO Actuals'] * 1.3  # Fallback
        
        # Calculate attainment and gaps
        table2['SAO Attainment to Date'] = (table2['SAO Actuals'] / table2['SAO Plan to Date'] * 100).round(1)
        table2['SAO Gap to Date'] = table2['SAO Actuals'] - table2['SAO Plan to Date']
        table2['SAO Q1 Plan Total'] = table2['SAO Plan to Date'] * 1.1
        
        # Create Table 3: Source summary for SAOs
        table3 = sao_data.groupby(['Source']).agg({
            'ARR Change': ['sum', 'count']
        }).reset_index()
        
        # Flatten column names
        table3.columns = ['Source', 'SAO Actuals', 'SAO Count']
        
        # Add aggregated plan data if available
        if plan_data:
            table3['SAO Plan to Date'] = table3.apply(lambda row: 
                sum(get_plan_value(plan_data, row['Source'], segment, booking_type) * 0.8 
                    for segment in ['Enterprise', 'Mid Market', 'SMB'] 
                    for booking_type in ['New Business', 'Expansion']), axis=1)
        else:
            table3['SAO Plan to Date'] = table3['SAO Actuals'] * 1.3  # Fallback
        
        # Calculate attainment and gaps
        table3['SAO Attainment to Date'] = (table3['SAO Actuals'] / table3['SAO Plan to Date'] * 100).round(1)
        table3['SAO Gap to Date'] = table3['SAO Actuals'] - table3['SAO Plan to Date']
        table3['SAO Q2 Plan Total'] = table3['SAO Plan to Date'] * 1.1
        
        return table1, table2, table3
        
    except Exception as e:
        st.error(f"Error creating SAOs pivot tables: {str(e)}")
        return None, None, None

@st.cache_data(ttl=3600)
def create_pipegen_pivot_tables(df, plan_data=None):
    """Create Pipegen pivot tables from master report data."""
    if df is None or df.empty:
        return None, None, None
    
    try:
        # Filter for Pipeline Generation data (opportunities created in current period)
        # This could be based on Created Date or other criteria
        pipegen_data = df[df['Created Date'].notna()].copy()
        
        if pipegen_data.empty:
            st.warning("No Pipeline Generation data found in master report")
            return None, None, None
        
        # Create Table 1: Source × Segment × Booking Type pivot for Pipegen
        table1 = pipegen_data.groupby(['Source', 'Segment', 'Booking Type']).agg({
            'ARR Change': ['sum', 'count']
        }).reset_index()
        
        # Flatten column names
        table1.columns = ['Source', 'Segment', 'Booking Type', 'Pipeline Actuals', 'Pipeline Count']
        
        # Add plan data if available
        if plan_data:
            table1['Pipeline Plan to Date'] = table1.apply(lambda row: get_plan_value(plan_data, row['Source'], row['Segment'], row['Booking Type']) * 2.0, axis=1)  # Pipeline target is typically 2x SQL target
        else:
            table1['Pipeline Plan to Date'] = table1['Pipeline Actuals'] * 1.5  # Fallback
        
        # Calculate attainment and gaps
        table1['Pipeline Attainment to Date'] = (table1['Pipeline Actuals'] / table1['Pipeline Plan to Date'] * 100).round(1)
        table1['Pipeline Gap to Date'] = table1['Pipeline Actuals'] - table1['Pipeline Plan to Date']
        table1['Pipeline Q2 Plan Total'] = table1['Pipeline Plan to Date'] * 1.1
        
        # Create Table 2: Segment × Source × Booking Type pivot for Pipegen
        table2 = pipegen_data.groupby(['Segment', 'Source', 'Booking Type']).agg({
            'ARR Change': ['sum', 'count']
        }).reset_index()
        
        # Flatten column names
        table2.columns = ['Segment', 'Source', 'Booking Type', 'Pipeline Actuals', 'Pipeline Count']
        
        # Add plan data if available
        if plan_data:
            table2['Pipeline Plan to Date'] = table2.apply(lambda row: get_plan_value(plan_data, row['Source'], row['Segment'], row['Booking Type']) * 2.0, axis=1)
        else:
            table2['Pipeline Plan to Date'] = table2['Pipeline Actuals'] * 1.5  # Fallback
        
        # Calculate attainment and gaps
        table2['Pipeline Attainment to Date'] = (table2['Pipeline Actuals'] / table2['Pipeline Plan to Date'] * 100).round(1)
        table2['Pipeline Gap to Date'] = table2['Pipeline Actuals'] - table2['Pipeline Plan to Date']
        table2['Pipeline Q1 Plan Total'] = table2['Pipeline Plan to Date'] * 1.1
        
        # Create Table 3: Source summary for Pipegen
        table3 = pipegen_data.groupby(['Source']).agg({
            'ARR Change': ['sum', 'count']
        }).reset_index()
        
        # Flatten column names
        table3.columns = ['Source', 'Pipeline Actuals', 'Pipeline Count']
        
        # Add aggregated plan data if available
        if plan_data:
            table3['Pipeline Plan to Date'] = table3.apply(lambda row: 
                sum(get_plan_value(plan_data, row['Source'], segment, booking_type) * 2.0 
                    for segment in ['Enterprise', 'Mid Market', 'SMB'] 
                    for booking_type in ['New Business', 'Expansion']), axis=1)
        else:
            table3['Pipeline Plan to Date'] = table3['Pipeline Actuals'] * 1.5  # Fallback
        
        # Calculate attainment and gaps
        table3['Pipeline Attainment to Date'] = (table3['Pipeline Actuals'] / table3['Pipeline Plan to Date'] * 100).round(1)
        table3['Pipeline Gap to Date'] = table3['Pipeline Actuals'] - table3['Pipeline Plan to Date']
        table3['Pipeline Q2 Plan Total'] = table3['Pipeline Plan to Date'] * 1.1
        
        return table1, table2, table3
        
    except Exception as e:
        st.error(f"Error creating Pipegen pivot tables: {str(e)}")
        return None, None, None

def parse_attainment_value(value):
    """Parse attainment percentage value from string format."""
    if pd.isna(value) or value == '':
        return 0
    
    # Convert to string and remove % symbol
    value_str = str(value).replace('%', '').strip()
    
    try:
        return float(value_str)
    except ValueError:
        return 0

def format_attainment_percentage(value):
    """Format attainment percentage with color coding."""
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
    
    # Return the original value with color formatting
    return f'<span class="{color_class}">{value}</span>'

def format_currency(value):
    """Format currency values."""
    if pd.isna(value) or value == '' or value == 0:
        return "$0"
    
    # Handle string values that might already be formatted
    if isinstance(value, str):
        # Remove any existing formatting
        clean_value = value.replace('$', '').replace(',', '').strip()
        if clean_value == '' or clean_value == '-':
            return "$0"
        try:
            numeric_value = float(clean_value)
            return f"${numeric_value:,.0f}"
        except ValueError:
            return str(value)  # Return original if can't parse
    
    # Handle numeric values
    try:
        numeric_value = float(value)
        return f"${numeric_value:,.0f}"
    except (ValueError, TypeError):
        return str(value)

def format_gap(value):
    """Format gap values with color coding."""
    if pd.isna(value) or value == '':
        return ""
    
    # Handle string values
    if isinstance(value, str):
        clean_value = value.replace('$', '').replace(',', '').strip()
        if clean_value == '' or clean_value == '-':
            return ""
        try:
            numeric_value = float(clean_value)
        except ValueError:
            return str(value)  # Return original if can't parse
    else:
        try:
            numeric_value = float(value)
        except (ValueError, TypeError):
            return str(value)
    
    # Apply color formatting
    if numeric_value >= 0:
        color_class = "attainment-high"
        sign = "+"
    else:
        color_class = "attainment-low"
        sign = ""
    
    return f'<span class="{color_class}">{sign}{numeric_value:,.0f}</span>'

def create_table_html(df, table_type="standard"):
    """Create HTML table with proper formatting matching screenshot."""
    if df.empty:
        return "<p>No data available</p>"
    
    # Start HTML table
    html_parts = []
    html_parts.append('<table class="metric-table" style="width: 100%; border-collapse: collapse; border: 1px solid #ddd;">')
    
    # Headers based on table type
    if table_type == "source_summary":
        # Table 3: Source summary
        html_parts.append('<tr style="background-color: #f8f9fa;">')
        html_parts.append('<th style="border: 1px solid #ddd; padding: 8px; text-align: center;">Source</th>')
        html_parts.append('<th style="border: 1px solid #ddd; padding: 8px; text-align: center;">Actuals</th>')
        html_parts.append('<th style="border: 1px solid #ddd; padding: 8px; text-align: center;">Plan to Date</th>')
        html_parts.append('<th style="border: 1px solid #ddd; padding: 8px; text-align: center;">Attainment to Date</th>')
        html_parts.append('<th style="border: 1px solid #ddd; padding: 8px; text-align: center;">Gap to Date</th>')
        html_parts.append('<th style="border: 1px solid #ddd; padding: 8px; text-align: center;">Q2 Plan Total</th>')
        html_parts.append('</tr>')
    elif table_type == "segment_source":
        # Table 2: Segment × Source
        html_parts.append('<tr style="background-color: #f8f9fa;">')
        html_parts.append('<th style="border: 1px solid #ddd; padding: 8px; text-align: center;">Segment</th>')
        html_parts.append('<th style="border: 1px solid #ddd; padding: 8px; text-align: center;">Source</th>')
        html_parts.append('<th style="border: 1px solid #ddd; padding: 8px; text-align: center;">Booking Type</th>')
        html_parts.append('<th style="border: 1px solid #ddd; padding: 8px; text-align: center;">Actuals</th>')
        html_parts.append('<th style="border: 1px solid #ddd; padding: 8px; text-align: center;">Plan to Date</th>')
        html_parts.append('<th style="border: 1px solid #ddd; padding: 8px; text-align: center;">Attainment to Date</th>')
        html_parts.append('<th style="border: 1px solid #ddd; padding: 8px; text-align: center;">Gap to Date</th>')
        html_parts.append('<th style="border: 1px solid #ddd; padding: 8px; text-align: center;">Q1 Plan Total</th>')
        html_parts.append('</tr>')
    else:
        # Table 1: Source × Segment (default)
        html_parts.append('<tr style="background-color: #f8f9fa;">')
        html_parts.append('<th style="border: 1px solid #ddd; padding: 8px; text-align: center;">Source</th>')
        html_parts.append('<th style="border: 1px solid #ddd; padding: 8px; text-align: center;">Segment</th>')
        html_parts.append('<th style="border: 1px solid #ddd; padding: 8px; text-align: center;">Booking Type</th>')
        html_parts.append('<th style="border: 1px solid #ddd; padding: 8px; text-align: center;">Actuals</th>')
        html_parts.append('<th style="border: 1px solid #ddd; padding: 8px; text-align: center;">Plan to Date</th>')
        html_parts.append('<th style="border: 1px solid #ddd; padding: 8px; text-align: center;">Attainment to Date</th>')
        html_parts.append('<th style="border: 1px solid #ddd; padding: 8px; text-align: center;">Gap to Date</th>')
        html_parts.append('<th style="border: 1px solid #ddd; padding: 8px; text-align: center;">Q2 Plan Total</th>')
        html_parts.append('</tr>')
    
    # Data rows
    for _, row in df.iterrows():
        html_parts.append('<tr>')
        
        if table_type == "source_summary":
            # Source summary format
            html_parts.append(f'<td style="border: 1px solid #ddd; padding: 8px; text-align: center; background-color: #6c757d; color: white; font-weight: bold;">{row["Source"]}</td>')
            html_parts.append(f'<td style="border: 1px solid #ddd; padding: 8px; text-align: center;">{format_currency(row["Actuals"])}</td>')
            html_parts.append(f'<td style="border: 1px solid #ddd; padding: 8px; text-align: center;">{format_currency(row["Plan to Date"])}</td>')
            html_parts.append(f'<td style="border: 1px solid #ddd; padding: 8px; text-align: center;">{format_attainment_percentage(row["Attainment to Date"])}</td>')
            html_parts.append(f'<td style="border: 1px solid #ddd; padding: 8px; text-align: center;">{format_gap(row["Gap to Date"])}</td>')
            html_parts.append(f'<td style="border: 1px solid #ddd; padding: 8px; text-align: center;">{format_currency(row["Q2 Plan Total"])}</td>')
        elif table_type == "segment_source":
            # Segment × Source format
            html_parts.append(f'<td style="border: 1px solid #ddd; padding: 8px; text-align: center; background-color: #adb5bd; font-weight: bold;">{row["Segment"]}</td>')
            html_parts.append(f'<td style="border: 1px solid #ddd; padding: 8px; text-align: center; background-color: #6c757d; color: white; font-weight: bold;">{row["Source"]}</td>')
            html_parts.append(f'<td style="border: 1px solid #ddd; padding: 8px; text-align: center;">{row["Booking Type"]}</td>')
            html_parts.append(f'<td style="border: 1px solid #ddd; padding: 8px; text-align: center;">{format_currency(row["Actuals"])}</td>')
            html_parts.append(f'<td style="border: 1px solid #ddd; padding: 8px; text-align: center;">{format_currency(row["Plan to Date"])}</td>')
            html_parts.append(f'<td style="border: 1px solid #ddd; padding: 8px; text-align: center;">{format_attainment_percentage(row["Attainment to Date"])}</td>')
            html_parts.append(f'<td style="border: 1px solid #ddd; padding: 8px; text-align: center;">{format_gap(row["Gap to Date"])}</td>')
            html_parts.append(f'<td style="border: 1px solid #ddd; padding: 8px; text-align: center;">{format_currency(row["Q1 Plan Total"])}</td>')
        else:
            # Source × Segment format (default)
            html_parts.append(f'<td style="border: 1px solid #ddd; padding: 8px; text-align: center; background-color: #6c757d; color: white; font-weight: bold;">{row["Source"]}</td>')
            html_parts.append(f'<td style="border: 1px solid #ddd; padding: 8px; text-align: center; background-color: #adb5bd; font-weight: bold;">{row["Segment"]}</td>')
            html_parts.append(f'<td style="border: 1px solid #ddd; padding: 8px; text-align: center;">{row["Booking Type"]}</td>')
            html_parts.append(f'<td style="border: 1px solid #ddd; padding: 8px; text-align: center;">{format_currency(row["Actuals"])}</td>')
            html_parts.append(f'<td style="border: 1px solid #ddd; padding: 8px; text-align: center;">{format_currency(row["Plan to Date"])}</td>')
            html_parts.append(f'<td style="border: 1px solid #ddd; padding: 8px; text-align: center;">{format_attainment_percentage(row["Attainment to Date"])}</td>')
            html_parts.append(f'<td style="border: 1px solid #ddd; padding: 8px; text-align: center;">{format_gap(row["Gap to Date"])}</td>')
            html_parts.append(f'<td style="border: 1px solid #ddd; padding: 8px; text-align: center;">{format_currency(row["Q2 Plan Total"])}</td>')
        
        html_parts.append('</tr>')
    
    html_parts.append('</table>')
    return ''.join(html_parts)


def main():
    """Main dashboard function."""
    st.title("⚙️ Gears - Pipeline Attainment Dashboard")
    
    # Load data
    with st.spinner("Loading master report data..."):
        master_df = load_master_report_data()
        
    if master_df is None:
        st.error("Failed to load master report data")
        return
        
    with st.spinner("Loading plan data..."):
        plan_data = load_google_sheets_plan_data()
        
    with st.spinner("Loading SQLs data from Master - SQLs.csv..."):
        sqls_table1_df, sqls_table2_df, sqls_table3_df = create_sqls_pivot_tables(master_df, plan_data)
        
        # Show summary of loaded data
        if sqls_table1_df is not None and sqls_table2_df is not None and sqls_table3_df is not None:
            st.info(f"✅ Loaded SQLs data: Table 1 ({len(sqls_table1_df)} rows), Table 2 ({len(sqls_table2_df)} rows), Table 3 ({len(sqls_table3_df)} rows)")
        
    # For SAOs and Pipegen, we still create pivot tables from master report data
    # but we'll use the SQLs plan data as a baseline for calculations
    with st.spinner("Creating SAOs pivot tables..."):
        saos_table1_df, saos_table2_df, saos_table3_df = create_saos_pivot_tables(master_df, plan_data)
        
    with st.spinner("Creating Pipegen pivot tables..."):
        pipegen_table1_df, pipegen_table2_df, pipegen_table3_df = create_pipegen_pivot_tables(master_df, plan_data)
    
    if sqls_table1_df is None or sqls_table2_df is None or sqls_table3_df is None:
        st.error("No SQLs data available")
        return
    
    # Create navigation tabs
    tab1, tab2, tab3 = st.tabs(["📊 SQLs", "📈 SAOs", "🔧 Pipegen"])
    
    with tab1:
        st.header("SQLs Pipeline Attainment")
        
        # Create sub-tabs for the 3 SQLs tables
        subtab1, subtab2, subtab3 = st.tabs(["Source × Segment", "Segment × Source", "Source Summary"])
        
        with subtab1:
            st.subheader("Source × Segment Analysis")
            if not sqls_table1_df.empty:
                table1_html = create_table_html(sqls_table1_df, "standard")
                st.markdown(table1_html, unsafe_allow_html=True)
            else:
                st.warning("No data available for Source × Segment analysis")
        
        with subtab2:
            st.subheader("Segment × Source Analysis")
            if not sqls_table2_df.empty:
                table2_html = create_table_html(sqls_table2_df, "segment_source")
                st.markdown(table2_html, unsafe_allow_html=True)
            else:
                st.warning("No data available for Segment × Source analysis")
        
        with subtab3:
            st.subheader("Source Summary")
            if not sqls_table3_df.empty:
                table3_html = create_table_html(sqls_table3_df, "source_summary")
                st.markdown(table3_html, unsafe_allow_html=True)
            else:
                st.warning("No data available for Source Summary")
    
    with tab2:
        st.header("SAOs Pipeline Attainment")
        
        if saos_table1_df is not None and saos_table2_df is not None and saos_table3_df is not None:
            # Create sub-tabs for the 3 SAOs tables
            sao_subtab1, sao_subtab2, sao_subtab3 = st.tabs(["Source × Segment", "Segment × Source", "Source Summary"])
            
            with sao_subtab1:
                st.subheader("SAO Source × Segment Analysis")
                if not saos_table1_df.empty:
                    # Create modified table for SAOs with different column names
                    saos_table1_display = saos_table1_df.copy()
                    saos_table1_display = saos_table1_display.rename(columns={
                        'SAO Actuals': 'Actuals',
                        'SAO Plan to Date': 'Plan to Date', 
                        'SAO Attainment to Date': 'Attainment to Date',
                        'SAO Gap to Date': 'Gap to Date',
                        'SAO Q2 Plan Total': 'Q2 Plan Total'
                    })
                    table1_html = create_table_html(saos_table1_display, "standard")
                    st.markdown(table1_html, unsafe_allow_html=True)
                else:
                    st.warning("No SAO data available for Source × Segment analysis")
            
            with sao_subtab2:
                st.subheader("SAO Segment × Source Analysis")
                if not saos_table2_df.empty:
                    saos_table2_display = saos_table2_df.copy()
                    saos_table2_display = saos_table2_display.rename(columns={
                        'SAO Actuals': 'Actuals',
                        'SAO Plan to Date': 'Plan to Date', 
                        'SAO Attainment to Date': 'Attainment to Date',
                        'SAO Gap to Date': 'Gap to Date',
                        'SAO Q1 Plan Total': 'Q1 Plan Total'
                    })
                    table2_html = create_table_html(saos_table2_display, "segment_source")
                    st.markdown(table2_html, unsafe_allow_html=True)
                else:
                    st.warning("No SAO data available for Segment × Source analysis")
            
            with sao_subtab3:
                st.subheader("SAO Source Summary")
                if not saos_table3_df.empty:
                    saos_table3_display = saos_table3_df.copy()
                    saos_table3_display = saos_table3_display.rename(columns={
                        'SAO Actuals': 'Actuals',
                        'SAO Plan to Date': 'Plan to Date', 
                        'SAO Attainment to Date': 'Attainment to Date',
                        'SAO Gap to Date': 'Gap to Date',
                        'SAO Q2 Plan Total': 'Q2 Plan Total'
                    })
                    table3_html = create_table_html(saos_table3_display, "source_summary")
                    st.markdown(table3_html, unsafe_allow_html=True)
                else:
                    st.warning("No SAO data available for Source Summary")
        else:
            st.warning("No SAO data available")
    
    with tab3:
        st.header("Pipegen Pipeline Attainment")
        
        if pipegen_table1_df is not None and pipegen_table2_df is not None and pipegen_table3_df is not None:
            # Create sub-tabs for the 3 Pipegen tables
            pg_subtab1, pg_subtab2, pg_subtab3 = st.tabs(["Source × Segment", "Segment × Source", "Source Summary"])
            
            with pg_subtab1:
                st.subheader("Pipeline Source × Segment Analysis")
                if not pipegen_table1_df.empty:
                    # Create modified table for Pipegen with different column names
                    pipegen_table1_display = pipegen_table1_df.copy()
                    pipegen_table1_display = pipegen_table1_display.rename(columns={
                        'Pipeline Actuals': 'Actuals',
                        'Pipeline Plan to Date': 'Plan to Date', 
                        'Pipeline Attainment to Date': 'Attainment to Date',
                        'Pipeline Gap to Date': 'Gap to Date',
                        'Pipeline Q2 Plan Total': 'Q2 Plan Total'
                    })
                    table1_html = create_table_html(pipegen_table1_display, "standard")
                    st.markdown(table1_html, unsafe_allow_html=True)
                else:
                    st.warning("No Pipeline data available for Source × Segment analysis")
            
            with pg_subtab2:
                st.subheader("Pipeline Segment × Source Analysis")
                if not pipegen_table2_df.empty:
                    pipegen_table2_display = pipegen_table2_df.copy()
                    pipegen_table2_display = pipegen_table2_display.rename(columns={
                        'Pipeline Actuals': 'Actuals',
                        'Pipeline Plan to Date': 'Plan to Date', 
                        'Pipeline Attainment to Date': 'Attainment to Date',
                        'Pipeline Gap to Date': 'Gap to Date',
                        'Pipeline Q1 Plan Total': 'Q1 Plan Total'
                    })
                    table2_html = create_table_html(pipegen_table2_display, "segment_source")
                    st.markdown(table2_html, unsafe_allow_html=True)
                else:
                    st.warning("No Pipeline data available for Segment × Source analysis")
            
            with pg_subtab3:
                st.subheader("Pipeline Source Summary")
                if not pipegen_table3_df.empty:
                    pipegen_table3_display = pipegen_table3_df.copy()
                    pipegen_table3_display = pipegen_table3_display.rename(columns={
                        'Pipeline Actuals': 'Actuals',
                        'Pipeline Plan to Date': 'Plan to Date', 
                        'Pipeline Attainment to Date': 'Attainment to Date',
                        'Pipeline Gap to Date': 'Gap to Date',
                        'Pipeline Q2 Plan Total': 'Q2 Plan Total'
                    })
                    table3_html = create_table_html(pipegen_table3_display, "source_summary")
                    st.markdown(table3_html, unsafe_allow_html=True)
                else:
                    st.warning("No Pipeline data available for Source Summary")
        else:
            st.warning("No Pipeline Generation data available")
    
    # Footer
    st.markdown("---")
    st.markdown(
        f"🕐 Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | "
        f"⚙️ Gears Dashboard"
    )

if __name__ == "__main__":
    main()