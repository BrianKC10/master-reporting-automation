#!/usr/bin/env python3
"""
Extract Plan Data from Google Sheets
Saves plan data as CSV files for the dashboard to load quickly
"""

import gspread
import pandas as pd
import os
from datetime import datetime

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

def extract_plan_data():
    """Extract plan data from Google Sheets and save as CSV files."""
    
    print("📊 Extracting Plan Data from Google Sheets")
    print("=" * 50)
    
    try:
        # Connect to Google Sheets
        print("🔗 Connecting to Google Sheets...")
        
        # Try multiple authentication methods
        gc = None
        auth_methods = [
            ('.streamlit/secrets.toml', 'service_account'),
            ('/Users/bchen/Downloads/gspread-428120-06947c66447d.json', 'json_file')
        ]
        
        for auth_file, method in auth_methods:
            try:
                if method == 'service_account':
                    # Try Streamlit secrets format
                    import streamlit as st
                    gc = gspread.service_account_from_dict(st.secrets["gcp_service_account"])
                    print(f"✅ Connected using Streamlit secrets")
                    break
                else:
                    # Try JSON file
                    gc = gspread.service_account(filename=auth_file)
                    print(f"✅ Connected using JSON file: {auth_file}")
                    break
            except Exception as e:
                print(f"❌ Failed with {method}: {e}")
                continue
        
        if gc is None:
            raise Exception("Could not authenticate with Google Sheets")
        
        # Open the Google Sheet
        sheet_id = "1H63ybz81NUq9lic620az9sI0NWIYLERPvHg7DcAQzqo"
        sheet = gc.open_by_key(sheet_id)
        print(f"✅ Opened spreadsheet: {sheet.title}")
        
        # Get the "Source View" worksheet
        worksheet = None
        for ws in sheet.worksheets():
            if ws.title == "Source View":
                worksheet = ws
                break
        
        if worksheet is None:
            raise Exception("'Source View' worksheet not found")
        
        print(f"✅ Found 'Source View' worksheet")
        
        # Get all data
        all_values = worksheet.get_all_values()
        print(f"✅ Loaded {len(all_values)} rows from worksheet")
        
        # Convert to DataFrame
        df = pd.DataFrame(all_values[1:], columns=[f'col_{i}' for i in range(len(all_values[0]))])
        
        # Get current quarter
        current_quarter = get_current_quarter()
        print(f"📅 Current quarter: {current_quarter}")
        
        # Extract plan data
        plan_data = extract_sql_plan_data(df, current_quarter)
        
        # Save to CSV files
        save_plan_data_to_csv(plan_data, current_quarter)
        
        print(f"✅ Plan data extraction completed successfully!")
        
    except Exception as e:
        print(f"❌ Error extracting plan data: {e}")
        import traceback
        traceback.print_exc()

def extract_sql_plan_data(df, current_quarter):
    """Extract SQL plan data from the worksheet."""
    
    print(f"📊 Extracting SQL plan data for {current_quarter}")
    
    # Determine which column contains our target quarter data
    quarter_col_map = {
        'FY26Q1': 11,  # 2026-Q1
        'FY26Q2': 12,  # 2026-Q2
        'FY26Q3': 14,  # 2026-Q3
        'FY26Q4': 15   # 2026-Q4
    }
    
    target_col = quarter_col_map.get(current_quarter, 12)  # Default to Q2
    print(f"🎯 Using column {target_col} for {current_quarter} data")
    
    # Extract SQL plan data
    sql_plan_data = []
    
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
                
                if plan_value > 0:  # Only include non-zero values
                    sql_plan_data.append({
                        'Source': source,
                        'Segment': segment,
                        'Booking Type': booking_type,
                        'SQL Plan': plan_value
                    })
    
    print(f"✅ Extracted {len(sql_plan_data)} SQL plan entries")
    
    return sql_plan_data

def save_plan_data_to_csv(sql_plan_data, current_quarter):
    """Save plan data to CSV files."""
    
    print(f"💾 Saving plan data to CSV files...")
    
    # Create plan data directory if it doesn't exist
    plan_dir = "../data_sources/plan_data"
    os.makedirs(plan_dir, exist_ok=True)
    
    # Save SQL plan data
    sql_df = pd.DataFrame(sql_plan_data)
    sql_csv_path = os.path.join(plan_dir, f"sql_plan_{current_quarter.lower()}.csv")
    sql_df.to_csv(sql_csv_path, index=False)
    print(f"✅ Saved SQL plan data to {sql_csv_path}")
    
    # Create summary by source
    source_summary = sql_df.groupby('Source')['SQL Plan'].sum().reset_index()
    source_summary_path = os.path.join(plan_dir, f"source_summary_{current_quarter.lower()}.csv")
    source_summary.to_csv(source_summary_path, index=False)
    print(f"✅ Saved source summary to {source_summary_path}")
    
    # Create summary by segment
    segment_summary = sql_df.groupby(['Segment', 'Source'])['SQL Plan'].sum().reset_index()
    segment_summary_path = os.path.join(plan_dir, f"segment_summary_{current_quarter.lower()}.csv")
    segment_summary.to_csv(segment_summary_path, index=False)
    print(f"✅ Saved segment summary to {segment_summary_path}")
    
    # Display sample data
    print(f"\n📊 Sample SQL Plan Data:")
    for i, entry in enumerate(sql_plan_data[:10]):
        print(f"  {i+1:2d}. {entry['Source']:10} - {entry['Segment']:12} - {entry['Booking Type']:12}: {entry['SQL Plan']:8.0f}")
    
    # Display source totals
    print(f"\n📈 Plan Totals by Source:")
    for _, row in source_summary.iterrows():
        print(f"  {row['Source']:10}: {row['SQL Plan']:8.0f}")
    
    total_plan = source_summary['SQL Plan'].sum()
    print(f"\n💰 Total SQL Plan: {total_plan:,.0f}")

if __name__ == "__main__":
    extract_plan_data()