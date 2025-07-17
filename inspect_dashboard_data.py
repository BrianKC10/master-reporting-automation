#!/usr/bin/env python3
"""
Inspect the dashboard data for accuracy
"""

import pandas as pd
import gspread
from datetime import datetime

def inspect_master_report_data():
    """Inspect the master report data structure."""
    print("📊 Master Report Data Inspection")
    print("=" * 50)
    
    try:
        # Load master report
        df = pd.read_csv("master_report.csv")
        print(f"✅ Loaded master report: {df.shape[0]} rows × {df.shape[1]} columns")
        
        # Check key columns
        key_columns = ['Source', 'Segment - historical', 'Bookings Type', 'ARR Change', 'Created Date', 'SQO Date', 'SAO Date']
        for col in key_columns:
            if col in df.columns:
                print(f"✅ {col}: {df[col].notna().sum()} non-null values")
            else:
                print(f"❌ {col}: Missing column")
        
        # Check data by source
        print(f"\n📈 Data by Source:")
        source_counts = df['Source'].value_counts()
        for source, count in source_counts.items():
            print(f"  {source}: {count} records")
        
        # Check data by segment
        print(f"\n🏢 Data by Segment:")
        segment_counts = df['Segment - historical'].value_counts()
        for segment, count in segment_counts.items():
            print(f"  {segment}: {count} records")
        
        # Check ARR totals
        print(f"\n💰 ARR Change Summary:")
        total_arr = df['ARR Change'].sum()
        print(f"  Total ARR Change: ${total_arr:,.2f}")
        
        # Check by booking type
        print(f"\n📋 Data by Booking Type:")
        booking_counts = df['Bookings Type'].value_counts()
        for booking_type, count in booking_counts.items():
            print(f"  {booking_type}: {count} records")
        
        return df
        
    except Exception as e:
        print(f"❌ Error loading master report: {e}")
        return None

def inspect_google_sheets_data():
    """Inspect the Google Sheets plan data."""
    print("\n📋 Google Sheets Plan Data Inspection")
    print("=" * 50)
    
    try:
        # Connect to Google Sheets
        gc = gspread.service_account(filename='/Users/bchen/Downloads/gspread-428120-06947c66447d.json')
        sheet_id = "1H63ybz81NUq9lic620az9sI0NWIYLERPvHg7DcAQzqo"
        sheet = gc.open_by_key(sheet_id)
        
        # Get Source View worksheet
        worksheet = None
        for ws in sheet.worksheets():
            if ws.title == "Source View":
                worksheet = ws
                break
        
        if worksheet is None:
            print("❌ Source View worksheet not found")
            return None
        
        print(f"✅ Found Source View worksheet: {worksheet.row_count} rows × {worksheet.col_count} cols")
        
        # Get all values
        all_values = worksheet.get_all_values()
        
        # Extract SQL plan data for current quarter (FY26Q2)
        target_col = 12  # 2026-Q2 column
        sql_plan_data = []
        
        for i, row in enumerate(all_values):
            if len(row) > 5 and str(row[5]).strip().upper() == 'SQL':
                source = str(row[2]).strip()
                segment = str(row[3]).strip()
                booking_type = str(row[4]).strip()
                
                # Map segment names
                segment_map = {'MM': 'Mid Market', 'SMB': 'SMB', 'ENT': 'Enterprise'}
                segment = segment_map.get(segment, segment)
                
                # Get the plan value
                if len(row) > target_col:
                    plan_value_str = str(row[target_col]).strip()
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
        
        # Show sample data
        print(f"\n📊 Sample SQL Plan Data:")
        for i, entry in enumerate(sql_plan_data[:10]):
            print(f"  {i+1}. {entry['Source']} - {entry['Segment']} - {entry['Booking Type']}: {entry['SQL Plan']:,.0f}")
        
        # Summarize by source
        print(f"\n📈 Plan Data by Source:")
        source_totals = {}
        for entry in sql_plan_data:
            source = entry['Source']
            if source not in source_totals:
                source_totals[source] = 0
            source_totals[source] += entry['SQL Plan']
        
        for source, total in source_totals.items():
            print(f"  {source}: {total:,.0f}")
        
        return sql_plan_data
        
    except Exception as e:
        print(f"❌ Error inspecting Google Sheets: {e}")
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

def main():
    print("🔍 Dashboard Data Accuracy Inspection")
    print("=" * 60)
    
    current_quarter = get_current_quarter()
    print(f"📅 Current Quarter: {current_quarter}")
    
    # Inspect master report data
    master_df = inspect_master_report_data()
    
    # Inspect Google Sheets plan data
    plan_data = inspect_google_sheets_data()
    
    # Summary
    print(f"\n📋 Inspection Summary:")
    print(f"  - Master Report: {'✅ OK' if master_df is not None else '❌ FAILED'}")
    print(f"  - Google Sheets: {'✅ OK' if plan_data is not None else '❌ FAILED'}")
    print(f"  - Current Quarter: {current_quarter}")
    
    if master_df is not None and plan_data is not None:
        print(f"\n🎯 Dashboard should show:")
        print(f"  - SQLs: Actual data from master report vs Plan from Google Sheets")
        print(f"  - SAOs: Filtered by SAO Date vs 80% of SQL plan")
        print(f"  - Pipegen: Filtered by Created Date vs 200% of SQL plan")
        print(f"\n✅ Data sources are ready for dashboard!")
    else:
        print(f"\n❌ Data issues detected. Check the errors above.")

if __name__ == "__main__":
    main()