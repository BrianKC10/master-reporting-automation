#!/usr/bin/env python3
"""
Examine the Source View worksheet structure to understand how to parse plan data
"""

import gspread
import pandas as pd

def examine_source_view():
    """Examine the Source View worksheet structure."""
    
    print("🔍 Examining Source View Worksheet Structure")
    print("=" * 60)
    
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
            return
        
        print(f"✅ Found Source View worksheet: {worksheet.row_count} rows × {worksheet.col_count} cols")
        
        # Get all values
        all_values = worksheet.get_all_values()
        
        # Show headers
        headers = all_values[0] if all_values else []
        print(f"\n📋 Headers ({len(headers)} columns):")
        for i, header in enumerate(headers):
            if header.strip():  # Only show non-empty headers
                print(f"  {i:2d}: {header}")
        
        # Show first 20 rows to understand structure
        print(f"\n📊 First 20 rows of data:")
        for i in range(min(20, len(all_values))):
            row = all_values[i]
            # Only show first 10 columns and non-empty values
            row_display = [str(val)[:20] if val else "" for val in row[:10]]
            print(f"  Row {i:2d}: {row_display}")
        
        # Look for source-related data
        print(f"\n🔍 Looking for source-related rows...")
        source_keywords = ['AE', 'BDR', 'CHANNEL', 'MARKETING', 'SUCCESS']
        
        for i, row in enumerate(all_values[:50]):  # Check first 50 rows
            row_text = ' '.join([str(val).upper() for val in row if val])
            if any(keyword in row_text for keyword in source_keywords):
                print(f"  Row {i:2d}: {[str(val)[:15] if val else '' for val in row[:8]]}")
        
        # Look for quarter-related columns
        print(f"\n📅 Looking for quarter/plan-related columns...")
        quarter_keywords = ['Q1', 'Q2', 'Q3', 'Q4', 'PLAN', 'TARGET', 'GOAL', 'FY26']
        
        for i, header in enumerate(headers):
            if any(keyword in str(header).upper() for keyword in quarter_keywords):
                print(f"  Col {i:2d}: {header}")
        
        # Look for segment-related columns
        print(f"\n🏢 Looking for segment-related columns...")
        segment_keywords = ['ENTERPRISE', 'MID MARKET', 'SMB', 'SEGMENT']
        
        for i, header in enumerate(headers):
            if any(keyword in str(header).upper() for keyword in segment_keywords):
                print(f"  Col {i:2d}: {header}")
        
        # Show a sample of the data with context
        print(f"\n📋 Sample data grid (first 10 rows, first 15 cols):")
        print("Row |", end="")
        for i in range(min(15, len(headers))):
            print(f"{i:8d}", end="")
        print()
        print("----" + "-" * 120)
        
        for i in range(min(10, len(all_values))):
            row = all_values[i]
            print(f"{i:3d} |", end="")
            for j in range(min(15, len(row))):
                val = str(row[j])[:8] if row[j] else ""
                print(f"{val:>8}", end="")
            print()
        
        return all_values, headers
        
    except Exception as e:
        print(f"❌ Error examining Source View: {e}")
        return None, None

if __name__ == "__main__":
    examine_source_view()