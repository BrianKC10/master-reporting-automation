#!/usr/bin/env python3
"""
Debug SQLs parsing from Master - SQLs.csv
"""

import pandas as pd

def debug_sqls_parsing():
    """Debug the SQLs parsing logic."""
    
    print("🔍 Debug SQLs Parsing")
    print("=" * 50)
    
    try:
        # Load the CSV file
        df = pd.read_csv("Master - SQLs.csv")
        print(f"✅ Loaded CSV: {df.shape[0]} rows × {df.shape[1]} columns")
        
        # Show first few rows to understand structure
        print(f"\n📊 First 10 rows:")
        for i in range(min(10, len(df))):
            row = df.iloc[i]
            print(f"Row {i}: {[str(val)[:15] if pd.notna(val) else '' for val in row[:35]]}")
        
        # Debug Table 1 parsing (columns 18-25)
        print(f"\n🔍 Table 1 Debug (columns 18-25):")
        table1_data = []
        
        for idx, row in df.iterrows():
            if len(row) > 25 and pd.notna(row.iloc[18]) and str(row.iloc[18]).strip() != '':
                source = str(row.iloc[18]).strip()
                segment = str(row.iloc[19]).strip() if pd.notna(row.iloc[19]) else ''
                booking_type = str(row.iloc[20]).strip() if pd.notna(row.iloc[20]) else ''
                
                print(f"  Row {idx}: Source='{source}', Segment='{segment}', BookingType='{booking_type}'")
                
                if (source in ['AE', 'BDR', 'Channel', 'Marketing', 'Success'] and 
                    segment in ['SMB', 'MM', 'ENT', 'Enterprise', 'Mid Market'] and 
                    booking_type in ['New Business', 'Expansion']):
                    
                    table1_data.append({
                        'Source': source,
                        'Segment': segment,
                        'Booking Type': booking_type,
                        'Actuals': row.iloc[21] if pd.notna(row.iloc[21]) else 0
                    })
                    print(f"    ✅ Added to table1_data")
        
        print(f"\n📊 Table 1 Results: {len(table1_data)} rows")
        for item in table1_data[:5]:
            print(f"  {item}")
        
        # Debug Table 3 parsing (columns 27-32)
        print(f"\n🔍 Table 3 Debug (columns 27-32):")
        table3_data = []
        
        for idx, row in df.iterrows():
            if len(row) > 32 and pd.notna(row.iloc[27]) and str(row.iloc[27]).strip() != '':
                source = str(row.iloc[27]).strip()
                
                print(f"  Row {idx}: Source='{source}'")
                
                if source in ['AE', 'BDR', 'Channel', 'Marketing', 'Success']:
                    table3_data.append({
                        'Source': source,
                        'Actuals': row.iloc[28] if pd.notna(row.iloc[28]) else 0
                    })
                    print(f"    ✅ Added to table3_data")
        
        print(f"\n📊 Table 3 Results: {len(table3_data)} rows")
        for item in table3_data[:5]:
            print(f"  {item}")
        
        # Check specific rows we know should have data
        print(f"\n🎯 Checking specific rows with known data:")
        test_rows = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
        for row_idx in test_rows:
            if row_idx < len(df):
                row = df.iloc[row_idx]
                print(f"Row {row_idx}:")
                print(f"  Col 18: '{row.iloc[18] if len(row) > 18 else 'N/A'}'")
                print(f"  Col 19: '{row.iloc[19] if len(row) > 19 else 'N/A'}'")
                print(f"  Col 20: '{row.iloc[20] if len(row) > 20 else 'N/A'}'")
                print(f"  Col 21: '{row.iloc[21] if len(row) > 21 else 'N/A'}'")
                print(f"  Col 27: '{row.iloc[27] if len(row) > 27 else 'N/A'}'")
                print(f"  Col 28: '{row.iloc[28] if len(row) > 28 else 'N/A'}'")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_sqls_parsing()