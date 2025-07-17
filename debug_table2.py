#!/usr/bin/env python3
"""
Debug Table 2 parsing to understand why rows are missing
"""

import pandas as pd

def debug_table2():
    """Debug Table 2 parsing."""
    
    print("🔍 Debug Table 2 Parsing")
    print("=" * 50)
    
    try:
        # Load the CSV file
        df = pd.read_csv("Master - SQLs.csv")
        print(f"✅ Loaded CSV: {df.shape[0]} rows × {df.shape[1]} columns")
        
        # Look at columns 18-26 for Table 2
        print(f"\n📊 Table 2 Debug (columns 18-26):")
        table2_data = []
        
        VALID_SOURCES = ['AE', 'BDR', 'Channel', 'Marketing', 'Success']
        VALID_BOOKING_TYPES = ['New Business', 'Expansion']
        
        for idx, row in df.iterrows():
            if len(row) > 26 and pd.notna(row.iloc[18]) and str(row.iloc[18]).strip() != '':
                # Skip header rows and empty rows
                if str(row.iloc[18]).strip() in ['Source', '']:
                    continue
                    
                source = str(row.iloc[19]).strip() if pd.notna(row.iloc[19]) else ''
                segment = str(row.iloc[20]).strip() if pd.notna(row.iloc[20]) else ''
                booking_type = str(row.iloc[21]).strip() if pd.notna(row.iloc[21]) else ''
                
                print(f"  Row {idx}: Col18='{row.iloc[18]}', Source='{source}', Segment='{segment}', BookingType='{booking_type}'")
                
                # Check if this would be included
                if (source in VALID_SOURCES and 
                    segment in ['SMB', 'MM', 'ENT', 'Enterprise', 'Mid Market'] and 
                    booking_type in VALID_BOOKING_TYPES):
                    
                    # Map segment names
                    if segment == 'MM':
                        segment = 'Mid Market'
                    elif segment == 'ENT':
                        segment = 'Enterprise'
                    
                    table2_data.append({
                        'Row': idx,
                        'Segment': segment,
                        'Source': source,
                        'Booking Type': booking_type,
                        'Actuals': row.iloc[22] if pd.notna(row.iloc[22]) else 0,
                        'Plan to Date': row.iloc[23] if pd.notna(row.iloc[23]) else 0,
                        'Attainment to Date': row.iloc[24] if pd.notna(row.iloc[24]) else 0,
                        'Gap to Date': row.iloc[25] if pd.notna(row.iloc[25]) else 0,
                        'Q2 Plan Total': row.iloc[26] if pd.notna(row.iloc[26]) else 0
                    })
                    print(f"    ✅ Added to table2_data")
                else:
                    print(f"    ❌ Filtered out: source={source in VALID_SOURCES}, segment={segment in ['SMB', 'MM', 'ENT', 'Enterprise', 'Mid Market']}, booking_type={booking_type in VALID_BOOKING_TYPES}")
        
        print(f"\n📊 Table 2 Results: {len(table2_data)} rows")
        for item in table2_data:
            print(f"  Row {item['Row']}: {item['Source']} - {item['Segment']} - {item['Booking Type']}: Actuals={item['Actuals']}, Plan={item['Plan to Date']}")
        
        # Now let's debug Table 1 and compare
        print(f"\n📊 Table 1 Debug (columns 1-8):")
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
                    
                    table1_data.append({
                        'Row': idx,
                        'Source': source,
                        'Segment': segment,
                        'Booking Type': booking_type,
                        'Actuals': actuals
                    })
        
        print(f"\n📊 Table 1 Results: {len(table1_data)} rows")
        for item in table1_data:
            print(f"  Row {item['Row']}: {item['Source']} - {item['Segment']} - {item['Booking Type']}: Actuals={item['Actuals']}")
        
        # Compare the two tables
        print(f"\n🔍 Comparison:")
        print(f"  Table 1 rows: {len(table1_data)}")
        print(f"  Table 2 rows: {len(table2_data)}")
        
        # Check if all Table 1 entries have corresponding Table 2 entries
        table1_keys = set(f"{item['Source']}-{item['Segment']}-{item['Booking Type']}" for item in table1_data)
        table2_keys = set(f"{item['Source']}-{item['Segment']}-{item['Booking Type']}" for item in table2_data)
        
        print(f"\n📊 Table 1 unique keys: {len(table1_keys)}")
        for key in sorted(table1_keys):
            print(f"  {key}")
        
        print(f"\n📊 Table 2 unique keys: {len(table2_keys)}")
        for key in sorted(table2_keys):
            print(f"  {key}")
        
        print(f"\n🔍 Keys in Table 1 but not in Table 2:")
        for key in table1_keys - table2_keys:
            print(f"  {key}")
        
        print(f"\n🔍 Keys in Table 2 but not in Table 1:")
        for key in table2_keys - table1_keys:
            print(f"  {key}")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_table2()