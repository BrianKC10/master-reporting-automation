#!/usr/bin/env python3
"""
Recreate Master - SQLs.csv from raw master_report.csv data
This generates the three tables that the dashboard expects
"""

import pandas as pd
import numpy as np

def recreate_master_sqls():
    """Recreate the Master - SQLs.csv structure from raw data."""
    
    print("🔄 Recreating Master - SQLs.csv from raw data...")
    
    # Load raw data
    df = pd.read_csv('data_sources/master_report.csv', low_memory=False)
    print(f"✅ Loaded {len(df)} rows of raw data")
    
    # Filter for SQLs (SQO Date is not null) in current quarter
    current_quarter = "2026-Q2"
    sql_data = df[
        (df['SQO Date'].notna()) & 
        (df['SQO Date_Quarter'] == current_quarter)
    ].copy()
    
    print(f"✅ Found {len(sql_data)} SQLs for {current_quarter}")
    
    # Load plan data
    plan_data = pd.read_csv('data_sources/plan_data/sql_plan_fy2026q2.csv')
    print(f"✅ Loaded {len(plan_data)} plan entries")
    
    # Create the three tables
    tables = []
    
    # Table 1: Source × Segment × Booking Type (detailed breakdown)
    print("\n📊 Creating Table 1: Source × Segment × Booking Type")
    table1_data = []
    
    # Group by Source, Segment, and Booking Type
    grouped = sql_data.groupby(['Source', 'Segment - historical', 'Bookings Type']).size().reset_index(name='Actuals')
    
    for _, row in grouped.iterrows():
        source = row['Source']
        segment = row['Segment - historical']
        booking_type = row['Bookings Type']
        actuals = row['Actuals']
        
        # Get plan data
        plan_match = plan_data[
            (plan_data['Source'] == source) & 
            (plan_data['Segment'] == segment) & 
            (plan_data['Booking Type'] == booking_type)
        ]
        
        plan_value = plan_match.iloc[0]['SQL Plan'] if not plan_match.empty else 0
        
        # Calculate metrics
        attainment = (actuals / plan_value * 100) if plan_value > 0 else 0
        gap = actuals - plan_value
        
        table1_data.append([
            f"{source}-{segment}-{booking_type}-SQL",
            source,
            segment,
            booking_type,
            '', '', '', '', actuals, '', '', '', '', '', '', '', '', '',
            f"{source}-{segment}-{booking_type}-SQL",
            source,
            segment,
            booking_type,
            actuals,
            plan_value,
            f"{attainment:.0f}%" if attainment == int(attainment) else f"{attainment:.1f}%",
            gap,
            plan_value,
            '', '',
            source,
            actuals,
            plan_value,
            f"{attainment:.0f}%" if attainment == int(attainment) else f"{attainment:.1f}%",
            gap,
            plan_value
        ])
    
    # Create headers
    headers = [
        '', 'Source', 'Segment - historical', 'Bookings Type', 
        '2025-Q2', '2025-Q3', '2025-Q4', '2026-Q1', '2026-Q2', 'Grand Total',
        '2026-Q2', '2026-Q2', '2026-Q2', '2026-Q2', '2026-Q2', '', '', '',
        'Source', 'Segment', 'Type', 'Actuals', 'Plan to Date', 'Attainment to Date', 'Gap to Date', 'Q2 Plan Total',
        '', '',
        'Source', 'Actuals', 'Plan to Date', 'Attainment to Date', 'Gap to Date', 'Q2 Plan Total'
    ]
    
    # Create DataFrame
    all_data = [headers] + table1_data
    
    # Add source summary data
    print("\n📊 Adding Table 3: Source Summary")
    source_summary = sql_data.groupby('Source').size().reset_index(name='Actuals')
    
    for _, row in source_summary.iterrows():
        source = row['Source']
        actuals = row['Actuals']
        
        # Get total plan for this source
        source_plan = plan_data[plan_data['Source'] == source]['SQL Plan'].sum()
        
        # Calculate metrics
        attainment = (actuals / source_plan * 100) if source_plan > 0 else 0
        gap = actuals - source_plan
        
        # Add to existing data structure
        source_row = [''] * 29  # Empty first 29 columns
        source_row.extend([
            source,
            actuals,
            source_plan,
            f"{attainment:.0f}%" if attainment == int(attainment) else f"{attainment:.1f}%",
            gap,
            source_plan
        ])
        
        all_data.append(source_row)
    
    # Create DataFrame and save
    df_output = pd.DataFrame(all_data)
    df_output.to_csv('data_sources/Master - SQLs.csv', index=False, header=False)
    
    print(f"✅ Created Master - SQLs.csv with {len(all_data)} rows")
    print(f"📁 Saved to: data_sources/Master - SQLs.csv")
    
    # Show summary
    print(f"\n📊 Summary:")
    print(f"  - Table 1 entries: {len(table1_data)}")
    print(f"  - Source summary entries: {len(source_summary)}")
    print(f"  - Total rows in output: {len(all_data)}")

if __name__ == "__main__":
    recreate_master_sqls()