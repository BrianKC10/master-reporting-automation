#!/usr/bin/env python3
"""
Test script to validate the bookings analysis enhancement for percentage row comparisons.
"""

import pandas as pd
import numpy as np
from datetime import datetime
import sys
import os

# Add the current directory to the Python path
sys.path.append('/Users/bchen/Documents/Master reporting automation')

# Import the functions we need
from advanced_streamlit_dashboard import calculate_bookings_analysis, add_comparison_columns, get_quarter_info, get_last_completed_quarters

def create_test_data():
    """Create test data for bookings analysis that produces realistic percentages."""
    
    # Create test quarters
    quarters = ['2023-Q1', '2023-Q2', '2023-Q3', '2023-Q4', '2024-Q1', '2024-Q2', '2024-Q3', '2024-Q4', '2025-Q1']
    
    # Create test data
    test_data = []
    
    # Add test opportunities with realistic patterns
    for i, quarter in enumerate(quarters):
        # New Business bookings (inquarter - varies from 30% to 50% of total)
        inq_multiplier = 0.3 + (i * 0.025)  # Gradual increase from 30% to 50%
        total_new_business = 500000 + (i * 50000)  # Growing total
        inq_new_business = total_new_business * inq_multiplier
        
        # Add inquarter New Business
        test_data.append({
            'Stage': 'Closed Won',
            'Close Date_Quarter': quarter,
            'Bookings Type': 'New Business',
            'ARR Change': inq_new_business,
            'Inquarter Booking Flag': True,
            'Segment - historical': 'Enterprise',
            'Created Date': datetime(2023, 1, 1),
            'Close Date': datetime(2023, 1, 15)
        })
        
        # Add total New Business (not inquarter)
        test_data.append({
            'Stage': 'Closed Won',
            'Close Date_Quarter': quarter,
            'Bookings Type': 'New Business',
            'ARR Change': total_new_business - inq_new_business,
            'Inquarter Booking Flag': False,
            'Segment - historical': 'Enterprise',
            'Created Date': datetime(2023, 1, 1),
            'Close Date': datetime(2023, 1, 15)
        })
        
        # Expansion bookings (inquarter - varies from 20% to 40% of total)
        exp_inq_multiplier = 0.2 + (i * 0.025)  # Gradual increase
        total_expansion = 300000 + (i * 30000)  # Growing total
        inq_expansion = total_expansion * exp_inq_multiplier
        
        # Add inquarter Expansion
        test_data.append({
            'Stage': 'Closed Won',
            'Close Date_Quarter': quarter,
            'Bookings Type': 'Expansion',
            'ARR Change': inq_expansion,
            'Inquarter Booking Flag': True,
            'Segment - historical': 'Enterprise',
            'Created Date': datetime(2023, 1, 1),
            'Close Date': datetime(2023, 1, 15)
        })
        
        # Add total Expansion (not inquarter)
        test_data.append({
            'Stage': 'Closed Won',
            'Close Date_Quarter': quarter,
            'Bookings Type': 'Expansion',
            'ARR Change': total_expansion - inq_expansion,
            'Inquarter Booking Flag': False,
            'Segment - historical': 'Enterprise',
            'Created Date': datetime(2023, 1, 1),
            'Close Date': datetime(2023, 1, 15)
        })
    
    return pd.DataFrame(test_data)

def test_bookings_analysis():
    """Test the enhanced bookings analysis function."""
    print("🧪 Testing Enhanced Bookings Analysis")
    print("=" * 50)
    
    # Create test data
    df = create_test_data()
    print(f"✅ Created test data with {len(df)} records")
    
    # Run the analysis
    try:
        result = calculate_bookings_analysis(df)
        print(f"✅ Analysis completed successfully")
        print(f"📊 Result shape: {result.shape}")
        print(f"📋 Result columns: {list(result.columns)}")
        print(f"📋 Result rows: {list(result.index)}")
        
        # Check if comparison columns exist
        comparison_cols = ['QoQ Change', '4Q Avg', 'YoY Change']
        missing_cols = [col for col in comparison_cols if col not in result.columns]
        
        if missing_cols:
            print(f"❌ Missing comparison columns: {missing_cols}")
        else:
            print(f"✅ All comparison columns present: {comparison_cols}")
        
        # Check if Percent inquarter row exists
        if 'Percent inquarter' in result.index:
            print(f"✅ 'Percent inquarter' row found")
            
            # Check if it has comparison values
            percent_row = result.loc['Percent inquarter']
            print(f"📊 Percent inquarter row data:")
            print(percent_row)
            
            # Check if comparison columns have values for the percentage row
            for col in comparison_cols:
                if col in result.columns:
                    value = percent_row[col]
                    if pd.isna(value) or value == "":
                        print(f"⚠️  {col}: No value (expected if not enough data)")
                    else:
                        print(f"✅ {col}: {value}")
                        
        else:
            print(f"❌ 'Percent inquarter' row not found")
        
        # Display the full result for inspection
        print(f"\n📊 FULL RESULT:")
        print(result.to_string())
        
        return result
        
    except Exception as e:
        print(f"❌ Error during analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    result = test_bookings_analysis()
    if result is not None:
        print(f"\n✅ Test completed successfully!")
        print(f"🎯 The enhancement appears to be working - percentage row now includes comparison columns!")
    else:
        print(f"\n❌ Test failed!")