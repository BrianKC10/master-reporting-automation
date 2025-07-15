#!/usr/bin/env python3
"""
Demo Data Generator
Creates sample data for testing the Streamlit dashboard without Salesforce connection.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

def generate_demo_data(num_records=5000):
    """Generate realistic demo data that matches the Salesforce report structure."""
    
    # Set random seed for reproducibility
    np.random.seed(42)
    random.seed(42)
    
    # Define realistic values
    sources = ['AE', 'BDR', 'Channel', 'Marketing', 'Success']
    segments = ['Enterprise', 'Mid Market', 'SMB']
    booking_types = ['New Business', 'Expansion']
    stages = ['Prospecting', 'Qualification', 'Proposal', 'Negotiation', 'Closed Won', 'Closed Lost']
    
    # Generate base data
    data = []
    start_date = datetime.now() - timedelta(days=730)  # 2 years of data
    
    for i in range(num_records):
        # Generate dates
        created_date = start_date + timedelta(days=random.randint(0, 730))
        
        # Generate other dates based on created date
        sqo_date = created_date + timedelta(days=random.randint(0, 14)) if random.random() > 0.3 else None
        sao_date = sqo_date + timedelta(days=random.randint(0, 21)) if sqo_date and random.random() > 0.4 else None
        close_date = sao_date + timedelta(days=random.randint(0, 90)) if sao_date and random.random() > 0.5 else None
        
        # Generate stage based on progression
        if close_date:
            stage = 'Closed Won' if random.random() > 0.25 else 'Closed Lost'
        elif sao_date:
            stage = random.choice(['Proposal', 'Negotiation'])
        elif sqo_date:
            stage = random.choice(['Qualification', 'Proposal'])
        else:
            stage = random.choice(['Prospecting', 'Qualification'])
        
        # Generate ARR based on segment and stage
        segment = random.choice(segments)
        if segment == 'Enterprise':
            base_arr = random.randint(50000, 500000)
        elif segment == 'Mid Market':
            base_arr = random.randint(10000, 100000)
        else:  # SMB
            base_arr = random.randint(1000, 25000)
        
        # Adjust ARR based on booking type
        booking_type = random.choice(booking_types)
        if booking_type == 'Expansion':
            arr_change = base_arr * random.uniform(0.2, 1.0)
        else:
            arr_change = base_arr
        
        # Only assign ARR if closed won
        if stage != 'Closed Won':
            arr_change = 0
        
        record = {
            'SFDC ID 18 Digit': f'00Q{i:015d}',
            'Created Date': created_date,
            'SQO Date': sqo_date,
            'SAO Date': sao_date,
            'Close Date': close_date,
            'Timestamp: Solution Validation': sao_date,
            'Source': random.choice(sources),
            'Segment - historical': segment,
            'Bookings Type': booking_type,
            'Stage': stage,
            'ARR Change': arr_change,
            'Account Name': f'Account_{i:04d}',
            'Opportunity Name': f'Opportunity_{i:04d}',
            'Owner Name': f'Rep_{random.randint(1, 50):02d}',
            'Product': random.choice(['Platform', 'Mobile', 'Integrations']),
            'Lead Source': random.choice(['Website', 'Referral', 'Event', 'Cold Call', 'Partner'])
        }
        
        data.append(record)
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Convert date columns to datetime
    date_columns = ['Created Date', 'SQO Date', 'SAO Date', 'Close Date', 'Timestamp: Solution Validation']
    for col in date_columns:
        df[col] = pd.to_datetime(df[col])
    
    return df

def save_demo_data():
    """Generate and save demo data to CSV file."""
    print("🎲 Generating demo data...")
    
    df = generate_demo_data(5000)
    
    # Save to CSV
    filename = 'demo_master_report.csv'
    df.to_csv(filename, index=False)
    
    print(f"✅ Demo data saved to {filename}")
    print(f"📊 Generated {len(df):,} records")
    print(f"📅 Date range: {df['Created Date'].min().date()} to {df['Created Date'].max().date()}")
    
    # Print summary statistics
    print("\n📈 Data Summary:")
    print(f"   Sources: {', '.join(df['Source'].value_counts().head().index.tolist())}")
    print(f"   Segments: {', '.join(df['Segment - historical'].value_counts().index.tolist())}")
    print(f"   Total ARR: ${df['ARR Change'].sum():,.0f}")
    print(f"   Avg Deal Size: ${df[df['ARR Change'] > 0]['ARR Change'].mean():,.0f}")
    print(f"   Closed Won Rate: {len(df[df['Stage'] == 'Closed Won']) / len(df) * 100:.1f}%")
    
    return df

if __name__ == "__main__":
    save_demo_data()