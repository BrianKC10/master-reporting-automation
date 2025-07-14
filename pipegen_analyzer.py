"""
Pipeline Generation Analyzer
Functions for analyzing pipeline generation from ARR change history
"""

import pandas as pd
import numpy as np
from date_utils import get_fiscal_quarter_string


def stage_end_arr(group, stage_name):
    """
    Find the ARR value at the end of a specified stage for a group of ARR history records.
    
    Parameters:
    - group: DataFrame group containing ARR history for one opportunity
    - stage_name: Name of the stage to find the end ARR for
    
    Returns:
    - ARR value at the end of the specified stage, or NaN if stage not found
    """
    # Check if Stage column exists
    if 'Stage' not in group.columns:
        return np.nan
    
    # Sort by Edit Date to get chronological order
    group_sorted = group.sort_values('Edit Date')
    
    # Find records where stage matches
    stage_records = group_sorted[group_sorted['Stage'] == stage_name]
    
    if stage_records.empty:
        return np.nan
    
    # Get the last ARR value for this stage (most recent edit in this stage)
    return stage_records.iloc[-1]['New Value']


def process_arr_change_history(raw_history_df, stage_timestamp_df=None):
    """
    Process ARR change history to calculate pipeline generation metrics.
    
    Pipeline Generation (Pipegen) Definition:
    - "ARR Change" that has been SAO'd (reached Sales Accepted Opportunity stage)
    - Identified by opportunities that have a SAO Date
    - Different from Bookings which is "ARR Change" for Closed Won opportunities
    
    Parameters:
    - raw_history_df: Raw DataFrame from Salesforce ARR change history report
    - stage_timestamp_df: DataFrame with stage timestamp information (processed_df)
    
    Returns:
    - Processed DataFrame with pipeline generation calculations
    """
    print("🔄 Processing ARR change history for pipeline generation analysis...")
    
    if raw_history_df.empty:
        print("❌ No ARR change history data provided")
        return pd.DataFrame()
    
    # Clean and prepare the data
    history_df = raw_history_df.copy()
    
    print(f"📋 Available columns in ARR change history: {list(history_df.columns)}")
    
    # Map common column variations to standard names
    column_mapping = {
        'Edit Date': ['Edit Date', 'EditDate', 'Modified Date', 'Date'],
        'SAO Date': ['SAO Date', 'SAODate', 'Sales Accepted Date', 'Stage Change Date'],
        'Close Date': ['Close Date', 'CloseDate', 'Opportunity Close Date'],
        'New Value': ['New Value', 'NewValue', 'ARR', 'ARR Change', 'Amount'],
        'SFDC ID 18 Digit': ['SFDC ID 18 Digit', 'Opportunity ID', 'OpportunityId', 'Id'],
        'Stage': ['Stage', 'Current Stage', 'Opportunity Stage'],
        'Segment': ['Segment', 'Account Segment', 'Opportunity Segment']
    }
    
    # Apply column mapping
    for standard_name, possible_names in column_mapping.items():
        for possible_name in possible_names:
            if possible_name in history_df.columns:
                if standard_name not in history_df.columns:
                    history_df[standard_name] = history_df[possible_name]
                break
    
    # Convert dates - use actual column names found
    date_columns = ['Edit Date', 'SAO Date', 'Close Date']
    for col in date_columns:
        if col in history_df.columns:
            history_df[col] = pd.to_datetime(history_df[col], errors='coerce')
    
    # Convert ARR values to numeric
    if 'New Value' in history_df.columns:
        history_df['New Value'] = pd.to_numeric(history_df['New Value'], errors='coerce')
    
    print(f"📊 Processing {len(history_df)} ARR change history records...")
    
    # Group by opportunity and consolidate multiple edits per day
    print("🔄 Consolidating multiple daily edits per opportunity...")
    
    # Create a daily consolidation key
    history_df['Edit Date Only'] = history_df['Edit Date'].dt.date
    
    # For each opportunity and each day, keep only the last edit (most recent ARR value)
    consolidated_history = history_df.sort_values(['SFDC ID 18 Digit', 'Edit Date Only', 'Edit Date']).groupby(['SFDC ID 18 Digit', 'Edit Date Only']).tail(1)
    
    print(f"📉 Consolidated to {len(consolidated_history)} records after removing multiple daily edits")
    
    # Calculate pipeline generation based on available data
    pipegen_results = []
    
    # Check if we have stage timestamp information to merge
    if stage_timestamp_df is not None and not stage_timestamp_df.empty:
        print("🔗 Merging ARR change history with stage timestamp data...")
        print(f"📋 Available columns in stage timestamp data: {list(stage_timestamp_df.columns)}")
        
        # Create a mapping of ARR edits by opportunity ID for efficient lookup
        arr_by_opp = {}
        for opp_id, opp_group in consolidated_history.groupby('SFDC ID 18 Digit'):
            arr_by_opp[opp_id] = opp_group.sort_values('Edit Date').copy()
        
        print(f"📊 Created ARR lookup for {len(arr_by_opp)} opportunities")
        
        # Debug: Check for matching opportunities
        stage_opp_ids = set(stage_timestamp_df['SFDC ID 18 Digit'].dropna())
        arr_opp_ids = set(arr_by_opp.keys())
        matching_opps = stage_opp_ids.intersection(arr_opp_ids)
        print(f"📊 Stage timestamp opportunities: {len(stage_opp_ids)}")
        print(f"📊 ARR change history opportunities: {len(arr_opp_ids)}")
        print(f"📊 Matching opportunities: {len(matching_opps)}")
        
        if len(matching_opps) == 0:
            print("⚠️ No matching opportunity IDs found between datasets!")
            print(f"📋 Sample stage IDs: {list(stage_opp_ids)[:5]}")
            print(f"📋 Sample ARR IDs: {list(arr_opp_ids)[:5]}")
        
        # Process each opportunity with stage timestamp information
        processed_count = 0
        for _, opp_row in stage_timestamp_df.iterrows():
            opp_id = opp_row.get('SFDC ID 18 Digit')
            if pd.isna(opp_id) or opp_id not in arr_by_opp:
                continue
                
            processed_count += 1
                
            opp_edits = arr_by_opp[opp_id]
            
            # Calculate ARR at the end of "Open" stage (equivalent to SAO Date)
            sao_date = opp_row.get('SAO Date')
            
            # Skip opportunities without SAO Date - they shouldn't be counted in pipeline generation
            if pd.isna(sao_date):
                if processed_count <= 3:
                    print(f"🔍 Debug {opp_id}: No SAO Date - skipping opportunity")
                continue
            
            # Debug the data types and values
            if processed_count <= 3:
                print(f"🔍 Debug {opp_id}: SAO Date {sao_date} (type: {type(sao_date)})")
                print(f"🔍 Debug {opp_id}: Edit dates sample: {opp_edits['Edit Date'].head(2).tolist()}")
                print(f"🔍 Debug {opp_id}: New Value sample: {opp_edits['New Value'].head(2).tolist()}")
            
            # Convert SAO Date to datetime if it's not already
            sao_date_dt = pd.to_datetime(sao_date, errors='coerce')
            
            if pd.isna(sao_date_dt):
                if processed_count <= 3:
                    print(f"🔍 Debug {opp_id}: Could not convert SAO Date to datetime - skipping")
                continue
            
            pipegen_arr = 0
            
            # Since we're looking for Pipeline Generation (ARR that has been SAO'd),
            # and the ARR change history has NaN values, we should use the 
            # ARR Change from the master report for opportunities with SAO Date
            
            # Use the ARR Change from the master report (this is the final ARR for the opportunity)
            master_arr_change = pd.to_numeric(opp_row.get('ARR Change', 0), errors='coerce')
            
            if pd.notna(master_arr_change) and master_arr_change != 0:
                pipegen_arr = master_arr_change
                if processed_count <= 3:
                    print(f"🔍 Debug {opp_id}: Using master report ARR Change: {pipegen_arr}")
            else:
                # Fallback: try to get from ARR change history if master report has no value
                sao_edits = opp_edits[opp_edits['Edit Date'] <= sao_date_dt]
                if not sao_edits.empty:
                    pipegen_arr = pd.to_numeric(sao_edits.iloc[-1]['New Value'], errors='coerce')
                    if processed_count <= 3:
                        print(f"🔍 Debug {opp_id}: Found {len(sao_edits)} edits before SAO Date, Pipegen ARR: {pipegen_arr}")
                else:
                    # If no edits before SAO Date, use earliest edit
                    if not opp_edits.empty:
                        pipegen_arr = pd.to_numeric(opp_edits.iloc[0]['New Value'], errors='coerce')
                        if processed_count <= 3:
                            print(f"🔍 Debug {opp_id}: No edits before SAO Date, using earliest: {pipegen_arr}")
                    else:
                        pipegen_arr = 0
            
            # Skip if we still don't have a valid ARR value
            if pd.isna(pipegen_arr) or pipegen_arr == 0:
                if processed_count <= 3:
                    print(f"🔍 Debug {opp_id}: No valid ARR value (ARR: {pipegen_arr}) - skipping")
                continue
            
            # Extract quarter from SAO Date
            sao_quarter = None
            if pd.notna(sao_date):
                try:
                    fiscal_quarter_str = get_fiscal_quarter_string(sao_date)
                    if fiscal_quarter_str:
                        sao_quarter = int(fiscal_quarter_str[-1])  # Extract Q number
                except Exception:
                    sao_quarter = None
            
            pipegen_results.append({
                'SFDC ID 18 Digit': opp_id,
                'Pipegen ARR': pipegen_arr if pd.notna(pipegen_arr) else 0,
                'SAO Date': sao_date,
                'SAO Date_Quarter': sao_quarter,
                'Segment': opp_row.get('Segment - historical', 'Unknown'),
                'Bookings Type': opp_row.get('Bookings Type', 'Unknown'),
                'Stage': opp_row.get('Stage', 'Unknown'),
                'Close Date': opp_row.get('Close Date'),
                'ARR Change': opp_row.get('ARR Change', 0)
            })
        
        print(f"📊 Successfully processed {processed_count} opportunities with stage timestamp data")
    else:
        print("⚠️ No stage timestamp data provided - using basic ARR change history only")
        # Fallback to original logic without stage timestamps
        for opp_id, opp_group in consolidated_history.groupby('SFDC ID 18 Digit'):
            opp_group_sorted = opp_group.sort_values('Edit Date')
            latest_record = opp_group_sorted.iloc[-1]
            
            # Use earliest ARR value as pipeline generation
            discovery_arr = opp_group_sorted.iloc[0]['New Value']
            
            # Get SAO Date quarter for grouping (may not be available in ARR change history)
            sao_date = latest_record.get('SAO Date')
            sao_quarter = None
            if pd.notna(sao_date):
                try:
                    # Extract quarter number (1-4) from SAO Date
                    fiscal_quarter_str = get_fiscal_quarter_string(sao_date)
                    if fiscal_quarter_str:
                        sao_quarter = int(fiscal_quarter_str[-1])  # Extract Q number
                except Exception as e:
                    print(f"⚠️ Could not process SAO Date for opportunity {opp_id}: {e}")
                    sao_quarter = None
            
            # Safely extract values with fallbacks for missing columns
            pipegen_results.append({
                'SFDC ID 18 Digit': opp_id,
                'Pipegen ARR': discovery_arr if pd.notna(discovery_arr) else 0,
                'SAO Date': sao_date,
                'SAO Date_Quarter': sao_quarter,
                'Segment': latest_record.get('Segment', 'Unknown'),
                'Bookings Type': latest_record.get('Bookings Type', 'Unknown'),
                'Stage': latest_record.get('Stage', 'Unknown'),
                'Close Date': latest_record.get('Close Date'),
                'ARR Change': latest_record.get('New Value', 0)
            })
    
    pipegen_df = pd.DataFrame(pipegen_results)
    
    print(f"✅ Pipeline generation analysis complete!")
    print(f"📊 Generated pipeline data for {len(pipegen_df)} opportunities")
    
    if not pipegen_df.empty and 'Pipegen ARR' in pipegen_df.columns:
        print(f"💰 Total pipeline generated: ${pipegen_df['Pipegen ARR'].sum():,.0f}")
    else:
        print(f"💰 Total pipeline generated: $0 (no valid opportunities found)")
    
    return pipegen_df


def calculate_pipegen_6_row_analysis(pipegen_df, current_quarter=None):
    """
    Calculate 6-row analysis structure for Pipeline Generation.
    
    Parameters:
    - pipegen_df: DataFrame with pipeline generation data
    - current_quarter: Current quarter for InQuarter calculations (auto-detected if None)
    
    Returns:
    - DataFrame with 6-row analysis results
    """
    if pipegen_df.empty:
        print("❌ No pipeline generation data available for analysis")
        return pd.DataFrame()
    
    print("🔍 PIPEGEN ANALYSIS - 6 Row Structure")
    print("=" * 50)
    
    # Auto-detect current quarter if not provided
    if current_quarter is None and 'SAO Date_Quarter' in pipegen_df.columns:
        current_quarter = pipegen_df['SAO Date_Quarter'].max()
    
    print(f"Current Quarter: Q{current_quarter}" if current_quarter else "Current Quarter: Not detected")
    
    # Create masks for filtering
    expansion_mask = pipegen_df['Segment'] == 'Expansion'
    new_business_mask = pipegen_df['Segment'] == 'New Business'
    current_quarter_mask = pipegen_df['SAO Date_Quarter'] == current_quarter if current_quarter else pd.Series([True] * len(pipegen_df))
    
    results = []
    
    # Row 1: Expansion Current Quarter (current quarter only)
    expansion_current = pipegen_df[expansion_mask & current_quarter_mask]['Pipegen ARR'].sum()
    results.append(('Expansion Current Quarter', expansion_current))
    
    # Row 2: New Business Current Quarter (current quarter only)  
    new_business_current = pipegen_df[new_business_mask & current_quarter_mask]['Pipegen ARR'].sum()
    results.append(('New Business Current Quarter', new_business_current))
    
    # Row 3: Total Current Quarter (current quarter only)
    total_current = expansion_current + new_business_current
    results.append(('Total Current Quarter', total_current))
    
    # Row 4: Expansion Total (all quarters)
    expansion_total = pipegen_df[expansion_mask]['Pipegen ARR'].sum()
    results.append(('Expansion Total', expansion_total))
    
    # Row 5: New Business Total (all quarters)
    new_business_total = pipegen_df[new_business_mask]['Pipegen ARR'].sum()
    results.append(('New Business Total', new_business_total))
    
    # Row 6: Total Pipegen (all quarters)
    total_pipegen = expansion_total + new_business_total
    results.append(('Total Pipegen', total_pipegen))
    
    # Create DataFrame for results
    pipegen_analysis_df = pd.DataFrame(results, columns=['Metric', 'Pipegen ARR'])
    
    # Display results
    print("\n📊 PIPEGEN ANALYSIS RESULTS")
    print("-" * 40)
    for index, row in pipegen_analysis_df.iterrows():
        print(f"{row['Metric']:<25}: ${row['Pipegen ARR']:>15,.0f}")
    
    # Create pivot for quarter breakdown
    if 'SAO Date_Quarter' in pipegen_df.columns:
        pipegen_pivot = pipegen_df.groupby(['Segment', 'SAO Date_Quarter']).agg({
            'Pipegen ARR': 'sum'
        }).reset_index().pivot_table(
            index='Segment', 
            columns='SAO Date_Quarter', 
            values='Pipegen ARR', 
            fill_value=0
        )
        
        print(f"\n📈 QUARTER BREAKDOWN")
        print("-" * 30)
        if not pipegen_pivot.empty:
            print(pipegen_pivot.round(0))
    
    # Additional insights
    print(f"\n💡 INSIGHTS")
    print("-" * 15)
    if total_current > 0 and total_pipegen > 0:
        current_quarter_percentage = (total_current / total_pipegen) * 100
        print(f"Current Quarter represents {current_quarter_percentage:.1f}% of total pipegen")
    
    if expansion_current > 0 and new_business_current > 0:
        expansion_ratio = expansion_current / (expansion_current + new_business_current) * 100
        print(f"Expansion: {expansion_ratio:.1f}% of Current Quarter pipegen")
        print(f"New Business: {100-expansion_ratio:.1f}% of Current Quarter pipegen")
    
    print(f"\n✅ Pipegen analysis complete! Total pipeline generated: ${total_pipegen:,.0f}")
    
    return pipegen_analysis_df


def display_pipegen_summary(pipegen_df):
    """Display a summary of pipeline generation data."""
    if pipegen_df.empty:
        print("No pipeline generation data to summarize")
        return
    
    print("\n📋 PIPELINE GENERATION SUMMARY")
    print("=" * 40)
    
    total_pipegen = pipegen_df['Pipegen ARR'].sum()
    total_opps = len(pipegen_df)
    
    print(f"Total Opportunities: {total_opps:,}")
    print(f"Total Pipeline Generated: ${total_pipegen:,.0f}")
    
    if total_opps > 0:
        avg_pipegen = total_pipegen / total_opps
        print(f"Average Pipeline per Opp: ${avg_pipegen:,.0f}")
    
    # Segment breakdown
    if 'Segment' in pipegen_df.columns:
        segment_summary = pipegen_df.groupby('Segment').agg({
            'Pipegen ARR': ['sum', 'count', 'mean']
        }).round(0)
        segment_summary.columns = ['Total Pipegen ARR', 'Count', 'Avg Pipegen ARR']
        
        print(f"\n📊 BY SEGMENT:")
        print(segment_summary)
    
    # Quarter breakdown
    if 'SAO Date_Quarter' in pipegen_df.columns:
        quarter_summary = pipegen_df.groupby('SAO Date_Quarter').agg({
            'Pipegen ARR': ['sum', 'count']
        }).round(0)
        quarter_summary.columns = ['Total Pipegen ARR', 'Count']
        
        print(f"\n📅 BY QUARTER:")
        print(quarter_summary)


def load_and_process_pipegen_data(simple_salesforce_conn, report_id='00OUO000009jhTS2AY'):
    """
    Load ARR change history from Salesforce and process for pipeline generation analysis.
    
    Parameters:
    - simple_salesforce_conn: Authenticated Salesforce connection
    - report_id: Salesforce report ID for ARR change history
    
    Returns:
    - Processed pipeline generation DataFrame
    """
    try:
        print(f"📊 Loading ARR change history from Salesforce report: {report_id}")
        
        # Load the report
        report = simple_salesforce_conn.restful(f'analytics/reports/{report_id}', params={'includeDetails': 'true'})
        
        # Extract data from report
        if 'factMap' in report and '0!T' in report['factMap']:
            rows = report['factMap']['0!T']['rows']
            
            if not rows:
                print("❌ No data found in the report")
                return pd.DataFrame()
            
            # Extract column information
            report_metadata = report['reportMetadata']
            detail_columns = report_metadata['detailColumns']
            
            # Create DataFrame
            data = []
            for row in rows:
                row_data = {}
                for i, cell in enumerate(row['dataCells']):
                    if i < len(detail_columns):
                        col_name = detail_columns[i]
                        # Convert column API names to friendly names
                        friendly_name = col_name.replace('__c', '').replace('_', ' ').title()
                        if 'Sfdc_Id_18_Digit' in col_name:
                            friendly_name = 'SFDC ID 18 Digit'
                        elif 'Edit_Date' in col_name:
                            friendly_name = 'Edit Date'
                        elif 'New_Value' in col_name:
                            friendly_name = 'New Value'
                        elif 'Sao_Date' in col_name:
                            friendly_name = 'SAO Date'
                        
                        row_data[friendly_name] = cell.get('value', '')
                data.append(row_data)
            
            raw_history_df = pd.DataFrame(data)
            print(f"✅ Loaded {len(raw_history_df)} records from Salesforce")
            
            # Process the data
            return process_arr_change_history(raw_history_df)
            
        else:
            print("❌ No data found in report response")
            return pd.DataFrame()
            
    except Exception as e:
        print(f"❌ Error loading ARR change history: {e}")
        return pd.DataFrame()