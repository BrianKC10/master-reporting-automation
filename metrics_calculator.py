"""
Metrics Calculator
Enhanced quarterly metrics calculations with Pipegen support
"""

import pandas as pd
import numpy as np

# Import fiscal quarter function for pipegen analysis
try:
    from date_utils import get_fiscal_quarter_info
except ImportError:
    # Fallback function if import fails
    def get_fiscal_quarter_info(date_input):
        """Fallback fiscal quarter function"""
        if pd.isnull(date_input):
            return pd.NaT, None, None
        date_to_process = pd.to_datetime(date_input, errors='coerce')
        if pd.isna(date_to_process):
            return pd.NaT, None, None
        year, month = date_to_process.year, date_to_process.month
        if month == 1:
            return pd.Timestamp(year=year-1, month=11, day=1), f'FY{year}Q4', year
        elif month in [2, 3, 4]:
            return pd.Timestamp(year=year, month=2, day=1), f'FY{year+1}Q1', year + 1
        elif month in [5, 6, 7]:
            return pd.Timestamp(year=year, month=5, day=1), f'FY{year+1}Q2', year + 1
        elif month in [8, 9, 10]:
            return pd.Timestamp(year=year, month=8, day=1), f'FY{year+1}Q3', year + 1
        elif month in [11, 12]:
            return pd.Timestamp(year=year, month=11, day=1), f'FY{year+1}Q4', year + 1


def calculate_quarterly_metrics_original(df_full, opp_id_col_name, group_by_cols=None, stage_won='Closed Won'):
    """
    ORIGINAL LOGIC: Calculates 7 key sales metrics exactly as in the original notebook.
    This function replicates the exact original logic without Pipegen integration.
    
    Parameters:
    - df_full: Full DataFrame containing opportunity data
    - opp_id_col_name: Name of the column containing opportunity IDs
    - group_by_cols: Optional grouping columns (string or list)
    - stage_won: String identifier for won opportunities
    
    Returns:
    - DataFrame with quarterly metrics (7 metrics only)
    """
    if df_full.empty: 
        return pd.DataFrame()
    
    # Required columns check
    required_cols = ['Fiscal Period - Corrected', 'ARR Change', 'Stage', 'Created Date', 'Close Date', opp_id_col_name]
    if not all(col in df_full.columns for col in required_cols):
        print(f"Warning: Missing one or more required columns in calculate_quarterly_metrics_original. Needed: {required_cols}")
        return pd.DataFrame()
    
    df_copy = df_full.copy()
    df_won = df_copy[df_copy['Stage'] == stage_won].copy()
    
    # Calculate sales cycle for won deals
    if not df_won.empty:
        df_won.loc[:, 'Sales Cycle Days'] = (df_won['Close Date'] - df_won['Created Date']).dt.days
        df_copy = df_copy.merge(df_won[['Sales Cycle Days']], left_index=True, right_index=True, how='left')
    else:
        df_copy['Sales Cycle Days'] = pd.NA
    
    # Set up grouping columns
    grouping_cols = []
    if group_by_cols:
        grouping_cols.extend([group_by_cols] if isinstance(group_by_cols, str) else list(group_by_cols))
    grouping_cols.append('Fiscal Period - Corrected')
    
    # ORIGINAL aggregation functions - exactly 7 metrics
    agg_funcs = {
        'Total ARR': ('ARR Change', 'sum'),
        '# of Opps': (opp_id_col_name, 'nunique'),
        '# of Won Opps': (opp_id_col_name, lambda x: x[df_copy.loc[x.index, 'Stage'] == stage_won].nunique()),
        'Total Relevant Opps for WR': (opp_id_col_name, lambda x: x[df_copy.loc[x.index, 'Stage'].isin([stage_won, 'Closed Lost'])].nunique()),
        'Won ARR': ('ARR Change', lambda x: x[df_copy.loc[x.index, 'Stage'] == stage_won].sum()),
        'Avg Sales Cycle': ('Sales Cycle Days', 'mean')
    }
    
    # Perform aggregation
    quarterly_summary = df_copy.groupby(grouping_cols, observed=False).agg(**agg_funcs)
    
    # Calculate derived metrics - ORIGINAL logic
    quarterly_summary['Avg Sales Price'] = (quarterly_summary['Won ARR'] / quarterly_summary['# of Won Opps'].replace(0, np.nan)).fillna(0)
    quarterly_summary['Win Rate (Count)'] = (quarterly_summary['# of Won Opps'] / quarterly_summary['Total Relevant Opps for WR'].replace(0, np.nan) * 100).fillna(0)
    
    # Return exactly 7 metrics as in original
    final_metrics = ['Total ARR', '# of Opps', '# of Won Opps', 'Won ARR', 'Avg Sales Price', 'Win Rate (Count)', 'Avg Sales Cycle']
    return quarterly_summary[final_metrics]


def calculate_quarterly_metrics(df_full, opp_id_col_name, group_by_cols=None, stage_won='Closed Won', pipegen_df=None):
    """
    Calculates all key sales metrics on a quarterly basis, with optional grouping and Pipegen support.
    
    Parameters:
    - df_full: Full DataFrame containing opportunity data
    - opp_id_col_name: Name of the column containing opportunity IDs
    - group_by_cols: Optional grouping columns (string or list)
    - stage_won: String identifier for won opportunities
    - pipegen_df: Optional DataFrame with pipeline generation data
    
    Returns:
    - DataFrame with quarterly metrics
    """
    if df_full.empty: 
        return pd.DataFrame()
    
    # Required columns check
    required_cols = ['Fiscal Period - Corrected', 'ARR Change', 'Stage', 'Created Date', 'Close Date', opp_id_col_name]
    if not all(col in df_full.columns for col in required_cols):
        print(f"Warning: Missing one or more required columns in calculate_quarterly_metrics. Needed: {required_cols}")
        return pd.DataFrame()
    
    df_copy = df_full.copy()
    df_won = df_copy[df_copy['Stage'] == stage_won].copy()
    
    # Calculate sales cycle for won deals
    if not df_won.empty:
        df_won.loc[:, 'Sales Cycle Days'] = (df_won['Close Date'] - df_won['Created Date']).dt.days
        df_copy = df_copy.merge(df_won[['Sales Cycle Days']], left_index=True, right_index=True, how='left')
    else:
        df_copy['Sales Cycle Days'] = pd.NA
    
    # Set up grouping columns
    grouping_cols = []
    if group_by_cols:
        grouping_cols.extend([group_by_cols] if isinstance(group_by_cols, str) else list(group_by_cols))
    grouping_cols.append('Fiscal Period - Corrected')
    
    # Define aggregation functions
    agg_funcs = {
        'Total ARR': ('ARR Change', 'sum'),
        '# of Opps': (opp_id_col_name, 'nunique'),
        '# of Won Opps': (opp_id_col_name, lambda x: x[df_copy.loc[x.index, 'Stage'] == stage_won].nunique()),
        'Total Relevant Opps for WR': (opp_id_col_name, lambda x: x[df_copy.loc[x.index, 'Stage'].isin([stage_won, 'Closed Lost'])].nunique()),
        'Won ARR': ('ARR Change', lambda x: x[df_copy.loc[x.index, 'Stage'] == stage_won].sum()),
        'Avg Sales Cycle': ('Sales Cycle Days', 'mean')
    }
    
    # Perform aggregation
    quarterly_summary = df_copy.groupby(grouping_cols, observed=False).agg(**agg_funcs)
    
    # Calculate derived metrics
    quarterly_summary['Avg Sales Price'] = (quarterly_summary['Won ARR'] / quarterly_summary['# of Won Opps'].replace(0, np.nan)).fillna(0)
    quarterly_summary['Win Rate (Count)'] = (quarterly_summary['# of Won Opps'] / quarterly_summary['Total Relevant Opps for WR'].replace(0, np.nan) * 100).fillna(0)
    
    # Add Pipegen metrics if pipegen_df is provided
    if pipegen_df is not None and not pipegen_df.empty and 'SAO Date_Quarter' in pipegen_df.columns:
        print("📊 Adding Pipegen metrics to quarterly analysis...")
        
        # Create pipegen grouping columns (map to fiscal period format)
        pipegen_grouping_cols = []
        if group_by_cols:
            # Use Segment instead of Segment - historical for pipegen data
            if isinstance(group_by_cols, str):
                pipegen_group_col = 'Segment' if group_by_cols == 'Segment - historical' else group_by_cols
                pipegen_grouping_cols.append(pipegen_group_col)
            else:
                for col in group_by_cols:
                    pipegen_group_col = 'Segment' if col == 'Segment - historical' else col
                    pipegen_grouping_cols.append(pipegen_group_col)
        
        # Convert SAO Date_Quarter to fiscal period format for matching
        pipegen_copy = pipegen_df.copy()
        
        # Use the actual SAO Date to determine the proper fiscal year and quarter
        def convert_sao_to_fiscal_period(row):
            if pd.isna(row.get('SAO Date')):
                return None
            try:
                from date_utils import get_fiscal_quarter_info
                _, fiscal_quarter_str, _ = get_fiscal_quarter_info(pd.to_datetime(row['SAO Date']))
                return fiscal_quarter_str
            except:
                # Fallback: if we only have quarter number, assume it's recent data
                quarter_num = row.get('SAO Date_Quarter')
                if pd.notna(quarter_num):
                    return f"FY2024Q{int(quarter_num)}"  # This is still a fallback
                return None
        
        pipegen_copy['Fiscal Period - Corrected'] = pipegen_copy.apply(convert_sao_to_fiscal_period, axis=1)
        
        pipegen_grouping_cols.append('Fiscal Period - Corrected')
        
        # Calculate pipegen metrics by the same grouping structure
        if all(col in pipegen_copy.columns for col in pipegen_grouping_cols):
            pipegen_agg = pipegen_copy.groupby(pipegen_grouping_cols, observed=False).agg({
                'Pipegen ARR': 'sum',
                'SFDC ID 18 Digit': 'nunique'  # Count of opportunities that generated pipeline
            }).rename(columns={'SFDC ID 18 Digit': '# of Pipegen Opps'})
            
            # Merge pipegen metrics with main quarterly summary
            try:
                quarterly_summary = quarterly_summary.merge(
                    pipegen_agg, 
                    left_index=True, 
                    right_index=True, 
                    how='left'
                )
                quarterly_summary['Pipegen ARR'] = quarterly_summary['Pipegen ARR'].fillna(0)
                quarterly_summary['# of Pipegen Opps'] = quarterly_summary['# of Pipegen Opps'].fillna(0)
                
                print("✅ Successfully added Pipegen metrics to quarterly analysis")
            except Exception as e:
                print(f"⚠️ Could not merge Pipegen data: {e}")
        else:
            print(f"⚠️ Missing required columns for Pipegen grouping: {pipegen_grouping_cols}")
    
    # Final metrics list including Pipegen if available
    final_metrics = ['Total ARR', '# of Opps', '# of Won Opps', 'Won ARR', 'Avg Sales Price', 'Win Rate (Count)', 'Avg Sales Cycle']
    if 'Pipegen ARR' in quarterly_summary.columns:
        final_metrics.extend(['Pipegen ARR', '# of Pipegen Opps'])
    
    return quarterly_summary[final_metrics]


def calculate_6_row_analysis(df, metric_col, segment_col='Segment', quarter_col='Quarter', current_quarter=None):
    """
    Calculate 6-row analysis structure for any metric.
    
    Parameters:
    - df: DataFrame with the data
    - metric_col: Column name containing the metric to analyze
    - segment_col: Column name for segmentation (default: 'Segment')
    - quarter_col: Column name for quarter information
    - current_quarter: Current quarter for InQuarter calculations (auto-detected if None)
    
    Returns:
    - DataFrame with 6-row analysis results
    """
    if df.empty:
        return pd.DataFrame()
    
    # Auto-detect current quarter if not provided
    if current_quarter is None and quarter_col in df.columns:
        current_quarter = df[quarter_col].max()
    
    # Create masks for filtering
    expansion_mask = df[segment_col] == 'Expansion'
    new_business_mask = df[segment_col] == 'New Business'
    current_quarter_mask = df[quarter_col] == current_quarter if current_quarter else pd.Series([True] * len(df))
    
    results = []
    
    # Row 1: Expansion InQuarter (current quarter only)
    expansion_inquarter = df[expansion_mask & current_quarter_mask][metric_col].sum()
    results.append(('Expansion InQuarter', expansion_inquarter))
    
    # Row 2: New Business InQuarter (current quarter only)  
    new_business_inquarter = df[new_business_mask & current_quarter_mask][metric_col].sum()
    results.append(('New Business InQuarter', new_business_inquarter))
    
    # Row 3: Total InQuarter (current quarter only)
    total_inquarter = expansion_inquarter + new_business_inquarter
    results.append(('Total InQuarter', total_inquarter))
    
    # Row 4: Expansion Total (all quarters)
    expansion_total = df[expansion_mask][metric_col].sum()
    results.append(('Expansion Total', expansion_total))
    
    # Row 5: New Business Total (all quarters)
    new_business_total = df[new_business_mask][metric_col].sum()
    results.append(('New Business Total', new_business_total))
    
    # Row 6: Total Metric (all quarters)
    total_metric = expansion_total + new_business_total
    results.append(('Total Metric', total_metric))
    
    # Create DataFrame for results
    analysis_df = pd.DataFrame(results, columns=['Metric', metric_col])
    
    return analysis_df


def calculate_win_rate_analysis(df, opp_id_col, stage_col='Stage', segment_col='Bookings Type', quarter_col='Fiscal Period - Corrected', current_quarter=None, stage_won='Closed Won', start_fiscal_year=2023):
    """Calculate quarterly win rate analysis pivoted by close date quarter."""
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()
    
    # Filter relevant deals (won or lost only)
    relevant_df = df[df[stage_col].isin([stage_won, 'Closed Lost'])].copy()
    if relevant_df.empty:
        return pd.DataFrame(), pd.DataFrame()
    
    # Get fiscal periods for filtering
    fy_start_str = f"FY{start_fiscal_year}Q1"
    relevant_df_filtered = relevant_df[
        (relevant_df[quarter_col].notna()) & 
        (relevant_df[quarter_col] >= fy_start_str)
    ].copy()
    
    if relevant_df_filtered.empty:
        return pd.DataFrame(), pd.DataFrame()
    
    # Get all fiscal periods and sort them
    fiscal_periods = sorted(relevant_df_filtered[quarter_col].unique())
    
    # Calculate win rates for each segment and quarter
    expansion_data = relevant_df_filtered[relevant_df_filtered[segment_col] == 'Expansion']
    new_business_data = relevant_df_filtered[relevant_df_filtered[segment_col] == 'New Business']
    
    # Helper function to calculate win rate by quarter
    def calc_quarter_win_rates(data, periods):
        win_rates = []
        for period in periods:
            period_data = data[data[quarter_col] == period]
            if len(period_data) > 0:
                won_count = len(period_data[period_data[stage_col] == stage_won])
                total_count = len(period_data)
                win_rate = (won_count / total_count * 100) if total_count > 0 else 0
            else:
                win_rate = 0
            win_rates.append(win_rate)
        return win_rates
    
    # Calculate inquarter win rates (using Inquarter Booking Flag)
    def calc_inquarter_win_rates(data, periods):
        win_rates = []
        for period in periods:
            period_data = data[(data[quarter_col] == period) & (data.get('Inquarter Booking Flag', False) == True)]
            if len(period_data) > 0:
                won_count = len(period_data[period_data[stage_col] == stage_won])
                total_count = len(period_data)
                win_rate = (won_count / total_count * 100) if total_count > 0 else 0
            else:
                win_rate = 0
            win_rates.append(win_rate)
        return win_rates
    
    # Calculate win rates for each segment
    expansion_rates = calc_quarter_win_rates(expansion_data, fiscal_periods)
    new_business_rates = calc_quarter_win_rates(new_business_data, fiscal_periods)
    
    # Calculate inquarter win rates
    expansion_inquarter_rates = calc_inquarter_win_rates(expansion_data, fiscal_periods)
    new_business_inquarter_rates = calc_inquarter_win_rates(new_business_data, fiscal_periods)
    total_inquarter_rates = calc_inquarter_win_rates(relevant_df_filtered, fiscal_periods)
    
    # Calculate total rates (combining both segments)
    total_rates = []
    for period in fiscal_periods:
        period_data = relevant_df_filtered[relevant_df_filtered[quarter_col] == period]
        if len(period_data) > 0:
            won_count = len(period_data[period_data[stage_col] == stage_won])
            total_count = len(period_data)
            win_rate = (won_count / total_count * 100) if total_count > 0 else 0
        else:
            win_rate = 0
        total_rates.append(win_rate)
    
    # Create the quarterly summary (bookings type view)
    quarterly_summary = []
    quarterly_summary.append(['Expansion'] + expansion_rates)
    quarterly_summary.append(['New Business'] + new_business_rates)
    quarterly_summary.append(['Total Inquarter'] + total_inquarter_rates)
    quarterly_summary.append(['Total Win Rate'] + total_rates)
    
    # Create DataFrame with fiscal periods as column headers
    columns = ['Metric'] + fiscal_periods
    summary_df = pd.DataFrame(quarterly_summary, columns=columns)
    summary_df = summary_df.set_index('Metric')
    
    # Add comparison columns (QoQ, vs Last 4Q Avg, YoY)
    summary_df = add_pipegen_comparison_columns(summary_df)
    
    # Create multi-index version (segment & bookings type view)
    segments_to_exclude = ['Self Serve', 'Self Service', 'Unknown']
    valid_segments = [seg for seg in sorted(df['Segment - historical'].dropna().unique()) if seg not in segments_to_exclude]
    
    all_segments_summary_list = []
    for segment_name in valid_segments:
        segment_data = relevant_df_filtered[relevant_df_filtered['Segment - historical'] == segment_name]
        if segment_data.empty:
            continue
            
        exp_segment_data = segment_data[segment_data[segment_col] == 'Expansion']
        nb_segment_data = segment_data[segment_data[segment_col] == 'New Business']
        
        exp_rates = calc_quarter_win_rates(exp_segment_data, fiscal_periods)
        nb_rates = calc_quarter_win_rates(nb_segment_data, fiscal_periods)
        exp_inq_rates = calc_inquarter_win_rates(exp_segment_data, fiscal_periods)
        nb_inq_rates = calc_inquarter_win_rates(nb_segment_data, fiscal_periods)
        total_inq_rates = calc_inquarter_win_rates(segment_data, fiscal_periods)
        total_seg_rates = calc_quarter_win_rates(segment_data, fiscal_periods)
        
        segment_summary_df = pd.DataFrame([
            ['Expansion'] + exp_rates,
            ['New Business'] + nb_rates,
            ['Total Inquarter'] + total_inq_rates,
            ['Total Win Rate'] + total_seg_rates
        ], columns=columns)
        segment_summary_df = segment_summary_df.set_index('Metric')
        
        # Add comparison columns
        segment_summary_df = add_pipegen_comparison_columns(segment_summary_df)
        
        # Create multi-index
        segment_summary_df.index = pd.MultiIndex.from_product([[segment_name], segment_summary_df.index], 
                                                              names=['Segment - historical', 'Bookings Type / Metric'])
        all_segments_summary_list.append(segment_summary_df)
    
    multi_index_summary = pd.concat(all_segments_summary_list) if all_segments_summary_list else pd.DataFrame()
    
    return summary_df, multi_index_summary


def calculate_asp_analysis(df, arr_col='ARR Change', opp_id_col='SFDC ID 18 Digit', stage_col='Stage', segment_col='Bookings Type', quarter_col='Fiscal Period - Corrected', current_quarter=None, stage_won='Closed Won', start_fiscal_year=2023):
    """Calculate quarterly Average Sales Price analysis pivoted by close date quarter."""
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()
    
    # Filter for won deals only
    won_df = df[df[stage_col] == stage_won].copy()
    if won_df.empty:
        return pd.DataFrame(), pd.DataFrame()
    
    # Get fiscal periods for filtering
    fy_start_str = f"FY{start_fiscal_year}Q1"
    won_df_filtered = won_df[
        (won_df[quarter_col].notna()) & 
        (won_df[quarter_col] >= fy_start_str)
    ].copy()
    
    if won_df_filtered.empty:
        return pd.DataFrame(), pd.DataFrame()
    
    # Get all fiscal periods and sort them
    fiscal_periods = sorted(won_df_filtered[quarter_col].unique())
    
    # Calculate ASP for each segment and quarter
    expansion_data = won_df_filtered[won_df_filtered[segment_col] == 'Expansion']
    new_business_data = won_df_filtered[won_df_filtered[segment_col] == 'New Business']
    
    # Helper function to calculate ASP by quarter
    def calc_quarter_asp(data, periods):
        asp_values = []
        for period in periods:
            period_data = data[data[quarter_col] == period]
            if len(period_data) > 0:
                total_arr = period_data[arr_col].sum()
                num_opps = period_data[opp_id_col].nunique()
                asp = total_arr / num_opps if num_opps > 0 else 0
            else:
                asp = 0
            asp_values.append(asp)
        return asp_values
    
    # Helper function to calculate inquarter ASP
    def calc_inquarter_asp(data, periods):
        asp_values = []
        for period in periods:
            period_data = data[(data[quarter_col] == period) & (data.get('Inquarter Booking Flag', False) == True)]
            if len(period_data) > 0:
                total_arr = period_data[arr_col].sum()
                num_opps = period_data[opp_id_col].nunique()
                asp = total_arr / num_opps if num_opps > 0 else 0
            else:
                asp = 0
            asp_values.append(asp)
        return asp_values
    
    # Calculate ASP for each segment
    expansion_asp = calc_quarter_asp(expansion_data, fiscal_periods)
    new_business_asp = calc_quarter_asp(new_business_data, fiscal_periods)
    
    # Calculate inquarter ASP
    total_inquarter_asp = calc_inquarter_asp(won_df_filtered, fiscal_periods)
    
    # Calculate total ASP (combining both segments)
    total_asp = []
    for period in fiscal_periods:
        period_data = won_df_filtered[won_df_filtered[quarter_col] == period]
        if len(period_data) > 0:
            total_arr = period_data[arr_col].sum()
            num_opps = period_data[opp_id_col].nunique()
            asp = total_arr / num_opps if num_opps > 0 else 0
        else:
            asp = 0
        total_asp.append(asp)
    
    # Create the quarterly summary (bookings type view)
    quarterly_summary = []
    quarterly_summary.append(['Expansion'] + expansion_asp)
    quarterly_summary.append(['New Business'] + new_business_asp)
    quarterly_summary.append(['Total Inquarter'] + total_inquarter_asp)
    quarterly_summary.append(['Total ASP'] + total_asp)
    
    # Create DataFrame with fiscal periods as column headers
    columns = ['Metric'] + fiscal_periods
    summary_df = pd.DataFrame(quarterly_summary, columns=columns)
    summary_df = summary_df.set_index('Metric')
    
    # Add comparison columns (QoQ, vs Last 4Q Avg, YoY)
    summary_df = add_pipegen_comparison_columns(summary_df)
    
    # Create multi-index version (segment & bookings type view)
    segments_to_exclude = ['Self Serve', 'Self Service', 'Unknown']
    valid_segments = [seg for seg in sorted(df['Segment - historical'].dropna().unique()) if seg not in segments_to_exclude]
    
    all_segments_summary_list = []
    for segment_name in valid_segments:
        segment_data = won_df_filtered[won_df_filtered['Segment - historical'] == segment_name]
        if segment_data.empty:
            continue
            
        exp_segment_data = segment_data[segment_data[segment_col] == 'Expansion']
        nb_segment_data = segment_data[segment_data[segment_col] == 'New Business']
        
        exp_asp = calc_quarter_asp(exp_segment_data, fiscal_periods)
        nb_asp = calc_quarter_asp(nb_segment_data, fiscal_periods)
        total_inq_asp = calc_inquarter_asp(segment_data, fiscal_periods)
        total_seg_asp = calc_quarter_asp(segment_data, fiscal_periods)
        
        segment_summary_df = pd.DataFrame([
            ['Expansion'] + exp_asp,
            ['New Business'] + nb_asp,
            ['Total Inquarter'] + total_inq_asp,
            ['Total ASP'] + total_seg_asp
        ], columns=columns)
        segment_summary_df = segment_summary_df.set_index('Metric')
        
        # Add comparison columns
        segment_summary_df = add_pipegen_comparison_columns(segment_summary_df)
        
        # Create multi-index
        segment_summary_df.index = pd.MultiIndex.from_product([[segment_name], segment_summary_df.index], 
                                                              names=['Segment - historical', 'Bookings Type / Metric'])
        all_segments_summary_list.append(segment_summary_df)
    
    multi_index_summary = pd.concat(all_segments_summary_list) if all_segments_summary_list else pd.DataFrame()
    
    return summary_df, multi_index_summary


def calculate_asc_analysis(df, opp_id_col='SFDC ID 18 Digit', stage_col='Stage', segment_col='Bookings Type', quarter_col='Fiscal Period - Corrected', created_col='Created Date', close_col='Close Date', current_quarter=None, stage_won='Closed Won', start_fiscal_year=2023):
    """Calculate quarterly Average Sales Cycle analysis pivoted by close date quarter."""
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()
    
    # Filter for won deals and calculate sales cycle
    won_df = df[df[stage_col] == stage_won].copy()
    if won_df.empty:
        return pd.DataFrame(), pd.DataFrame()
    
    won_df['Sales Cycle Days'] = (won_df[close_col] - won_df[created_col]).dt.days
    
    # Get fiscal periods for filtering
    fy_start_str = f"FY{start_fiscal_year}Q1"
    won_df_filtered = won_df[
        (won_df[quarter_col].notna()) & 
        (won_df[quarter_col] >= fy_start_str)
    ].copy()
    
    if won_df_filtered.empty:
        return pd.DataFrame(), pd.DataFrame()
    
    # Get all fiscal periods and sort them
    fiscal_periods = sorted(won_df_filtered[quarter_col].unique())
    
    # Calculate ASC for each segment and quarter
    expansion_data = won_df_filtered[won_df_filtered[segment_col] == 'Expansion']
    new_business_data = won_df_filtered[won_df_filtered[segment_col] == 'New Business']
    
    # Helper function to calculate ASC by quarter
    def calc_quarter_asc(data, periods):
        asc_values = []
        for period in periods:
            period_data = data[data[quarter_col] == period]
            if len(period_data) > 0:
                asc = period_data['Sales Cycle Days'].mean()
                asc = asc if pd.notna(asc) else 0
            else:
                asc = 0
            asc_values.append(asc)
        return asc_values
    
    # Helper function to calculate inquarter ASC
    def calc_inquarter_asc(data, periods):
        asc_values = []
        for period in periods:
            period_data = data[(data[quarter_col] == period) & (data.get('Inquarter Booking Flag', False) == True)]
            if len(period_data) > 0:
                asc = period_data['Sales Cycle Days'].mean()
                asc = asc if pd.notna(asc) else 0
            else:
                asc = 0
            asc_values.append(asc)
        return asc_values
    
    # Calculate ASC for each segment
    expansion_asc = calc_quarter_asc(expansion_data, fiscal_periods)
    new_business_asc = calc_quarter_asc(new_business_data, fiscal_periods)
    
    # Calculate inquarter ASC
    total_inquarter_asc = calc_inquarter_asc(won_df_filtered, fiscal_periods)
    
    # Calculate total ASC (combining both segments)
    total_asc = []
    for period in fiscal_periods:
        period_data = won_df_filtered[won_df_filtered[quarter_col] == period]
        if len(period_data) > 0:
            asc = period_data['Sales Cycle Days'].mean()
            asc = asc if pd.notna(asc) else 0
        else:
            asc = 0
        total_asc.append(asc)
    
    # Create the quarterly summary (bookings type view)
    quarterly_summary = []
    quarterly_summary.append(['Expansion'] + expansion_asc)
    quarterly_summary.append(['New Business'] + new_business_asc)
    quarterly_summary.append(['Total Inquarter'] + total_inquarter_asc)
    quarterly_summary.append(['Total ASC'] + total_asc)
    
    # Create DataFrame with fiscal periods as column headers
    columns = ['Metric'] + fiscal_periods
    summary_df = pd.DataFrame(quarterly_summary, columns=columns)
    summary_df = summary_df.set_index('Metric')
    
    # Add comparison columns (QoQ, vs Last 4Q Avg, YoY)
    summary_df = add_pipegen_comparison_columns(summary_df)
    
    # Create multi-index version (segment & bookings type view)
    segments_to_exclude = ['Self Serve', 'Self Service', 'Unknown']
    valid_segments = [seg for seg in sorted(df['Segment - historical'].dropna().unique()) if seg not in segments_to_exclude]
    
    all_segments_summary_list = []
    for segment_name in valid_segments:
        segment_data = won_df_filtered[won_df_filtered['Segment - historical'] == segment_name]
        if segment_data.empty:
            continue
            
        exp_segment_data = segment_data[segment_data[segment_col] == 'Expansion']
        nb_segment_data = segment_data[segment_data[segment_col] == 'New Business']
        
        exp_asc = calc_quarter_asc(exp_segment_data, fiscal_periods)
        nb_asc = calc_quarter_asc(nb_segment_data, fiscal_periods)
        total_inq_asc = calc_inquarter_asc(segment_data, fiscal_periods)
        total_seg_asc = calc_quarter_asc(segment_data, fiscal_periods)
        
        segment_summary_df = pd.DataFrame([
            ['Expansion'] + exp_asc,
            ['New Business'] + nb_asc,
            ['Total Inquarter'] + total_inq_asc,
            ['Total ASC'] + total_seg_asc
        ], columns=columns)
        segment_summary_df = segment_summary_df.set_index('Metric')
        
        # Add comparison columns
        segment_summary_df = add_pipegen_comparison_columns(segment_summary_df)
        
        # Create multi-index
        segment_summary_df.index = pd.MultiIndex.from_product([[segment_name], segment_summary_df.index], 
                                                              names=['Segment - historical', 'Bookings Type / Metric'])
        all_segments_summary_list.append(segment_summary_df)
    
    multi_index_summary = pd.concat(all_segments_summary_list) if all_segments_summary_list else pd.DataFrame()
    
    return summary_df, multi_index_summary


def calculate_pipegen_analysis(pipegen_df, segment_col='Segment', quarter_col='SAO Date_Quarter', current_quarter=None):
    """Calculate 6-row Pipeline Generation analysis."""
    if pipegen_df.empty:
        return pd.DataFrame()
    
    # Auto-detect current quarter if not provided
    if current_quarter is None and quarter_col in pipegen_df.columns:
        current_quarter = pipegen_df[quarter_col].max()
    
    # Create masks
    expansion_mask = pipegen_df[segment_col] == 'Expansion'
    new_business_mask = pipegen_df[segment_col] == 'New Business'
    current_quarter_mask = pipegen_df[quarter_col] == current_quarter if current_quarter else pd.Series([True] * len(pipegen_df))
    
    results = []
    
    # Row 1: Expansion InQuarter
    exp_inq_pipegen = pipegen_df[expansion_mask & current_quarter_mask]['Pipegen ARR'].sum()
    results.append(('Expansion InQuarter', exp_inq_pipegen))
    
    # Row 2: New Business InQuarter
    nb_inq_pipegen = pipegen_df[new_business_mask & current_quarter_mask]['Pipegen ARR'].sum()
    results.append(('New Business InQuarter', nb_inq_pipegen))
    
    # Row 3: Total InQuarter
    total_inq_pipegen = exp_inq_pipegen + nb_inq_pipegen
    results.append(('Total InQuarter', total_inq_pipegen))
    
    # Row 4: Expansion Total
    exp_total_pipegen = pipegen_df[expansion_mask]['Pipegen ARR'].sum()
    results.append(('Expansion Total', exp_total_pipegen))
    
    # Row 5: New Business Total
    nb_total_pipegen = pipegen_df[new_business_mask]['Pipegen ARR'].sum()
    results.append(('New Business Total', nb_total_pipegen))
    
    # Row 6: Total Pipegen
    total_pipegen = exp_total_pipegen + nb_total_pipegen
    results.append(('Total Pipegen', total_pipegen))
    
    return pd.DataFrame(results, columns=['Metric', 'Pipegen ARR ($)'])


def calculate_pipegen_quarterly_summary(pipegen_df, bookings_type_col='Bookings Type', start_fiscal_year=2023):
    """
    Calculate quarterly Pipeline Generation summary with fiscal period columns.
    Returns a DataFrame with fiscal periods as columns (FY2023Q1, FY2023Q2, etc.) and 6-row structure.
    """
    if pipegen_df.empty:
        return pd.DataFrame()
    
    # Map the SAO_Date_Quarter numbers to actual fiscal periods from the SAO Date
    pipegen_df_copy = pipegen_df.copy()
    
    # Use the SAO Date to determine fiscal periods (this matches the bookings table logic)
    if 'SAO Date' in pipegen_df_copy.columns:
        # Apply fiscal quarter calculation to SAO Date to get proper fiscal periods
        def get_fy_quarter(date_val):
            if pd.isna(date_val):
                return None
            fiscal_info = get_fiscal_quarter_info(date_val)
            if fiscal_info[1] is None:
                return None
            # Convert format like "2025-Q4" to "FY2025Q4"
            quarter_str = fiscal_info[1]
            if '-Q' in quarter_str:
                year_part, q_part = quarter_str.split('-Q')
                return f"FY{year_part}Q{q_part}"
            else:
                return quarter_str
        
        pipegen_df_copy['Fiscal Period'] = pipegen_df_copy['SAO Date'].apply(get_fy_quarter)
        print(f"📊 Sample fiscal period mapping from SAO Date: {pipegen_df_copy[['SAO Date', 'SAO Date_Quarter', 'Fiscal Period']].head()}")
    else:
        # Fallback: if no SAO Date, we can't determine proper fiscal periods
        print("⚠️ No SAO Date column found - cannot map to fiscal periods")
        return pd.DataFrame()
    
    # Filter to start from specified fiscal year
    fy_start_str = f"FY{start_fiscal_year}Q1"
    pipegen_df_filtered = pipegen_df_copy[
        (pipegen_df_copy['Fiscal Period'].notna()) & 
        (pipegen_df_copy['Fiscal Period'] >= fy_start_str)
    ].copy()
    
    if pipegen_df_filtered.empty:
        return pd.DataFrame()
    
    # Get all fiscal periods in the data and sort them
    fiscal_periods = sorted(pipegen_df_filtered['Fiscal Period'].unique())
    print(f"📊 Found fiscal periods in pipegen data: {fiscal_periods}")
    
    # Debug: Check what bookings type values we have
    print(f"📊 Available {bookings_type_col} values in pipegen data: {pipegen_df_filtered[bookings_type_col].unique()}")
    print(f"📊 Looking for {bookings_type_col} values: 'Expansion' and 'New Business'")
    
    # Create pivot tables for each bookings type
    expansion_data = pipegen_df_filtered[pipegen_df_filtered[bookings_type_col] == 'Expansion']
    new_business_data = pipegen_df_filtered[pipegen_df_filtered[bookings_type_col] == 'New Business']
    
    print(f"📊 Expansion data found: {len(expansion_data)} opportunities")
    print(f"📊 New Business data found: {len(new_business_data)} opportunities")
    
    # Pivot on Fiscal Period
    expansion_pivot = expansion_data.groupby('Fiscal Period')['Pipegen ARR'].sum().reindex(fiscal_periods, fill_value=0)
    new_business_pivot = new_business_data.groupby('Fiscal Period')['Pipegen ARR'].sum().reindex(fiscal_periods, fill_value=0)
    
    # Create quarterly breakdown structure (3 rows only - no repetitive totals)
    quarterly_summary = []
    
    # Row 1: Expansion (by fiscal period)
    quarterly_summary.append(['Expansion'] + expansion_pivot.tolist())
    
    # Row 2: New Business (by fiscal period)
    quarterly_summary.append(['New Business'] + new_business_pivot.tolist())
    
    # Row 3: Total by Period (sum of Expansion and New Business per period)
    total_by_period = expansion_pivot + new_business_pivot
    quarterly_summary.append(['Total by Period'] + total_by_period.tolist())
    
    # Create DataFrame with fiscal periods as column headers
    columns = ['Metric'] + fiscal_periods
    summary_df = pd.DataFrame(quarterly_summary, columns=columns)
    summary_df = summary_df.set_index('Metric')
    
    return summary_df


def add_pipegen_comparison_columns(df):
    """Add QoQ Change, vs Last 4 Quarter Average, and YoY Change columns to pipegen summary."""
    if df.empty:
        return df
    
    df_with_comparisons = df.copy()
    
    # Get fiscal quarter columns (FY2023Q1, FY2023Q2, etc. format)
    quarter_cols = [col for col in df_with_comparisons.columns if isinstance(col, str) and col.startswith('FY') and 'Q' in col]
    quarter_cols.sort()  # This will sort chronologically
    
    if len(quarter_cols) < 2:
        return df_with_comparisons
    
    # Get the most recent quarter (last column)
    current_quarter = quarter_cols[-1]
    
    # Convert columns to numeric for calculations
    for col in quarter_cols:
        df_with_comparisons[col] = pd.to_numeric(df_with_comparisons[col], errors='coerce')
    
    # QoQ Change (Quarter over Quarter)
    if len(quarter_cols) >= 2:
        prev_quarter = quarter_cols[-2]
        qoq_values = ((df_with_comparisons[current_quarter] - df_with_comparisons[prev_quarter]) / 
                      df_with_comparisons[prev_quarter].replace(0, np.nan) * 100)
        df_with_comparisons['QoQ Change'] = qoq_values.round(2)
    else:
        df_with_comparisons['QoQ Change'] = np.nan
    
    # vs Last 4 Quarter Average (if we have at least 5 quarters total)
    if len(quarter_cols) >= 5:
        last_4_quarters = quarter_cols[-5:-1]  # Get 4 quarters before current
        avg_last_4 = df_with_comparisons[last_4_quarters].mean(axis=1)
        vs_avg_values = ((df_with_comparisons[current_quarter] - avg_last_4) / 
                        avg_last_4.replace(0, np.nan) * 100)
        df_with_comparisons['vs Last 4Q Avg'] = vs_avg_values.round(2)
    else:
        df_with_comparisons['vs Last 4Q Avg'] = np.nan
    
    # YoY Change (Year over Year) - compare same quarter from previous year
    current_year = int(current_quarter[2:6])  # Extract year from FY2024Q1
    current_q = current_quarter[-2:]  # Extract Q1, Q2, etc.
    prev_year_quarter = f"FY{current_year-1}{current_q}"
    
    if prev_year_quarter in quarter_cols:
        yoy_values = ((df_with_comparisons[current_quarter] - df_with_comparisons[prev_year_quarter]) / 
                     df_with_comparisons[prev_year_quarter].replace(0, np.nan) * 100)
        df_with_comparisons['YoY Change'] = yoy_values.round(2)
    else:
        df_with_comparisons['YoY Change'] = np.nan
    
    return df_with_comparisons