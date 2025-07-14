"""
Trend Analyzer
Comprehensive trend analysis with QoQ, YoY, and 4-Quarter Average comparisons
"""

import pandas as pd
import numpy as np
from metrics_calculator import calculate_quarterly_metrics, calculate_quarterly_metrics_original


def build_comparison_view(all_q_metrics, current_q, comparison_data, comparison_label, group_tuple=None):
    """Build comparison table between current quarter and comparison data."""
    if group_tuple is not None:
        # Ensure group_tuple is always a tuple
        if isinstance(group_tuple, str):
            group_tuple = (group_tuple,)
        elif not isinstance(group_tuple, tuple):
            group_tuple = tuple(group_tuple) if hasattr(group_tuple, '__iter__') else (group_tuple,)
        
        current_data = all_q_metrics.loc[group_tuple + (current_q,)]
    else:
        current_data = all_q_metrics.loc[current_q]
    
    comparison_table = pd.DataFrame({
        'Current': current_data,
        'Comparison': comparison_data
    })
    comparison_table['Abs Change'] = comparison_table['Current'] - comparison_table['Comparison']
    comparison_table['% Change'] = ((comparison_table['Current'] - comparison_table['Comparison']) / 
                                   comparison_table['Comparison'].replace(0, np.nan) * 100).fillna(0)
    comparison_table['Comparison Label'] = comparison_label
    
    return comparison_table


def collect_all_changes(comparison_table, current_q, comparison_label, group_name):
    """
    Collect ALL changes (regardless of significance) for complete analysis.
    
    Returns:
    - List of all changes
    """
    all_changes = []
    
    for metric_name in comparison_table.index:
        pct_change = comparison_table.loc[metric_name, '% Change']
        abs_change = comparison_table.loc[metric_name, 'Abs Change']
        
        # Convert to numeric for calculations
        pct_num = pd.to_numeric(pct_change, errors='coerce') if pd.notnull(pct_change) else 0
        abs_num = pd.to_numeric(abs_change, errors='coerce') if pd.notnull(abs_change) else 0
        
        # Determine metric type
        if metric_name in ['Total ARR', 'Won ARR', 'Pipegen ARR']:
            metric_type = 'ARR'
        elif metric_name in ['# of Opps', '# of Won Opps', '# of Pipegen Opps']:
            metric_type = 'Count'
        elif metric_name == 'Win Rate (Count)':
            metric_type = 'Percentage'
        elif metric_name == 'Avg Sales Cycle':
            metric_type = 'Days'
        elif metric_name == 'Avg Sales Price':
            metric_type = 'Currency'
        else:
            metric_type = 'Other'
        
        # Determine change direction
        if abs_num > 0:
            change_type = 'Increase'
        elif abs_num < 0:
            change_type = 'Decrease'
        else:
            change_type = 'No Change'
        
        all_changes.append({
            'Quarter': current_q,
            'Comparison': comparison_label,
            'Group': group_name,
            'Metric': metric_name,
            'Type': change_type,
            '% Change': f"{pct_num:.2f}%",
            'Abs Change': abs_num,
            'Current Value': comparison_table.loc[metric_name, 'Current'],
            'Comparison Value': comparison_table.loc[metric_name, 'Comparison']
        })
    
    return all_changes


def identify_and_report_significant_changes(comparison_table, current_q, comparison_label, group_name, thresholds=None):
    """
    Identify significant changes based on thresholds.
    
    Parameters:
    - comparison_table: DataFrame with comparison data
    - current_q: Current quarter string
    - comparison_label: Label describing the comparison
    - group_name: Name of the group being analyzed
    - thresholds: Dictionary of thresholds (will use defaults if None)
    
    Returns:
    - List of significant changes
    """
    if thresholds is None:
        thresholds = {
            'pct_change_threshold': 20.0,
            'abs_change_threshold_arr': 100000,
            'abs_change_threshold_opps': 10,
            'abs_change_threshold_wr': 10.0,
            'abs_change_threshold_asc': 30.0,
            'abs_change_threshold_asp': 5000,
            'abs_change_threshold_pipegen': 50000
        }
    
    significant_changes = []
    
    # Metrics to check with their thresholds
    metric_checks = [
        ('Total ARR', thresholds['abs_change_threshold_arr'], 'ARR'),
        ('Won ARR', thresholds['abs_change_threshold_arr'], 'ARR'),
        ('# of Opps', thresholds['abs_change_threshold_opps'], 'Count'),
        ('# of Won Opps', thresholds['abs_change_threshold_opps'], 'Count'),
        ('Win Rate (Count)', thresholds['abs_change_threshold_wr'], 'Percentage'),
        ('Avg Sales Cycle', thresholds['abs_change_threshold_asc'], 'Days'),
        ('Avg Sales Price', thresholds['abs_change_threshold_asp'], 'Currency'),
        ('Pipegen ARR', thresholds['abs_change_threshold_pipegen'], 'ARR'),
        ('# of Pipegen Opps', thresholds['abs_change_threshold_opps'], 'Count')
    ]
    
    for metric_name, abs_threshold, metric_type in metric_checks:
        if metric_name not in comparison_table.index:
            continue
            
        pct_change = comparison_table.loc[metric_name, '% Change']
        abs_change = comparison_table.loc[metric_name, 'Abs Change']
        
        # Convert to numeric for calculations
        pct_num = pd.to_numeric(pct_change, errors='coerce') if pd.notnull(pct_change) else 0
        abs_num = pd.to_numeric(abs_change, errors='coerce') if pd.notnull(abs_change) else 0
        
        # Use ORIGINAL notebook threshold logic for each metric type
        is_sig_drop = False
        is_sig_inc = False
        
        if metric_name in ['Total ARR', 'Won ARR']:
            is_sig_drop = (pct_num <= -thresholds['pct_change_threshold'] and abs(abs_num) >= thresholds['abs_change_threshold_arr'])
            is_sig_inc = (pct_num >= thresholds['pct_change_threshold'] and abs(abs_num) >= thresholds['abs_change_threshold_arr'])
        elif metric_name in ['# of Opps', '# of Won Opps', '# of Pipegen Opps']:
            is_sig_drop = (pct_num <= -thresholds['pct_change_threshold'] and abs(abs_num) >= thresholds['abs_change_threshold_opps'])
            is_sig_inc = (pct_num >= thresholds['pct_change_threshold'] and abs(abs_num) >= thresholds['abs_change_threshold_opps'])
        elif metric_name == 'Avg Sales Price':
            is_sig_drop = (pct_num <= -thresholds['pct_change_threshold'] and abs(abs_num) >= thresholds['abs_change_threshold_asp'])
            is_sig_inc = (pct_num >= thresholds['pct_change_threshold'] and abs(abs_num) >= thresholds['abs_change_threshold_asp'])
        elif metric_name == 'Win Rate (Count)':
            # Win rate uses absolute percentage point change, not relative percentage change
            is_sig_drop = (abs_num <= -thresholds['abs_change_threshold_wr'])
            is_sig_inc = (abs_num >= thresholds['abs_change_threshold_wr'])
        elif metric_name == 'Avg Sales Cycle':
            # Sales cycle uses EITHER absolute day change OR percentage change (original logic)
            is_sig_drop = ((abs_num <= -thresholds['abs_change_threshold_asc']) or (pct_num <= -15.0))
            is_sig_inc = ((abs_num >= thresholds['abs_change_threshold_asc']) or (pct_num >= 15.0))
        elif metric_name == 'Pipegen ARR':
            is_sig_drop = (pct_num <= -thresholds['pct_change_threshold'] and abs(abs_num) >= thresholds['abs_change_threshold_pipegen'])
            is_sig_inc = (pct_num >= thresholds['pct_change_threshold'] and abs(abs_num) >= thresholds['abs_change_threshold_pipegen'])
        
        if is_sig_drop or is_sig_inc:
            change_type = 'Decrease' if is_sig_drop else 'Increase'
            
            significant_changes.append({
                'Quarter': current_q,
                'Comparison': comparison_label,
                'Group': group_name,
                'Metric': metric_name,
                'Type': change_type,
                '% Change': f"{pct_num:.2f}%",
                'Abs Change': abs_num
            })
    
    return significant_changes


def perform_trend_analysis(processed_df, inq_df, opp_id_col_name, start_fiscal_year, stage_won='Closed Won', pipegen_df=None, thresholds=None, use_original_logic=False):
    """
    Perform comprehensive trend analysis with QoQ, YoY, and 4-Quarter Average comparisons.
    
    Parameters:
    - processed_df: Main processed DataFrame
    - inq_df: In-quarter DataFrame
    - opp_id_col_name: Opportunity ID column name
    - start_fiscal_year: Starting fiscal year for analysis
    - stage_won: Won stage identifier
    - pipegen_df: Optional pipeline generation DataFrame
    - thresholds: Optional custom thresholds dictionary
    - use_original_logic: Whether to use original 7-metric logic
    
    Returns:
    - Tuple: (complete_trends_df, significant_trends_df)
    """
    print("🚀 COMPREHENSIVE TREND ANALYSIS WITH PIPEGEN SUPPORT")
    print("=" * 60)
    
    all_changes = []  # Store ALL changes
    all_significant_changes = []  # Store only significant changes
    
    # Prepare data for trend analysis
    fy_start_str = f"FY{start_fiscal_year}Q1"
    
    processed_df_for_trends = pd.DataFrame()
    inq_df_for_trends = pd.DataFrame()
    pipegen_df_for_trends = pd.DataFrame()
    
    if not processed_df.empty and 'Fiscal Period - Corrected' in processed_df.columns:
        processed_df_for_trends = processed_df[processed_df['Fiscal Period - Corrected'] >= fy_start_str].copy()
        if not inq_df.empty:
            inq_df_for_trends = inq_df[inq_df['Fiscal Period - Corrected'] >= fy_start_str].copy()
    
    # Prepare Pipegen data if available
    if pipegen_df is not None and not pipegen_df.empty:
        pipegen_df_for_trends = pipegen_df.copy()
        print("✅ Pipegen data available for trend analysis")
    else:
        print("⚠️ Pipegen data not available - will analyze standard metrics only")
    
    if processed_df_for_trends.empty or not opp_id_col_name or opp_id_col_name not in processed_df_for_trends.columns:
        print("❌ Cannot perform trend analysis: insufficient data")
        return pd.DataFrame()
    
    # Define grouping configurations
    grouping_definitions = {
        "Overall Inquarter": {'df': inq_df_for_trends, 'cols': None},
        "Overall Total": {'df': processed_df_for_trends, 'cols': None},
        "By Bookings Type": {'df': processed_df_for_trends, 'cols': 'Bookings Type'},
        "By Segment": {'df': processed_df_for_trends, 'cols': 'Segment - historical'},
        "By Segment & Bookings Type": {'df': processed_df_for_trends, 'cols': ['Segment - historical', 'Bookings Type']}
    }
    
    for group_name, G in grouping_definitions.items():
        cols_to_check = G['cols']
        if isinstance(cols_to_check, str): 
            cols_to_check = [cols_to_check]
        
        if G['df'].empty or (cols_to_check and any(c not in G['df'].columns for c in cols_to_check)):
            print(f"Skipping grouping '{group_name}' due to missing data or columns.")
            continue
        
        print(f"\n{'='*25} Analyzing Enhanced Trends for: {group_name} {'='*25}")
        
        # Calculate quarterly metrics - use original logic if requested
        if use_original_logic:
            print(f"🔄 Using ORIGINAL logic (7 metrics) for {group_name}")
            all_q_metrics = calculate_quarterly_metrics_original(
                G['df'], 
                opp_id_col_name, 
                group_by_cols=G['cols'],
                stage_won=stage_won
            )
        else:
            print(f"🔄 Using ENHANCED logic (9 metrics with Pipegen) for {group_name}")
            all_q_metrics = calculate_quarterly_metrics(
                G['df'], 
                opp_id_col_name, 
                group_by_cols=G['cols'],
                stage_won=stage_won,
                pipegen_df=pipegen_df_for_trends if not pipegen_df_for_trends.empty else None
            )
        
        if all_q_metrics.empty:
            print(f"Could not calculate quarterly metrics for {group_name}.")
            continue
        
        quarters_in_data = sorted([q for q in all_q_metrics.index.get_level_values('Fiscal Period - Corrected').unique() if q is not None])
        
        # Analyze each quarter
        for i, current_q in enumerate(quarters_in_data):
            current_q_groups = all_q_metrics[all_q_metrics.index.get_level_values('Fiscal Period - Corrected') == current_q]
            if current_q_groups.empty: 
                continue
            
            # --- Comparison 1: QoQ (Quarter-over-Quarter) ---
            if i > 0:
                prev_q = quarters_in_data[i-1]
                comparison_label = f"vs Prev Q ({prev_q})"
                if G['cols']: 
                    for group_tuple in current_q_groups.index.droplevel('Fiscal Period - Corrected'):
                        group_as_tuple = group_tuple if isinstance(group_tuple, tuple) else (group_tuple,)
                        idx_key = group_as_tuple + (prev_q,)
                        if idx_key in all_q_metrics.index:
                             comparison_table = build_comparison_view(all_q_metrics, current_q, all_q_metrics.loc[idx_key], comparison_label, group_tuple=group_as_tuple)
                             group_label_str = " - ".join(map(str, group_as_tuple))
                             # Collect ALL changes
                             all_changes.extend(collect_all_changes(comparison_table, current_q, comparison_label, f"{group_name}: {group_label_str}"))
                             # Collect significant changes
                             all_significant_changes.extend(identify_and_report_significant_changes(comparison_table, current_q, comparison_label, f"{group_name}: {group_label_str}", thresholds))
                else: # Not grouped
                    if prev_q in all_q_metrics.index:
                        comparison_table = build_comparison_view(all_q_metrics, current_q, all_q_metrics.loc[prev_q], comparison_label)
                        # Collect ALL changes
                        all_changes.extend(collect_all_changes(comparison_table, current_q, comparison_label, group_name))
                        # Collect significant changes
                        all_significant_changes.extend(identify_and_report_significant_changes(comparison_table, current_q, comparison_label, group_name, thresholds))

            # --- Comparison 2: YoY (Year-over-Year) ---
            year, q_num = int(current_q[2:6]), int(current_q[-1])
            prev_year_q = f"FY{year-1}Q{q_num}"
            if prev_year_q in quarters_in_data:
                comparison_label = f"vs Prev Year ({prev_year_q})"
                if G['cols']:
                    for group_tuple in current_q_groups.index.droplevel('Fiscal Period - Corrected'):
                        group_as_tuple = group_tuple if isinstance(group_tuple, tuple) else (group_tuple,)
                        idx_key = group_as_tuple + (prev_year_q,)
                        if idx_key in all_q_metrics.index:
                            comparison_table = build_comparison_view(all_q_metrics, current_q, all_q_metrics.loc[idx_key], comparison_label, group_tuple=group_as_tuple)
                            group_label_str = " - ".join(map(str, group_as_tuple))
                            # Collect ALL changes
                            all_changes.extend(collect_all_changes(comparison_table, current_q, comparison_label, f"{group_name}: {group_label_str}"))
                            # Collect significant changes
                            all_significant_changes.extend(identify_and_report_significant_changes(comparison_table, current_q, comparison_label, f"{group_name}: {group_label_str}", thresholds))
                else: # Not grouped
                    if prev_year_q in all_q_metrics.index:
                        comparison_table = build_comparison_view(all_q_metrics, current_q, all_q_metrics.loc[prev_year_q], comparison_label)
                        # Collect ALL changes
                        all_changes.extend(collect_all_changes(comparison_table, current_q, comparison_label, group_name))
                        # Collect significant changes
                        all_significant_changes.extend(identify_and_report_significant_changes(comparison_table, current_q, comparison_label, group_name, thresholds))

            # --- Comparison 3: vs. Prior 4-Quarter Average ---
            if i >= 4:
                prior_4_quarters = quarters_in_data[i-4:i]
                comparison_label = f"vs Avg of {prior_4_quarters[0]}-{prior_4_quarters[-1]}"
                if G['cols']:
                    for group_tuple in current_q_groups.index.droplevel('Fiscal Period - Corrected'):
                        try:
                            # Ensure group_tuple is properly handled as tuple
                            group_as_tuple = group_tuple if isinstance(group_tuple, tuple) else (group_tuple,)
                            
                            # Select all data for the specific group, then filter for the prior 4 quarters
                            data_for_group = all_q_metrics.loc[group_tuple]
                            prior_4q_group_data = data_for_group.loc[data_for_group.index.isin(prior_4_quarters)]
                            
                            # Only proceed if we have data for all 4 prior quarters for this specific group
                            if len(prior_4q_group_data) == 4:
                                prior_4q_avg_group = prior_4q_group_data.mean()
                                comparison_table = build_comparison_view(all_q_metrics, current_q, prior_4q_avg_group, comparison_label, group_tuple=group_as_tuple)
                                group_label_str = " - ".join(map(str, group_as_tuple))
                                # Collect ALL changes
                                all_changes.extend(collect_all_changes(comparison_table, current_q, comparison_label, f"{group_name}: {group_label_str}"))
                                # Collect significant changes
                                all_significant_changes.extend(identify_and_report_significant_changes(comparison_table, current_q, comparison_label, f"{group_name}: {group_label_str}", thresholds))
                        except (KeyError, IndexError):
                            continue
                else: # Not grouped
                    prior_4q_avg = all_q_metrics.loc[prior_4_quarters].mean()
                    comparison_table = build_comparison_view(all_q_metrics, current_q, prior_4q_avg, comparison_label)
                    # Collect ALL changes
                    all_changes.extend(collect_all_changes(comparison_table, current_q, comparison_label, group_name))
                    # Collect significant changes
                    all_significant_changes.extend(identify_and_report_significant_changes(comparison_table, current_q, comparison_label, group_name, thresholds))

    # Create both DataFrames
    print(f"\n{'='*25} CREATING FINAL TREND ANALYSIS RESULTS {'='*25}")
    
    # 1. Complete trends DataFrame (ALL changes)
    complete_trends_df = pd.DataFrame(all_changes) if all_changes else pd.DataFrame()
    
    # 2. Significant trends DataFrame (threshold-filtered changes)
    significant_trends_df = pd.DataFrame(all_significant_changes) if all_significant_changes else pd.DataFrame()
    
    # Display summaries
    print(f"📊 COMPLETE TRENDS ANALYSIS:")
    print(f"   • Total trend comparisons: {len(complete_trends_df)}")
    
    print(f"\n📊 SIGNIFICANT TRENDS ANALYSIS:")
    print(f"   • Trends meeting thresholds: {len(significant_trends_df)}")
    
    if not significant_trends_df.empty:
        pipegen_trends_count = len(significant_trends_df[significant_trends_df['Metric'].str.contains('Pipegen', na=False)])
        standard_trends_count = len(significant_trends_df) - pipegen_trends_count
        
        print(f"   • Standard metrics trends: {standard_trends_count}")
        print(f"   • Pipegen metrics trends: {pipegen_trends_count}")
        
        if pipegen_trends_count > 0:
            print(f"\n💡 Pipegen trends are integrated with standard metrics")
    
    if not complete_trends_df.empty:
        print(f"\n📋 METRICS INCLUDED:")
        metrics = complete_trends_df['Metric'].unique()
        print(f"   • {sorted(metrics)}")
    
    return complete_trends_df, significant_trends_df


def save_trend_results(complete_trends_df, significant_trends_df, base_filename='trends_analysis'):
    """Save both trend analysis DataFrames to CSV files."""
    saved_files = []
    
    # Save complete trends
    if not complete_trends_df.empty:
        complete_filename = f"{base_filename}_complete.csv"
        complete_trends_df.to_csv(complete_filename, index=False)
        saved_files.append(complete_filename)
        print(f"✅ Saved complete trends: '{complete_filename}' ({len(complete_trends_df)} rows)")
    
    # Save significant trends
    if not significant_trends_df.empty:
        significant_filename = f"{base_filename}_significant.csv"
        significant_trends_df.to_csv(significant_filename, index=False)
        saved_files.append(significant_filename)
        print(f"✅ Saved significant trends: '{significant_filename}' ({len(significant_trends_df)} rows)")
        
        # Summary stats for significant trends
        pipegen_count = len(significant_trends_df[significant_trends_df['Metric'].str.contains('Pipegen', na=False)])
        if pipegen_count > 0:
            print(f"   🎯 Including {pipegen_count} Pipegen-related trends")
    
    if saved_files:
        print(f"\n💾 Successfully saved {len(saved_files)} trend analysis files")
        return saved_files
    else:
        print("❌ No trend data to save")
        return []