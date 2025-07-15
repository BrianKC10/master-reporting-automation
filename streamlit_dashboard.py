#!/usr/bin/env python3
"""
Master Report Streamlit Dashboard
Interactive analytics dashboard for Salesforce data with real-time insights.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os
import sys
from io import StringIO

# Import functions from master_report.py
from master_report import (
    connect_to_salesforce, 
    fetch_salesforce_report,
    process_date_columns,
    get_quarter_info,
    get_last_completed_quarters,
    compute_day_of_quarter,
    create_sql_pivot,
    create_sao_pivot,
    create_pipegen_pivot
)

# Page configuration
st.set_page_config(
    page_title="Master Report Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
.main > div {
    padding-top: 2rem;
}
.stMetric {
    background-color: #f0f2f6;
    padding: 1rem;
    border-radius: 0.5rem;
    border-left: 4px solid #ff6b6b;
}
.quarter-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 1.5rem;
    border-radius: 0.5rem;
    color: white;
    margin: 1rem 0;
}
</style>
""", unsafe_allow_html=True)

@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_data():
    """Load and process Salesforce data with caching."""
    try:
        sf = connect_to_salesforce()
        df = fetch_salesforce_report(sf)
        df = process_date_columns(df)
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

@st.cache_data
def get_quarter_metrics(df):
    """Calculate key metrics for the current quarter."""
    today = datetime.today()
    current_quarter = get_quarter_info(today)[0]
    
    # Filter for current quarter
    current_df = df[df['Created Date_Quarter'] == current_quarter]
    
    # Key metrics
    total_sqls = len(current_df)
    total_saos = len(current_df[current_df['SAO Date'].notna()])
    total_bookings = current_df[current_df['Stage'] == 'Closed Won']['ARR Change'].sum()
    avg_deal_size = current_df[current_df['Stage'] == 'Closed Won']['ARR Change'].mean()
    
    return {
        'total_sqls': total_sqls,
        'total_saos': total_saos,
        'total_bookings': total_bookings,
        'avg_deal_size': avg_deal_size,
        'current_quarter': current_quarter
    }

def create_pacing_chart(df, metric_type='bookings'):
    """Create quarter pacing visualization."""
    today = datetime.today()
    current_quarter = get_quarter_info(today)[0]
    
    if metric_type == 'bookings':
        # Bookings pacing
        bookings_df = df[df['Stage'] == 'Closed Won'].copy()
        bookings_df['Day_of_Quarter'] = bookings_df['Close Date'].apply(
            lambda x: compute_day_of_quarter(x)['Day_of_Quarter'] if pd.notnull(x) else None
        )
        bookings_df['Pct_Day'] = bookings_df['Close Date'].apply(
            lambda x: compute_day_of_quarter(x)['Pct_Day'] if pd.notnull(x) else None
        )
        
        # Get current quarter data
        current_data = bookings_df[bookings_df['Close Date_Quarter'] == current_quarter]
        
        # Calculate cumulative bookings by day
        daily_bookings = current_data.groupby('Pct_Day')['ARR Change'].sum().cumsum()
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=daily_bookings.index,
            y=daily_bookings.values,
            mode='lines+markers',
            name='Cumulative Bookings',
            line=dict(color='#1f77b4', width=3)
        ))
        
        fig.update_layout(
            title=f'Bookings Pacing - {current_quarter}',
            xaxis_title='% of Quarter Completed',
            yaxis_title='Cumulative ARR ($)',
            height=400
        )
        
    else:
        # SQL pacing
        sql_data = df[df['Created Date_Quarter'] == current_quarter]
        sql_data['Pct_Day'] = sql_data['Created Date'].apply(
            lambda x: compute_day_of_quarter(x)['Pct_Day'] if pd.notnull(x) else None
        )
        
        daily_sqls = sql_data.groupby('Pct_Day').size().cumsum()
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=daily_sqls.index,
            y=daily_sqls.values,
            mode='lines+markers',
            name='Cumulative SQLs',
            line=dict(color='#ff7f0e', width=3)
        ))
        
        fig.update_layout(
            title=f'SQL Pacing - {current_quarter}',
            xaxis_title='% of Quarter Completed',
            yaxis_title='Cumulative SQLs',
            height=400
        )
    
    return fig

def create_segment_analysis(df):
    """Create segment performance analysis."""
    segments = ['Enterprise', 'Mid Market', 'SMB']
    today = datetime.today()
    current_quarter = get_quarter_info(today)[0]
    
    # Filter for current quarter and valid segments
    current_df = df[
        (df['Created Date_Quarter'] == current_quarter) & 
        (df['Segment - historical'].isin(segments))
    ]
    
    # Calculate metrics by segment
    segment_metrics = []
    for segment in segments:
        seg_data = current_df[current_df['Segment - historical'] == segment]
        
        sqls = len(seg_data)
        saos = len(seg_data[seg_data['SAO Date'].notna()])
        bookings = seg_data[seg_data['Stage'] == 'Closed Won']['ARR Change'].sum()
        conversion_rate = (saos / sqls * 100) if sqls > 0 else 0
        
        segment_metrics.append({
            'Segment': segment,
            'SQLs': sqls,
            'SAOs': saos,
            'Bookings': bookings,
            'Conversion Rate': conversion_rate
        })
    
    seg_df = pd.DataFrame(segment_metrics)
    
    # Create visualization
    fig = px.bar(
        seg_df,
        x='Segment',
        y='SQLs',
        title=f'SQL Performance by Segment - {current_quarter}',
        color='Conversion Rate',
        color_continuous_scale='RdYlBu_r',
        text='SQLs'
    )
    
    fig.update_traces(texttemplate='%{text}', textposition='outside')
    fig.update_layout(height=400)
    
    return fig, seg_df

def create_source_analysis(df):
    """Create source performance analysis."""
    sources = ['AE', 'BDR', 'Channel', 'Marketing', 'Success']
    today = datetime.today()
    current_quarter = get_quarter_info(today)[0]
    
    # Filter for current quarter and valid sources
    current_df = df[
        (df['Created Date_Quarter'] == current_quarter) & 
        (df['Source'].isin(sources))
    ]
    
    # Calculate metrics by source
    source_metrics = current_df.groupby('Source').agg({
        'SFDC ID 18 Digit': 'count',
        'SAO Date': lambda x: x.notna().sum(),
        'ARR Change': lambda x: x[current_df['Stage'] == 'Closed Won'].sum()
    }).rename(columns={
        'SFDC ID 18 Digit': 'SQLs',
        'SAO Date': 'SAOs',
        'ARR Change': 'Bookings'
    })
    
    source_metrics['Conversion Rate'] = (source_metrics['SAOs'] / source_metrics['SQLs'] * 100).fillna(0)
    
    # Create sunburst chart
    fig = px.sunburst(
        values=source_metrics['SQLs'].tolist(),
        names=source_metrics.index.tolist(),
        title=f'SQL Distribution by Source - {current_quarter}'
    )
    
    fig.update_layout(height=400)
    
    return fig, source_metrics

def main():
    """Main dashboard function."""
    st.title("📊 Master Report Dashboard")
    st.markdown("Interactive analytics for Salesforce performance data")
    
    # Sidebar filters
    st.sidebar.header("🎛️ Filters")
    
    # Load data
    with st.spinner("Loading data..."):
        df = load_data()
    
    if df is None:
        st.error("Failed to load data. Please check your connection.")
        return
    
    # Date range filter
    min_date = df['Created Date'].min().date()
    max_date = df['Created Date'].max().date()
    
    date_range = st.sidebar.date_input(
        "Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )
    
    # Segment filter
    segments = ['All'] + sorted(df['Segment - historical'].dropna().unique().tolist())
    selected_segment = st.sidebar.selectbox("Segment", segments)
    
    # Source filter
    sources = ['All'] + sorted(df['Source'].dropna().unique().tolist())
    selected_source = st.sidebar.selectbox("Source", sources)
    
    # Apply filters
    filtered_df = df.copy()
    
    if len(date_range) == 2:
        filtered_df = filtered_df[
            (filtered_df['Created Date'].dt.date >= date_range[0]) &
            (filtered_df['Created Date'].dt.date <= date_range[1])
        ]
    
    if selected_segment != 'All':
        filtered_df = filtered_df[filtered_df['Segment - historical'] == selected_segment]
    
    if selected_source != 'All':
        filtered_df = filtered_df[filtered_df['Source'] == selected_source]
    
    # Key metrics
    metrics = get_quarter_metrics(filtered_df)
    
    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total SQLs",
            f"{metrics['total_sqls']:,}",
            delta=f"Q: {metrics['current_quarter']}"
        )
    
    with col2:
        st.metric(
            "Total SAOs",
            f"{metrics['total_saos']:,}",
            delta=f"{metrics['total_saos']/metrics['total_sqls']*100:.1f}% conversion" if metrics['total_sqls'] > 0 else "0% conversion"
        )
    
    with col3:
        st.metric(
            "Total Bookings",
            f"${metrics['total_bookings']:,.0f}",
            delta=f"ARR Change"
        )
    
    with col4:
        st.metric(
            "Avg Deal Size",
            f"${metrics['avg_deal_size']:,.0f}" if not np.isnan(metrics['avg_deal_size']) else "$0",
            delta="Per Deal"
        )
    
    # Tabs for different views
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Pacing", "🎯 Segments", "🔄 Sources", "📊 Pivot Tables", "🔍 Raw Data"
    ])
    
    with tab1:
        st.header("Quarter Pacing Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            bookings_chart = create_pacing_chart(filtered_df, 'bookings')
            st.plotly_chart(bookings_chart, use_container_width=True)
        
        with col2:
            sql_chart = create_pacing_chart(filtered_df, 'sqls')
            st.plotly_chart(sql_chart, use_container_width=True)
    
    with tab2:
        st.header("Segment Performance")
        
        segment_chart, segment_data = create_segment_analysis(filtered_df)
        st.plotly_chart(segment_chart, use_container_width=True)
        
        st.subheader("Segment Details")
        st.dataframe(segment_data, use_container_width=True)
    
    with tab3:
        st.header("Source Analysis")
        
        source_chart, source_data = create_source_analysis(filtered_df)
        st.plotly_chart(source_chart, use_container_width=True)
        
        st.subheader("Source Performance")
        st.dataframe(source_data, use_container_width=True)
    
    with tab4:
        st.header("Pivot Tables")
        
        # Create pivot tables
        sql_pivot = create_sql_pivot(filtered_df)
        sao_pivot = create_sao_pivot(filtered_df)
        pipegen_pivot = create_pipegen_pivot(filtered_df)
        
        pivot_tab1, pivot_tab2, pivot_tab3 = st.tabs(["SQLs", "SAOs", "Pipegen"])
        
        with pivot_tab1:
            st.subheader("SQL Pivot Table")
            st.dataframe(sql_pivot, use_container_width=True)
        
        with pivot_tab2:
            st.subheader("SAO Pivot Table")
            st.dataframe(sao_pivot, use_container_width=True)
        
        with pivot_tab3:
            st.subheader("Pipeline Generation Pivot Table")
            st.dataframe(pipegen_pivot, use_container_width=True)
    
    with tab5:
        st.header("Raw Data Explorer")
        
        # Data summary
        st.subheader("Data Summary")
        st.write(f"Total Records: {len(filtered_df):,}")
        st.write(f"Date Range: {filtered_df['Created Date'].min().date()} to {filtered_df['Created Date'].max().date()}")
        
        # Column selector
        all_columns = filtered_df.columns.tolist()
        selected_columns = st.multiselect(
            "Select columns to display:",
            all_columns,
            default=['Created Date', 'Source', 'Segment - historical', 'Bookings Type', 'Stage', 'ARR Change'][:6]
        )
        
        if selected_columns:
            st.dataframe(filtered_df[selected_columns], use_container_width=True)
        
        # Download button
        csv = filtered_df.to_csv(index=False)
        st.download_button(
            label="Download filtered data as CSV",
            data=csv,
            file_name=f"master_report_filtered_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    
    # Footer
    st.markdown("---")
    st.markdown(
        f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | "
        f"Data points: {len(filtered_df):,} | "
        f"🤖 Master Report Dashboard"
    )

if __name__ == "__main__":
    main()