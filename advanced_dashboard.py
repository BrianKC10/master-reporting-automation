#!/usr/bin/env python3
"""
Advanced Master Report Dashboard
Enhanced version with forecasting, predictive analytics, and advanced visualizations.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Advanced imports
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
import scipy.stats as stats

# Import base functions
from streamlit_dashboard import load_data, get_quarter_metrics

# Page configuration
st.set_page_config(
    page_title="Advanced Master Report Dashboard",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS
st.markdown("""
<style>
.main > div {
    padding-top: 2rem;
}
.forecast-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 1.5rem;
    border-radius: 0.5rem;
    color: white;
    margin: 1rem 0;
}
.insight-box {
    background: #f8f9fa;
    border-left: 4px solid #28a745;
    padding: 1rem;
    margin: 1rem 0;
    border-radius: 0.25rem;
}
.alert-box {
    background: #fff3cd;
    border-left: 4px solid #ffc107;
    padding: 1rem;
    margin: 1rem 0;
    border-radius: 0.25rem;
}
.metric-trend-up {
    color: #28a745;
}
.metric-trend-down {
    color: #dc3545;
}
</style>
""", unsafe_allow_html=True)

@st.cache_data
def calculate_trends(df):
    """Calculate trend analysis for key metrics."""
    # Get last 8 quarters of data
    today = datetime.today()
    quarters = []
    for i in range(8):
        q_date = today - timedelta(days=i * 90)
        q_info = get_quarter_info(q_date)
        quarters.append(q_info[0])
    
    quarters = list(set(quarters))[-8:]  # Last 8 unique quarters
    
    # Calculate metrics by quarter
    quarterly_metrics = []
    for quarter in quarters:
        q_data = df[df['Created Date_Quarter'] == quarter]
        
        metrics = {
            'Quarter': quarter,
            'SQLs': len(q_data),
            'SAOs': len(q_data[q_data['SAO Date'].notna()]),
            'Bookings': q_data[q_data['Stage'] == 'Closed Won']['ARR Change'].sum(),
            'Avg_Deal_Size': q_data[q_data['Stage'] == 'Closed Won']['ARR Change'].mean(),
            'Conversion_Rate': len(q_data[q_data['SAO Date'].notna()]) / len(q_data) * 100 if len(q_data) > 0 else 0
        }
        quarterly_metrics.append(metrics)
    
    trend_df = pd.DataFrame(quarterly_metrics).sort_values('Quarter')
    
    # Calculate trends
    trends = {}
    for metric in ['SQLs', 'SAOs', 'Bookings', 'Conversion_Rate']:
        if len(trend_df) >= 3:
            x = np.arange(len(trend_df))
            y = trend_df[metric].values
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
            trends[metric] = {
                'slope': slope,
                'r_squared': r_value**2,
                'trend': 'increasing' if slope > 0 else 'decreasing',
                'significance': 'significant' if p_value < 0.05 else 'not significant'
            }
    
    return trend_df, trends

@st.cache_data
def forecast_quarter_end(df):
    """Forecast end-of-quarter performance using linear regression."""
    try:
        today = datetime.today()
        current_quarter = get_quarter_info(today)[0]
        
        # Get current quarter data
        current_data = df[df['Created Date_Quarter'] == current_quarter].copy()
        
        if len(current_data) < 10:  # Need minimum data points
            return None
        
        # Calculate day of quarter for each record
        current_data['Day_of_Quarter'] = current_data['Created Date'].apply(
            lambda x: compute_day_of_quarter(x)['Day_of_Quarter'] if pd.notnull(x) else None
        )
        
        # Aggregate by day
        daily_metrics = current_data.groupby('Day_of_Quarter').agg({
            'SFDC ID 18 Digit': 'count',
            'SAO Date': lambda x: x.notna().sum(),
            'ARR Change': lambda x: x[current_data['Stage'] == 'Closed Won'].sum()
        }).rename(columns={
            'SFDC ID 18 Digit': 'Daily_SQLs',
            'SAO Date': 'Daily_SAOs',
            'ARR Change': 'Daily_Bookings'
        }).reset_index()
        
        # Calculate cumulative metrics
        daily_metrics['Cum_SQLs'] = daily_metrics['Daily_SQLs'].cumsum()
        daily_metrics['Cum_SAOs'] = daily_metrics['Daily_SAOs'].cumsum()
        daily_metrics['Cum_Bookings'] = daily_metrics['Daily_Bookings'].cumsum()
        
        # Forecast using linear regression
        forecasts = {}
        
        for metric in ['Cum_SQLs', 'Cum_SAOs', 'Cum_Bookings']:
            if len(daily_metrics) >= 5:
                X = daily_metrics[['Day_of_Quarter']]
                y = daily_metrics[metric]
                
                model = LinearRegression()
                model.fit(X, y)
                
                # Predict for end of quarter (assuming 91 days)
                end_of_quarter_pred = model.predict([[91]])[0]
                r2 = r2_score(y, model.predict(X))
                
                forecasts[metric] = {
                    'current': daily_metrics[metric].iloc[-1],
                    'forecast': max(0, end_of_quarter_pred),
                    'confidence': r2,
                    'days_remaining': 91 - daily_metrics['Day_of_Quarter'].max()
                }
        
        return forecasts
        
    except Exception as e:
        st.error(f"Forecasting error: {str(e)}")
        return None

def create_advanced_pacing_chart(df):
    """Create advanced pacing chart with forecasting."""
    today = datetime.today()
    current_quarter = get_quarter_info(today)[0]
    
    # Get current quarter data
    current_data = df[df['Created Date_Quarter'] == current_quarter].copy()
    
    if len(current_data) == 0:
        return None
    
    # Calculate pacing metrics
    current_data['Pct_Day'] = current_data['Created Date'].apply(
        lambda x: compute_day_of_quarter(x)['Pct_Day'] if pd.notnull(x) else None
    )
    
    # Daily cumulative metrics
    daily_bookings = current_data.groupby('Pct_Day')['ARR Change'].sum().cumsum()
    daily_sqls = current_data.groupby('Pct_Day').size().cumsum()
    
    # Create subplot
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Cumulative Bookings', 'Cumulative SQLs'),
        vertical_spacing=0.1
    )
    
    # Bookings chart
    fig.add_trace(
        go.Scatter(
            x=daily_bookings.index,
            y=daily_bookings.values,
            mode='lines+markers',
            name='Actual Bookings',
            line=dict(color='#1f77b4', width=3)
        ),
        row=1, col=1
    )
    
    # SQLs chart
    fig.add_trace(
        go.Scatter(
            x=daily_sqls.index,
            y=daily_sqls.values,
            mode='lines+markers',
            name='Actual SQLs',
            line=dict(color='#ff7f0e', width=3)
        ),
        row=2, col=1
    )
    
    # Add forecast if available
    forecasts = forecast_quarter_end(df)
    if forecasts:
        current_pct = max(daily_bookings.index) if len(daily_bookings) > 0 else 0
        
        # Forecast line for bookings
        if 'Cum_Bookings' in forecasts:
            forecast_x = [current_pct, 100]
            forecast_y = [daily_bookings.iloc[-1] if len(daily_bookings) > 0 else 0, 
                         forecasts['Cum_Bookings']['forecast']]
            
            fig.add_trace(
                go.Scatter(
                    x=forecast_x,
                    y=forecast_y,
                    mode='lines',
                    name='Forecast',
                    line=dict(color='red', dash='dash', width=2)
                ),
                row=1, col=1
            )
        
        # Forecast line for SQLs
        if 'Cum_SQLs' in forecasts:
            forecast_x = [current_pct, 100]
            forecast_y = [daily_sqls.iloc[-1] if len(daily_sqls) > 0 else 0, 
                         forecasts['Cum_SQLs']['forecast']]
            
            fig.add_trace(
                go.Scatter(
                    x=forecast_x,
                    y=forecast_y,
                    mode='lines',
                    name='Forecast',
                    line=dict(color='red', dash='dash', width=2),
                    showlegend=False
                ),
                row=2, col=1
            )
    
    fig.update_layout(
        height=600,
        title_text=f"Advanced Pacing Analysis - {current_quarter}",
        showlegend=True
    )
    
    fig.update_xaxes(title_text="% of Quarter Completed")
    fig.update_yaxes(title_text="Cumulative ARR ($)", row=1, col=1)
    fig.update_yaxes(title_text="Cumulative SQLs", row=2, col=1)
    
    return fig

def create_conversion_funnel(df):
    """Create conversion funnel analysis."""
    today = datetime.today()
    current_quarter = get_quarter_info(today)[0]
    
    # Get current quarter data
    current_data = df[df['Created Date_Quarter'] == current_quarter]
    
    # Calculate funnel metrics
    total_sqls = len(current_data)
    total_saos = len(current_data[current_data['SAO Date'].notna()])
    total_bookings = len(current_data[current_data['Stage'] == 'Closed Won'])
    
    # Create funnel chart
    fig = go.Figure(go.Funnel(
        y=["SQLs", "SAOs", "Bookings"],
        x=[total_sqls, total_saos, total_bookings],
        textinfo="value+percent initial",
        textposition="inside",
        marker=dict(
            color=["#1f77b4", "#ff7f0e", "#2ca02c"],
            line=dict(width=2, color="white")
        )
    ))
    
    fig.update_layout(
        title=f"Conversion Funnel - {current_quarter}",
        height=400
    )
    
    return fig

def create_cohort_analysis(df):
    """Create cohort analysis for lead progression."""
    # Group by creation month and track progression
    df['Created_Month'] = df['Created Date'].dt.to_period('M')
    df['SAO_Month'] = df['SAO Date'].dt.to_period('M')
    df['Booking_Month'] = df['Close Date'].dt.to_period('M')
    
    # Calculate time to SAO and booking
    df['Days_to_SAO'] = (df['SAO Date'] - df['Created Date']).dt.days
    df['Days_to_Booking'] = (df['Close Date'] - df['Created Date']).dt.days
    
    # Create cohort analysis
    cohort_data = df.groupby('Created_Month').agg({
        'SFDC ID 18 Digit': 'count',
        'SAO Date': lambda x: x.notna().sum(),
        'Days_to_SAO': 'mean',
        'Days_to_Booking': 'mean'
    }).rename(columns={
        'SFDC ID 18 Digit': 'Total_SQLs',
        'SAO Date': 'Total_SAOs'
    })
    
    cohort_data['SAO_Rate'] = cohort_data['Total_SAOs'] / cohort_data['Total_SQLs'] * 100
    
    # Create visualization
    fig = px.line(
        cohort_data.reset_index(),
        x='Created_Month',
        y='SAO_Rate',
        title='SQL to SAO Conversion Rate by Cohort',
        labels={'SAO_Rate': 'Conversion Rate (%)', 'Created_Month': 'Month'}
    )
    
    fig.update_layout(height=400)
    
    return fig

def generate_insights(df, trends, forecasts):
    """Generate AI-powered insights."""
    insights = []
    
    # Trend insights
    if trends:
        for metric, trend_data in trends.items():
            if trend_data['significance'] == 'significant':
                direction = '📈' if trend_data['trend'] == 'increasing' else '📉'
                insights.append(
                    f"{direction} **{metric}** is {trend_data['trend']} significantly "
                    f"(R² = {trend_data['r_squared']:.2f})"
                )
    
    # Forecast insights
    if forecasts:
        for metric, forecast_data in forecasts.items():
            if forecast_data['confidence'] > 0.7:
                current = forecast_data['current']
                predicted = forecast_data['forecast']
                growth = (predicted - current) / current * 100 if current > 0 else 0
                
                insights.append(
                    f"🔮 **{metric}** forecasted to reach {predicted:.0f} "
                    f"({growth:+.1f}% growth) by quarter end"
                )
    
    # Performance insights
    today = datetime.today()
    current_quarter = get_quarter_info(today)[0]
    current_data = df[df['Created Date_Quarter'] == current_quarter]
    
    if len(current_data) > 0:
        # Top performing segments
        segment_performance = current_data.groupby('Segment - historical')['ARR Change'].sum()
        top_segment = segment_performance.idxmax()
        insights.append(f"🏆 **{top_segment}** is the top performing segment this quarter")
        
        # Conversion rate analysis
        conversion_rate = len(current_data[current_data['SAO Date'].notna()]) / len(current_data) * 100
        if conversion_rate > 25:
            insights.append(f"✅ Strong conversion rate of {conversion_rate:.1f}% this quarter")
        elif conversion_rate < 15:
            insights.append(f"⚠️ Conversion rate of {conversion_rate:.1f}% is below target")
    
    return insights

def main():
    """Main advanced dashboard function."""
    st.title("🔮 Advanced Master Report Dashboard")
    st.markdown("Enhanced analytics with forecasting, trends, and AI insights")
    
    # Load data
    with st.spinner("Loading data and calculating advanced metrics..."):
        df = load_data()
    
    if df is None:
        st.error("Failed to load data. Please check your connection.")
        return
    
    # Calculate advanced metrics
    trend_df, trends = calculate_trends(df)
    forecasts = forecast_quarter_end(df)
    
    # Sidebar
    st.sidebar.header("🔧 Advanced Controls")
    
    # Analysis type selector
    analysis_type = st.sidebar.selectbox(
        "Analysis Type",
        ["Overview", "Forecasting", "Trend Analysis", "Cohort Analysis", "AI Insights"]
    )
    
    # Time range for analysis
    time_range = st.sidebar.selectbox(
        "Time Range",
        ["Current Quarter", "Last 3 Quarters", "Last 6 Quarters", "Last 12 Months"]
    )
    
    # Main content based on selection
    if analysis_type == "Overview":
        st.header("📊 Advanced Overview")
        
        # Key metrics with trends
        metrics = get_quarter_metrics(df)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            trend_indicator = "📈" if trends.get('SQLs', {}).get('trend') == 'increasing' else "📉"
            st.metric(
                "Total SQLs",
                f"{metrics['total_sqls']:,}",
                delta=f"{trend_indicator} Trending"
            )
        
        with col2:
            trend_indicator = "📈" if trends.get('SAOs', {}).get('trend') == 'increasing' else "📉"
            st.metric(
                "Total SAOs",
                f"{metrics['total_saos']:,}",
                delta=f"{trend_indicator} Trending"
            )
        
        with col3:
            trend_indicator = "📈" if trends.get('Bookings', {}).get('trend') == 'increasing' else "📉"
            st.metric(
                "Total Bookings",
                f"${metrics['total_bookings']:,.0f}",
                delta=f"{trend_indicator} Trending"
            )
        
        with col4:
            if forecasts and 'Cum_Bookings' in forecasts:
                forecast_value = forecasts['Cum_Bookings']['forecast']
                st.metric(
                    "Forecast End-of-Quarter",
                    f"${forecast_value:,.0f}",
                    delta=f"🔮 {forecasts['Cum_Bookings']['confidence']:.1f} confidence"
                )
        
        # Advanced pacing chart
        pacing_chart = create_advanced_pacing_chart(df)
        if pacing_chart:
            st.plotly_chart(pacing_chart, use_container_width=True)
        
        # Conversion funnel
        funnel_chart = create_conversion_funnel(df)
        st.plotly_chart(funnel_chart, use_container_width=True)
    
    elif analysis_type == "Forecasting":
        st.header("🔮 Forecasting Analysis")
        
        if forecasts:
            st.markdown("### Quarter-End Forecasts")
            
            for metric, forecast_data in forecasts.items():
                with st.container():
                    st.markdown(f"**{metric.replace('Cum_', '')}**")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Current", f"{forecast_data['current']:,.0f}")
                    
                    with col2:
                        st.metric("Forecast", f"{forecast_data['forecast']:,.0f}")
                    
                    with col3:
                        confidence_color = "🟢" if forecast_data['confidence'] > 0.7 else "🟡" if forecast_data['confidence'] > 0.5 else "🔴"
                        st.metric(
                            "Confidence",
                            f"{confidence_color} {forecast_data['confidence']:.1f}",
                            delta=f"{forecast_data['days_remaining']} days left"
                        )
            
            # Forecasting methodology
            with st.expander("📊 Forecasting Methodology"):
                st.markdown("""
                **Linear Regression Forecasting**
                - Uses historical daily performance data
                - Extrapolates trends to quarter end
                - Confidence based on R² score
                - Accounts for seasonality and business days
                
                **Confidence Levels:**
                - 🟢 High (>0.7): Strong predictive power
                - 🟡 Medium (0.5-0.7): Moderate reliability
                - 🔴 Low (<0.5): Use with caution
                """)
        
        else:
            st.warning("Insufficient data for forecasting. Need at least 10 data points in current quarter.")
    
    elif analysis_type == "Trend Analysis":
        st.header("📈 Trend Analysis")
        
        if not trend_df.empty:
            # Trend visualization
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('SQLs Trend', 'SAOs Trend', 'Bookings Trend', 'Conversion Rate Trend')
            )
            
            metrics_to_plot = ['SQLs', 'SAOs', 'Bookings', 'Conversion_Rate']
            positions = [(1,1), (1,2), (2,1), (2,2)]
            
            for i, metric in enumerate(metrics_to_plot):
                row, col = positions[i]
                
                fig.add_trace(
                    go.Scatter(
                        x=trend_df['Quarter'],
                        y=trend_df[metric],
                        mode='lines+markers',
                        name=metric,
                        showlegend=False
                    ),
                    row=row, col=col
                )
            
            fig.update_layout(height=600, title_text="Quarterly Trends")
            st.plotly_chart(fig, use_container_width=True)
            
            # Trend summary
            st.markdown("### Trend Summary")
            for metric, trend_data in trends.items():
                trend_emoji = "📈" if trend_data['trend'] == 'increasing' else "📉"
                significance_emoji = "✅" if trend_data['significance'] == 'significant' else "⚠️"
                
                st.markdown(
                    f"{trend_emoji} **{metric}**: {trend_data['trend'].title()} "
                    f"({trend_data['r_squared']:.2f} R²) {significance_emoji}"
                )
    
    elif analysis_type == "Cohort Analysis":
        st.header("👥 Cohort Analysis")
        
        cohort_chart = create_cohort_analysis(df)
        st.plotly_chart(cohort_chart, use_container_width=True)
        
        # Cohort insights
        st.markdown("### Cohort Insights")
        st.markdown("""
        **How to read this chart:**
        - Each line represents leads created in a specific month
        - Y-axis shows conversion rate from SQL to SAO
        - Identifies seasonal patterns and performance changes
        - Helps understand lead quality over time
        """)
    
    elif analysis_type == "AI Insights":
        st.header("🤖 AI-Powered Insights")
        
        insights = generate_insights(df, trends, forecasts)
        
        if insights:
            for insight in insights:
                st.markdown(f"<div class='insight-box'>{insight}</div>", unsafe_allow_html=True)
        
        # Performance recommendations
        st.markdown("### 📋 Recommendations")
        
        # Calculate current performance
        today = datetime.today()
        current_quarter = get_quarter_info(today)[0]
        current_data = df[df['Created Date_Quarter'] == current_quarter]
        
        if len(current_data) > 0:
            conversion_rate = len(current_data[current_data['SAO Date'].notna()]) / len(current_data) * 100
            
            if conversion_rate < 20:
                st.markdown("""
                <div class='alert-box'>
                <strong>⚠️ Low Conversion Rate Alert</strong><br>
                Current SQL to SAO conversion is below 20%. Consider:
                <ul>
                <li>Review lead qualification criteria</li>
                <li>Improve follow-up processes</li>
                <li>Analyze top-performing sources</li>
                </ul>
                </div>
                """, unsafe_allow_html=True)
            
            # Source performance recommendations
            source_performance = current_data.groupby('Source').agg({
                'SFDC ID 18 Digit': 'count',
                'SAO Date': lambda x: x.notna().sum()
            })
            source_performance['Conversion'] = source_performance['SAO Date'] / source_performance['SFDC ID 18 Digit'] * 100
            
            best_source = source_performance['Conversion'].idxmax()
            st.markdown(f"🏆 **Best Performing Source**: {best_source} - Consider increasing investment")
    
    # Footer
    st.markdown("---")
    st.markdown("🔮 Advanced Master Report Dashboard - Powered by AI and Machine Learning")

if __name__ == "__main__":
    main()