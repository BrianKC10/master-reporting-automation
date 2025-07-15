# 📊 Master Report Streamlit Dashboard

Transform your static master report into an interactive, real-time analytics dashboard with advanced visualizations and filtering capabilities.

## 🚀 Features

### 📈 Real-Time Analytics
- **Live Data Connection**: Direct integration with Salesforce
- **Auto-Refresh**: Cached data updates every hour
- **Interactive Filters**: Date range, segment, and source filtering
- **Responsive Design**: Works on desktop, tablet, and mobile

### 📊 Advanced Visualizations
- **Quarter Pacing Charts**: Track performance against targets
- **Segment Analysis**: Deep dive into Enterprise/Mid Market/SMB performance
- **Source Attribution**: Understand which channels drive results
- **Conversion Funnels**: Visualize SQL → SAO → Booking progression

### 🎯 Key Metrics Dashboard
- **Current Quarter Summary**: SQLs, SAOs, Bookings, Avg Deal Size
- **Trending Indicators**: QoQ growth, conversion rates, pacing
- **Pivot Tables**: Interactive versions of your existing reports
- **Data Export**: Download filtered data as CSV

### 🔍 Interactive Features
- **Drill-Down Analysis**: Click charts to filter data
- **Multi-Tab Interface**: Organized views for different analysis types
- **Real-Time Calculations**: Dynamic metrics based on filters
- **Responsive Tables**: Sortable, searchable data grids

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.8+
- Existing master report automation setup
- Salesforce credentials (for live data)

### Quick Start

1. **Install Dependencies**
   ```bash
   pip install -r requirements_streamlit.txt
   ```

2. **Set Environment Variables** (for live data)
   ```bash
   export SF_USERNAME="your_salesforce_username"
   export SF_PASSWORD="your_salesforce_password"
   export SF_SECURITY_TOKEN="your_security_token"
   ```

3. **Launch Dashboard**
   ```bash
   python run_dashboard.py
   ```
   
   Or directly with Streamlit:
   ```bash
   streamlit run streamlit_dashboard.py
   ```

4. **Open in Browser**
   - Navigate to `http://localhost:8501`
   - Dashboard will auto-open in your default browser

### Demo Mode

Test the dashboard without Salesforce connection:

```bash
# Generate demo data
python demo_data_generator.py

# Run dashboard with demo data
python run_dashboard.py
```

## 📱 Dashboard Sections

### 1. 🏠 Overview
- **Key Metrics Cards**: Current quarter performance
- **Quick Filters**: Date range, segment, source
- **Data Freshness**: Last update timestamp

### 2. 📈 Pacing Analysis
- **Bookings Pacing**: Cumulative ARR vs. quarter timeline
- **SQL Pacing**: Lead generation vs. quarter timeline
- **Historical Comparison**: Current vs. previous quarters
- **Target Tracking**: Performance against goals

### 3. 🎯 Segment Performance
- **Segment Breakdown**: Enterprise, Mid Market, SMB analysis
- **Conversion Rates**: SQL → SAO → Booking by segment
- **ARR Distribution**: Revenue contribution by segment
- **Trend Analysis**: QoQ growth by segment

### 4. 🔄 Source Analysis
- **Channel Performance**: AE, BDR, Channel, Marketing, Success
- **Source Attribution**: SQLs and bookings by source
- **Efficiency Metrics**: Conversion rates by source
- **Pipeline Contribution**: SAO generation by source

### 5. 📊 Pivot Tables
- **Interactive SQLs Table**: Dynamic SQL creation analysis
- **SAO Pivot**: Sales accepted opportunity breakdown
- **Pipeline Generation**: ARR pipeline by source/segment
- **Exportable Data**: Download pivot tables as CSV

### 6. 🔍 Raw Data Explorer
- **Filterable Dataset**: Full data exploration
- **Column Selector**: Choose which fields to display
- **Search & Sort**: Find specific records
- **Export Capability**: Download filtered data

## 🎨 Customization

### Styling
- **Color Scheme**: Modify `.streamlit/config.toml`
- **Custom CSS**: Edit styles in `streamlit_dashboard.py`
- **Themes**: Light/dark mode support

### Metrics
- **Custom KPIs**: Add new metrics in `get_quarter_metrics()`
- **Calculations**: Modify formulas in analysis functions
- **Visualizations**: Add new charts using Plotly

### Data Sources
- **Additional APIs**: Extend data loading functions
- **File Uploads**: Add CSV/Excel upload capability
- **Database Integration**: Connect to other data sources

## 🔧 Configuration Options

### Environment Variables
```bash
# Salesforce Connection
SF_USERNAME=your_username
SF_PASSWORD=your_password
SF_SECURITY_TOKEN=your_token

# Google Sheets Integration (optional)
GOOGLE_CREDENTIALS_PATH=/path/to/credentials.json
GOOGLE_SHEET_KEY=your_sheet_id

# Dashboard Settings
STREAMLIT_PORT=8501
CACHE_TTL=3600  # 1 hour cache
```

### Streamlit Configuration
Edit `.streamlit/config.toml`:
```toml
[server]
port = 8501
enableCORS = false

[theme]
primaryColor = "#ff6b6b"
backgroundColor = "#ffffff"
```

## 📊 Performance Features

### Data Caching
- **Smart Caching**: 1-hour cache for data queries
- **Selective Refresh**: Only reload changed data
- **Performance Optimization**: Fast filtering and aggregation

### Responsive Design
- **Mobile-First**: Optimized for all screen sizes
- **Progressive Loading**: Staged data loading
- **Efficient Rendering**: Streamlined chart generation

## 🚀 Advanced Features

### Real-Time Updates
- **Auto-Refresh**: Configurable data refresh intervals
- **Live Metrics**: Real-time KPI calculations
- **Change Detection**: Highlight data changes

### Export Capabilities
- **Multiple Formats**: CSV, Excel, JSON export
- **Filtered Data**: Export only selected data
- **Scheduled Reports**: Automated export functionality

### Integration Options
- **Slack Integration**: Send reports to Slack channels
- **Email Reports**: Automated email delivery
- **API Endpoints**: RESTful API for external access

## 🛡️ Security

### Data Protection
- **Environment Variables**: Secure credential storage
- **Local Processing**: No data leaves your environment
- **Access Control**: Optional authentication integration

### Best Practices
- **Credential Management**: Use environment variables
- **Network Security**: Run on private networks
- **Data Governance**: Respect data access policies

## 🔧 Troubleshooting

### Common Issues

**Dashboard won't start**
```bash
# Check Python version
python --version  # Requires 3.8+

# Install dependencies
pip install -r requirements_streamlit.txt

# Check port availability
lsof -i :8501
```

**No data loading**
```bash
# Check environment variables
echo $SF_USERNAME

# Test Salesforce connection
python -c "from master_report import connect_to_salesforce; print('Connection OK')"

# Use demo data
python demo_data_generator.py
```

**Performance issues**
```bash
# Clear Streamlit cache
streamlit cache clear

# Reduce data range
# Use date filters in sidebar

# Check system resources
htop  # or Task Manager on Windows
```

## 📈 Future Enhancements

### Planned Features
- **Predictive Analytics**: ML-powered forecasting
- **Automated Insights**: AI-generated recommendations
- **Advanced Filtering**: Complex query builder
- **Collaborative Features**: Comments and annotations

### Integration Roadmap
- **Salesforce Reports**: Direct report integration
- **Marketing Automation**: HubSpot/Marketo connections
- **Business Intelligence**: Tableau/Power BI integration
- **Data Warehouse**: Snowflake/BigQuery connections

## 🤝 Support

### Getting Help
- **Issues**: Report bugs via GitHub issues
- **Feature Requests**: Submit enhancement requests
- **Documentation**: Check inline code documentation
- **Community**: Join Streamlit community forums

### Contributing
- **Code Contributions**: Submit pull requests
- **Documentation**: Improve this README
- **Testing**: Add test cases and examples
- **Feedback**: Share usage experiences

## 📄 License

This dashboard is part of the Master Report Automation project. Use in accordance with your organization's data and software policies.

---

**🎯 Ready to transform your static reports into dynamic insights?**

Launch the dashboard and start exploring your data in ways you never could with static reports!

```bash
python run_dashboard.py
```