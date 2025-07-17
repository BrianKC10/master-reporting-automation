# Master Reporting Automation

This project contains the Gears Dashboard for pipeline attainment analysis.

## Directory Structure

```
Master reporting automation/
├── streamlit/                    # Streamlit dashboard application
│   ├── gears_dashboard_v2.py    # Main dashboard application
│   ├── extract_plan_data.py     # Plan data extraction script
│   ├── run_dashboard.py         # Dashboard launcher script
│   └── .streamlit/              # Streamlit configuration
├── data_sources/                # Data files for the dashboard
│   ├── Master - SQLs.csv       # Main SQLs data
│   ├── master_report.csv       # Master report data
│   └── plan_data/              # Plan data CSVs
└── [other files...]            # Analysis notebooks, utilities, etc.
```

## Quick Start

### Running the Dashboard

1. **Navigate to the streamlit directory:**
   ```bash
   cd "Master reporting automation/streamlit"
   ```

2. **Run the dashboard:**
   ```bash
   streamlit run gears_dashboard_v2.py
   ```

   Or use the launcher script:
   ```bash
   python run_dashboard.py
   ```

3. **Access the dashboard:**
   Open your browser and go to `http://localhost:8501`

### Updating Plan Data

When the Google Sheets plan data is updated, run:
```bash
cd "Master reporting automation/streamlit"
python extract_plan_data.py
```

## Dashboard Features

- **Source Summary**: Aggregated data by source
- **Source × Segment × Booking Type**: Detailed breakdown by source, then segment, then booking type
- **Segment × Source × Booking Type**: Data organized by segment first (SMB, Mid Market, Enterprise)

## Data Sources

- **Master - SQLs.csv**: Contains the three pre-calculated tables with actual SQLs data
- **plan_data/**: CSV files with plan data extracted from Google Sheets
- **master_report.csv**: Master report data for additional analysis

## Notes

- The dashboard loads plan data from CSV files for improved performance
- Plan data is extracted from Google Sheets using the `extract_plan_data.py` script
- All tables display proper formatting with color-coded attainment levels
- Segments are ordered as SMB → Mid Market → Enterprise