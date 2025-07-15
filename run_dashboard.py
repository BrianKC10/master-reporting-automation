#!/usr/bin/env python3
"""
Dashboard Launcher Script
Simplified way to run the Streamlit dashboard with environment setup.
"""

import subprocess
import sys
import os
from pathlib import Path

def setup_environment():
    """Set up environment variables for the dashboard."""
    # Add current directory to Python path
    current_dir = Path(__file__).parent
    sys.path.insert(0, str(current_dir))
    
    # Set environment variables if not already set
    if not os.getenv('SF_USERNAME'):
        print("⚠️  Warning: Salesforce credentials not found in environment variables")
        print("   Set SF_USERNAME, SF_PASSWORD, and SF_SECURITY_TOKEN for live data")
        print("   Or the dashboard will use demo/cached data")
    
    if not os.getenv('GOOGLE_CREDENTIALS_PATH'):
        print("⚠️  Warning: Google credentials not found")
        print("   Set GOOGLE_CREDENTIALS_PATH and GOOGLE_SHEET_KEY for Google Sheets integration")

def main():
    """Main launcher function."""
    print("🚀 Starting Master Report Dashboard...")
    print("=" * 50)
    
    setup_environment()
    
    # Install requirements if needed
    try:
        import streamlit
        import plotly
    except ImportError:
        print("📦 Installing required packages...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", "requirements_streamlit.txt"
        ])
    
    # Launch Streamlit app
    print("🌐 Opening dashboard in your browser...")
    print("   Local URL: http://localhost:8501")
    print("   Network URL: http://YOUR_IP:8501")
    print("=" * 50)
    
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            "streamlit_dashboard.py",
            "--server.headless", "false",
            "--server.runOnSave", "true",
            "--browser.gatherUsageStats", "false"
        ])
    except KeyboardInterrupt:
        print("\n👋 Dashboard stopped by user")
    except Exception as e:
        print(f"❌ Error running dashboard: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()