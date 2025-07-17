#!/usr/bin/env python3
"""
Simple launcher script for the Gears Dashboard
"""

import subprocess
import sys

def run_dashboard():
    """Run the Gears Dashboard."""
    try:
        print("🚀 Starting Gears Dashboard...")
        print("🌐 Dashboard will be available at: http://localhost:8501")
        print("🔄 Press Ctrl+C to stop the dashboard")
        print("-" * 50)
        
        # Run the Streamlit app
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            "gears_dashboard_v2.py", 
            "--server.port=8501"
        ])
        
    except KeyboardInterrupt:
        print("\n🛑 Dashboard stopped by user")
    except Exception as e:
        print(f"❌ Error running dashboard: {e}")

if __name__ == "__main__":
    run_dashboard()