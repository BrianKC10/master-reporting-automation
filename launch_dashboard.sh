#!/bin/bash
# Master Report Dashboard Launcher

echo "🚀 Master Report Dashboard Launcher"
echo "===================================="

# Check if we're in the right directory
if [ ! -f "streamlit_dashboard.py" ]; then
    echo "❌ Error: streamlit_dashboard.py not found"
    echo "   Please run this script from the master reporting automation directory"
    exit 1
fi

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "🐍 Python version: $python_version"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source venv/bin/activate

# Install requirements
echo "📥 Installing requirements..."
pip install -r requirements_streamlit.txt

# Set environment variables if .env file exists
if [ -f ".env" ]; then
    echo "🔑 Loading environment variables..."
    export $(cat .env | xargs)
fi

# Check for Salesforce credentials
if [ -z "$SF_USERNAME" ]; then
    echo "⚠️  Warning: SF_USERNAME not set"
    echo "   Dashboard will run in demo mode"
    echo "   Generate demo data? (y/n)"
    read -r response
    if [[ "$response" =~ ^[Yy]$ ]]; then
        echo "🎲 Generating demo data..."
        python3 demo_data_generator.py
    fi
fi

# Launch dashboard
echo "🌐 Starting dashboard..."
echo "   Local URL: http://localhost:8501"
echo "   Press Ctrl+C to stop"
echo "===================================="

python3 -m streamlit run streamlit_dashboard.py --server.headless false

echo "👋 Dashboard stopped"