#!/usr/bin/env python3
"""
Test Google Sheets connection
"""

import gspread
from google.oauth2.service_account import Credentials
import traceback

def test_gsheets_connection():
    """Test Google Sheets connection with detailed error reporting."""
    
    print("🔧 Testing Google Sheets Connection")
    print("=" * 50)
    
    # Method 1: Try with file path
    try:
        print("📁 Trying authentication with file path...")
        gc = gspread.service_account(filename='/Users/bchen/Downloads/gspread-428120-06947c66447d.json')
        print("✅ Authentication successful with file path!")
        
        # Test sheet access
        sheet_id = "1H63ybz81NUq9lic620az9sI0NWIYLERPvHg7DcAQzqo"
        print(f"📊 Trying to access sheet: {sheet_id}")
        
        sheet = gc.open_by_key(sheet_id)
        print(f"✅ Successfully opened sheet: {sheet.title}")
        
        # Get worksheets
        worksheets = sheet.worksheets()
        print(f"📋 Found {len(worksheets)} worksheets:")
        for i, ws in enumerate(worksheets):
            print(f"  {i+1}. {ws.title} ({ws.row_count} rows × {ws.col_count} cols)")
        
        # Try to read some data
        if worksheets:
            ws = worksheets[0]
            print(f"\n📖 Reading first 5 rows from '{ws.title}':")
            try:
                values = ws.get_all_values()[:5]  # First 5 rows
                for i, row in enumerate(values):
                    print(f"  Row {i+1}: {row[:5]}...")  # First 5 columns
            except Exception as e:
                print(f"❌ Error reading data: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Authentication/access failed: {e}")
        print(f"🔍 Error details: {traceback.format_exc()}")
        
        # Common issues and solutions
        print("\n🚨 Common issues:")
        print("1. Sheet not shared with service account email")
        print("2. Service account doesn't have proper permissions")
        print("3. Sheet ID is incorrect")
        print("4. Service account key file is invalid")
        
        # Show service account email
        try:
            import json
            with open('/Users/bchen/Downloads/gspread-428120-06947c66447d.json', 'r') as f:
                creds = json.load(f)
                print(f"\n📧 Service account email: {creds['client_email']}")
                print("👉 Make sure you've shared the Google Sheet with this email!")
        except:
            print("❌ Could not read service account email from file")
        
        return False

if __name__ == "__main__":
    success = test_gsheets_connection()
    
    if success:
        print("\n🎉 Google Sheets connection test PASSED!")
        print("✅ You can now use the dashboard with real Google Sheets data")
    else:
        print("\n❌ Google Sheets connection test FAILED!")
        print("🔧 Please fix the issues above before using the dashboard")