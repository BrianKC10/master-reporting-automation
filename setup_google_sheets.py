#!/usr/bin/env python3
"""
Google Sheets Setup Script
This script helps set up Google Sheets authentication for the Gears dashboard.
"""

import os
import gspread
from google.oauth2.service_account import Credentials
import json

def setup_google_sheets():
    """Set up Google Sheets authentication."""
    
    print("🔧 Setting up Google Sheets Authentication")
    print("=" * 50)
    
    # Check if credentials already exist
    config_dir = os.path.expanduser("~/.config/gspread")
    service_account_file = os.path.join(config_dir, "service_account.json")
    
    if os.path.exists(service_account_file):
        print(f"✅ Service account file already exists at: {service_account_file}")
        try:
            gc = gspread.service_account()
            print("✅ Google Sheets authentication successful!")
            return True
        except Exception as e:
            print(f"❌ Authentication failed: {e}")
            return False
    
    print(f"📁 Creating config directory: {config_dir}")
    os.makedirs(config_dir, exist_ok=True)
    
    print("\n🔑 Google Sheets Authentication Setup")
    print("To use Google Sheets, you need to:")
    print("1. Go to https://console.cloud.google.com/")
    print("2. Create a new project or select an existing one")
    print("3. Enable the Google Sheets API")
    print("4. Create a service account")
    print("5. Download the service account key JSON file")
    print("6. Share your Google Sheet with the service account email")
    
    print(f"\n📋 Place your service account JSON file at: {service_account_file}")
    
    # Check if user wants to paste the JSON content
    response = input("\nDo you want to paste the service account JSON content now? (y/N): ").lower()
    
    if response == 'y':
        print("\n📝 Paste your service account JSON content below (press Enter twice when done):")
        lines = []
        while True:
            line = input()
            if line == "" and len(lines) > 0:
                break
            lines.append(line)
        
        json_content = "\n".join(lines)
        
        try:
            # Validate JSON
            json.loads(json_content)
            
            # Write to file
            with open(service_account_file, 'w') as f:
                f.write(json_content)
            
            print(f"✅ Service account file created at: {service_account_file}")
            
            # Test authentication
            gc = gspread.service_account()
            print("✅ Google Sheets authentication successful!")
            return True
            
        except json.JSONDecodeError:
            print("❌ Invalid JSON format. Please check your service account file.")
            return False
        except Exception as e:
            print(f"❌ Authentication failed: {e}")
            return False
    
    else:
        print(f"\n📝 Manual setup required:")
        print(f"1. Download your service account JSON file")
        print(f"2. Save it as: {service_account_file}")
        print(f"3. Run this script again to test")
        return False

def test_sheet_access():
    """Test access to the specific Google Sheet."""
    try:
        gc = gspread.service_account()
        sheet_id = "1H63ybz81NUq9lic620az9sI0NWIYLERPvHg7DcAQzqo"
        sheet = gc.open_by_key(sheet_id)
        
        print(f"✅ Successfully accessed sheet: {sheet.title}")
        print(f"📊 Worksheets: {[ws.title for ws in sheet.worksheets()]}")
        
        # Get first worksheet and show basic info
        ws = sheet.worksheets()[0]
        print(f"📋 First worksheet: {ws.title}")
        print(f"📏 Dimensions: {ws.row_count} rows × {ws.col_count} columns")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to access sheet: {e}")
        print("\n🔍 Possible solutions:")
        print("1. Make sure you've shared the sheet with your service account email")
        print("2. Check that the sheet ID is correct")
        print("3. Verify the service account has proper permissions")
        return False

if __name__ == "__main__":
    if setup_google_sheets():
        print("\n🧪 Testing sheet access...")
        test_sheet_access()
    
    print("\n🎯 Next steps:")
    print("1. Make sure your Google Sheet is shared with the service account email")
    print("2. Run the Gears dashboard to load real plan data")
    print("3. Update the parse_google_sheets_plan_data() function with actual parsing logic")