# Master Report Automation Setup Guide

This guide will help you set up automated daily master reports using GitHub Actions.

## Overview

- **Schedule**: Daily at 6 AM EST (11 AM UTC)
- **Data Source**: Salesforce reports
- **Output**: Google Sheets with processed data and pivot tables
- **Cost**: Free (uses GitHub Actions free tier)

## Setup Steps

### 1. Create Google Service Account

1. **Go to Google Cloud Console**: https://console.cloud.google.com
2. **Create or select a project**
3. **Enable APIs**:
   - Google Sheets API
   - Google Drive API
4. **Create Service Account**:
   - IAM & Admin → Service Accounts → Create Service Account
   - Name: `master-report-automation`
   - Role: `Editor` (or custom role with Sheets/Drive access)
5. **Generate Key**:
   - Click on service account → Keys → Add Key → Create New Key → JSON
   - Download the JSON file

### 2. Share Google Sheet with Service Account

1. **Open your Google Sheet**
2. **Click Share**
3. **Add the service account email** (from JSON file, looks like `xxx@xxx.iam.gserviceaccount.com`)
4. **Give "Editor" permission**

### 3. Configure GitHub Secrets

Go to your GitHub repo → Settings → Secrets and Variables → Actions

Add these secrets:

#### **Salesforce Credentials**
- `SF_USERNAME`: Your Salesforce username (e.g., `bchen@envoy.com`)
- `SF_PASSWORD`: Your Salesforce password
- `SF_SECURITY_TOKEN`: Your Salesforce security token

#### **Google Credentials**
- `GOOGLE_SERVICE_ACCOUNT_KEY`: Paste the entire JSON content from step 1
- `GOOGLE_SHEET_KEY`: Your Google Sheet ID (from URL: `https://docs.google.com/spreadsheets/d/SHEET_ID/edit`)

### 4. Verify Salesforce Report ID

In the `master_report.py` file, update the report ID if needed:

```python
reportId = '00OUO000009IZVD2A4'  # Update this to your report ID
```

To find your report ID:
1. Go to Salesforce → Reports
2. Open your report
3. Copy the ID from the URL

### 5. Test the Workflow

1. **Go to GitHub** → Actions → "Daily Master Report"
2. **Click "Run workflow"** → Run workflow (manual trigger)
3. **Monitor the run** - should complete in ~5-10 minutes
4. **Check Google Sheets** for updated data

## How It Works

### Daily Automation
- **Runs every day at 6 AM EST**
- **Fetches Salesforce report data**
- **Processes date columns** (adds quarter, week info)
- **Creates pivot tables** for SQLs, SAOs, and Pipegen
- **Updates Google Sheets** with all data

### Manual Trigger
- You can run it anytime via GitHub Actions UI
- Useful for testing or immediate updates

### Data Processing
The script processes these worksheets:
- **Data**: Raw Salesforce report with enhanced date columns
- **SQLs**: Lead creation pivot by source/segment
- **SAOs**: SAO date pivot by source/segment  
- **Pipegen**: Pipeline generation (ARR) by source/segment

## Monitoring

### Success Indicators
- ✅ Green checkmark in GitHub Actions
- ✅ Google Sheets updated with latest data
- ✅ CSV file generated and stored as artifact

### Failure Handling
- ❌ GitHub will email you on failures
- ❌ Logs available in Actions tab
- ❌ Can retry failed runs manually

## Troubleshooting

### Common Issues

1. **"Salesforce login failed"**:
   - Check SF_USERNAME, SF_PASSWORD, SF_SECURITY_TOKEN
   - Verify credentials work in Salesforce

2. **"Google Sheets access denied"**:
   - Check service account has access to sheet
   - Verify JSON key format in GitHub secrets

3. **"Report not found"**:
   - Update reportId in master_report.py
   - Verify report exists and is accessible

### Manual Debug
```bash
# Test locally (with proper credentials)
python master_report.py
```

## Cost Estimate

- **GitHub Actions**: Free (2,000 minutes/month)
- **Google Sheets API**: Free (100 requests/100 seconds)
- **Salesforce API**: Included in your Salesforce plan
- **Total**: $0/month

## Security Notes

- **Never commit credentials** to git
- **Use GitHub secrets** for all sensitive data
- **Rotate credentials periodically**
- **Monitor API usage** for unusual activity

## Next Steps

After setup:
1. ✅ Initialize git repository and push to GitHub
2. ✅ Set up GitHub secrets for credentials
3. ✅ Test the workflow manually
4. ✅ Monitor daily automation
5. ✅ Customize report processing if needed

The system will now automatically maintain your master report with the latest Salesforce data every day!