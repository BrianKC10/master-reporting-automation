# Streamlit Cloud Deployment Guide

## Prerequisites

1. **GitHub Repository**: Your code should be in a GitHub repository
2. **Streamlit Cloud Account**: Sign up at [share.streamlit.io](https://share.streamlit.io)
3. **Google Sheets Access**: Service account credentials for Google Sheets API

## Step 1: Prepare Your Repository

### 1.1 Commit and Push Your Changes

```bash
cd "Master reporting automation"
git add .
git commit -m "Add organized dashboard structure and deployment files"
git push origin main
```

### 1.2 Repository Structure

Your repository should have this structure:
```
Master reporting automation/
├── streamlit/
│   ├── gears_dashboard_v2.py    # Main app
│   ├── extract_plan_data.py     # Plan data extraction
│   ├── requirements.txt         # Dependencies
│   └── .streamlit/
│       └── secrets.toml         # Google Sheets credentials
├── data_sources/
│   ├── Master - SQLs.csv
│   ├── master_report.csv
│   └── plan_data/
└── README.md
```

## Step 2: Deploy to Streamlit Cloud

### 2.1 Create a New App

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Click "New app"
3. Connect your GitHub account if not already connected
4. Select your repository: `master-reporting-automation`
5. Set the main file path: `streamlit/gears_dashboard_v2.py`
6. Click "Deploy"

### 2.2 Configure Secrets

1. In your Streamlit Cloud app dashboard, go to "Settings" → "Secrets"
2. Add your Google Sheets service account credentials:

```toml
[gcp_service_account]
type = "service_account"
project_id = "your-project-id"
private_key_id = "your-private-key-id"
private_key = "-----BEGIN PRIVATE KEY-----\nYour private key here\n-----END PRIVATE KEY-----\n"
client_email = "your-service-account-email@your-project.iam.gserviceaccount.com"
client_id = "your-client-id"
auth_uri = "https://accounts.google.com/o/oauth2/auth"
token_uri = "https://oauth2.googleapis.com/token"
auth_provider_x509_cert_url = "https://www.googleapis.com/oauth2/v1/certs"
client_x509_cert_url = "https://www.googleapis.com/robot/v1/metadata/x509/your-service-account-email%40your-project.iam.gserviceaccount.com"
```

## Step 3: Test Your Deployment

1. Wait for the deployment to complete
2. Test all three tables in the dashboard
3. Verify that Google Sheets integration works
4. Check that all formatting is correct

## Step 4: Update Plan Data

To update plan data after deployment:

1. Run the plan extraction script locally:
   ```bash
   cd streamlit
   python extract_plan_data.py
   ```

2. Commit and push the updated CSV files:
   ```bash
   git add data_sources/plan_data/
   git commit -m "Update plan data"
   git push origin main
   ```

3. Streamlit Cloud will automatically redeploy with the new data

## Troubleshooting

### Common Issues

1. **Import Errors**: Check that all dependencies are in `requirements.txt`
2. **File Not Found**: Ensure file paths are correct relative to the main app
3. **Google Sheets Access**: Verify service account permissions and secrets configuration
4. **Data Loading**: Check that CSV files are properly committed to the repository

### Debugging

- Check the logs in Streamlit Cloud dashboard
- Test locally first: `streamlit run gears_dashboard_v2.py`
- Verify file paths and permissions

## Security Notes

- Never commit Google Sheets credentials to the repository
- Use Streamlit Cloud secrets for sensitive configuration
- Keep the service account key secure and rotate it regularly

## Performance Optimization

- CSV files are cached for better performance
- Plan data is pre-extracted to avoid real-time Google Sheets queries
- Consider implementing data refresh schedules if needed