# API Setup Instructions

This guide walks you through setting up authentication for Google Ads API, Google Analytics 4 API, and Anthropic API.

## Prerequisites

1. Python 3.11 or 3.12 installed
2. Poetry installed (`curl -sSL https://install.python-poetry.org | python3 -`)
3. Project dependencies installed (`poetry install`)

## Google Ads API Setup

### Step 1: Create Google Cloud Project

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select an existing one
3. Enable the Google Ads API:
   - Go to "APIs & Services" > "Library"
   - Search for "Google Ads API"
   - Click "Enable"

### Step 2: Create OAuth2 Credentials

1. Go to "APIs & Services" > "Credentials"
2. Click "Create Credentials" > "OAuth 2.0 Client IDs"
3. Choose "Web application"
4. Add authorized redirect URI: `http://localhost:8080`
5. Save the Client ID and Client Secret

### Step 3: Get Developer Token

1. Go to [Google Ads API Center](https://developers.google.com/google-ads/api/)
2. Sign in with your Google Ads account
3. Apply for a developer token
4. Wait for approval (can take several days)

### Step 4: Generate Refresh Token

Run the OAuth flow to get a refresh token:

```python
from marketing_ai_agent.auth.config_manager import ConfigManager
from marketing_ai_agent.api_clients import GoogleAdsAPIClient

# Set up temporary environment variables or create .env file
config_manager = ConfigManager()
ads_client = GoogleAdsAPIClient()

# Initiate OAuth flow
auth_url = ads_client.oauth_manager.initiate_oauth_flow()
print(f"Visit this URL: {auth_url}")

# Get authorization code from callback
auth_code = input("Enter authorization code: ")

# Complete OAuth flow
if ads_client.oauth_manager.complete_oauth_flow(auth_code):
    print("✅ Google Ads authentication successful!")
```

## Google Analytics 4 API Setup

### Option 1: OAuth2 (Recommended for Interactive Use)

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Enable Google Analytics Data API
3. Create OAuth2 credentials (same as Google Ads)
4. Generate refresh token using similar process

### Option 2: Service Account (Recommended for Server Applications)

1. In Google Cloud Console, create a service account:
   - Go to "IAM & Admin" > "Service Accounts"
   - Click "Create Service Account"
   - Download the JSON key file
2. In Google Analytics 4 Admin:
   - Go to Admin > Account/Property > Property Access Management
   - Add the service account email with "Viewer" role
3. Save the JSON file securely and reference it in configuration

### Step 3: Generate Refresh Token (OAuth2 only)

```python
from marketing_ai_agent.api_clients import GA4APIClient

ga4_client = GA4APIClient()

# Initiate OAuth flow
auth_url = ga4_client.oauth_manager.initiate_oauth_flow()
print(f"Visit this URL: {auth_url}")

# Get authorization code from callback
auth_code = input("Enter authorization code: ")

# Complete OAuth flow
if ga4_client.oauth_manager.complete_oauth_flow(auth_code):
    print("✅ Google Analytics authentication successful!")
```

## Anthropic API Setup

1. Go to [Anthropic Console](https://console.anthropic.com/)
2. Sign up or log in
3. Create an API key
4. Copy the API key (starts with `sk-ant-api03-`)

## Configuration

### Option 1: Environment Variables

Create a `.env` file in your project root:

```bash
cp config/examples/.env.example .env
```

Edit `.env` with your actual credentials.

### Option 2: Configuration Files

Copy example configurations and customize:

```bash
mkdir -p config/secrets
cp config/examples/google_ads_config.yaml config/
cp config/examples/google_analytics_config.yaml config/
# Edit with your credentials
```

## Testing Authentication

Test each API connection:

```python
from marketing_ai_agent.api_clients import GoogleAdsAPIClient, GA4APIClient

# Test Google Ads
ads_client = GoogleAdsAPIClient()
if ads_client.test_connection():
    print("✅ Google Ads API connected successfully")
else:
    print("❌ Google Ads API connection failed")

# Test Google Analytics
ga4_client = GA4APIClient()
if ga4_client.test_connection():
    print("✅ Google Analytics API connected successfully")
else:
    print("❌ Google Analytics API connection failed")
```

## Security Best Practices

1. **Never commit credentials to version control**
2. **Use environment variables for sensitive data**
3. **Store service account files outside the project directory**
4. **Rotate API keys and tokens regularly**
5. **Use service accounts for production deployments**
6. **Limit API access to minimum required scopes**

## Troubleshooting

### Common Issues

1. **"Invalid developer token"**
   - Ensure your developer token is approved
   - Check that you're using the correct customer ID

2. **"Insufficient permissions"**
   - Verify your Google account has access to the Google Ads account
   - For GA4, ensure the service account has Viewer permissions

3. **"Quota exceeded"**
   - Check your rate limiting configuration
   - Ensure you're not exceeding API quotas

4. **"Invalid credentials"**
   - Verify OAuth2 credentials are correct
   - Check that refresh tokens haven't expired
   - For service accounts, ensure JSON file is valid and accessible

### Getting Help

1. Check the [Google Ads API documentation](https://developers.google.com/google-ads/api/)
2. Review [Google Analytics Data API docs](https://developers.google.com/analytics/data/api)
3. Visit [Anthropic API documentation](https://docs.anthropic.com/)
4. Check project logs for detailed error messages