"""Tests for authentication system."""

import os
from unittest.mock import Mock, patch

import pytest
from marketing_ai_agent.api_clients.ga4_client import GA4APIClient
from marketing_ai_agent.api_clients.google_ads_client import GoogleAdsAPIClient
from marketing_ai_agent.auth.config_manager import (
    AnthropicConfig,
    ConfigManager,
    GoogleAdsConfig,
    GoogleAnalyticsConfig,
)
from marketing_ai_agent.auth.oauth2_manager import (
    OAuth2Config,
    OAuth2Manager,
    TokenInfo,
)


class TestOAuth2Manager:
    """Test OAuth2 manager functionality."""

    @pytest.fixture
    def oauth_config(self):
        """Create test OAuth2 configuration."""
        return OAuth2Config(
            client_id="test_client_id",
            client_secret="test_client_secret",
            scopes=["https://www.googleapis.com/auth/adwords"],
            service_name="test_service",
        )

    def test_token_info_model(self):
        """Test TokenInfo pydantic model."""
        token_info = TokenInfo(
            access_token="test_token",
            refresh_token="test_refresh",
            client_id="test_client",
            client_secret="test_secret",
            scopes=["test_scope"],
        )

        assert token_info.access_token == "test_token"
        assert token_info.refresh_token == "test_refresh"
        assert token_info.client_id == "test_client"
        assert token_info.scopes == ["test_scope"]

    @patch("keyring.set_password")
    @patch("keyring.get_password")
    def test_credential_storage(
        self, mock_get_password, mock_set_password, oauth_config
    ):
        """Test credential storage and retrieval."""

        # Create a concrete implementation for testing
        class TestOAuth2Manager(OAuth2Manager):
            @property
            def service_name(self):
                return "test_service"

            @property
            def required_scopes(self):
                return ["test_scope"]

        manager = TestOAuth2Manager(oauth_config)

        # Mock credentials
        mock_credentials = Mock()
        mock_credentials.token = "test_token"
        mock_credentials.refresh_token = "test_refresh"
        mock_credentials.expiry = None

        # Test storing credentials
        manager.store_credentials(mock_credentials, "test_account")
        mock_set_password.assert_called_once()

        # Test loading credentials
        mock_token_info = TokenInfo(
            access_token="test_token",
            refresh_token="test_refresh",
            client_id="test_client_id",
            client_secret="test_client_secret",
            scopes=["test_scope"],
        )

        mock_get_password.return_value = mock_token_info.model_dump_json()
        credentials = manager.load_credentials("test_account")

        assert credentials is not None
        assert credentials.token == "test_token"
        assert credentials.refresh_token == "test_refresh"


class TestConfigManager:
    """Test configuration manager."""

    def test_config_validation(self):
        """Test configuration validation."""
        with patch.dict(os.environ, {}, clear=True):
            config_manager = ConfigManager()
            results = config_manager.validate_configuration()

            # All should be False without environment variables
            assert not results["google_ads"]
            assert not results["google_analytics"]
            assert not results["anthropic"]
            assert results["app"]  # App config has defaults

    @patch.dict(
        os.environ,
        {
            "GOOGLE_ADS_DEVELOPER_TOKEN": "test_token",
            "GOOGLE_ADS_CLIENT_ID": "test_client_id",
            "GOOGLE_ADS_CLIENT_SECRET": "test_client_secret",
        },
    )
    def test_google_ads_config(self):
        """Test Google Ads configuration loading."""
        config_manager = ConfigManager()
        config = config_manager.load_google_ads_config()

        assert isinstance(config, GoogleAdsConfig)
        assert config.developer_token == "test_token"
        assert config.client_id == "test_client_id"
        assert config.client_secret == "test_client_secret"

    @patch.dict(os.environ, {"GOOGLE_ANALYTICS_PROPERTY_ID": "123456789"})
    def test_google_analytics_config(self):
        """Test Google Analytics configuration loading."""
        config_manager = ConfigManager()
        config = config_manager.load_google_analytics_config()

        assert isinstance(config, GoogleAnalyticsConfig)
        assert config.property_id == "123456789"

    @patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test_api_key"})
    def test_anthropic_config(self):
        """Test Anthropic configuration loading."""
        config_manager = ConfigManager()
        config = config_manager.load_anthropic_config()

        assert isinstance(config, AnthropicConfig)
        assert config.api_key == "test_api_key"
        assert config.model == "claude-3-sonnet-20240229"


class TestGoogleAdsClient:
    """Test Google Ads API client."""

    @pytest.fixture
    def mock_config(self):
        """Create mock Google Ads configuration."""
        return GoogleAdsConfig(
            developer_token="test_developer_token",
            client_id="test_client_id",
            client_secret="test_client_secret",
            customer_id="1234567890",
        )

    def test_client_initialization(self, mock_config):
        """Test client initialization."""
        client = GoogleAdsAPIClient(config=mock_config)

        assert client.config == mock_config
        assert client.rate_limit == 60
        assert client._client is None

    @patch("marketing_ai_agent.api_clients.google_ads_client.GoogleAdsClient")
    def test_rate_limiting(self, mock_google_ads_client, mock_config):
        """Test rate limiting functionality."""
        client = GoogleAdsAPIClient(
            config=mock_config, rate_limit_requests_per_minute=2
        )

        # Mock the OAuth manager
        client.oauth_manager = Mock()
        client.oauth_manager.get_valid_credentials.return_value = Mock()

        # Mock the Google Ads client
        mock_client_instance = Mock()
        mock_google_ads_client.load_from_dict.return_value = mock_client_instance

        # Should be able to make 2 requests quickly
        assert client.authenticate()
        client._apply_rate_limit()
        client._apply_rate_limit()

        # Third request should trigger rate limiting (but we won't actually wait)
        assert client._request_count == 2


class TestGA4Client:
    """Test Google Analytics 4 API client."""

    @pytest.fixture
    def mock_config(self):
        """Create mock GA4 configuration."""
        return GoogleAnalyticsConfig(
            property_id="123456789",
            client_id="test_client_id",
            client_secret="test_client_secret",
        )

    def test_client_initialization(self, mock_config):
        """Test client initialization."""
        client = GA4APIClient(config=mock_config)

        assert client.config == mock_config
        assert client.rate_limit == 60
        assert client._client is None

    @patch("marketing_ai_agent.api_clients.ga4_client.BetaAnalyticsDataClient")
    def test_authentication(self, mock_analytics_client, mock_config):
        """Test authentication."""
        client = GA4APIClient(config=mock_config)

        # Mock the OAuth manager
        client.oauth_manager = Mock()
        mock_credentials = Mock()
        client.oauth_manager.get_valid_credentials.return_value = mock_credentials

        # Mock the Analytics client
        mock_client_instance = Mock()
        mock_analytics_client.return_value = mock_client_instance

        # Test authentication
        assert client.authenticate()
        assert client._client == mock_client_instance
        assert client._credentials == mock_credentials


def test_import_structure():
    """Test that all modules can be imported correctly."""
    from marketing_ai_agent.api_clients import GA4APIClient, GoogleAdsAPIClient
    from marketing_ai_agent.auth import ConfigManager, OAuth2Manager

    assert OAuth2Manager is not None
    assert ConfigManager is not None
    assert GoogleAdsAPIClient is not None
    assert GA4APIClient is not None


if __name__ == "__main__":
    # Run basic import test
    test_import_structure()
    print("✅ All imports successful")

    # Test configuration validation
    config_manager = ConfigManager()
    results = config_manager.validate_configuration()

    print("\n📋 Configuration validation results:")
    for service, valid in results.items():
        status = "✅" if valid else "❌"
        print(f"  {status} {service}: {'Valid' if valid else 'Invalid/Missing'}")

    print("\n🔧 To set up authentication:")
    print("1. Copy config/examples/.env.example to .env")
    print("2. Fill in your API credentials")
    print("3. Run setup instructions in config/examples/setup_instructions.md")
