"""Configuration management for API credentials and settings."""

import logging
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from pydantic import BaseModel, Field, validator

logger = logging.getLogger(__name__)


class GoogleAdsConfig(BaseModel):
    """Google Ads API configuration."""

    developer_token: str = Field(..., description="Google Ads API developer token")
    client_id: str = Field(..., description="OAuth2 client ID")
    client_secret: str = Field(..., description="OAuth2 client secret")
    refresh_token: str | None = Field(
        default=None, description="OAuth2 refresh token"
    )
    customer_id: str | None = Field(
        default=None, description="Google Ads customer ID"
    )
    login_customer_id: str | None = Field(
        default=None, description="Manager account customer ID"
    )
    use_proto_plus: bool = Field(default=True, description="Use proto-plus library")


class GoogleAnalyticsConfig(BaseModel):
    """Google Analytics 4 configuration."""

    property_id: str = Field(..., description="GA4 property ID")
    client_id: str | None = Field(default=None, description="OAuth2 client ID")
    client_secret: str | None = Field(
        default=None, description="OAuth2 client secret"
    )
    refresh_token: str | None = Field(
        default=None, description="OAuth2 refresh token"
    )
    credentials_path: str | None = Field(
        default=None, description="Path to service account credentials"
    )


class AnthropicConfig(BaseModel):
    """Anthropic API configuration."""

    api_key: str = Field(..., description="Anthropic API key")
    model: str = Field(
        default="claude-3-sonnet-20240229", description="Default model to use"
    )
    max_tokens: int = Field(default=8000, description="Maximum tokens per request")
    temperature: float = Field(default=0.3, description="Model temperature")


class AppConfig(BaseModel):
    """Application configuration."""

    log_level: str = Field(default="INFO", description="Logging level")
    output_dir: str = Field(
        default="./reports", description="Output directory for reports"
    )
    cache_dir: str = Field(default="./.cache", description="Cache directory")
    rate_limit_requests_per_minute: int = Field(
        default=60, description="Rate limit for API requests"
    )

    @validator("log_level")
    def validate_log_level(cls, v):
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"log_level must be one of {valid_levels}")
        return v.upper()


class ConfigManager:
    """Configuration manager for the marketing AI agent."""

    def __init__(self, config_path: str | Path | None = None):
        """
        Initialize configuration manager.

        Args:
            config_path: Optional path to configuration file
        """
        self.config_path = Path(config_path) if config_path else None
        self._config_cache: dict[str, Any] = {}

        # Load environment variables
        load_dotenv()

    def _get_env_var(
        self, key: str, default: Any = None, required: bool = False
    ) -> Any:
        """
        Get environment variable with optional validation.

        Args:
            key: Environment variable key
            default: Default value if not found
            required: Whether the variable is required

        Returns:
            Environment variable value or default

        Raises:
            ValueError: If required variable is not found
        """
        value = os.getenv(key, default)

        if required and value is None:
            raise ValueError(f"Required environment variable {key} not found")

        return value

    def load_google_ads_config(self) -> GoogleAdsConfig:
        """
        Load Google Ads API configuration.

        Returns:
            Google Ads configuration
        """
        try:
            config = GoogleAdsConfig(
                developer_token=self._get_env_var(
                    "GOOGLE_ADS_DEVELOPER_TOKEN", required=True
                ),
                client_id=self._get_env_var("GOOGLE_ADS_CLIENT_ID", required=True),
                client_secret=self._get_env_var(
                    "GOOGLE_ADS_CLIENT_SECRET", required=True
                ),
                refresh_token=self._get_env_var("GOOGLE_ADS_REFRESH_TOKEN"),
                customer_id=self._get_env_var("GOOGLE_ADS_CUSTOMER_ID"),
                login_customer_id=self._get_env_var("GOOGLE_ADS_LOGIN_CUSTOMER_ID"),
                use_proto_plus=self._get_env_var("GOOGLE_ADS_USE_PROTO_PLUS", True),
            )

            logger.info("Loaded Google Ads configuration")
            return config

        except Exception as e:
            logger.error(f"Failed to load Google Ads configuration: {e}")
            raise

    def load_google_analytics_config(self) -> GoogleAnalyticsConfig:
        """
        Load Google Analytics 4 configuration.

        Returns:
            Google Analytics configuration
        """
        try:
            config = GoogleAnalyticsConfig(
                property_id=self._get_env_var(
                    "GOOGLE_ANALYTICS_PROPERTY_ID", required=True
                ),
                client_id=self._get_env_var("GOOGLE_ANALYTICS_CLIENT_ID"),
                client_secret=self._get_env_var("GOOGLE_ANALYTICS_CLIENT_SECRET"),
                refresh_token=self._get_env_var("GOOGLE_ANALYTICS_REFRESH_TOKEN"),
                credentials_path=self._get_env_var("GOOGLE_ANALYTICS_CREDENTIALS_PATH"),
            )

            logger.info("Loaded Google Analytics configuration")
            return config

        except Exception as e:
            logger.error(f"Failed to load Google Analytics configuration: {e}")
            raise

    def load_anthropic_config(self) -> AnthropicConfig:
        """
        Load Anthropic API configuration.

        Returns:
            Anthropic configuration
        """
        try:
            config = AnthropicConfig(
                api_key=self._get_env_var("ANTHROPIC_API_KEY", required=True),
                model=self._get_env_var("ANTHROPIC_MODEL", "claude-3-sonnet-20240229"),
                max_tokens=int(self._get_env_var("ANTHROPIC_MAX_TOKENS", 8000)),
                temperature=float(self._get_env_var("ANTHROPIC_TEMPERATURE", 0.3)),
            )

            logger.info("Loaded Anthropic configuration")
            return config

        except Exception as e:
            logger.error(f"Failed to load Anthropic configuration: {e}")
            raise

    def load_app_config(self) -> AppConfig:
        """
        Load application configuration.

        Returns:
            Application configuration
        """
        try:
            config = AppConfig(
                log_level=self._get_env_var("LOG_LEVEL", "INFO"),
                output_dir=self._get_env_var("OUTPUT_DIR", "./reports"),
                cache_dir=self._get_env_var("CACHE_DIR", "./.cache"),
                rate_limit_requests_per_minute=int(
                    self._get_env_var("RATE_LIMIT_REQUESTS_PER_MINUTE", 60)
                ),
            )

            logger.info("Loaded application configuration")
            return config

        except Exception as e:
            logger.error(f"Failed to load application configuration: {e}")
            raise

    def save_config_template(self, path: str | Path | None = None) -> None:
        """
        Save configuration template file.

        Args:
            path: Optional path for template file
        """
        template_path = Path(path) if path else Path(".env.template")

        template_content = """# Google Ads API Configuration
GOOGLE_ADS_DEVELOPER_TOKEN=your_developer_token_here
GOOGLE_ADS_CLIENT_ID=your_client_id_here
GOOGLE_ADS_CLIENT_SECRET=your_client_secret_here
GOOGLE_ADS_REFRESH_TOKEN=your_refresh_token_here
GOOGLE_ADS_CUSTOMER_ID=your_customer_id_here
GOOGLE_ADS_LOGIN_CUSTOMER_ID=your_manager_account_id_here
GOOGLE_ADS_USE_PROTO_PLUS=true

# Google Analytics 4 Configuration
GOOGLE_ANALYTICS_PROPERTY_ID=your_property_id_here
GOOGLE_ANALYTICS_CLIENT_ID=your_client_id_here
GOOGLE_ANALYTICS_CLIENT_SECRET=your_client_secret_here
GOOGLE_ANALYTICS_REFRESH_TOKEN=your_refresh_token_here
GOOGLE_ANALYTICS_CREDENTIALS_PATH=path/to/service-account.json

# Anthropic API Configuration
ANTHROPIC_API_KEY=your_anthropic_api_key_here
ANTHROPIC_MODEL=claude-3-sonnet-20240229
ANTHROPIC_MAX_TOKENS=8000
ANTHROPIC_TEMPERATURE=0.3

# Application Configuration
LOG_LEVEL=INFO
OUTPUT_DIR=./reports
CACHE_DIR=./.cache
RATE_LIMIT_REQUESTS_PER_MINUTE=60
"""

        try:
            template_path.write_text(template_content)
            logger.info(f"Saved configuration template to {template_path}")

        except Exception as e:
            logger.error(f"Failed to save configuration template: {e}")
            raise

    def validate_configuration(self) -> dict[str, bool]:
        """
        Validate all configuration sections.

        Returns:
            Dictionary with validation results for each section
        """
        results = {}

        # Validate Google Ads config
        try:
            self.load_google_ads_config()
            results["google_ads"] = True
        except Exception as e:
            logger.warning(f"Google Ads configuration invalid: {e}")
            results["google_ads"] = False

        # Validate Google Analytics config
        try:
            self.load_google_analytics_config()
            results["google_analytics"] = True
        except Exception as e:
            logger.warning(f"Google Analytics configuration invalid: {e}")
            results["google_analytics"] = False

        # Validate Anthropic config
        try:
            self.load_anthropic_config()
            results["anthropic"] = True
        except Exception as e:
            logger.warning(f"Anthropic configuration invalid: {e}")
            results["anthropic"] = False

        # Validate app config
        try:
            self.load_app_config()
            results["app"] = True
        except Exception as e:
            logger.warning(f"App configuration invalid: {e}")
            results["app"] = False

        return results
