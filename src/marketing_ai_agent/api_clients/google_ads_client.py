"""Google Ads API client with authentication and rate limiting."""

import logging
import time
from datetime import datetime, timedelta
from typing import Any

from google.ads.googleads.client import GoogleAdsClient
from google.ads.googleads.errors import GoogleAdsException
from google.auth.credentials import Credentials

from ..auth.config_manager import ConfigManager, GoogleAdsConfig
from ..auth.oauth2_manager import OAuth2Config, OAuth2Manager

logger = logging.getLogger(__name__)


class GoogleAdsOAuth2Manager(OAuth2Manager):
    """OAuth2 manager specifically for Google Ads API."""

    @property
    def service_name(self) -> str:
        return "google_ads"

    @property
    def required_scopes(self) -> list[str]:
        return ["https://www.googleapis.com/auth/adwords"]


class GoogleAdsAPIClient:
    """
    Google Ads API client with authentication and rate limiting.
    """

    def __init__(
        self,
        config: GoogleAdsConfig | None = None,
        credentials_path: str | None = None,
        rate_limit_requests_per_minute: int = 60,
    ):
        """
        Initialize Google Ads API client.

        Args:
            config: Google Ads configuration
            credentials_path: Path to service account credentials
            rate_limit_requests_per_minute: Rate limit for API requests
        """
        # Load configuration if not provided
        if config is None:
            config_manager = ConfigManager()
            config = config_manager.load_google_ads_config()

        self.config = config
        self.credentials_path = credentials_path
        self.rate_limit = rate_limit_requests_per_minute
        self._last_request_time = 0.0
        self._request_count = 0
        self._request_window_start = time.time()

        # Initialize OAuth2 manager
        oauth_config = OAuth2Config(
            client_id=config.client_id,
            client_secret=config.client_secret,
            scopes=["https://www.googleapis.com/auth/adwords"],
            service_name="google_ads",
        )

        self.oauth_manager = GoogleAdsOAuth2Manager(oauth_config, credentials_path)

        self._client: GoogleAdsClient | None = None
        self._credentials: Credentials | None = None

    def _apply_rate_limit(self) -> None:
        """Apply rate limiting to API requests."""
        current_time = time.time()

        # Reset counter if we're in a new minute window
        if current_time - self._request_window_start >= 60:
            self._request_count = 0
            self._request_window_start = current_time

        # Check if we've exceeded rate limit
        if self._request_count >= self.rate_limit:
            sleep_time = 60 - (current_time - self._request_window_start)
            if sleep_time > 0:
                logger.info(
                    f"Rate limit reached, sleeping for {sleep_time:.2f} seconds"
                )
                time.sleep(sleep_time)
                self._request_count = 0
                self._request_window_start = time.time()

        self._request_count += 1
        self._last_request_time = current_time

    def authenticate(self, account_id: str = "default") -> bool:
        """
        Authenticate with Google Ads API.

        Args:
            account_id: Account identifier for multi-account support

        Returns:
            True if authentication successful, False otherwise
        """
        try:
            # Get valid credentials
            credentials = self.oauth_manager.get_valid_credentials(account_id)

            if not credentials:
                logger.error(
                    f"No valid credentials available for account: {account_id}"
                )
                return False

            # Create client configuration
            client_config = {
                "developer_token": self.config.developer_token,
                "use_proto_plus": self.config.use_proto_plus,
            }

            # Add customer ID if provided
            if self.config.customer_id:
                client_config["customer_id"] = self.config.customer_id

            # Add login customer ID if provided (for manager accounts)
            if self.config.login_customer_id:
                client_config["login_customer_id"] = self.config.login_customer_id

            # Create Google Ads client
            self._client = GoogleAdsClient.load_from_dict(
                client_config, credentials=credentials
            )

            self._credentials = credentials

            logger.info(
                f"Successfully authenticated Google Ads API for account: {account_id}"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to authenticate Google Ads API: {e}")
            return False

    def get_client(self, account_id: str = "default") -> GoogleAdsClient | None:
        """
        Get authenticated Google Ads client.

        Args:
            account_id: Account identifier

        Returns:
            Authenticated Google Ads client or None
        """
        if not self._client or not self._credentials:
            if not self.authenticate(account_id):
                return None

        # Check if credentials need refresh
        if self._credentials.expired or (
            self._credentials.expiry
            and self._credentials.expiry < datetime.utcnow() + timedelta(minutes=5)
        ):
            logger.info("Credentials expired or expiring soon, re-authenticating")
            if not self.authenticate(account_id):
                return None

        return self._client

    def search(
        self,
        query: str,
        customer_id: str | None = None,
        page_size: int = 10000,
        account_id: str = "default",
    ) -> list[Any]:
        """
        Execute a search query against Google Ads API.

        Args:
            query: GAQL query string
            customer_id: Google Ads customer ID (overrides config)
            page_size: Maximum results per page
            account_id: Account identifier

        Returns:
            List of search results
        """
        client = self.get_client(account_id)
        if not client:
            raise ValueError("Failed to get authenticated client")

        # Apply rate limiting
        self._apply_rate_limit()

        # Use provided customer ID or fall back to config
        target_customer_id = customer_id or self.config.customer_id
        if not target_customer_id:
            raise ValueError("No customer ID provided or configured")

        try:
            ga_service = client.get_service("GoogleAdsService")

            response = ga_service.search(
                customer_id=target_customer_id, query=query, page_size=page_size
            )

            results = list(response)
            logger.info(f"Retrieved {len(results)} results from Google Ads API")

            return results

        except GoogleAdsException as ex:
            logger.error(f"Google Ads API error: {ex}")
            # Log specific error details
            for error in ex.failure.errors:
                logger.error(f"Error: {error.message}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in search: {e}")
            raise

    def get_campaigns(
        self, customer_id: str | None = None, account_id: str = "default"
    ) -> list[Any]:
        """
        Get all campaigns for a customer.

        Args:
            customer_id: Google Ads customer ID
            account_id: Account identifier

        Returns:
            List of campaigns
        """
        query = """
            SELECT
                campaign.id,
                campaign.name,
                campaign.status,
                campaign.advertising_channel_type,
                campaign.start_date,
                campaign.end_date,
                metrics.impressions,
                metrics.clicks,
                metrics.conversions,
                metrics.cost_micros
            FROM campaign
            WHERE campaign.status != 'REMOVED'
        """

        return self.search(query, customer_id, account_id=account_id)

    def get_ad_groups(
        self,
        campaign_id: str | None = None,
        customer_id: str | None = None,
        account_id: str = "default",
    ) -> list[Any]:
        """
        Get ad groups for a customer or specific campaign.

        Args:
            campaign_id: Optional campaign ID to filter by
            customer_id: Google Ads customer ID
            account_id: Account identifier

        Returns:
            List of ad groups
        """
        query = """
            SELECT
                ad_group.id,
                ad_group.name,
                ad_group.status,
                ad_group.type,
                campaign.id,
                campaign.name,
                metrics.impressions,
                metrics.clicks,
                metrics.conversions,
                metrics.cost_micros
            FROM ad_group
            WHERE ad_group.status != 'REMOVED'
        """

        if campaign_id:
            query += f" AND campaign.id = {campaign_id}"

        return self.search(query, customer_id, account_id=account_id)

    def get_keywords(
        self,
        ad_group_id: str | None = None,
        campaign_id: str | None = None,
        customer_id: str | None = None,
        account_id: str = "default",
    ) -> list[Any]:
        """
        Get keywords for a customer, campaign, or ad group.

        Args:
            ad_group_id: Optional ad group ID to filter by
            campaign_id: Optional campaign ID to filter by
            customer_id: Google Ads customer ID
            account_id: Account identifier

        Returns:
            List of keywords
        """
        query = """
            SELECT
                ad_group_criterion.keyword.text,
                ad_group_criterion.keyword.match_type,
                ad_group_criterion.status,
                ad_group_criterion.quality_info.quality_score,
                ad_group.id,
                ad_group.name,
                campaign.id,
                campaign.name,
                metrics.impressions,
                metrics.clicks,
                metrics.conversions,
                metrics.cost_micros,
                metrics.average_cpc
            FROM keyword_view
            WHERE ad_group_criterion.status != 'REMOVED'
        """

        conditions = []
        if ad_group_id:
            conditions.append(f"ad_group.id = {ad_group_id}")
        if campaign_id:
            conditions.append(f"campaign.id = {campaign_id}")

        if conditions:
            query += " AND " + " AND ".join(conditions)

        return self.search(query, customer_id, account_id=account_id)

    def get_performance_data(
        self,
        date_range: str = "LAST_30_DAYS",
        customer_id: str | None = None,
        account_id: str = "default",
    ) -> list[Any]:
        """
        Get performance data for a date range.

        Args:
            date_range: Date range (LAST_7_DAYS, LAST_30_DAYS, etc.)
            customer_id: Google Ads customer ID
            account_id: Account identifier

        Returns:
            List of performance data
        """
        query = f"""
            SELECT
                segments.date,
                campaign.id,
                campaign.name,
                campaign.advertising_channel_type,
                metrics.impressions,
                metrics.clicks,
                metrics.conversions,
                metrics.cost_micros,
                metrics.average_cpc,
                metrics.ctr,
                metrics.conversion_rate,
                metrics.cost_per_conversion
            FROM campaign
            WHERE segments.date DURING {date_range}
            AND campaign.status != 'REMOVED'
            ORDER BY segments.date DESC
        """

        return self.search(query, customer_id, account_id=account_id)

    def test_connection(self, account_id: str = "default") -> bool:
        """
        Test connection to Google Ads API.

        Args:
            account_id: Account identifier

        Returns:
            True if connection successful, False otherwise
        """
        try:
            client = self.get_client(account_id)
            if not client:
                return False

            # Simple query to test connection
            query = "SELECT customer.id FROM customer LIMIT 1"
            self.search(query, account_id=account_id)

            logger.info("Google Ads API connection test successful")
            return True

        except Exception as e:
            logger.error(f"Google Ads API connection test failed: {e}")
            return False
