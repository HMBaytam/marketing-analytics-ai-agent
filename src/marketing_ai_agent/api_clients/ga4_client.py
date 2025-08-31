"""Google Analytics 4 API client with authentication and rate limiting."""

import logging
import time
from datetime import datetime, timedelta
from typing import Any

from google.analytics.data_v1beta import BetaAnalyticsDataClient
from google.analytics.data_v1beta.types import (
    DateRange,
    Dimension,
    Metric,
    RunRealtimeReportRequest,
    RunReportRequest,
)
from google.api_core.exceptions import GoogleAPIError
from google.auth.credentials import Credentials

from ..auth.config_manager import ConfigManager, GoogleAnalyticsConfig
from ..auth.oauth2_manager import OAuth2Config, OAuth2Manager

logger = logging.getLogger(__name__)


class GoogleAnalyticsOAuth2Manager(OAuth2Manager):
    """OAuth2 manager specifically for Google Analytics API."""

    @property
    def service_name(self) -> str:
        return "google_analytics"

    @property
    def required_scopes(self) -> list[str]:
        return [
            "https://www.googleapis.com/auth/analytics.readonly",
            "https://www.googleapis.com/auth/analytics",
        ]


class GA4APIClient:
    """
    Google Analytics 4 API client with authentication and rate limiting.
    """

    def __init__(
        self,
        config: GoogleAnalyticsConfig | None = None,
        credentials_path: str | None = None,
        rate_limit_requests_per_minute: int = 60,
    ):
        """
        Initialize Google Analytics 4 API client.

        Args:
            config: Google Analytics configuration
            credentials_path: Path to service account credentials
            rate_limit_requests_per_minute: Rate limit for API requests
        """
        # Load configuration if not provided
        if config is None:
            config_manager = ConfigManager()
            config = config_manager.load_google_analytics_config()

        self.config = config
        self.credentials_path = credentials_path or config.credentials_path
        self.rate_limit = rate_limit_requests_per_minute
        self._last_request_time = 0.0
        self._request_count = 0
        self._request_window_start = time.time()

        # Initialize OAuth2 manager if OAuth credentials are provided
        self.oauth_manager = None
        if config.client_id and config.client_secret:
            oauth_config = OAuth2Config(
                client_id=config.client_id,
                client_secret=config.client_secret,
                scopes=[
                    "https://www.googleapis.com/auth/analytics.readonly",
                    "https://www.googleapis.com/auth/analytics",
                ],
                service_name="google_analytics",
            )

            self.oauth_manager = GoogleAnalyticsOAuth2Manager(
                oauth_config, self.credentials_path
            )

        self._client: BetaAnalyticsDataClient | None = None
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
        Authenticate with Google Analytics API.

        Args:
            account_id: Account identifier for multi-account support

        Returns:
            True if authentication successful, False otherwise
        """
        try:
            credentials = None

            # Try OAuth2 credentials first if available
            if self.oauth_manager:
                credentials = self.oauth_manager.get_valid_credentials(account_id)
                if credentials:
                    logger.info("Using OAuth2 credentials for Google Analytics API")

            # Fall back to service account credentials
            if not credentials and self.oauth_manager:
                credentials = self.oauth_manager.load_service_account_credentials()
                if credentials:
                    logger.info(
                        "Using service account credentials for Google Analytics API"
                    )

            if not credentials:
                logger.error(
                    f"No valid credentials available for account: {account_id}"
                )
                return False

            # Create client with credentials
            self._client = BetaAnalyticsDataClient(credentials=credentials)
            self._credentials = credentials

            logger.info(
                f"Successfully authenticated Google Analytics API for account: {account_id}"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to authenticate Google Analytics API: {e}")
            return False

    def get_client(self, account_id: str = "default") -> BetaAnalyticsDataClient | None:
        """
        Get authenticated Google Analytics client.

        Args:
            account_id: Account identifier

        Returns:
            Authenticated Google Analytics client or None
        """
        if not self._client or not self._credentials:
            if not self.authenticate(account_id):
                return None

        # Check if credentials need refresh (for OAuth2 only)
        if (
            hasattr(self._credentials, "expired")
            and self._credentials.expired
            or (
                hasattr(self._credentials, "expiry")
                and self._credentials.expiry
                and self._credentials.expiry < datetime.utcnow() + timedelta(minutes=5)
            )
        ):
            logger.info("Credentials expired or expiring soon, re-authenticating")
            if not self.authenticate(account_id):
                return None

        return self._client

    def run_report(
        self,
        dimensions: list[str],
        metrics: list[str],
        date_ranges: list[dict[str, str]],
        property_id: str | None = None,
        dimension_filter: dict | None = None,
        metric_filter: dict | None = None,
        order_bys: list[dict] | None = None,
        limit: int | None = None,
        account_id: str = "default",
    ) -> Any:
        """
        Run a report against Google Analytics 4.

        Args:
            dimensions: List of dimension names
            metrics: List of metric names
            date_ranges: List of date range dictionaries with 'start_date' and 'end_date'
            property_id: GA4 property ID (overrides config)
            dimension_filter: Optional dimension filter
            metric_filter: Optional metric filter
            order_bys: Optional ordering specifications
            limit: Optional limit on results
            account_id: Account identifier

        Returns:
            Report response
        """
        client = self.get_client(account_id)
        if not client:
            raise ValueError("Failed to get authenticated client")

        # Apply rate limiting
        self._apply_rate_limit()

        # Use provided property ID or fall back to config
        target_property_id = property_id or self.config.property_id
        if not target_property_id:
            raise ValueError("No property ID provided or configured")

        try:
            # Build request
            request = RunReportRequest(
                property=f"properties/{target_property_id}",
                dimensions=[Dimension(name=dim) for dim in dimensions],
                metrics=[Metric(name=metric) for metric in metrics],
                date_ranges=[
                    DateRange(start_date=dr["start_date"], end_date=dr["end_date"])
                    for dr in date_ranges
                ],
            )

            # Add filters if provided
            if dimension_filter:
                # Simplified filter implementation - will be implemented when needed
                # For now, skip advanced filtering to avoid import issues
                logger.info(
                    "Advanced filtering not implemented yet, skipping dimension_filter"
                )

            # Add limit if provided
            if limit:
                request.limit = limit

            response = client.run_report(request=request)

            logger.info(
                f"Retrieved {len(response.rows)} rows from Google Analytics API"
            )

            return response

        except GoogleAPIError as e:
            logger.error(f"Google Analytics API error: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in run_report: {e}")
            raise

    def get_traffic_data(
        self,
        start_date: str,
        end_date: str,
        property_id: str | None = None,
        account_id: str = "default",
    ) -> Any:
        """
        Get basic traffic data for a date range.

        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            property_id: GA4 property ID
            account_id: Account identifier

        Returns:
            Traffic data report
        """
        dimensions = [
            "date",
            "country",
            "deviceCategory",
            "firstUserDefaultChannelGrouping",
        ]

        metrics = [
            "sessions",
            "users",
            "newUsers",
            "pageviews",
            "bounceRate",
            "averageSessionDuration",
        ]

        date_ranges = [{"start_date": start_date, "end_date": end_date}]

        return self.run_report(
            dimensions=dimensions,
            metrics=metrics,
            date_ranges=date_ranges,
            property_id=property_id,
            account_id=account_id,
        )

    def get_conversion_data(
        self,
        start_date: str,
        end_date: str,
        property_id: str | None = None,
        account_id: str = "default",
    ) -> Any:
        """
        Get conversion data for a date range.

        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            property_id: GA4 property ID
            account_id: Account identifier

        Returns:
            Conversion data report
        """
        dimensions = ["date", "firstUserDefaultChannelGrouping", "eventName"]

        metrics = ["conversions", "totalRevenue", "purchaseRevenue", "eventCount"]

        date_ranges = [{"start_date": start_date, "end_date": end_date}]

        # Filter for conversion events - will be implemented later
        dimension_filter = None  # Filtering not implemented yet

        return self.run_report(
            dimensions=dimensions,
            metrics=metrics,
            date_ranges=date_ranges,
            property_id=property_id,
            dimension_filter=dimension_filter,
            account_id=account_id,
        )

    def get_channel_performance(
        self,
        start_date: str,
        end_date: str,
        property_id: str | None = None,
        account_id: str = "default",
    ) -> Any:
        """
        Get channel performance data.

        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            property_id: GA4 property ID
            account_id: Account identifier

        Returns:
            Channel performance report
        """
        dimensions = [
            "firstUserDefaultChannelGrouping",
            "firstUserSource",
            "firstUserMedium",
        ]

        metrics = [
            "sessions",
            "users",
            "newUsers",
            "conversions",
            "totalRevenue",
            "engagementRate",
            "averageSessionDuration",
        ]

        date_ranges = [{"start_date": start_date, "end_date": end_date}]

        return self.run_report(
            dimensions=dimensions,
            metrics=metrics,
            date_ranges=date_ranges,
            property_id=property_id,
            account_id=account_id,
        )

    def get_realtime_data(
        self, property_id: str | None = None, account_id: str = "default"
    ) -> Any:
        """
        Get real-time analytics data.

        Args:
            property_id: GA4 property ID
            account_id: Account identifier

        Returns:
            Real-time data report
        """
        client = self.get_client(account_id)
        if not client:
            raise ValueError("Failed to get authenticated client")

        # Apply rate limiting
        self._apply_rate_limit()

        # Use provided property ID or fall back to config
        target_property_id = property_id or self.config.property_id
        if not target_property_id:
            raise ValueError("No property ID provided or configured")

        try:
            request = RunRealtimeReportRequest(
                property=f"properties/{target_property_id}",
                dimensions=[
                    Dimension(name="country"),
                    Dimension(name="deviceCategory"),
                ],
                metrics=[Metric(name="activeUsers"), Metric(name="screenPageViews")],
            )

            response = client.run_realtime_report(request=request)

            logger.info("Retrieved real-time data from Google Analytics API")

            return response

        except GoogleAPIError as e:
            logger.error(f"Google Analytics API error: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in get_realtime_data: {e}")
            raise

    def test_connection(self, account_id: str = "default") -> bool:
        """
        Test connection to Google Analytics API.

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
            self.run_report(
                dimensions=["date"],
                metrics=["sessions"],
                date_ranges=[{"start_date": "7daysAgo", "end_date": "today"}],
                limit=1,
                account_id=account_id,
            )

            logger.info("Google Analytics API connection test successful")
            return True

        except Exception as e:
            logger.error(f"Google Analytics API connection test failed: {e}")
            return False
