"""OAuth2 authentication manager for Google APIs."""

import logging
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from pathlib import Path

import keyring
from google.auth.credentials import Credentials
from google.auth.exceptions import RefreshError
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials as OAuth2Credentials
from google.oauth2.service_account import Credentials as ServiceAccountCredentials
from google_auth_oauthlib.flow import Flow
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class OAuth2Config(BaseModel):
    """OAuth2 configuration model."""

    client_id: str = Field(..., description="OAuth2 client ID")
    client_secret: str = Field(..., description="OAuth2 client secret")
    redirect_uri: str = Field(
        default="http://localhost:8080", description="OAuth2 redirect URI"
    )
    scopes: list[str] = Field(..., description="OAuth2 scopes")
    service_name: str = Field(..., description="Service name for keyring storage")


class TokenInfo(BaseModel):
    """Token information model."""

    access_token: str
    refresh_token: str | None = None
    token_uri: str = "https://oauth2.googleapis.com/token"
    client_id: str
    client_secret: str
    scopes: list[str]
    expires_at: datetime | None = None

    class Config:
        arbitrary_types_allowed = True


class OAuth2Manager(ABC):
    """Abstract base class for OAuth2 authentication management."""

    def __init__(
        self, config: OAuth2Config, credentials_path: str | Path | None = None
    ):
        """
        Initialize OAuth2 manager.

        Args:
            config: OAuth2 configuration
            credentials_path: Optional path to service account credentials
        """
        self.config = config
        self.credentials_path = Path(credentials_path) if credentials_path else None
        self._credentials: Credentials | None = None

    @property
    @abstractmethod
    def service_name(self) -> str:
        """Return the service name for this manager."""
        pass

    @property
    @abstractmethod
    def required_scopes(self) -> list[str]:
        """Return required scopes for this service."""
        pass

    def _get_keyring_key(self, account_id: str = "default") -> str:
        """Generate keyring key for storing credentials."""
        return f"{self.service_name}_{account_id}_oauth2"

    def store_credentials(
        self, credentials: Credentials, account_id: str = "default"
    ) -> None:
        """
        Store credentials securely using keyring.

        Args:
            credentials: OAuth2 credentials to store
            account_id: Account identifier for multi-account support
        """
        try:
            token_info = TokenInfo(
                access_token=credentials.token,
                refresh_token=getattr(credentials, "refresh_token", None),
                client_id=self.config.client_id,
                client_secret=self.config.client_secret,
                scopes=self.required_scopes,
                expires_at=credentials.expiry,
            )

            keyring_key = self._get_keyring_key(account_id)
            keyring.set_password(
                self.config.service_name, keyring_key, token_info.model_dump_json()
            )

            logger.info(
                f"Stored credentials for {self.service_name} account: {account_id}"
            )

        except Exception as e:
            logger.error(f"Failed to store credentials: {e}")
            raise

    def load_credentials(self, account_id: str = "default") -> OAuth2Credentials | None:
        """
        Load credentials from secure storage.

        Args:
            account_id: Account identifier for multi-account support

        Returns:
            OAuth2 credentials if found, None otherwise
        """
        try:
            keyring_key = self._get_keyring_key(account_id)
            stored_data = keyring.get_password(self.config.service_name, keyring_key)

            if not stored_data:
                logger.info(
                    f"No stored credentials found for {self.service_name} account: {account_id}"
                )
                return None

            token_info = TokenInfo.model_validate_json(stored_data)

            credentials = OAuth2Credentials(
                token=token_info.access_token,
                refresh_token=token_info.refresh_token,
                token_uri=token_info.token_uri,
                client_id=token_info.client_id,
                client_secret=token_info.client_secret,
                scopes=token_info.scopes,
            )

            # Set expiry if available
            if token_info.expires_at:
                credentials.expiry = token_info.expires_at

            logger.info(
                f"Loaded credentials for {self.service_name} account: {account_id}"
            )
            return credentials

        except Exception as e:
            logger.error(f"Failed to load credentials for {account_id}: {e}")
            return None

    def load_service_account_credentials(self) -> ServiceAccountCredentials | None:
        """
        Load service account credentials from file.

        Returns:
            Service account credentials if available, None otherwise
        """
        if not self.credentials_path or not self.credentials_path.exists():
            logger.info("No service account credentials file found")
            return None

        try:
            credentials = ServiceAccountCredentials.from_service_account_file(
                str(self.credentials_path), scopes=self.required_scopes
            )

            logger.info(f"Loaded service account credentials for {self.service_name}")
            return credentials

        except Exception as e:
            logger.error(f"Failed to load service account credentials: {e}")
            return None

    def refresh_credentials(
        self, credentials: OAuth2Credentials, account_id: str = "default"
    ) -> bool:
        """
        Refresh OAuth2 credentials.

        Args:
            credentials: Credentials to refresh
            account_id: Account identifier

        Returns:
            True if refresh successful, False otherwise
        """
        try:
            request = Request()
            credentials.refresh(request)

            # Store refreshed credentials
            self.store_credentials(credentials, account_id)

            logger.info(
                f"Successfully refreshed credentials for {self.service_name} account: {account_id}"
            )
            return True

        except RefreshError as e:
            logger.error(f"Failed to refresh credentials for {account_id}: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error refreshing credentials: {e}")
            return False

    def get_valid_credentials(self, account_id: str = "default") -> Credentials | None:
        """
        Get valid credentials, refreshing if necessary.

        Args:
            account_id: Account identifier

        Returns:
            Valid credentials or None if unavailable
        """
        # Try service account first
        service_creds = self.load_service_account_credentials()
        if service_creds:
            return service_creds

        # Try OAuth2 credentials
        oauth_creds = self.load_credentials(account_id)
        if not oauth_creds:
            logger.warning(
                f"No credentials found for {self.service_name} account: {account_id}"
            )
            return None

        # Check if refresh is needed
        if oauth_creds.expired or (
            oauth_creds.expiry
            and oauth_creds.expiry < datetime.utcnow() + timedelta(minutes=5)
        ):
            logger.info("Credentials expired or expiring soon, attempting refresh")
            if not self.refresh_credentials(oauth_creds, account_id):
                logger.error("Failed to refresh expired credentials")
                return None

        return oauth_creds

    def initiate_oauth_flow(self, port: int = 8080) -> str:
        """
        Initiate OAuth2 flow and return authorization URL.

        Args:
            port: Port for local server

        Returns:
            Authorization URL for user to visit
        """
        try:
            flow = Flow.from_client_config(
                {
                    "web": {
                        "client_id": self.config.client_id,
                        "client_secret": self.config.client_secret,
                        "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                        "token_uri": "https://oauth2.googleapis.com/token",
                        "redirect_uris": [f"http://localhost:{port}"],
                    }
                },
                scopes=self.required_scopes,
            )

            flow.redirect_uri = f"http://localhost:{port}"

            auth_url, _ = flow.authorization_url(
                access_type="offline", include_granted_scopes="true"
            )

            return auth_url

        except Exception as e:
            logger.error(f"Failed to initiate OAuth flow: {e}")
            raise

    def complete_oauth_flow(
        self, authorization_code: str, account_id: str = "default"
    ) -> bool:
        """
        Complete OAuth2 flow with authorization code.

        Args:
            authorization_code: Authorization code from OAuth flow
            account_id: Account identifier

        Returns:
            True if successful, False otherwise
        """
        try:
            flow = Flow.from_client_config(
                {
                    "web": {
                        "client_id": self.config.client_id,
                        "client_secret": self.config.client_secret,
                        "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                        "token_uri": "https://oauth2.googleapis.com/token",
                        "redirect_uris": [self.config.redirect_uri],
                    }
                },
                scopes=self.required_scopes,
            )

            flow.redirect_uri = self.config.redirect_uri
            flow.fetch_token(code=authorization_code)

            # Store the credentials
            self.store_credentials(flow.credentials, account_id)

            logger.info(
                f"Successfully completed OAuth flow for {self.service_name} account: {account_id}"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to complete OAuth flow: {e}")
            return False

    def revoke_credentials(self, account_id: str = "default") -> bool:
        """
        Revoke stored credentials.

        Args:
            account_id: Account identifier

        Returns:
            True if successful, False otherwise
        """
        try:
            keyring_key = self._get_keyring_key(account_id)
            keyring.delete_password(self.config.service_name, keyring_key)

            logger.info(
                f"Revoked credentials for {self.service_name} account: {account_id}"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to revoke credentials: {e}")
            return False

    def list_accounts(self) -> list[str]:
        """
        List available accounts.

        Returns:
            List of account identifiers
        """
        # This is a simplified implementation
        # In practice, you might want to store account metadata separately
        try:
            # Try to load default account to see if any credentials exist
            if self.load_credentials("default"):
                return ["default"]
            return []
        except Exception:
            return []
