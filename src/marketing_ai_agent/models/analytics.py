"""Data models for Google Analytics 4 events, conversions, and traffic data."""

from datetime import date as Date
from decimal import Decimal
from enum import Enum

from pydantic import BaseModel, Field


class ChannelGrouping(str, Enum):
    """Default channel grouping enumeration."""

    DIRECT = "Direct"
    ORGANIC_SEARCH = "Organic Search"
    PAID_SEARCH = "Paid Search"
    PAID_SOCIAL = "Paid Social"
    ORGANIC_SOCIAL = "Organic Social"
    EMAIL = "Email"
    AFFILIATES = "Affiliates"
    REFERRAL = "Referral"
    PAID_VIDEO = "Paid Video"
    ORGANIC_VIDEO = "Organic Video"
    DISPLAY = "Display"
    PAID_SHOPPING = "Paid Shopping"
    ORGANIC_SHOPPING = "Organic Shopping"
    PUSH = "Push"
    SMS = "SMS"
    AUDIO = "Audio"
    UNASSIGNED = "(not set)"


class DeviceCategory(str, Enum):
    """Device category enumeration."""

    DESKTOP = "desktop"
    MOBILE = "mobile"
    TABLET = "tablet"
    SMART_TV = "smart tv"
    UNKNOWN = "(not set)"


class EventCategory(str, Enum):
    """Common GA4 event categories."""

    PAGE_VIEW = "page_view"
    PURCHASE = "purchase"
    SIGN_UP = "sign_up"
    LOGIN = "login"
    FORM_SUBMIT = "form_submit"
    DOWNLOAD = "file_download"
    VIDEO_PLAY = "video_play"
    SEARCH = "search"
    ADD_TO_CART = "add_to_cart"
    BEGIN_CHECKOUT = "begin_checkout"
    CUSTOM = "custom"


class TrafficData(BaseModel):
    """Google Analytics 4 traffic data model."""

    date: Date = Field(..., description="Traffic Date")
    property_id: str = Field(..., description="GA4 property ID")

    # Dimensions
    country: str | None = Field(None, description="Country")
    device_category: DeviceCategory | None = Field(None, description="Device category")
    channel_grouping: ChannelGrouping | None = Field(
        None, description="Default channel grouping"
    )
    source: str | None = Field(None, description="Traffic source")
    medium: str | None = Field(None, description="Traffic medium")
    campaign_name: str | None = Field(None, description="Campaign name")

    # Core metrics
    sessions: int = Field(0, description="Number of sessions")
    users: int = Field(0, description="Number of users")
    new_users: int = Field(0, description="Number of new users")
    pageviews: int = Field(0, description="Number of pageviews")
    bounce_rate: float | None = Field(None, description="Bounce rate (0-1)")
    average_session_duration: float | None = Field(
        None, description="Average session duration in seconds"
    )

    # Engagement metrics
    engaged_sessions: int | None = Field(None, description="Number of engaged sessions")
    engagement_rate: float | None = Field(None, description="Engagement rate (0-1)")
    events_per_session: float | None = Field(
        None, description="Average events per session"
    )

    @property
    def returning_users(self) -> int:
        """Calculate returning users."""
        return max(0, self.users - self.new_users)

    @property
    def pages_per_session(self) -> float | None:
        """Calculate pages per session."""
        if self.sessions > 0:
            return self.pageviews / self.sessions
        return None


class ConversionEvent(BaseModel):
    """Google Analytics 4 conversion event model."""

    date: Date = Field(..., description="Conversion Date")
    property_id: str = Field(..., description="GA4 property ID")

    # Event details
    event_name: str = Field(..., description="Event name")
    event_category: EventCategory | None = Field(None, description="Event category")

    # Dimensions
    channel_grouping: ChannelGrouping | None = Field(
        None, description="Default channel grouping"
    )
    source: str | None = Field(None, description="Traffic source")
    medium: str | None = Field(None, description="Traffic medium")
    campaign_name: str | None = Field(None, description="Campaign name")
    country: str | None = Field(None, description="Country")
    device_category: DeviceCategory | None = Field(None, description="Device category")

    # Conversion metrics
    conversions: float = Field(0.0, description="Number of conversions")
    total_revenue: Decimal | None = Field(
        None, description="Total revenue from conversions"
    )
    purchase_revenue: Decimal | None = Field(None, description="Purchase revenue")
    event_count: int = Field(0, description="Total event count")

    # Conversion details
    conversion_value: Decimal | None = Field(
        None, description="Average conversion value"
    )
    items_purchased: int | None = Field(None, description="Number of items purchased")
    transaction_id: str | None = Field(None, description="Transaction ID for purchases")

    @property
    def average_order_value(self) -> Decimal | None:
        """Calculate average order value for purchase events."""
        if self.conversions > 0 and self.purchase_revenue:
            return self.purchase_revenue / Decimal(str(self.conversions))
        return None


class ChannelPerformance(BaseModel):
    """Channel performance analysis model."""

    Date_range_start: Date = Field(..., description="Start Date")
    Date_range_end: Date = Field(..., description="End Date")
    property_id: str = Field(..., description="GA4 property ID")

    # Channel details
    channel_grouping: ChannelGrouping = Field(..., description="Channel grouping")
    source: str | None = Field(None, description="Primary source")
    medium: str | None = Field(None, description="Primary medium")

    # Traffic metrics
    sessions: int = Field(0, description="Total sessions")
    users: int = Field(0, description="Total users")
    new_users: int = Field(0, description="Total new users")
    pageviews: int = Field(0, description="Total pageviews")

    # Conversion metrics
    conversions: float = Field(0.0, description="Total conversions")
    total_revenue: Decimal | None = Field(None, description="Total revenue")
    purchase_revenue: Decimal | None = Field(None, description="Purchase revenue")

    # Engagement metrics
    engagement_rate: float | None = Field(None, description="Engagement rate")
    average_session_duration: float | None = Field(
        None, description="Average session duration"
    )
    bounce_rate: float | None = Field(None, description="Bounce rate")

    @property
    def conversion_rate(self) -> float | None:
        """Calculate conversion rate."""
        if self.sessions > 0:
            return self.conversions / self.sessions
        return None

    @property
    def revenue_per_user(self) -> Decimal | None:
        """Calculate revenue per user."""
        if self.users > 0 and self.total_revenue:
            return self.total_revenue / Decimal(str(self.users))
        return None

    @property
    def revenue_per_session(self) -> Decimal | None:
        """Calculate revenue per session."""
        if self.sessions > 0 and self.total_revenue:
            return self.total_revenue / Decimal(str(self.sessions))
        return None


class AudienceSegment(BaseModel):
    """Audience segment analysis model."""

    segment_name: str = Field(..., description="Segment name")
    property_id: str = Field(..., description="GA4 property ID")
    Date_range_start: Date = Field(..., description="Start Date")
    Date_range_end: Date = Field(..., description="End Date")

    # Segment dimensions
    age_group: str | None = Field(None, description="Age group")
    gender: str | None = Field(None, description="Gender")
    country: str | None = Field(None, description="Country")
    city: str | None = Field(None, description="City")
    device_category: DeviceCategory | None = Field(None, description="Device category")

    # Behavior metrics
    users: int = Field(0, description="Users in segment")
    sessions: int = Field(0, description="Sessions from segment")
    pageviews: int = Field(0, description="Pageviews from segment")
    events: int = Field(0, description="Events from segment")
    conversions: float = Field(0.0, description="Conversions from segment")
    revenue: Decimal | None = Field(None, description="Revenue from segment")

    # Engagement metrics
    average_session_duration: float | None = Field(
        None, description="Average session duration"
    )
    pages_per_session: float | None = Field(None, description="Pages per session")
    bounce_rate: float | None = Field(None, description="Bounce rate")

    @property
    def sessions_per_user(self) -> float | None:
        """Calculate sessions per user."""
        if self.users > 0:
            return self.sessions / self.users
        return None


class AttributionData(BaseModel):
    """Attribution analysis data model."""

    conversion_event: str = Field(..., description="Conversion event name")
    property_id: str = Field(..., description="GA4 property ID")
    Date_range_start: Date = Field(..., description="Start Date")
    Date_range_end: Date = Field(..., description="End Date")

    # Attribution model
    attribution_model: str = Field(
        default="last_click", description="Attribution model used"
    )

    # Channel attribution
    channel_contributions: list[dict[str, str | float | Decimal]] = Field(
        default_factory=list, description="Channel contribution breakdown"
    )

    # Campaign attribution (for paid channels)
    campaign_contributions: list[dict[str, str | float | Decimal]] = Field(
        default_factory=list, description="Campaign contribution breakdown"
    )

    # Path analysis
    conversion_paths: list[dict[str, str | int]] = Field(
        default_factory=list, description="Common conversion paths"
    )

    # Summary metrics
    total_conversions: float = Field(0.0, description="Total conversions attributed")
    total_revenue: Decimal | None = Field(None, description="Total revenue attributed")
    unique_paths: int = Field(0, description="Number of unique conversion paths")
    average_path_length: float | None = Field(
        None, description="Average path length to conversion"
    )

    def get_channel_attribution(self, channel: str) -> dict | None:
        """Get attribution data for a specific channel."""
        for contrib in self.channel_contributions:
            if contrib.get("channel") == channel:
                return contrib
        return None

    def get_top_channels(self, limit: int = 5) -> list[dict]:
        """Get top performing channels by conversions."""
        sorted_channels = sorted(
            self.channel_contributions,
            key=lambda x: x.get("conversions", 0),
            reverse=True,
        )
        return sorted_channels[:limit]


class EcommerceData(BaseModel):
    """Ecommerce transaction data model."""

    transaction_id: str = Field(..., description="Transaction ID")
    property_id: str = Field(..., description="GA4 property ID")
    date: Date = Field(..., description="Transaction Date")

    # Transaction details
    purchase_revenue: Decimal = Field(..., description="Purchase revenue")
    tax: Decimal | None = Field(None, description="Tax amount")
    shipping: Decimal | None = Field(None, description="Shipping cost")
    total_revenue: Decimal = Field(
        ..., description="Total revenue including tax and shipping"
    )

    # Transaction attributes
    currency: str = Field(default="USD", description="Currency code")
    affiliation: str | None = Field(None, description="Store or affiliation")
    coupon: str | None = Field(None, description="Coupon code used")

    # Items
    items: list[dict[str, str | int | Decimal]] = Field(
        default_factory=list, description="Items purchased in transaction"
    )

    # Attribution
    channel_grouping: ChannelGrouping | None = Field(
        None, description="Attribution channel"
    )
    source: str | None = Field(None, description="Attribution source")
    medium: str | None = Field(None, description="Attribution medium")
    campaign_name: str | None = Field(None, description="Attribution campaign")

    # User info
    user_id: str | None = Field(None, description="User ID")
    session_id: str | None = Field(None, description="Session ID")
    device_category: DeviceCategory | None = Field(None, description="Device category")

    @property
    def item_count(self) -> int:
        """Get total number of items."""
        return sum(item.get("quantity", 0) for item in self.items)

    @property
    def average_item_value(self) -> Decimal | None:
        """Calculate average item value."""
        if self.items:
            return self.purchase_revenue / len(self.items)
        return None

    def get_product_performance(self) -> dict[str, dict[str, int | Decimal]]:
        """Get performance summary by product."""
        product_performance = {}

        for item in self.items:
            product_name = item.get("item_name", "Unknown")
            quantity = item.get("quantity", 0)
            price = item.get("price", Decimal(0))

            if product_name not in product_performance:
                product_performance[product_name] = {
                    "quantity": 0,
                    "revenue": Decimal(0),
                    "transactions": 0,
                }

            product_performance[product_name]["quantity"] += quantity
            product_performance[product_name]["revenue"] += price * quantity
            product_performance[product_name]["transactions"] += 1

        return product_performance
