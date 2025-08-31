"""Data models for Google Ads campaigns, ad groups, and keywords."""

from datetime import datetime
from datetime import date as Date
from decimal import Decimal
from enum import Enum
from typing import Dict, List, Optional, Union

from pydantic import BaseModel, Field, validator


class CampaignStatus(str, Enum):
    """Campaign status enumeration."""
    ENABLED = "ENABLED"
    PAUSED = "PAUSED"
    REMOVED = "REMOVED"
    UNKNOWN = "UNKNOWN"


class AdvertisingChannelType(str, Enum):
    """Advertising channel type enumeration."""
    SEARCH = "SEARCH"
    DISPLAY = "DISPLAY"
    SHOPPING = "SHOPPING"
    HOTEL = "HOTEL"
    VIDEO = "VIDEO"
    MULTI_CHANNEL = "MULTI_CHANNEL"
    LOCAL = "LOCAL"
    SMART = "SMART"
    PERFORMANCE_MAX = "PERFORMANCE_MAX"
    UNKNOWN = "UNKNOWN"


class AdGroupStatus(str, Enum):
    """Ad group status enumeration."""
    ENABLED = "ENABLED"
    PAUSED = "PAUSED"
    REMOVED = "REMOVED"
    UNKNOWN = "UNKNOWN"


class AdGroupType(str, Enum):
    """Ad group type enumeration."""
    STANDARD = "STANDARD"
    DISPLAY_STANDARD = "DISPLAY_STANDARD"
    SHOPPING_PRODUCT_ADS = "SHOPPING_PRODUCT_ADS"
    SHOPPING_SHOWCASE_ADS = "SHOPPING_SHOWCASE_ADS"
    HOTEL_ADS = "HOTEL_ADS"
    SMART_DISPLAY_ADS = "SMART_DISPLAY_ADS"
    VIDEO_BUMPER = "VIDEO_BUMPER"
    VIDEO_TRUE_VIEW_DISCOVERY = "VIDEO_TRUE_VIEW_DISCOVERY"
    VIDEO_TRUE_VIEW_IN_STREAM = "VIDEO_TRUE_VIEW_IN_STREAM"
    VIDEO_ACTION = "VIDEO_ACTION"
    SEARCH_DYNAMIC_ADS = "SEARCH_DYNAMIC_ADS"
    UNKNOWN = "UNKNOWN"


class KeywordMatchType(str, Enum):
    """Keyword match type enumeration."""
    EXACT = "EXACT"
    PHRASE = "PHRASE"
    BROAD = "BROAD"
    UNKNOWN = "UNKNOWN"


class KeywordStatus(str, Enum):
    """Keyword status enumeration."""
    ENABLED = "ENABLED"
    PAUSED = "PAUSED"
    REMOVED = "REMOVED"
    UNKNOWN = "UNKNOWN"


class Campaign(BaseModel):
    """Google Ads campaign data model."""
    
    id: str = Field(..., description="Campaign ID")
    name: str = Field(..., description="Campaign name")
    status: CampaignStatus = Field(..., description="Campaign status")
    advertising_channel_type: AdvertisingChannelType = Field(..., description="Advertising channel type")
    start_date: Optional[Date] = Field(None, description="Campaign start date")
    end_date: Optional[Date] = Field(None, description="Campaign end date")
    budget_amount_micros: Optional[int] = Field(None, description="Daily budget in micros")
    target_cpa_micros: Optional[int] = Field(None, description="Target CPA in micros")
    target_roas: Optional[float] = Field(None, description="Target ROAS")
    customer_id: str = Field(..., description="Customer account ID")
    
    # Performance metrics (optional, populated from metrics queries)
    impressions: Optional[int] = Field(None, description="Total impressions")
    clicks: Optional[int] = Field(None, description="Total clicks")
    conversions: Optional[float] = Field(None, description="Total conversions")
    cost_micros: Optional[int] = Field(None, description="Total cost in micros")
    
    # Computed fields
    ctr: Optional[float] = Field(None, description="Click-through rate")
    average_cpc_micros: Optional[int] = Field(None, description="Average cost per click in micros")
    cost_per_conversion_micros: Optional[int] = Field(None, description="Cost per conversion in micros")
    conversion_rate: Optional[float] = Field(None, description="Conversion rate")
    
    @property
    def budget_amount(self) -> Optional[Decimal]:
        """Convert budget from micros to currency amount."""
        if self.budget_amount_micros is None:
            return None
        return Decimal(self.budget_amount_micros) / Decimal(1_000_000)
    
    @property
    def cost(self) -> Optional[Decimal]:
        """Convert cost from micros to currency amount."""
        if self.cost_micros is None:
            return None
        return Decimal(self.cost_micros) / Decimal(1_000_000)
    
    @property
    def average_cpc(self) -> Optional[Decimal]:
        """Convert average CPC from micros to currency amount."""
        if self.average_cpc_micros is None:
            return None
        return Decimal(self.average_cpc_micros) / Decimal(1_000_000)
    
    @property
    def cost_per_conversion(self) -> Optional[Decimal]:
        """Convert cost per conversion from micros to currency amount."""
        if self.cost_per_conversion_micros is None:
            return None
        return Decimal(self.cost_per_conversion_micros) / Decimal(1_000_000)
    
    def calculate_metrics(self) -> None:
        """Calculate derived metrics from base metrics."""
        if self.impressions and self.clicks:
            self.ctr = self.clicks / self.impressions
        
        if self.clicks and self.conversions:
            self.conversion_rate = self.conversions / self.clicks
        
        if self.clicks and self.cost_micros:
            self.average_cpc_micros = int(self.cost_micros / self.clicks)
        
        if self.conversions and self.cost_micros and self.conversions > 0:
            self.cost_per_conversion_micros = int(self.cost_micros / self.conversions)


class AdGroup(BaseModel):
    """Google Ads ad group data model."""
    
    id: str = Field(..., description="Ad group ID")
    name: str = Field(..., description="Ad group name")
    status: AdGroupStatus = Field(..., description="Ad group status")
    type: AdGroupType = Field(..., description="Ad group type")
    campaign_id: str = Field(..., description="Parent campaign ID")
    campaign_name: str = Field(..., description="Parent campaign name")
    customer_id: str = Field(..., description="Customer account ID")
    
    # Bidding
    cpc_bid_micros: Optional[int] = Field(None, description="CPC bid in micros")
    cpm_bid_micros: Optional[int] = Field(None, description="CPM bid in micros")
    target_cpa_micros: Optional[int] = Field(None, description="Target CPA in micros")
    
    # Performance metrics
    impressions: Optional[int] = Field(None, description="Total impressions")
    clicks: Optional[int] = Field(None, description="Total clicks")
    conversions: Optional[float] = Field(None, description="Total conversions")
    cost_micros: Optional[int] = Field(None, description="Total cost in micros")
    
    # Computed fields
    ctr: Optional[float] = Field(None, description="Click-through rate")
    average_cpc_micros: Optional[int] = Field(None, description="Average cost per click in micros")
    cost_per_conversion_micros: Optional[int] = Field(None, description="Cost per conversion in micros")
    conversion_rate: Optional[float] = Field(None, description="Conversion rate")
    
    @property
    def cpc_bid(self) -> Optional[Decimal]:
        """Convert CPC bid from micros to currency amount."""
        if self.cpc_bid_micros is None:
            return None
        return Decimal(self.cpc_bid_micros) / Decimal(1_000_000)
    
    @property
    def cost(self) -> Optional[Decimal]:
        """Convert cost from micros to currency amount."""
        if self.cost_micros is None:
            return None
        return Decimal(self.cost_micros) / Decimal(1_000_000)
    
    def calculate_metrics(self) -> None:
        """Calculate derived metrics from base metrics."""
        if self.impressions and self.clicks:
            self.ctr = self.clicks / self.impressions
        
        if self.clicks and self.conversions:
            self.conversion_rate = self.conversions / self.clicks
        
        if self.clicks and self.cost_micros:
            self.average_cpc_micros = int(self.cost_micros / self.clicks)
        
        if self.conversions and self.cost_micros and self.conversions > 0:
            self.cost_per_conversion_micros = int(self.cost_micros / self.conversions)


class Keyword(BaseModel):
    """Google Ads keyword data model."""
    
    text: str = Field(..., description="Keyword text")
    match_type: KeywordMatchType = Field(..., description="Keyword match type")
    status: KeywordStatus = Field(..., description="Keyword status")
    ad_group_id: str = Field(..., description="Parent ad group ID")
    ad_group_name: str = Field(..., description="Parent ad group name")
    campaign_id: str = Field(..., description="Parent campaign ID")
    campaign_name: str = Field(..., description="Parent campaign name")
    customer_id: str = Field(..., description="Customer account ID")
    
    # Quality and bidding
    quality_score: Optional[int] = Field(None, description="Quality score (1-10)")
    cpc_bid_micros: Optional[int] = Field(None, description="CPC bid in micros")
    first_page_cpc_micros: Optional[int] = Field(None, description="First page CPC estimate")
    top_of_page_cpc_micros: Optional[int] = Field(None, description="Top of page CPC estimate")
    
    # Performance metrics
    impressions: Optional[int] = Field(None, description="Total impressions")
    clicks: Optional[int] = Field(None, description="Total clicks")
    conversions: Optional[float] = Field(None, description="Total conversions")
    cost_micros: Optional[int] = Field(None, description="Total cost in micros")
    
    # Search terms and positions
    average_position: Optional[float] = Field(None, description="Average position")
    search_impression_share: Optional[float] = Field(None, description="Search impression share")
    search_exact_match_impression_share: Optional[float] = Field(None, description="Exact match impression share")
    
    # Computed fields
    ctr: Optional[float] = Field(None, description="Click-through rate")
    average_cpc_micros: Optional[int] = Field(None, description="Average cost per click in micros")
    cost_per_conversion_micros: Optional[int] = Field(None, description="Cost per conversion in micros")
    conversion_rate: Optional[float] = Field(None, description="Conversion rate")
    
    @property
    def cpc_bid(self) -> Optional[Decimal]:
        """Convert CPC bid from micros to currency amount."""
        if self.cpc_bid_micros is None:
            return None
        return Decimal(self.cpc_bid_micros) / Decimal(1_000_000)
    
    @property
    def cost(self) -> Optional[Decimal]:
        """Convert cost from micros to currency amount."""
        if self.cost_micros is None:
            return None
        return Decimal(self.cost_micros) / Decimal(1_000_000)
    
    @property
    def average_cpc(self) -> Optional[Decimal]:
        """Convert average CPC from micros to currency amount."""
        if self.average_cpc_micros is None:
            return None
        return Decimal(self.average_cpc_micros) / Decimal(1_000_000)
    
    def calculate_metrics(self) -> None:
        """Calculate derived metrics from base metrics."""
        if self.impressions and self.clicks:
            self.ctr = self.clicks / self.impressions
        
        if self.clicks and self.conversions:
            self.conversion_rate = self.conversions / self.clicks
        
        if self.clicks and self.cost_micros:
            self.average_cpc_micros = int(self.cost_micros / self.clicks)
        
        if self.conversions and self.cost_micros and self.conversions > 0:
            self.cost_per_conversion_micros = int(self.cost_micros / self.conversions)
    
    @validator('quality_score')
    def validate_quality_score(cls, v):
        """Validate quality score is between 1 and 10."""
        if v is not None and (v < 1 or v > 10):
            raise ValueError("Quality score must be between 1 and 10")
        return v


class CampaignPerformance(BaseModel):
    """Campaign performance data with date dimension."""
    
    date: Date = Field(..., description="Performance date")
    campaign_id: str = Field(..., description="Campaign ID")
    campaign_name: str = Field(..., description="Campaign name")
    customer_id: str = Field(..., description="Customer account ID")
    
    # Core metrics
    impressions: int = Field(0, description="Total impressions")
    clicks: int = Field(0, description="Total clicks")
    conversions: float = Field(0.0, description="Total conversions")
    cost_micros: int = Field(0, description="Total cost in micros")
    
    # Derived metrics
    ctr: Optional[float] = Field(None, description="Click-through rate")
    average_cpc_micros: Optional[int] = Field(None, description="Average cost per click in micros")
    cost_per_conversion_micros: Optional[int] = Field(None, description="Cost per conversion in micros")
    conversion_rate: Optional[float] = Field(None, description="Conversion rate")
    
    @property
    def cost(self) -> Decimal:
        """Convert cost from micros to currency amount."""
        return Decimal(self.cost_micros) / Decimal(1_000_000)
    
    def calculate_metrics(self) -> None:
        """Calculate derived metrics from base metrics."""
        if self.impressions > 0 and self.clicks > 0:
            self.ctr = self.clicks / self.impressions
        
        if self.clicks > 0 and self.conversions > 0:
            self.conversion_rate = self.conversions / self.clicks
        
        if self.clicks > 0 and self.cost_micros > 0:
            self.average_cpc_micros = int(self.cost_micros / self.clicks)
        
        if self.conversions > 0 and self.cost_micros > 0:
            self.cost_per_conversion_micros = int(self.cost_micros / self.conversions)