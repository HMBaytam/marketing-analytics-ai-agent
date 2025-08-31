"""Data models and schemas for marketing data."""

from .analytics import (
    AttributionData,
    AudienceSegment,
    ChannelGrouping,
    ChannelPerformance,
    ConversionEvent,
    DeviceCategory,
    EcommerceData,
    EventCategory,
    TrafficData,
)
from .cache import CacheConfig, CacheManager
from .campaign import (
    AdGroup,
    AdGroupStatus,
    AdGroupType,
    AdvertisingChannelType,
    Campaign,
    CampaignPerformance,
    CampaignStatus,
    Keyword,
    KeywordMatchType,
    KeywordStatus,
)
from .exporters import DataExporter, ExportConfig, MultiSheetExporter, ReportGenerator
from .metrics import (
    AgeGroupMetrics,
    AggregatedMetrics,
    DailyMetrics,
    DeviceMetrics,
    GenderMetrics,
    HourlyMetrics,
    LocationMetrics,
    MetricType,
    PerformanceTrend,
    SearchTermMetrics,
)
from .transformers import DataCleaner, GA4Transformer, GoogleAdsTransformer

__all__ = [
    # Campaign models
    "Campaign",
    "AdGroup",
    "Keyword",
    "CampaignPerformance",
    "CampaignStatus",
    "AdGroupStatus",
    "KeywordStatus",
    "KeywordMatchType",
    "AdvertisingChannelType",
    "AdGroupType",
    # Metrics models
    "MetricType",
    "DailyMetrics",
    "HourlyMetrics",
    "DeviceMetrics",
    "LocationMetrics",
    "AgeGroupMetrics",
    "GenderMetrics",
    "SearchTermMetrics",
    "AggregatedMetrics",
    "PerformanceTrend",
    # Analytics models
    "TrafficData",
    "ConversionEvent",
    "ChannelPerformance",
    "AudienceSegment",
    "AttributionData",
    "EcommerceData",
    "ChannelGrouping",
    "DeviceCategory",
    "EventCategory",
    # Utility classes
    "GoogleAdsTransformer",
    "GA4Transformer",
    "DataCleaner",
    "CacheManager",
    "CacheConfig",
    "DataExporter",
    "MultiSheetExporter",
    "ReportGenerator",
    "ExportConfig",
]
