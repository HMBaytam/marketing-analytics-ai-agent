"""Data models and schemas for marketing data."""

from .campaign import (
    Campaign,
    AdGroup,
    Keyword,
    CampaignPerformance,
    CampaignStatus,
    AdGroupStatus,
    KeywordStatus,
    KeywordMatchType,
    AdvertisingChannelType,
    AdGroupType
)
from .metrics import (
    MetricType,
    DailyMetrics,
    HourlyMetrics,
    DeviceMetrics,
    LocationMetrics,
    AgeGroupMetrics,
    GenderMetrics,
    SearchTermMetrics,
    AggregatedMetrics,
    PerformanceTrend
)
from .analytics import (
    TrafficData,
    ConversionEvent,
    ChannelPerformance,
    AudienceSegment,
    AttributionData,
    EcommerceData,
    ChannelGrouping,
    DeviceCategory,
    EventCategory
)
from .transformers import GoogleAdsTransformer, GA4Transformer, DataCleaner
from .cache import CacheManager, CacheConfig
from .exporters import DataExporter, MultiSheetExporter, ReportGenerator, ExportConfig

__all__ = [
    # Campaign models
    "Campaign", "AdGroup", "Keyword", "CampaignPerformance",
    "CampaignStatus", "AdGroupStatus", "KeywordStatus", "KeywordMatchType",
    "AdvertisingChannelType", "AdGroupType",
    
    # Metrics models
    "MetricType", "DailyMetrics", "HourlyMetrics", "DeviceMetrics",
    "LocationMetrics", "AgeGroupMetrics", "GenderMetrics", "SearchTermMetrics",
    "AggregatedMetrics", "PerformanceTrend",
    
    # Analytics models
    "TrafficData", "ConversionEvent", "ChannelPerformance", "AudienceSegment",
    "AttributionData", "EcommerceData", "ChannelGrouping", "DeviceCategory",
    "EventCategory",
    
    # Utility classes
    "GoogleAdsTransformer", "GA4Transformer", "DataCleaner",
    "CacheManager", "CacheConfig",
    "DataExporter", "MultiSheetExporter", "ReportGenerator", "ExportConfig"
]