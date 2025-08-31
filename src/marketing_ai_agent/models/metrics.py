"""Data models for performance metrics and aggregations."""

from datetime import datetime
from datetime import date as Date
from decimal import Decimal
from typing import Dict, List, Optional, Union

from pydantic import BaseModel, Field, validator


class MetricType(BaseModel):
    """Base metric type with common properties."""
    
    impressions: int = Field(0, description="Total impressions")
    clicks: int = Field(0, description="Total clicks")
    conversions: float = Field(0.0, description="Total conversions")
    cost_micros: int = Field(0, description="Total cost in micros")
    
    @property
    def cost(self) -> Decimal:
        """Convert cost from micros to currency amount."""
        return Decimal(self.cost_micros) / Decimal(1_000_000)
    
    @property
    def ctr(self) -> Optional[float]:
        """Calculate click-through rate."""
        if self.impressions > 0:
            return self.clicks / self.impressions
        return None
    
    @property
    def conversion_rate(self) -> Optional[float]:
        """Calculate conversion rate."""
        if self.clicks > 0:
            return self.conversions / self.clicks
        return None
    
    @property
    def average_cpc(self) -> Optional[Decimal]:
        """Calculate average cost per click."""
        if self.clicks > 0:
            return Decimal(self.cost_micros / self.clicks) / Decimal(1_000_000)
        return None
    
    @property
    def cost_per_conversion(self) -> Optional[Decimal]:
        """Calculate cost per conversion."""
        if self.conversions > 0:
            return Decimal(self.cost_micros / self.conversions) / Decimal(1_000_000)
        return None


class DailyMetrics(MetricType):
    """Daily performance metrics."""
    
    date: Date = Field(..., description="Performance date")
    entity_id: str = Field(..., description="Entity ID (campaign, ad group, keyword)")
    entity_name: str = Field(..., description="Entity name")
    entity_type: str = Field(..., description="Entity type (campaign, ad_group, keyword)")
    customer_id: str = Field(..., description="Customer account ID")


class HourlyMetrics(MetricType):
    """Hourly performance metrics."""
    
    timestamp: datetime = Field(..., description="Performance datetime")
    hour: int = Field(..., description="Hour of day (0-23)")
    entity_id: str = Field(..., description="Entity ID")
    entity_name: str = Field(..., description="Entity name")
    entity_type: str = Field(..., description="Entity type")
    customer_id: str = Field(..., description="Customer account ID")
    
    @validator('hour')
    def valiDate_hour(cls, v):
        """ValiDate hour is between 0 and 23."""
        if v < 0 or v > 23:
            raise ValueError("Hour must be between 0 and 23")
        return v


class DeviceMetrics(MetricType):
    """Device-based performance metrics."""
    
    device: str = Field(..., description="Device type (desktop, mobile, tablet)")
    date: Date = Field(..., description="Performance date")
    entity_id: str = Field(..., description="Entity ID")
    entity_name: str = Field(..., description="Entity name")
    entity_type: str = Field(..., description="Entity type")
    customer_id: str = Field(..., description="Customer account ID")


class LocationMetrics(MetricType):
    """Location-based performance metrics."""
    
    location_name: str = Field(..., description="Location name")
    location_type: str = Field(..., description="Location type (country, region, city)")
    date: Date = Field(..., description="Performance date")
    entity_id: str = Field(..., description="Entity ID")
    entity_name: str = Field(..., description="Entity name")
    entity_type: str = Field(..., description="Entity type")
    customer_id: str = Field(..., description="Customer account ID")


class AgeGroupMetrics(MetricType):
    """Age group performance metrics."""
    
    age_range: str = Field(..., description="Age range (18-24, 25-34, etc.)")
    date: Date = Field(..., description="Performance date")
    entity_id: str = Field(..., description="Entity ID")
    entity_name: str = Field(..., description="Entity name")
    entity_type: str = Field(..., description="Entity type")
    customer_id: str = Field(..., description="Customer account ID")


class GenderMetrics(MetricType):
    """Gender-based performance metrics."""
    
    gender: str = Field(..., description="Gender (male, female, undetermined)")
    date: Date = Field(..., description="Performance date")
    entity_id: str = Field(..., description="Entity ID")
    entity_name: str = Field(..., description="Entity name")
    entity_type: str = Field(..., description="Entity type")
    customer_id: str = Field(..., description="Customer account ID")


class SearchTermMetrics(MetricType):
    """Search term performance metrics."""
    
    search_term: str = Field(..., description="Search term query")
    match_type: str = Field(..., description="Match type for the search term")
    keyword_text: str = Field(..., description="Matched keyword text")
    keyword_match_type: str = Field(..., description="Keyword match type")
    ad_group_id: str = Field(..., description="Ad group ID")
    ad_group_name: str = Field(..., description="Ad group name")
    campaign_id: str = Field(..., description="Campaign ID")
    campaign_name: str = Field(..., description="Campaign name")
    customer_id: str = Field(..., description="Customer account ID")
    date: Date = Field(..., description="Performance date")
    
    # Additional search term specific metrics
    search_term_view_status: Optional[str] = Field(None, description="Search term view status")


class AggregatedMetrics(BaseModel):
    """Aggregated metrics summary."""
    
    entity_type: str = Field(..., description="Entity type being aggregated")
    entity_ids: List[str] = Field(..., description="List of entity IDs included")
    Date_range_start: Date = Field(..., description="Start Date of aggregation")
    Date_range_end: Date = Field(..., description="End Date of aggregation")
    customer_id: str = Field(..., description="Customer account ID")
    
    # Aggregated totals
    total_impressions: int = Field(0, description="Total impressions")
    total_clicks: int = Field(0, description="Total clicks")
    total_conversions: float = Field(0.0, description="Total conversions")
    total_cost_micros: int = Field(0, description="Total cost in micros")
    
    # Computed aggregates
    average_ctr: Optional[float] = Field(None, description="Average CTR across entities")
    average_conversion_rate: Optional[float] = Field(None, description="Average conversion rate")
    average_cpc_micros: Optional[int] = Field(None, description="Average CPC in micros")
    average_cost_per_conversion_micros: Optional[int] = Field(None, description="Average cost per conversion")
    
    # Entity counts
    entity_count: int = Field(..., description="Number of entities aggregated")
    
    @property
    def total_cost(self) -> Decimal:
        """Convert total cost from micros to currency amount."""
        return Decimal(self.total_cost_micros) / Decimal(1_000_000)
    
    @property
    def total_ctr(self) -> Optional[float]:
        """Calculate overall CTR."""
        if self.total_impressions > 0:
            return self.total_clicks / self.total_impressions
        return None
    
    @property
    def total_conversion_rate(self) -> Optional[float]:
        """Calculate overall conversion rate."""
        if self.total_clicks > 0:
            return self.total_conversions / self.total_clicks
        return None
    
    def calculate_averages(self, daily_metrics: List[DailyMetrics]) -> None:
        """Calculate average metrics from daily metrics list."""
        if not daily_metrics:
            return
        
        # Calculate averages for non-zero values only
        ctrs = [m.ctr for m in daily_metrics if m.ctr is not None]
        conversion_rates = [m.conversion_rate for m in daily_metrics if m.conversion_rate is not None]
        cpcs = [m.average_cpc for m in daily_metrics if m.average_cpc is not None]
        cpas = [m.cost_per_conversion for m in daily_metrics if m.cost_per_conversion is not None]
        
        self.average_ctr = sum(ctrs) / len(ctrs) if ctrs else None
        self.average_conversion_rate = sum(conversion_rates) / len(conversion_rates) if conversion_rates else None
        
        if cpcs:
            avg_cpc = sum(cpcs) / len(cpcs)
            self.average_cpc_micros = int(avg_cpc * 1_000_000)
        
        if cpas:
            avg_cpa = sum(cpas) / len(cpas)
            self.average_cost_per_conversion_micros = int(avg_cpa * 1_000_000)


class PerformanceTrend(BaseModel):
    """Performance trend analysis."""
    
    entity_id: str = Field(..., description="Entity ID")
    entity_name: str = Field(..., description="Entity name")
    entity_type: str = Field(..., description="Entity type")
    metric_name: str = Field(..., description="Metric being analyzed")
    
    # Trend data
    start_date: Date = Field(..., description="Trend start Date")
    end_date: Date = Field(..., description="Trend end Date")
    data_points: List[Dict[str, Union[Date, float]]] = Field(..., description="Time series data points")
    
    # Trend analysis
    trend_direction: Optional[str] = Field(None, description="Trend direction (up, down, flat)")
    trend_strength: Optional[float] = Field(None, description="Trend strength (0-1)")
    change_percent: Optional[float] = Field(None, description="Percentage change from start to end")
    volatility: Optional[float] = Field(None, description="Metric volatility")
    
    # Statistical measures
    mean_value: Optional[float] = Field(None, description="Mean metric value")
    median_value: Optional[float] = Field(None, description="Median metric value")
    std_deviation: Optional[float] = Field(None, description="Standard deviation")
    min_value: Optional[float] = Field(None, description="Minimum value")
    max_value: Optional[float] = Field(None, description="Maximum value")
    
    def analyze_trend(self) -> None:
        """Analyze trend from data points."""
        if not self.data_points or len(self.data_points) < 2:
            return
        
        values = [point.get("value", 0) for point in self.data_points if point.get("value") is not None]
        
        if not values:
            return
        
        # Calculate statistical measures
        self.mean_value = sum(values) / len(values)
        sorted_values = sorted(values)
        n = len(sorted_values)
        self.median_value = sorted_values[n // 2] if n % 2 == 1 else (sorted_values[n // 2 - 1] + sorted_values[n // 2]) / 2
        
        # Calculate variance and standard deviation
        variance = sum((x - self.mean_value) ** 2 for x in values) / len(values)
        self.std_deviation = variance ** 0.5
        
        self.min_value = min(values)
        self.max_value = max(values)
        
        # Calculate trend direction and change
        start_value = values[0]
        end_value = values[-1]
        
        if start_value > 0:
            self.change_percent = ((end_value - start_value) / start_value) * 100
        
        # Simple trend direction
        if abs(self.change_percent) < 5:  # Less than 5% change
            self.trend_direction = "flat"
        elif self.change_percent > 0:
            self.trend_direction = "up"
        else:
            self.trend_direction = "down"
        
        # Trend strength based on consistency of direction
        positive_changes = sum(1 for i in range(1, len(values)) if values[i] > values[i-1])
        negative_changes = sum(1 for i in range(1, len(values)) if values[i] < values[i-1])
        total_changes = len(values) - 1
        
        if total_changes > 0:
            if self.trend_direction == "up":
                self.trend_strength = positive_changes / total_changes
            elif self.trend_direction == "down":
                self.trend_strength = negative_changes / total_changes
            else:
                self.trend_strength = 1 - (abs(positive_changes - negative_changes) / total_changes)
        
        # Volatility as coefficient of variation
        if self.mean_value > 0:
            self.volatility = self.std_deviation / self.mean_value