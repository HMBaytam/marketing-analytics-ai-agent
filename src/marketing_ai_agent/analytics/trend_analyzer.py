"""Trend analysis and forecasting engine for marketing performance data."""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
from scipy import stats
from pydantic import BaseModel, Field

from ..models.metrics import DailyMetrics
from ..models.campaign import Campaign

logger = logging.getLogger(__name__)


class TrendConfig(BaseModel):
    """Configuration for trend analysis."""
    
    # Analysis parameters
    min_data_points: int = Field(7, description="Minimum data points for trend analysis")
    confidence_level: float = Field(0.95, description="Confidence level for trend significance")
    seasonal_period: int = Field(7, description="Seasonal period in days (7 for weekly)")
    
    # Forecasting parameters
    forecast_days: int = Field(30, description="Number of days to forecast")
    smoothing_factor: float = Field(0.3, description="Exponential smoothing factor")
    trend_dampening: float = Field(0.8, description="Trend dampening factor")
    
    # Change detection thresholds
    significant_change_threshold: float = Field(0.15, description="Threshold for significant change (15%)")
    anomaly_threshold: float = Field(2.0, description="Z-score threshold for anomaly detection")
    
    class Config:
        arbitrary_types_allowed = True


class TrendPoint(BaseModel):
    """Individual trend data point."""
    
    date: datetime = Field(..., description="Data point date")
    value: float = Field(..., description="Metric value")
    smoothed_value: Optional[float] = Field(None, description="Smoothed value")
    forecast: bool = Field(False, description="Whether this is a forecast point")
    confidence_interval: Optional[Tuple[float, float]] = Field(None, description="Confidence interval for forecasts")


class TrendAnalysis(BaseModel):
    """Comprehensive trend analysis result."""
    
    metric_name: str = Field(..., description="Name of analyzed metric")
    entity_id: str = Field(..., description="Entity ID")
    entity_name: str = Field(..., description="Entity name")
    
    # Trend characteristics
    trend_direction: str = Field(..., description="Overall trend direction (up, down, stable)")
    trend_strength: float = Field(..., description="Trend strength (0-1)")
    trend_significance: float = Field(..., description="Statistical significance (p-value)")
    
    # Trend metrics
    slope: float = Field(..., description="Linear trend slope")
    r_squared: float = Field(..., description="R-squared value")
    correlation: float = Field(..., description="Correlation coefficient")
    
    # Change analysis
    period_change: float = Field(..., description="Total change over analysis period (%)")
    recent_change: float = Field(..., description="Recent change (last week %)")
    acceleration: float = Field(..., description="Trend acceleration")
    
    # Forecasting
    forecast_points: List[TrendPoint] = Field(default_factory=list, description="Forecast data points")
    forecast_confidence: float = Field(..., description="Forecast confidence score")
    
    # Seasonality
    seasonal_pattern: Optional[List[float]] = Field(None, description="Detected seasonal pattern")
    seasonality_strength: float = Field(0.0, description="Seasonality strength")
    
    # Data points
    historical_points: List[TrendPoint] = Field(default_factory=list, description="Historical data points")
    
    # Analysis metadata
    analysis_period_days: int = Field(..., description="Analysis period in days")
    data_quality_score: float = Field(..., description="Data quality assessment")
    calculated_at: datetime = Field(default_factory=datetime.utcnow, description="Analysis timestamp")


class TrendAnalyzer:
    """Advanced trend analysis and forecasting engine."""
    
    def __init__(self, config: Optional[TrendConfig] = None):
        """
        Initialize trend analyzer.
        
        Args:
            config: Trend analysis configuration
        """
        self.config = config or TrendConfig()
    
    def analyze_campaign_trends(
        self, 
        campaign: Campaign, 
        historical_metrics: List[DailyMetrics]
    ) -> Dict[str, TrendAnalysis]:
        """
        Analyze trends for all key metrics of a campaign.
        
        Args:
            campaign: Campaign to analyze
            historical_metrics: Historical performance data
            
        Returns:
            Dictionary of trend analyses by metric name
        """
        try:
            if len(historical_metrics) < self.config.min_data_points:
                logger.warning(f"Insufficient data for trend analysis: {len(historical_metrics)} points")
                return {}
            
            # Sort metrics by date
            sorted_metrics = sorted(historical_metrics, key=lambda x: x.date)
            
            trend_analyses = {}
            
            # Analyze key metrics
            metrics_to_analyze = [
                ("impressions", "impressions"),
                ("clicks", "clicks"), 
                ("conversions", "conversions"),
                ("cost", "cost"),
                ("ctr", "ctr"),
                ("conversion_rate", "conversion_rate")
            ]
            
            for metric_name, metric_attr in metrics_to_analyze:
                try:
                    # Extract metric values
                    values = []
                    dates = []
                    
                    for metric in sorted_metrics:
                        value = getattr(metric, metric_attr, None)
                        if value is not None and value > 0:  # Skip zero/null values
                            values.append(float(value))
                            dates.append(metric.date)
                    
                    if len(values) >= self.config.min_data_points:
                        analysis = self.analyze_metric_trend(
                            metric_name=metric_name,
                            entity_id=campaign.id,
                            entity_name=campaign.name,
                            dates=dates,
                            values=values
                        )
                        trend_analyses[metric_name] = analysis
                        
                except Exception as e:
                    logger.error(f"Failed to analyze {metric_name} trend: {e}")
                    continue
            
            return trend_analyses
            
        except Exception as e:
            logger.error(f"Campaign trend analysis failed: {e}")
            return {}
    
    def analyze_metric_trend(
        self,
        metric_name: str,
        entity_id: str,
        entity_name: str,
        dates: List[datetime],
        values: List[float]
    ) -> TrendAnalysis:
        """
        Analyze trend for a specific metric.
        
        Args:
            metric_name: Name of the metric
            entity_id: Entity identifier
            entity_name: Entity name
            dates: List of dates
            values: List of metric values
            
        Returns:
            Comprehensive trend analysis
        """
        try:
            # Convert dates to numeric for analysis
            date_nums = np.array([(d - dates[0]).days for d in dates])
            value_array = np.array(values)
            
            # Linear regression for trend
            slope, intercept, r_value, p_value, std_err = stats.linregress(date_nums, value_array)
            
            # Trend characteristics
            trend_direction = self._determine_trend_direction(slope, p_value)
            trend_strength = min(1.0, abs(r_value))
            
            # Change analysis
            period_change = ((values[-1] - values[0]) / values[0] * 100) if values[0] != 0 else 0.0
            recent_change = self._calculate_recent_change(values, window=7)
            acceleration = self._calculate_acceleration(date_nums, value_array)
            
            # Smoothing and forecasting
            smoothed_values = self._exponential_smoothing(value_array)
            forecast_points = self._generate_forecast(dates, smoothed_values, slope)
            
            # Seasonality analysis
            seasonal_pattern, seasonality_strength = self._detect_seasonality(value_array)
            
            # Create historical points
            historical_points = []
            for i, (date, value) in enumerate(zip(dates, values)):
                historical_points.append(TrendPoint(
                    date=date,
                    value=value,
                    smoothed_value=smoothed_values[i] if i < len(smoothed_values) else None,
                    forecast=False
                ))
            
            # Data quality assessment
            data_quality_score = self._assess_data_quality(values, dates)
            
            return TrendAnalysis(
                metric_name=metric_name,
                entity_id=entity_id,
                entity_name=entity_name,
                trend_direction=trend_direction,
                trend_strength=trend_strength,
                trend_significance=p_value,
                slope=slope,
                r_squared=r_value ** 2,
                correlation=r_value,
                period_change=period_change,
                recent_change=recent_change,
                acceleration=acceleration,
                forecast_points=forecast_points,
                forecast_confidence=self._calculate_forecast_confidence(r_value, len(values)),
                seasonal_pattern=seasonal_pattern,
                seasonality_strength=seasonality_strength,
                historical_points=historical_points,
                analysis_period_days=(dates[-1] - dates[0]).days,
                data_quality_score=data_quality_score
            )
            
        except Exception as e:
            logger.error(f"Metric trend analysis failed for {metric_name}: {e}")
            # Return minimal analysis
            return TrendAnalysis(
                metric_name=metric_name,
                entity_id=entity_id,
                entity_name=entity_name,
                trend_direction="unknown",
                trend_strength=0.0,
                trend_significance=1.0,
                slope=0.0,
                r_squared=0.0,
                correlation=0.0,
                period_change=0.0,
                recent_change=0.0,
                acceleration=0.0,
                forecast_confidence=0.0,
                analysis_period_days=0,
                data_quality_score=0.0
            )
    
    def _determine_trend_direction(self, slope: float, p_value: float) -> str:
        """Determine trend direction based on slope and significance."""
        if p_value > (1 - self.config.confidence_level):
            return "stable"
        elif slope > 0:
            return "up"
        else:
            return "down"
    
    def _calculate_recent_change(self, values: List[float], window: int = 7) -> float:
        """Calculate recent change percentage over a window."""
        try:
            if len(values) < window:
                return 0.0
            
            recent_values = values[-window:]
            if len(recent_values) < 2:
                return 0.0
            
            start_value = recent_values[0]
            end_value = recent_values[-1]
            
            if start_value != 0:
                return ((end_value - start_value) / start_value) * 100
            else:
                return 0.0
                
        except Exception:
            return 0.0
    
    def _calculate_acceleration(self, x: np.ndarray, y: np.ndarray) -> float:
        """Calculate trend acceleration (second derivative)."""
        try:
            if len(x) < 3:
                return 0.0
            
            # Fit quadratic polynomial
            coeffs = np.polyfit(x, y, 2)
            # Second derivative is 2 * a (where polynomial is ax^2 + bx + c)
            acceleration = 2 * coeffs[0]
            
            return float(acceleration)
            
        except Exception:
            return 0.0
    
    def _exponential_smoothing(self, values: np.ndarray) -> np.ndarray:
        """Apply exponential smoothing to the data."""
        try:
            alpha = self.config.smoothing_factor
            smoothed = np.zeros_like(values)
            smoothed[0] = values[0]
            
            for i in range(1, len(values)):
                smoothed[i] = alpha * values[i] + (1 - alpha) * smoothed[i-1]
            
            return smoothed
            
        except Exception:
            return values.copy()
    
    def _generate_forecast(
        self, 
        dates: List[datetime], 
        smoothed_values: np.ndarray, 
        slope: float
    ) -> List[TrendPoint]:
        """Generate forecast points."""
        try:
            forecast_points = []
            last_date = dates[-1]
            last_value = smoothed_values[-1]
            
            # Calculate daily slope adjustment
            daily_slope = slope * self.config.trend_dampening
            
            # Generate forecast points
            for i in range(1, self.config.forecast_days + 1):
                forecast_date = last_date + timedelta(days=i)
                
                # Simple linear extrapolation with dampening
                forecast_value = last_value + (daily_slope * i)
                
                # Ensure forecast is positive
                forecast_value = max(0.0, forecast_value)
                
                # Calculate confidence interval (simple approximation)
                confidence_width = forecast_value * 0.1 * i  # Increasing uncertainty
                confidence_interval = (
                    max(0.0, forecast_value - confidence_width),
                    forecast_value + confidence_width
                )
                
                forecast_points.append(TrendPoint(
                    date=forecast_date,
                    value=forecast_value,
                    forecast=True,
                    confidence_interval=confidence_interval
                ))
            
            return forecast_points
            
        except Exception as e:
            logger.error(f"Forecast generation failed: {e}")
            return []
    
    def _detect_seasonality(self, values: np.ndarray) -> Tuple[Optional[List[float]], float]:
        """Detect seasonal patterns in the data."""
        try:
            if len(values) < self.config.seasonal_period * 2:
                return None, 0.0
            
            # Simple seasonal decomposition
            period = self.config.seasonal_period
            
            # Calculate seasonal averages
            seasonal_pattern = []
            for i in range(period):
                period_values = []
                for j in range(i, len(values), period):
                    period_values.append(values[j])
                
                if period_values:
                    seasonal_pattern.append(np.mean(period_values))
            
            # Calculate seasonality strength
            if seasonal_pattern:
                overall_mean = np.mean(values)
                seasonal_variance = np.var(seasonal_pattern)
                total_variance = np.var(values)
                
                if total_variance > 0:
                    seasonality_strength = min(1.0, seasonal_variance / total_variance)
                else:
                    seasonality_strength = 0.0
            else:
                seasonality_strength = 0.0
            
            return seasonal_pattern, seasonality_strength
            
        except Exception:
            return None, 0.0
    
    def _calculate_forecast_confidence(self, r_value: float, data_points: int) -> float:
        """Calculate forecast confidence based on trend fit and data volume."""
        try:
            # Base confidence on R-squared
            base_confidence = abs(r_value) ** 2
            
            # Adjust for data volume
            data_confidence = min(1.0, data_points / 30.0)  # Full confidence at 30+ points
            
            # Combined confidence
            overall_confidence = (base_confidence * 0.7) + (data_confidence * 0.3)
            
            return min(1.0, overall_confidence)
            
        except Exception:
            return 0.0
    
    def _assess_data_quality(self, values: List[float], dates: List[datetime]) -> float:
        """Assess data quality for trend analysis."""
        try:
            quality_score = 0.0
            
            # Completeness score (no gaps)
            if dates:
                expected_days = (dates[-1] - dates[0]).days + 1
                actual_days = len(dates)
                completeness = min(1.0, actual_days / expected_days)
                quality_score += completeness * 0.4
            
            # Consistency score (low variance in differences)
            if len(values) > 1:
                diffs = np.diff(values)
                if np.std(diffs) > 0:
                    consistency = 1.0 / (1.0 + np.std(diffs) / np.mean(np.abs(diffs)))
                    quality_score += consistency * 0.3
                else:
                    quality_score += 0.3
            
            # Volume adequacy
            volume_score = min(1.0, len(values) / self.config.min_data_points)
            quality_score += volume_score * 0.3
            
            return min(1.0, quality_score)
            
        except Exception:
            return 0.0
    
    def detect_significant_changes(self, trend_analyses: Dict[str, TrendAnalysis]) -> List[Dict[str, Any]]:
        """Detect significant changes across trend analyses."""
        significant_changes = []
        
        try:
            for metric_name, analysis in trend_analyses.items():
                # Recent change threshold
                if abs(analysis.recent_change) > self.config.significant_change_threshold * 100:
                    change_type = "increase" if analysis.recent_change > 0 else "decrease"
                    significance = "high" if abs(analysis.recent_change) > 30 else "medium"
                    
                    significant_changes.append({
                        "metric": metric_name,
                        "entity_id": analysis.entity_id,
                        "entity_name": analysis.entity_name,
                        "change_type": change_type,
                        "change_magnitude": abs(analysis.recent_change),
                        "significance": significance,
                        "trend_direction": analysis.trend_direction,
                        "trend_strength": analysis.trend_strength
                    })
                
                # Trend acceleration
                if abs(analysis.acceleration) > 0.1:  # Arbitrary threshold
                    accel_type = "accelerating" if analysis.acceleration > 0 else "decelerating"
                    
                    significant_changes.append({
                        "metric": metric_name,
                        "entity_id": analysis.entity_id,
                        "entity_name": analysis.entity_name,
                        "change_type": accel_type,
                        "change_magnitude": abs(analysis.acceleration),
                        "significance": "medium",
                        "trend_direction": analysis.trend_direction,
                        "trend_strength": analysis.trend_strength
                    })
            
            return sorted(significant_changes, key=lambda x: x["change_magnitude"], reverse=True)
            
        except Exception as e:
            logger.error(f"Significant change detection failed: {e}")
            return []
    
    def get_trend_summary(self, trend_analyses: Dict[str, TrendAnalysis]) -> Dict[str, Any]:
        """Generate summary insights from trend analyses."""
        try:
            if not trend_analyses:
                return {"error": "No trend analyses to summarize"}
            
            summary = {
                "metrics_analyzed": len(trend_analyses),
                "overall_trend": "mixed",
                "strong_trends": 0,
                "trending_up": 0,
                "trending_down": 0,
                "stable_metrics": 0,
                "forecast_reliability": 0.0,
                "seasonal_metrics": 0,
                "key_insights": []
            }
            
            # Aggregate statistics
            trend_strengths = []
            forecast_confidences = []
            
            for metric_name, analysis in trend_analyses.items():
                trend_strengths.append(analysis.trend_strength)
                forecast_confidences.append(analysis.forecast_confidence)
                
                # Count trends by direction
                if analysis.trend_direction == "up":
                    summary["trending_up"] += 1
                elif analysis.trend_direction == "down":
                    summary["trending_down"] += 1
                else:
                    summary["stable_metrics"] += 1
                
                # Strong trends
                if analysis.trend_strength > 0.7:
                    summary["strong_trends"] += 1
                
                # Seasonal metrics
                if analysis.seasonality_strength > 0.3:
                    summary["seasonal_metrics"] += 1
            
            # Overall trend assessment
            if summary["trending_up"] > summary["trending_down"]:
                summary["overall_trend"] = "positive"
            elif summary["trending_down"] > summary["trending_up"]:
                summary["overall_trend"] = "negative"
            
            # Average forecast reliability
            if forecast_confidences:
                summary["forecast_reliability"] = np.mean(forecast_confidences)
            
            # Key insights
            if summary["strong_trends"] > len(trend_analyses) * 0.5:
                summary["key_insights"].append("Strong trends detected in majority of metrics")
            
            if summary["seasonal_metrics"] > 0:
                summary["key_insights"].append(f"Seasonal patterns found in {summary['seasonal_metrics']} metrics")
            
            if summary["forecast_reliability"] > 0.7:
                summary["key_insights"].append("High confidence forecasts available")
            elif summary["forecast_reliability"] < 0.3:
                summary["key_insights"].append("Low forecast confidence due to volatile trends")
            
            return summary
            
        except Exception as e:
            logger.error(f"Trend summary generation failed: {e}")
            return {"error": str(e)}