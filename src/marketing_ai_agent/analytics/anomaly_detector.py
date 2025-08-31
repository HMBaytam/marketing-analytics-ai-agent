"""Anomaly detection system for identifying unusual patterns in marketing data."""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union

import numpy as np
from scipy import stats
from pydantic import BaseModel, Field

from ..models.metrics import DailyMetrics
from ..models.campaign import Campaign

logger = logging.getLogger(__name__)


class AnomalyConfig(BaseModel):
    """Configuration for anomaly detection."""
    
    # Statistical thresholds
    z_score_threshold: float = Field(2.5, description="Z-score threshold for statistical anomalies")
    iqr_multiplier: float = Field(1.5, description="IQR multiplier for outlier detection")
    confidence_level: float = Field(0.95, description="Confidence level for anomaly detection")
    
    # Change detection
    sudden_change_threshold: float = Field(0.3, description="Threshold for sudden change detection (30%)")
    gradual_change_window: int = Field(7, description="Window size for gradual change detection")
    
    # Volume thresholds
    min_spend_anomaly: float = Field(100.0, description="Minimum spend for spend anomaly detection")
    min_volume_anomaly: int = Field(100, description="Minimum volume for volume anomaly detection")
    
    # Pattern detection
    pattern_window: int = Field(14, description="Window size for pattern anomaly detection")
    seasonal_window: int = Field(7, description="Window size for seasonal comparison")
    
    # Severity levels
    critical_multiplier: float = Field(3.0, description="Multiplier for critical anomaly threshold")
    high_multiplier: float = Field(2.0, description="Multiplier for high severity threshold")
    
    class Config:
        arbitrary_types_allowed = True


class AnomalyResult(BaseModel):
    """Anomaly detection result."""
    
    entity_id: str = Field(..., description="Entity ID")
    entity_name: str = Field(..., description="Entity name")
    metric_name: str = Field(..., description="Affected metric")
    
    # Anomaly details
    anomaly_type: str = Field(..., description="Type of anomaly (spike, drop, trend, pattern)")
    severity: str = Field(..., description="Severity level (critical, high, medium, low)")
    confidence: float = Field(..., description="Detection confidence (0-1)")
    
    # Values
    current_value: float = Field(..., description="Current anomalous value")
    expected_value: float = Field(..., description="Expected normal value")
    deviation: float = Field(..., description="Deviation from expected (%)")
    z_score: float = Field(..., description="Z-score of the anomaly")
    
    # Context
    detection_date: datetime = Field(..., description="Date when anomaly was detected")
    affected_period: Tuple[datetime, datetime] = Field(..., description="Period affected by anomaly")
    
    # Impact assessment
    impact_score: float = Field(..., description="Impact score (0-100)")
    estimated_impact: str = Field(..., description="Estimated business impact")
    
    # Recommendations
    immediate_actions: List[str] = Field(default_factory=list, description="Immediate actions to take")
    investigation_areas: List[str] = Field(default_factory=list, description="Areas to investigate")
    
    # Additional context
    context_data: Dict[str, Any] = Field(default_factory=dict, description="Additional context data")
    detected_at: datetime = Field(default_factory=datetime.utcnow, description="Detection timestamp")


class AnomalyDetector:
    """Advanced anomaly detection system for marketing performance data."""
    
    def __init__(self, config: Optional[AnomalyConfig] = None):
        """
        Initialize anomaly detector.
        
        Args:
            config: Anomaly detection configuration
        """
        self.config = config or AnomalyConfig()
    
    def detect_campaign_anomalies(
        self, 
        campaign: Campaign, 
        historical_metrics: List[DailyMetrics],
        reference_campaigns: Optional[List[Campaign]] = None
    ) -> List[AnomalyResult]:
        """
        Detect anomalies in campaign performance.
        
        Args:
            campaign: Campaign to analyze
            historical_metrics: Historical performance data
            reference_campaigns: Reference campaigns for benchmarking
            
        Returns:
            List of detected anomalies
        """
        try:
            anomalies = []
            
            if len(historical_metrics) < 7:  # Need minimum data
                logger.warning(f"Insufficient data for anomaly detection: {len(historical_metrics)} points")
                return anomalies
            
            # Sort metrics by date
            sorted_metrics = sorted(historical_metrics, key=lambda x: x.date)
            
            # Detect anomalies for key metrics
            metrics_to_check = [
                ("impressions", "Volume"),
                ("clicks", "Volume"),
                ("conversions", "Conversions"),
                ("cost", "Spend"),
                ("ctr", "Efficiency"),
                ("conversion_rate", "Efficiency")
            ]
            
            for metric_name, category in metrics_to_check:
                try:
                    metric_anomalies = self._detect_metric_anomalies(
                        campaign=campaign,
                        metrics=sorted_metrics,
                        metric_name=metric_name,
                        category=category,
                        reference_campaigns=reference_campaigns
                    )
                    anomalies.extend(metric_anomalies)
                    
                except Exception as e:
                    logger.error(f"Failed to detect {metric_name} anomalies: {e}")
                    continue
            
            # Sort by severity and confidence
            anomalies.sort(key=lambda x: (self._severity_score(x.severity), x.confidence), reverse=True)
            
            return anomalies
            
        except Exception as e:
            logger.error(f"Campaign anomaly detection failed: {e}")
            return []
    
    def _detect_metric_anomalies(
        self,
        campaign: Campaign,
        metrics: List[DailyMetrics],
        metric_name: str,
        category: str,
        reference_campaigns: Optional[List[Campaign]] = None
    ) -> List[AnomalyResult]:
        """Detect anomalies for a specific metric."""
        anomalies = []
        
        try:
            # Extract metric values and dates
            values = []
            dates = []
            
            for metric in metrics:
                value = getattr(metric, metric_name, None)
                if value is not None and value >= 0:  # Include zero but not negative
                    values.append(float(value))
                    dates.append(metric.date)
            
            if len(values) < 7:
                return anomalies
            
            values_array = np.array(values)
            
            # Statistical anomaly detection
            anomalies.extend(self._detect_statistical_anomalies(
                campaign, metric_name, category, dates, values_array
            ))
            
            # Trend anomaly detection
            anomalies.extend(self._detect_trend_anomalies(
                campaign, metric_name, category, dates, values_array
            ))
            
            # Pattern anomaly detection
            anomalies.extend(self._detect_pattern_anomalies(
                campaign, metric_name, category, dates, values_array
            ))
            
            # Sudden change detection
            anomalies.extend(self._detect_sudden_changes(
                campaign, metric_name, category, dates, values_array
            ))
            
            return anomalies
            
        except Exception as e:
            logger.error(f"Metric anomaly detection failed for {metric_name}: {e}")
            return []
    
    def _detect_statistical_anomalies(
        self,
        campaign: Campaign,
        metric_name: str,
        category: str,
        dates: List[datetime],
        values: np.ndarray
    ) -> List[AnomalyResult]:
        """Detect statistical outliers using Z-score and IQR methods."""
        anomalies = []
        
        try:
            if len(values) < 7:
                return anomalies
            
            # Z-score method
            z_scores = np.abs(stats.zscore(values))
            z_outliers = np.where(z_scores > self.config.z_score_threshold)[0]
            
            # IQR method
            Q1 = np.percentile(values, 25)
            Q3 = np.percentile(values, 75)
            IQR = Q3 - Q1
            lower_bound = Q1 - (self.config.iqr_multiplier * IQR)
            upper_bound = Q3 + (self.config.iqr_multiplier * IQR)
            iqr_outliers = np.where((values < lower_bound) | (values > upper_bound))[0]
            
            # Combine outliers
            all_outliers = np.unique(np.concatenate([z_outliers, iqr_outliers]))
            
            for idx in all_outliers:
                if idx >= len(values) or idx >= len(dates):
                    continue
                
                current_value = values[idx]
                expected_value = np.median(values)  # Use median as expected
                z_score = z_scores[idx]
                
                # Calculate deviation percentage
                if expected_value != 0:
                    deviation = ((current_value - expected_value) / expected_value) * 100
                else:
                    deviation = 0.0
                
                # Determine anomaly type
                anomaly_type = "spike" if current_value > expected_value else "drop"
                
                # Determine severity
                severity = self._determine_severity(z_score, abs(deviation))
                
                # Calculate confidence
                confidence = min(1.0, z_score / 5.0)  # Scale z-score to confidence
                
                # Skip low-confidence anomalies for noisy metrics
                if confidence < 0.6:
                    continue
                
                # Create anomaly result
                anomaly = AnomalyResult(
                    entity_id=campaign.id,
                    entity_name=campaign.name,
                    metric_name=metric_name,
                    anomaly_type=anomaly_type,
                    severity=severity,
                    confidence=confidence,
                    current_value=current_value,
                    expected_value=expected_value,
                    deviation=deviation,
                    z_score=z_score,
                    detection_date=dates[idx],
                    affected_period=(dates[idx], dates[idx]),
                    impact_score=self._calculate_impact_score(metric_name, deviation, category),
                    estimated_impact=self._estimate_impact(category, deviation),
                    immediate_actions=self._generate_immediate_actions(anomaly_type, metric_name, category),
                    investigation_areas=self._generate_investigation_areas(anomaly_type, metric_name),
                    context_data={"method": "statistical", "z_score": float(z_score)}
                )
                
                anomalies.append(anomaly)
            
            return anomalies
            
        except Exception as e:
            logger.error(f"Statistical anomaly detection failed: {e}")
            return []
    
    def _detect_trend_anomalies(
        self,
        campaign: Campaign,
        metric_name: str,
        category: str,
        dates: List[datetime],
        values: np.ndarray
    ) -> List[AnomalyResult]:
        """Detect trend-based anomalies."""
        anomalies = []
        
        try:
            if len(values) < 10:  # Need enough data for trend analysis
                return anomalies
            
            # Calculate moving averages
            window_size = min(7, len(values) // 3)
            moving_avg = np.convolve(values, np.ones(window_size)/window_size, mode='valid')
            
            # Extend moving average to match values length
            moving_avg = np.pad(moving_avg, (window_size//2, len(values) - len(moving_avg) - window_size//2), 
                              mode='edge')
            
            # Calculate trend deviations
            deviations = np.abs(values - moving_avg)
            mean_deviation = np.mean(deviations)
            std_deviation = np.std(deviations)
            
            # Find trend anomalies
            threshold = mean_deviation + (2.0 * std_deviation)
            trend_outliers = np.where(deviations > threshold)[0]
            
            for idx in trend_outliers:
                if idx >= len(values) or idx >= len(dates):
                    continue
                
                current_value = values[idx]
                expected_value = moving_avg[idx]
                deviation_pct = ((current_value - expected_value) / expected_value * 100) if expected_value != 0 else 0.0
                
                # Skip minor deviations
                if abs(deviation_pct) < 20:
                    continue
                
                anomaly_type = "trend_spike" if current_value > expected_value else "trend_drop"
                severity = self._determine_severity(deviations[idx] / std_deviation, abs(deviation_pct))
                confidence = min(1.0, deviations[idx] / (mean_deviation + std_deviation))
                
                anomaly = AnomalyResult(
                    entity_id=campaign.id,
                    entity_name=campaign.name,
                    metric_name=metric_name,
                    anomaly_type=anomaly_type,
                    severity=severity,
                    confidence=confidence,
                    current_value=current_value,
                    expected_value=expected_value,
                    deviation=deviation_pct,
                    z_score=deviations[idx] / std_deviation if std_deviation > 0 else 0.0,
                    detection_date=dates[idx],
                    affected_period=(dates[idx], dates[idx]),
                    impact_score=self._calculate_impact_score(metric_name, deviation_pct, category),
                    estimated_impact=self._estimate_impact(category, deviation_pct),
                    immediate_actions=self._generate_immediate_actions(anomaly_type, metric_name, category),
                    investigation_areas=self._generate_investigation_areas(anomaly_type, metric_name),
                    context_data={"method": "trend", "moving_average": float(expected_value)}
                )
                
                anomalies.append(anomaly)
            
            return anomalies
            
        except Exception as e:
            logger.error(f"Trend anomaly detection failed: {e}")
            return []
    
    def _detect_pattern_anomalies(
        self,
        campaign: Campaign,
        metric_name: str,
        category: str,
        dates: List[datetime],
        values: np.ndarray
    ) -> List[AnomalyResult]:
        """Detect pattern-based anomalies (seasonal, cyclical)."""
        anomalies = []
        
        try:
            if len(values) < 14:  # Need enough data for pattern detection
                return anomalies
            
            # Day-of-week pattern analysis
            dow_patterns = {}
            for i, date in enumerate(dates):
                dow = date.weekday()  # 0=Monday, 6=Sunday
                if dow not in dow_patterns:
                    dow_patterns[dow] = []
                if i < len(values):
                    dow_patterns[dow].append(values[i])
            
            # Detect day-of-week anomalies
            for i, date in enumerate(dates[-7:], start=len(dates)-7):  # Check last week
                if i >= len(values):
                    continue
                    
                dow = date.weekday()
                current_value = values[i]
                
                if dow in dow_patterns and len(dow_patterns[dow]) > 1:
                    expected_value = np.median(dow_patterns[dow][:-1])  # Exclude current value
                    dow_std = np.std(dow_patterns[dow][:-1]) if len(dow_patterns[dow]) > 2 else 0
                    
                    if dow_std > 0 and expected_value > 0:
                        deviation = abs(current_value - expected_value)
                        z_score = deviation / dow_std
                        deviation_pct = (deviation / expected_value) * 100
                        
                        if z_score > 2.0 and deviation_pct > 25:
                            anomaly_type = "pattern_break"
                            severity = self._determine_severity(z_score, deviation_pct)
                            confidence = min(1.0, z_score / 3.0)
                            
                            anomaly = AnomalyResult(
                                entity_id=campaign.id,
                                entity_name=campaign.name,
                                metric_name=metric_name,
                                anomaly_type=anomaly_type,
                                severity=severity,
                                confidence=confidence,
                                current_value=current_value,
                                expected_value=expected_value,
                                deviation=deviation_pct if current_value > expected_value else -deviation_pct,
                                z_score=z_score,
                                detection_date=date,
                                affected_period=(date, date),
                                impact_score=self._calculate_impact_score(metric_name, deviation_pct, category),
                                estimated_impact=self._estimate_impact(category, deviation_pct),
                                immediate_actions=self._generate_immediate_actions(anomaly_type, metric_name, category),
                                investigation_areas=self._generate_investigation_areas(anomaly_type, metric_name),
                                context_data={
                                    "method": "pattern", 
                                    "pattern_type": "day_of_week",
                                    "day_of_week": dow
                                }
                            )
                            
                            anomalies.append(anomaly)
            
            return anomalies
            
        except Exception as e:
            logger.error(f"Pattern anomaly detection failed: {e}")
            return []
    
    def _detect_sudden_changes(
        self,
        campaign: Campaign,
        metric_name: str,
        category: str,
        dates: List[datetime],
        values: np.ndarray
    ) -> List[AnomalyResult]:
        """Detect sudden changes in metric values."""
        anomalies = []
        
        try:
            if len(values) < 5:
                return anomalies
            
            # Calculate day-over-day changes
            changes = np.diff(values)
            pct_changes = []
            
            for i in range(len(changes)):
                if i < len(values) - 1 and values[i] != 0:
                    pct_change = (changes[i] / values[i]) * 100
                    pct_changes.append(pct_change)
                else:
                    pct_changes.append(0.0)
            
            pct_changes = np.array(pct_changes)
            
            # Find sudden changes
            threshold = self.config.sudden_change_threshold * 100  # Convert to percentage
            sudden_changes = np.where(np.abs(pct_changes) > threshold)[0]
            
            for idx in sudden_changes:
                if idx + 1 >= len(values) or idx + 1 >= len(dates):
                    continue
                
                current_value = values[idx + 1]
                previous_value = values[idx]
                change_pct = pct_changes[idx]
                
                # Skip if values are too small to be meaningful
                if category == "Spend" and current_value < self.config.min_spend_anomaly:
                    continue
                elif category == "Volume" and current_value < self.config.min_volume_anomaly:
                    continue
                
                anomaly_type = "sudden_increase" if change_pct > 0 else "sudden_decrease"
                severity = self._determine_severity(abs(change_pct) / 50, abs(change_pct))  # Scale by 50%
                confidence = min(1.0, abs(change_pct) / 100)  # Scale to 0-1
                
                # Determine affected period (might extend beyond single day)
                period_start = dates[idx + 1]
                period_end = dates[idx + 1]
                
                # Check if change persists
                if idx + 2 < len(values):
                    next_change = pct_changes[idx + 1] if idx + 1 < len(pct_changes) else 0
                    if abs(next_change) > threshold / 2:  # Continues changing
                        period_end = dates[min(idx + 2, len(dates) - 1)]
                
                anomaly = AnomalyResult(
                    entity_id=campaign.id,
                    entity_name=campaign.name,
                    metric_name=metric_name,
                    anomaly_type=anomaly_type,
                    severity=severity,
                    confidence=confidence,
                    current_value=current_value,
                    expected_value=previous_value,  # Use previous value as expected
                    deviation=change_pct,
                    z_score=abs(change_pct) / 25,  # Approximate z-score
                    detection_date=dates[idx + 1],
                    affected_period=(period_start, period_end),
                    impact_score=self._calculate_impact_score(metric_name, abs(change_pct), category),
                    estimated_impact=self._estimate_impact(category, abs(change_pct)),
                    immediate_actions=self._generate_immediate_actions(anomaly_type, metric_name, category),
                    investigation_areas=self._generate_investigation_areas(anomaly_type, metric_name),
                    context_data={"method": "sudden_change", "change_percentage": float(change_pct)}
                )
                
                anomalies.append(anomaly)
            
            return anomalies
            
        except Exception as e:
            logger.error(f"Sudden change detection failed: {e}")
            return []
    
    def _determine_severity(self, statistical_score: float, deviation_pct: float) -> str:
        """Determine anomaly severity based on statistical measures and business impact."""
        if (statistical_score > self.config.critical_multiplier or 
            abs(deviation_pct) > 75):
            return "critical"
        elif (statistical_score > self.config.high_multiplier or 
              abs(deviation_pct) > 50):
            return "high"
        elif (statistical_score > 1.5 or 
              abs(deviation_pct) > 25):
            return "medium"
        else:
            return "low"
    
    def _severity_score(self, severity: str) -> int:
        """Convert severity to numeric score for sorting."""
        severity_map = {"critical": 4, "high": 3, "medium": 2, "low": 1}
        return severity_map.get(severity, 0)
    
    def _calculate_impact_score(self, metric_name: str, deviation_pct: float, category: str) -> float:
        """Calculate business impact score (0-100)."""
        base_score = min(100, abs(deviation_pct))
        
        # Weight by metric importance
        metric_weights = {
            "cost": 1.0,
            "conversions": 0.9,
            "clicks": 0.7,
            "conversion_rate": 0.8,
            "ctr": 0.6,
            "impressions": 0.5
        }
        
        weight = metric_weights.get(metric_name, 0.5)
        return min(100, base_score * weight)
    
    def _estimate_impact(self, category: str, deviation_pct: float) -> str:
        """Estimate business impact description."""
        abs_dev = abs(deviation_pct)
        
        if abs_dev > 75:
            if category == "Spend":
                return "Severe budget impact - immediate attention required"
            elif category == "Conversions":
                return "Critical conversion impact - revenue at risk"
            else:
                return "Major performance impact - urgent investigation needed"
        elif abs_dev > 50:
            return "Significant impact - requires prompt attention"
        elif abs_dev > 25:
            return "Moderate impact - monitor closely"
        else:
            return "Minor impact - routine monitoring"
    
    def _generate_immediate_actions(self, anomaly_type: str, metric_name: str, category: str) -> List[str]:
        """Generate immediate action recommendations."""
        actions = []
        
        if "spike" in anomaly_type:
            if metric_name == "cost":
                actions.extend([
                    "Check for bid increases or budget changes",
                    "Review recent campaign modifications",
                    "Consider pausing high-spend keywords if unexpected"
                ])
            elif metric_name in ["clicks", "impressions"]:
                actions.extend([
                    "Verify if this aligns with campaign objectives",
                    "Check for new keyword additions or bid changes",
                    "Monitor for quality and conversion impact"
                ])
            elif metric_name == "conversions":
                actions.extend([
                    "Celebrate and analyze what's working",
                    "Scale successful elements if sustainable",
                    "Document changes for replication"
                ])
        
        elif "drop" in anomaly_type:
            if metric_name == "conversions":
                actions.extend([
                    "Check landing page functionality",
                    "Verify tracking implementation",
                    "Review recent campaign changes"
                ])
            elif metric_name in ["clicks", "impressions"]:
                actions.extend([
                    "Check for disapproved ads or keywords",
                    "Review bid competitiveness",
                    "Verify budget availability"
                ])
        
        elif "sudden" in anomaly_type:
            actions.extend([
                "Investigate cause of sudden change",
                "Check for external factors (seasonality, competitors)",
                "Review recent account modifications"
            ])
        
        return actions[:3]  # Limit to top 3 actions
    
    def _generate_investigation_areas(self, anomaly_type: str, metric_name: str) -> List[str]:
        """Generate investigation area recommendations."""
        areas = [
            "Recent campaign modifications",
            "External market conditions",
            "Competitive landscape changes",
            "Seasonal or temporal factors",
            "Technical tracking issues",
            "Landing page performance",
            "Ad quality and relevance",
            "Keyword performance shifts"
        ]
        
        # Prioritize based on anomaly type and metric
        if metric_name == "conversions":
            priority_areas = [
                "Technical tracking issues",
                "Landing page performance", 
                "Recent campaign modifications"
            ]
        elif metric_name == "cost":
            priority_areas = [
                "Recent campaign modifications",
                "Competitive landscape changes",
                "Keyword performance shifts"
            ]
        else:
            priority_areas = [
                "Recent campaign modifications",
                "External market conditions",
                "Ad quality and relevance"
            ]
        
        # Return prioritized areas
        return priority_areas[:3]
    
    def get_anomaly_summary(self, anomalies: List[AnomalyResult]) -> Dict[str, Any]:
        """Generate summary of detected anomalies."""
        try:
            if not anomalies:
                return {"total_anomalies": 0, "message": "No anomalies detected"}
            
            summary = {
                "total_anomalies": len(anomalies),
                "by_severity": {},
                "by_type": {},
                "by_metric": {},
                "critical_count": 0,
                "high_impact_count": 0,
                "average_confidence": 0.0,
                "most_affected_entity": "",
                "top_concerns": []
            }
            
            # Aggregate statistics
            confidences = []
            entity_counts = {}
            
            for anomaly in anomalies:
                # Severity distribution
                severity = anomaly.severity
                summary["by_severity"][severity] = summary["by_severity"].get(severity, 0) + 1
                
                # Type distribution
                atype = anomaly.anomaly_type
                summary["by_type"][atype] = summary["by_type"].get(atype, 0) + 1
                
                # Metric distribution
                metric = anomaly.metric_name
                summary["by_metric"][metric] = summary["by_metric"].get(metric, 0) + 1
                
                # Counts
                if severity == "critical":
                    summary["critical_count"] += 1
                if anomaly.impact_score > 70:
                    summary["high_impact_count"] += 1
                
                # Confidence tracking
                confidences.append(anomaly.confidence)
                
                # Entity tracking
                entity_counts[anomaly.entity_name] = entity_counts.get(anomaly.entity_name, 0) + 1
            
            # Calculate averages and most affected
            if confidences:
                summary["average_confidence"] = np.mean(confidences)
            
            if entity_counts:
                summary["most_affected_entity"] = max(entity_counts.items(), key=lambda x: x[1])[0]
            
            # Generate top concerns
            critical_anomalies = [a for a in anomalies if a.severity == "critical"]
            high_impact_anomalies = [a for a in anomalies if a.impact_score > 70]
            
            for anomaly in critical_anomalies[:3]:
                summary["top_concerns"].append(f"Critical {anomaly.anomaly_type} in {anomaly.metric_name}")
            
            for anomaly in high_impact_anomalies[:2]:
                if len(summary["top_concerns"]) < 5:
                    summary["top_concerns"].append(f"High impact {anomaly.anomaly_type} in {anomaly.metric_name}")
            
            return summary
            
        except Exception as e:
            logger.error(f"Anomaly summary generation failed: {e}")
            return {"error": str(e)}