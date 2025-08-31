"""Competitive benchmarking system for marketing performance analysis."""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from statistics import mean, median

import numpy as np
from pydantic import BaseModel, Field

from ..models.campaign import Campaign, AdvertisingChannelType
from ..models.metrics import DailyMetrics

logger = logging.getLogger(__name__)


class BenchmarkConfig(BaseModel):
    """Configuration for benchmarking analysis."""
    
    # Benchmark categories
    peer_group_size: int = Field(10, description="Target size for peer groups")
    min_peer_campaigns: int = Field(3, description="Minimum campaigns for valid benchmarking")
    
    # Industry benchmarks (default values - should be updated with real data)
    industry_benchmarks: Dict[str, Dict[str, float]] = Field(
        default_factory=lambda: {
            "search": {
                "ctr": 3.17,
                "conversion_rate": 3.75,
                "cpc": 2.69,
                "cpa": 48.96
            },
            "display": {
                "ctr": 0.46,
                "conversion_rate": 0.89,
                "cpc": 0.63,
                "cpa": 75.51
            },
            "shopping": {
                "ctr": 0.66,
                "conversion_rate": 1.91,
                "cpc": 0.66,
                "cpa": 38.87
            },
            "youtube": {
                "ctr": 0.84,
                "conversion_rate": 0.61,
                "cpc": 3.21,
                "cpa": 72.25
            }
        }
    )
    
    # Percentile thresholds for performance classification
    top_performer_threshold: float = Field(0.90, description="Top 10% threshold")
    good_performer_threshold: float = Field(0.75, description="Top 25% threshold")
    average_performer_threshold: float = Field(0.50, description="Median threshold")
    
    # Statistical parameters
    confidence_level: float = Field(0.95, description="Confidence level for benchmarks")
    outlier_threshold: float = Field(2.0, description="Z-score threshold for outlier removal")
    
    class Config:
        arbitrary_types_allowed = True


class BenchmarkResult(BaseModel):
    """Benchmarking analysis result."""
    
    entity_id: str = Field(..., description="Entity ID")
    entity_name: str = Field(..., description="Entity name")
    entity_type: str = Field(..., description="Entity type")
    
    # Performance classification
    overall_rank: str = Field(..., description="Overall performance ranking")
    percentile_rank: float = Field(..., description="Percentile ranking (0-100)")
    
    # Metric benchmarks
    metric_benchmarks: Dict[str, Dict[str, float]] = Field(
        default_factory=dict,
        description="Benchmarks by metric"
    )
    
    # Peer comparison
    peer_group_size: int = Field(..., description="Size of comparison peer group")
    peer_comparison: Dict[str, str] = Field(
        default_factory=dict,
        description="Performance vs peers by metric"
    )
    
    # Industry comparison
    industry_comparison: Dict[str, Dict[str, Union[str, float]]] = Field(
        default_factory=dict,
        description="Performance vs industry benchmarks"
    )
    
    # Competitive gaps
    improvement_opportunities: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Areas for improvement with benchmarks"
    )
    
    # Strengths
    competitive_advantages: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Areas where performance exceeds benchmarks"
    )
    
    # Benchmark metadata
    benchmark_period: Tuple[datetime, datetime] = Field(..., description="Benchmarking period")
    calculated_at: datetime = Field(default_factory=datetime.utcnow, description="Analysis timestamp")


class BenchmarkingEngine:
    """Advanced benchmarking engine for competitive performance analysis."""
    
    def __init__(self, config: Optional[BenchmarkConfig] = None):
        """
        Initialize benchmarking engine.
        
        Args:
            config: Benchmarking configuration
        """
        self.config = config or BenchmarkConfig()
    
    def benchmark_campaigns(
        self,
        target_campaigns: List[Campaign],
        comparison_campaigns: List[Campaign],
        historical_metrics: Optional[List[DailyMetrics]] = None
    ) -> List[BenchmarkResult]:
        """
        Benchmark campaign performance against peer campaigns.
        
        Args:
            target_campaigns: Campaigns to benchmark
            comparison_campaigns: Peer campaigns for comparison
            historical_metrics: Historical performance data
            
        Returns:
            List of benchmark results
        """
        try:
            benchmark_results = []
            
            if len(comparison_campaigns) < self.config.min_peer_campaigns:
                logger.warning(f"Insufficient peer campaigns for benchmarking: {len(comparison_campaigns)}")
                return benchmark_results
            
            # Group campaigns by channel type for fair comparison
            channel_groups = self._group_campaigns_by_channel(comparison_campaigns)
            
            for campaign in target_campaigns:
                try:
                    channel_type = self._get_channel_key(campaign.advertising_channel_type)
                    peer_campaigns = channel_groups.get(channel_type, comparison_campaigns)
                    
                    if len(peer_campaigns) < self.config.min_peer_campaigns:
                        peer_campaigns = comparison_campaigns  # Fall back to all campaigns
                    
                    benchmark = self._benchmark_single_campaign(
                        campaign=campaign,
                        peer_campaigns=peer_campaigns,
                        historical_metrics=historical_metrics
                    )
                    
                    benchmark_results.append(benchmark)
                    
                except Exception as e:
                    logger.error(f"Failed to benchmark campaign {campaign.id}: {e}")
                    continue
            
            # Sort by percentile rank (highest first)
            benchmark_results.sort(key=lambda x: x.percentile_rank, reverse=True)
            
            return benchmark_results
            
        except Exception as e:
            logger.error(f"Campaign benchmarking failed: {e}")
            return []
    
    def _benchmark_single_campaign(
        self,
        campaign: Campaign,
        peer_campaigns: List[Campaign],
        historical_metrics: Optional[List[DailyMetrics]] = None
    ) -> BenchmarkResult:
        """Benchmark a single campaign against peers."""
        
        try:
            # Calculate peer statistics
            peer_stats = self._calculate_peer_statistics(peer_campaigns)
            
            # Calculate campaign metrics
            campaign_metrics = self._extract_campaign_metrics(campaign)
            
            # Benchmark each metric
            metric_benchmarks = {}
            peer_comparison = {}
            overall_scores = []
            
            for metric_name, campaign_value in campaign_metrics.items():
                if metric_name in peer_stats and campaign_value is not None:
                    
                    peer_values = peer_stats[metric_name]
                    
                    # Calculate percentile rank
                    percentile = self._calculate_percentile(campaign_value, peer_values, metric_name)
                    overall_scores.append(percentile)
                    
                    # Determine performance level
                    performance_level = self._classify_performance(percentile)
                    
                    metric_benchmarks[metric_name] = {
                        "value": campaign_value,
                        "peer_median": np.median(peer_values),
                        "peer_p25": np.percentile(peer_values, 25),
                        "peer_p75": np.percentile(peer_values, 75),
                        "peer_p90": np.percentile(peer_values, 90),
                        "percentile_rank": percentile
                    }
                    
                    peer_comparison[metric_name] = performance_level
            
            # Calculate overall performance
            overall_percentile = np.mean(overall_scores) if overall_scores else 0.0
            overall_rank = self._classify_performance(overall_percentile)
            
            # Industry comparison
            channel_type = self._get_channel_key(campaign.advertising_channel_type)
            industry_comparison = self._compare_to_industry(campaign_metrics, channel_type)
            
            # Identify opportunities and advantages
            improvement_opportunities = self._identify_improvement_opportunities(
                metric_benchmarks, industry_comparison, channel_type
            )
            competitive_advantages = self._identify_competitive_advantages(
                metric_benchmarks, industry_comparison
            )
            
            # Determine benchmark period
            benchmark_period = self._get_benchmark_period(historical_metrics)
            
            return BenchmarkResult(
                entity_id=campaign.id,
                entity_name=campaign.name,
                entity_type="campaign",
                overall_rank=overall_rank,
                percentile_rank=round(overall_percentile, 2),
                metric_benchmarks=metric_benchmarks,
                peer_group_size=len(peer_campaigns),
                peer_comparison=peer_comparison,
                industry_comparison=industry_comparison,
                improvement_opportunities=improvement_opportunities,
                competitive_advantages=competitive_advantages,
                benchmark_period=benchmark_period
            )
            
        except Exception as e:
            logger.error(f"Single campaign benchmarking failed: {e}")
            # Return minimal benchmark result
            return BenchmarkResult(
                entity_id=campaign.id,
                entity_name=campaign.name,
                entity_type="campaign",
                overall_rank="unknown",
                percentile_rank=0.0,
                peer_group_size=0,
                benchmark_period=(datetime.now() - timedelta(days=30), datetime.now())
            )
    
    def _group_campaigns_by_channel(self, campaigns: List[Campaign]) -> Dict[str, List[Campaign]]:
        """Group campaigns by advertising channel type."""
        groups = {}
        
        for campaign in campaigns:
            channel_key = self._get_channel_key(campaign.advertising_channel_type)
            if channel_key not in groups:
                groups[channel_key] = []
            groups[channel_key].append(campaign)
        
        return groups
    
    def _get_channel_key(self, channel_type: AdvertisingChannelType) -> str:
        """Convert channel type to benchmark key."""
        channel_map = {
            AdvertisingChannelType.SEARCH: "search",
            AdvertisingChannelType.DISPLAY: "display", 
            AdvertisingChannelType.SHOPPING: "shopping",
            AdvertisingChannelType.VIDEO: "youtube",
            AdvertisingChannelType.UNKNOWN: "search"  # Default fallback
        }
        return channel_map.get(channel_type, "search")
    
    def _calculate_peer_statistics(self, peer_campaigns: List[Campaign]) -> Dict[str, List[float]]:
        """Calculate statistics from peer campaigns."""
        peer_stats = {
            "impressions": [],
            "clicks": [],
            "conversions": [],
            "cost": [],
            "ctr": [],
            "conversion_rate": [],
            "cpc": [],
            "cpa": []
        }
        
        for campaign in peer_campaigns:
            # Basic metrics
            if campaign.impressions is not None and campaign.impressions > 0:
                peer_stats["impressions"].append(float(campaign.impressions))
            
            if campaign.clicks is not None and campaign.clicks > 0:
                peer_stats["clicks"].append(float(campaign.clicks))
            
            if campaign.conversions is not None and campaign.conversions > 0:
                peer_stats["conversions"].append(float(campaign.conversions))
            
            if campaign.cost is not None:
                peer_stats["cost"].append(float(campaign.cost))
            
            # Calculated metrics
            if campaign.ctr is not None:
                peer_stats["ctr"].append(campaign.ctr)
            
            if campaign.conversion_rate is not None:
                peer_stats["conversion_rate"].append(campaign.conversion_rate)
            
            # CPC calculation
            if campaign.clicks and campaign.clicks > 0 and campaign.cost:
                cpc = float(campaign.cost) / campaign.clicks
                peer_stats["cpc"].append(cpc)
            
            # CPA calculation
            if campaign.conversions and campaign.conversions > 0 and campaign.cost:
                cpa = float(campaign.cost) / campaign.conversions
                peer_stats["cpa"].append(cpa)
        
        # Remove outliers
        for metric in peer_stats:
            if len(peer_stats[metric]) > 5:  # Only remove outliers if sufficient data
                peer_stats[metric] = self._remove_outliers(peer_stats[metric])
        
        return peer_stats
    
    def _extract_campaign_metrics(self, campaign: Campaign) -> Dict[str, Optional[float]]:
        """Extract comparable metrics from campaign."""
        metrics = {
            "impressions": float(campaign.impressions) if campaign.impressions else None,
            "clicks": float(campaign.clicks) if campaign.clicks else None,
            "conversions": float(campaign.conversions) if campaign.conversions else None,
            "cost": float(campaign.cost) if campaign.cost else None,
            "ctr": campaign.ctr,
            "conversion_rate": campaign.conversion_rate
        }
        
        # Calculate derived metrics
        if campaign.clicks and campaign.clicks > 0 and campaign.cost:
            metrics["cpc"] = float(campaign.cost) / campaign.clicks
        else:
            metrics["cpc"] = None
        
        if campaign.conversions and campaign.conversions > 0 and campaign.cost:
            metrics["cpa"] = float(campaign.cost) / campaign.conversions
        else:
            metrics["cpa"] = None
        
        return metrics
    
    def _remove_outliers(self, values: List[float]) -> List[float]:
        """Remove statistical outliers from values."""
        try:
            if len(values) < 5:
                return values
            
            values_array = np.array(values)
            z_scores = np.abs((values_array - np.mean(values_array)) / np.std(values_array))
            
            # Keep values within threshold
            filtered_values = values_array[z_scores < self.config.outlier_threshold]
            
            return filtered_values.tolist()
            
        except Exception:
            return values
    
    def _calculate_percentile(self, value: float, peer_values: List[float], metric_name: str) -> float:
        """Calculate percentile rank for a value against peers."""
        if not peer_values or value is None:
            return 0.0
        
        try:
            # For "lower is better" metrics, invert the ranking
            lower_is_better = metric_name in ["cpc", "cpa", "cost"]
            
            if lower_is_better:
                # Count values greater than current value
                better_count = sum(1 for v in peer_values if v > value)
            else:
                # Count values less than current value  
                better_count = sum(1 for v in peer_values if v < value)
            
            percentile = (better_count / len(peer_values)) * 100
            return min(100.0, max(0.0, percentile))
            
        except Exception:
            return 0.0
    
    def _classify_performance(self, percentile: float) -> str:
        """Classify performance based on percentile."""
        if percentile >= self.config.top_performer_threshold * 100:
            return "top_performer"
        elif percentile >= self.config.good_performer_threshold * 100:
            return "good_performer"
        elif percentile >= self.config.average_performer_threshold * 100:
            return "average_performer"
        else:
            return "below_average"
    
    def _compare_to_industry(self, campaign_metrics: Dict[str, Optional[float]], channel_type: str) -> Dict[str, Dict[str, Union[str, float]]]:
        """Compare campaign metrics to industry benchmarks."""
        industry_comparison = {}
        
        if channel_type not in self.config.industry_benchmarks:
            return industry_comparison
        
        industry_benchmarks = self.config.industry_benchmarks[channel_type]
        
        for metric_name, campaign_value in campaign_metrics.items():
            if metric_name in industry_benchmarks and campaign_value is not None:
                industry_value = industry_benchmarks[metric_name]
                
                # Calculate performance vs industry
                if industry_value > 0:
                    difference_pct = ((campaign_value - industry_value) / industry_value) * 100
                else:
                    difference_pct = 0.0
                
                # Determine performance level
                lower_is_better = metric_name in ["cpc", "cpa", "cost"]
                
                if lower_is_better:
                    if difference_pct <= -20:
                        performance = "excellent"
                    elif difference_pct <= -10:
                        performance = "good"
                    elif difference_pct <= 10:
                        performance = "average"
                    else:
                        performance = "poor"
                else:
                    if difference_pct >= 20:
                        performance = "excellent"
                    elif difference_pct >= 10:
                        performance = "good"
                    elif difference_pct >= -10:
                        performance = "average"
                    else:
                        performance = "poor"
                
                industry_comparison[metric_name] = {
                    "campaign_value": campaign_value,
                    "industry_benchmark": industry_value,
                    "difference_percent": difference_pct,
                    "performance": performance
                }
        
        return industry_comparison
    
    def _identify_improvement_opportunities(
        self,
        metric_benchmarks: Dict[str, Dict[str, float]],
        industry_comparison: Dict[str, Dict[str, Union[str, float]]],
        channel_type: str
    ) -> List[Dict[str, Any]]:
        """Identify areas for improvement based on benchmarks."""
        opportunities = []
        
        # Peer-based opportunities
        for metric_name, benchmark_data in metric_benchmarks.items():
            if benchmark_data["percentile_rank"] < 50:  # Below median
                gap = benchmark_data["peer_median"] - benchmark_data["value"]
                
                # For "lower is better" metrics
                if metric_name in ["cpc", "cpa"]:
                    if benchmark_data["value"] > benchmark_data["peer_p75"]:  # Worse than 75th percentile
                        opportunities.append({
                            "metric": metric_name,
                            "type": "peer_gap",
                            "current_value": benchmark_data["value"],
                            "target_value": benchmark_data["peer_p75"],
                            "improvement_needed": benchmark_data["value"] - benchmark_data["peer_p75"],
                            "priority": "high" if benchmark_data["percentile_rank"] < 25 else "medium"
                        })
                else:
                    if benchmark_data["value"] < benchmark_data["peer_p25"]:  # Worse than 25th percentile
                        opportunities.append({
                            "metric": metric_name,
                            "type": "peer_gap",
                            "current_value": benchmark_data["value"],
                            "target_value": benchmark_data["peer_p75"],
                            "improvement_needed": benchmark_data["peer_p75"] - benchmark_data["value"],
                            "priority": "high" if benchmark_data["percentile_rank"] < 25 else "medium"
                        })
        
        # Industry-based opportunities
        for metric_name, industry_data in industry_comparison.items():
            if isinstance(industry_data, dict) and industry_data.get("performance") in ["poor", "average"]:
                opportunities.append({
                    "metric": metric_name,
                    "type": "industry_gap",
                    "current_value": industry_data["campaign_value"],
                    "target_value": industry_data["industry_benchmark"],
                    "improvement_needed": abs(industry_data["difference_percent"]),
                    "priority": "high" if industry_data["performance"] == "poor" else "medium"
                })
        
        # Sort by priority and improvement potential
        opportunities.sort(key=lambda x: (x["priority"] == "high", x.get("improvement_needed", 0)), reverse=True)
        
        return opportunities[:5]  # Return top 5 opportunities
    
    def _identify_competitive_advantages(
        self,
        metric_benchmarks: Dict[str, Dict[str, float]],
        industry_comparison: Dict[str, Dict[str, Union[str, float]]]
    ) -> List[Dict[str, Any]]:
        """Identify competitive advantages based on benchmarks."""
        advantages = []
        
        # Peer-based advantages
        for metric_name, benchmark_data in metric_benchmarks.items():
            if benchmark_data["percentile_rank"] >= 75:  # Top 25%
                advantages.append({
                    "metric": metric_name,
                    "type": "peer_advantage",
                    "current_value": benchmark_data["value"],
                    "peer_median": benchmark_data["peer_median"],
                    "percentile_rank": benchmark_data["percentile_rank"],
                    "strength": "strong" if benchmark_data["percentile_rank"] >= 90 else "moderate"
                })
        
        # Industry-based advantages
        for metric_name, industry_data in industry_comparison.items():
            if isinstance(industry_data, dict) and industry_data.get("performance") in ["excellent", "good"]:
                advantages.append({
                    "metric": metric_name,
                    "type": "industry_advantage",
                    "current_value": industry_data["campaign_value"],
                    "industry_benchmark": industry_data["industry_benchmark"],
                    "outperformance_percent": abs(industry_data["difference_percent"]),
                    "strength": "strong" if industry_data["performance"] == "excellent" else "moderate"
                })
        
        return advantages
    
    def _get_benchmark_period(self, historical_metrics: Optional[List[DailyMetrics]]) -> Tuple[datetime, datetime]:
        """Determine benchmarking period from historical data."""
        if historical_metrics:
            dates = [m.date for m in historical_metrics]
            return (min(dates), max(dates))
        else:
            # Default to last 30 days
            end_date = datetime.now()
            start_date = end_date - timedelta(days=30)
            return (start_date, end_date)
    
    def get_benchmarking_insights(self, benchmark_results: List[BenchmarkResult]) -> Dict[str, Any]:
        """Generate insights from benchmarking results."""
        try:
            if not benchmark_results:
                return {"error": "No benchmark results to analyze"}
            
            insights = {
                "total_entities": len(benchmark_results),
                "performance_distribution": {},
                "average_percentile": 0.0,
                "top_performers": [],
                "common_strengths": [],
                "common_opportunities": [],
                "industry_comparison_summary": {},
                "key_insights": []
            }
            
            # Performance distribution
            for result in benchmark_results:
                rank = result.overall_rank
                insights["performance_distribution"][rank] = insights["performance_distribution"].get(rank, 0) + 1
            
            # Average percentile
            percentiles = [r.percentile_rank for r in benchmark_results]
            insights["average_percentile"] = np.mean(percentiles)
            
            # Top performers
            top_performers = [r for r in benchmark_results if r.percentile_rank >= 75]
            insights["top_performers"] = [
                {"name": tp.entity_name, "percentile": tp.percentile_rank}
                for tp in top_performers[:5]
            ]
            
            # Common strengths and opportunities
            all_advantages = []
            all_opportunities = []
            
            for result in benchmark_results:
                all_advantages.extend([adv["metric"] for adv in result.competitive_advantages])
                all_opportunities.extend([opp["metric"] for opp in result.improvement_opportunities])
            
            # Count frequency
            strength_counts = {}
            opportunity_counts = {}
            
            for advantage in all_advantages:
                strength_counts[advantage] = strength_counts.get(advantage, 0) + 1
            
            for opportunity in all_opportunities:
                opportunity_counts[opportunity] = opportunity_counts.get(opportunity, 0) + 1
            
            # Get top common items
            if strength_counts:
                insights["common_strengths"] = sorted(strength_counts.items(), key=lambda x: x[1], reverse=True)[:3]
            
            if opportunity_counts:
                insights["common_opportunities"] = sorted(opportunity_counts.items(), key=lambda x: x[1], reverse=True)[:3]
            
            # Generate key insights
            top_performer_pct = len(top_performers) / len(benchmark_results) * 100
            
            if top_performer_pct > 50:
                insights["key_insights"].append("Strong overall performance - majority of entities are top performers")
            elif top_performer_pct < 20:
                insights["key_insights"].append("Performance improvement opportunity - few entities are top performers")
            
            if insights["common_opportunities"]:
                top_opportunity = insights["common_opportunities"][0][0]
                insights["key_insights"].append(f"Focus on {top_opportunity} - most common improvement opportunity")
            
            if insights["average_percentile"] > 75:
                insights["key_insights"].append("Above-average competitive position overall")
            elif insights["average_percentile"] < 50:
                insights["key_insights"].append("Below-average competitive position - consider strategic review")
            
            return insights
            
        except Exception as e:
            logger.error(f"Benchmarking insights generation failed: {e}")
            return {"error": str(e)}