"""Performance scoring engine for campaigns, ad groups, and keywords."""

import logging
import math
from datetime import datetime
from typing import Any

import numpy as np
from pydantic import BaseModel, Field

from ..models.campaign import Campaign
from ..models.metrics import DailyMetrics

logger = logging.getLogger(__name__)


class ScoringConfig(BaseModel):
    """Configuration for performance scoring."""

    # Scoring weights (should sum to 1.0)
    efficiency_weight: float = Field(
        0.3, description="Weight for cost efficiency metrics"
    )
    volume_weight: float = Field(0.25, description="Weight for volume metrics")
    quality_weight: float = Field(0.25, description="Weight for quality metrics")
    trend_weight: float = Field(0.2, description="Weight for trend metrics")

    # Benchmark values for normalization
    target_ctr: float = Field(
        2.0, description="Target CTR percentage for normalization"
    )
    target_conversion_rate: float = Field(
        3.0, description="Target conversion rate for normalization"
    )
    target_cpc: float = Field(2.0, description="Target CPC for normalization")
    target_cpa: float = Field(50.0, description="Target CPA for normalization")

    # Volume thresholds
    min_impressions: int = Field(
        1000, description="Minimum impressions for reliable scoring"
    )
    min_clicks: int = Field(10, description="Minimum clicks for reliable scoring")

    # Scoring parameters
    max_score: float = Field(100.0, description="Maximum possible score")
    decay_factor: float = Field(0.1, description="Decay factor for trend scoring")

    class Config:
        arbitrary_types_allowed = True


class PerformanceScore(BaseModel):
    """Performance score result."""

    entity_id: str = Field(..., description="Entity ID (campaign, ad group, keyword)")
    entity_name: str = Field(..., description="Entity name")
    entity_type: str = Field(..., description="Entity type")

    # Overall score
    overall_score: float = Field(..., description="Overall performance score (0-100)")
    score_grade: str = Field(..., description="Score grade (A+, A, B+, B, C+, C, D, F)")

    # Component scores
    efficiency_score: float = Field(..., description="Cost efficiency score")
    volume_score: float = Field(..., description="Volume score")
    quality_score: float = Field(..., description="Quality score")
    trend_score: float = Field(..., description="Trend score")

    # Supporting metrics
    key_metrics: dict[str, float] = Field(
        default_factory=dict, description="Key performance metrics"
    )
    strengths: list[str] = Field(
        default_factory=list, description="Performance strengths"
    )
    weaknesses: list[str] = Field(
        default_factory=list, description="Performance weaknesses"
    )

    # Reliability indicators
    data_reliability: str = Field(
        ..., description="Data reliability (high, medium, low)"
    )
    sample_size_score: float = Field(..., description="Sample size adequacy score")

    # Timestamp
    calculated_at: datetime = Field(
        default_factory=datetime.utcnow, description="Calculation timestamp"
    )


class PerformanceScorer:
    """Advanced performance scoring engine using multiple weighted metrics."""

    def __init__(self, config: ScoringConfig | None = None):
        """
        Initialize performance scorer.

        Args:
            config: Scoring configuration
        """
        self.config = config or ScoringConfig()
        self._validate_config()

    def _validate_config(self):
        """Validate scoring configuration."""
        total_weight = (
            self.config.efficiency_weight
            + self.config.volume_weight
            + self.config.quality_weight
            + self.config.trend_weight
        )

        if abs(total_weight - 1.0) > 0.01:
            logger.warning(
                f"Scoring weights sum to {total_weight:.3f}, not 1.0. Normalizing weights."
            )
            # Normalize weights
            self.config.efficiency_weight /= total_weight
            self.config.volume_weight /= total_weight
            self.config.quality_weight /= total_weight
            self.config.trend_weight /= total_weight

    def score_campaigns(
        self,
        campaigns: list[Campaign],
        historical_metrics: list[DailyMetrics] | None = None,
    ) -> list[PerformanceScore]:
        """
        Score campaign performance.

        Args:
            campaigns: List of campaigns to score
            historical_metrics: Historical performance data for trend analysis

        Returns:
            List of performance scores
        """
        try:
            scores = []

            # Create historical metrics lookup
            metrics_lookup = {}
            if historical_metrics:
                for metric in historical_metrics:
                    if metric.entity_id not in metrics_lookup:
                        metrics_lookup[metric.entity_id] = []
                    metrics_lookup[metric.entity_id].append(metric)

            for campaign in campaigns:
                try:
                    score = self._score_campaign(
                        campaign, metrics_lookup.get(campaign.id, [])
                    )
                    scores.append(score)
                except Exception as e:
                    logger.error(f"Failed to score campaign {campaign.id}: {e}")
                    # Create a fallback score
                    scores.append(
                        self._create_fallback_score(
                            campaign.id, campaign.name, "campaign"
                        )
                    )

            return sorted(scores, key=lambda x: x.overall_score, reverse=True)

        except Exception as e:
            logger.error(f"Campaign scoring failed: {e}")
            return []

    def _score_campaign(
        self, campaign: Campaign, historical_metrics: list[DailyMetrics]
    ) -> PerformanceScore:
        """Score individual campaign performance."""

        # Calculate component scores
        efficiency_score = self._calculate_efficiency_score(campaign)
        volume_score = self._calculate_volume_score(campaign)
        quality_score = self._calculate_quality_score(campaign)
        trend_score = self._calculate_trend_score(historical_metrics)

        # Calculate overall weighted score
        overall_score = (
            efficiency_score * self.config.efficiency_weight
            + volume_score * self.config.volume_weight
            + quality_score * self.config.quality_weight
            + trend_score * self.config.trend_weight
        )

        # Determine grade
        score_grade = self._calculate_grade(overall_score)

        # Extract key metrics
        key_metrics = {
            "impressions": float(campaign.impressions or 0),
            "clicks": float(campaign.clicks or 0),
            "conversions": float(campaign.conversions or 0),
            "cost": float(campaign.cost),
            "ctr": campaign.ctr or 0.0,
            "conversion_rate": campaign.conversion_rate or 0.0,
            "cpc": float(campaign.cost) / campaign.clicks
            if campaign.clicks > 0
            else 0.0,
        }

        # Determine strengths and weaknesses
        strengths, weaknesses = self._analyze_strengths_weaknesses(
            campaign,
            {
                "efficiency": efficiency_score,
                "volume": volume_score,
                "quality": quality_score,
                "trend": trend_score,
            },
        )

        # Data reliability assessment
        data_reliability, sample_size_score = self._assess_data_reliability(campaign)

        return PerformanceScore(
            entity_id=campaign.id,
            entity_name=campaign.name,
            entity_type="campaign",
            overall_score=round(overall_score, 2),
            score_grade=score_grade,
            efficiency_score=round(efficiency_score, 2),
            volume_score=round(volume_score, 2),
            quality_score=round(quality_score, 2),
            trend_score=round(trend_score, 2),
            key_metrics=key_metrics,
            strengths=strengths,
            weaknesses=weaknesses,
            data_reliability=data_reliability,
            sample_size_score=sample_size_score,
        )

    def _calculate_efficiency_score(self, campaign: Campaign) -> float:
        """Calculate cost efficiency score (0-100)."""
        try:
            if not campaign.clicks or not campaign.cost:
                return 0.0

            cpc = float(campaign.cost) / campaign.clicks

            # Normalize CPC score (lower is better)
            cpc_score = max(0, 100 * (1 - (cpc / self.config.target_cpc)))

            # CPA score if conversions available
            cpa_score = 50.0  # Default neutral score
            if campaign.conversions and campaign.conversions > 0:
                cpa = float(campaign.cost) / campaign.conversions
                cpa_score = max(0, 100 * (1 - (cpa / self.config.target_cpa)))

            # Weight CPC more heavily if no conversion data
            if campaign.conversions and campaign.conversions > 0:
                return (cpc_score * 0.4) + (cpa_score * 0.6)
            else:
                return cpc_score * 0.8  # Penalize lack of conversion data

        except Exception as e:
            logger.error(f"Efficiency score calculation failed: {e}")
            return 0.0

    def _calculate_volume_score(self, campaign: Campaign) -> float:
        """Calculate volume score based on impressions and clicks (0-100)."""
        try:
            impressions = campaign.impressions or 0
            clicks = campaign.clicks or 0

            # Logarithmic scaling for volume
            impression_score = min(
                100, 100 * (math.log(impressions + 1) / math.log(100000))
            )
            click_score = min(100, 100 * (math.log(clicks + 1) / math.log(1000)))

            # Combined volume score
            volume_score = (impression_score * 0.6) + (click_score * 0.4)

            return volume_score

        except Exception as e:
            logger.error(f"Volume score calculation failed: {e}")
            return 0.0

    def _calculate_quality_score(self, campaign: Campaign) -> float:
        """Calculate quality score based on CTR and conversion rates (0-100)."""
        try:
            ctr_score = 0.0
            if campaign.ctr:
                # Normalize CTR score
                ctr_score = min(100, 100 * (campaign.ctr / self.config.target_ctr))

            conversion_score = 50.0  # Default neutral score
            if campaign.conversion_rate:
                conversion_score = min(
                    100,
                    100
                    * (campaign.conversion_rate / self.config.target_conversion_rate),
                )

            # Quality score from Google Ads if available
            quality_score = 50.0  # Default
            if hasattr(campaign, "quality_score") and campaign.quality_score:
                quality_score = campaign.quality_score * 10  # Convert 1-10 to 0-100

            # Combine quality indicators
            return (ctr_score * 0.4) + (conversion_score * 0.4) + (quality_score * 0.2)

        except Exception as e:
            logger.error(f"Quality score calculation failed: {e}")
            return 0.0

    def _calculate_trend_score(self, historical_metrics: list[DailyMetrics]) -> float:
        """Calculate trend score based on historical performance (0-100)."""
        try:
            if len(historical_metrics) < 7:  # Need at least a week of data
                return 50.0  # Neutral score for insufficient data

            # Sort by date
            sorted_metrics = sorted(historical_metrics, key=lambda x: x.date)

            # Calculate trends for key metrics
            trends = {}

            # CTR trend
            ctr_values = [m.ctr for m in sorted_metrics if m.ctr is not None]
            if len(ctr_values) >= 5:
                trends["ctr"] = self._calculate_metric_trend(ctr_values)

            # Conversion rate trend
            conv_values = [
                m.conversion_rate
                for m in sorted_metrics
                if m.conversion_rate is not None
            ]
            if len(conv_values) >= 5:
                trends["conversion_rate"] = self._calculate_metric_trend(conv_values)

            # Cost efficiency trend (inverse of CPC trend)
            cpc_values = []
            for m in sorted_metrics:
                if m.clicks and m.clicks > 0 and m.cost:
                    cpc_values.append(float(m.cost) / m.clicks)

            if len(cpc_values) >= 5:
                cpc_trend = self._calculate_metric_trend(cpc_values)
                trends["cost_efficiency"] = -cpc_trend  # Invert (lower CPC is better)

            # Aggregate trend score
            if trends:
                avg_trend = sum(trends.values()) / len(trends)
                # Convert trend (-1 to 1) to score (0 to 100)
                return max(0, min(100, 50 + (avg_trend * 50)))

            return 50.0

        except Exception as e:
            logger.error(f"Trend score calculation failed: {e}")
            return 50.0

    def _calculate_metric_trend(self, values: list[float]) -> float:
        """Calculate trend direction for a metric series (-1 to 1)."""
        try:
            if len(values) < 3:
                return 0.0

            # Use linear regression slope
            n = len(values)
            x = np.arange(n)
            y = np.array(values)

            # Calculate slope
            slope = (n * np.sum(x * y) - np.sum(x) * np.sum(y)) / (
                n * np.sum(x * x) - np.sum(x) ** 2
            )

            # Normalize slope to -1 to 1 range
            max_value = max(values)
            if max_value > 0:
                normalized_slope = slope / max_value * n  # Scale by series length
                return max(-1, min(1, normalized_slope))

            return 0.0

        except Exception as e:
            logger.error(f"Trend calculation failed: {e}")
            return 0.0

    def _calculate_grade(self, score: float) -> str:
        """Convert numeric score to letter grade."""
        if score >= 95:
            return "A+"
        elif score >= 90:
            return "A"
        elif score >= 85:
            return "B+"
        elif score >= 80:
            return "B"
        elif score >= 75:
            return "C+"
        elif score >= 70:
            return "C"
        elif score >= 60:
            return "D"
        else:
            return "F"

    def _analyze_strengths_weaknesses(
        self, campaign: Campaign, component_scores: dict[str, float]
    ) -> tuple[list[str], list[str]]:
        """Analyze campaign strengths and weaknesses."""
        strengths = []
        weaknesses = []

        # Component score analysis
        for component, score in component_scores.items():
            if score >= 80:
                strengths.append(f"Strong {component.replace('_', ' ')} performance")
            elif score <= 40:
                weaknesses.append(f"Poor {component.replace('_', ' ')} performance")

        # Metric-specific analysis
        if campaign.ctr and campaign.ctr >= self.config.target_ctr:
            strengths.append(f"High CTR ({campaign.ctr:.2f}%)")
        elif campaign.ctr and campaign.ctr < self.config.target_ctr * 0.5:
            weaknesses.append(f"Low CTR ({campaign.ctr:.2f}%)")

        if (
            campaign.conversion_rate
            and campaign.conversion_rate >= self.config.target_conversion_rate
        ):
            strengths.append(f"High conversion rate ({campaign.conversion_rate:.2f}%)")
        elif (
            campaign.conversion_rate
            and campaign.conversion_rate < self.config.target_conversion_rate * 0.5
        ):
            weaknesses.append(f"Low conversion rate ({campaign.conversion_rate:.2f}%)")

        # Cost efficiency
        if campaign.clicks and campaign.cost:
            cpc = float(campaign.cost) / campaign.clicks
            if cpc <= self.config.target_cpc:
                strengths.append(f"Efficient CPC (${cpc:.2f})")
            elif cpc >= self.config.target_cpc * 2:
                weaknesses.append(f"High CPC (${cpc:.2f})")

        return strengths, weaknesses

    def _assess_data_reliability(self, campaign: Campaign) -> tuple[str, float]:
        """Assess data reliability based on sample size and data completeness."""
        impressions = campaign.impressions or 0
        clicks = campaign.clicks or 0

        # Sample size score
        sample_score = 0.0

        if impressions >= self.config.min_impressions:
            sample_score += 50
        else:
            sample_score += (impressions / self.config.min_impressions) * 50

        if clicks >= self.config.min_clicks:
            sample_score += 50
        else:
            sample_score += (clicks / self.config.min_clicks) * 50

        # Reliability assessment
        if sample_score >= 90:
            reliability = "high"
        elif sample_score >= 60:
            reliability = "medium"
        else:
            reliability = "low"

        return reliability, sample_score

    def _create_fallback_score(
        self, entity_id: str, entity_name: str, entity_type: str
    ) -> PerformanceScore:
        """Create fallback score for entities that couldn't be scored."""
        return PerformanceScore(
            entity_id=entity_id,
            entity_name=entity_name,
            entity_type=entity_type,
            overall_score=0.0,
            score_grade="F",
            efficiency_score=0.0,
            volume_score=0.0,
            quality_score=0.0,
            trend_score=0.0,
            key_metrics={},
            strengths=[],
            weaknesses=["Insufficient data for scoring"],
            data_reliability="low",
            sample_size_score=0.0,
        )

    def get_performance_insights(
        self, scores: list[PerformanceScore]
    ) -> dict[str, Any]:
        """Generate performance insights from a list of scores."""
        if not scores:
            return {"error": "No scores to analyze"}

        try:
            # Overall statistics
            overall_scores = [s.overall_score for s in scores]

            insights = {
                "summary": {
                    "total_entities": len(scores),
                    "average_score": np.mean(overall_scores),
                    "median_score": np.median(overall_scores),
                    "score_std": np.std(overall_scores),
                    "top_performers": len([s for s in scores if s.overall_score >= 80]),
                    "underperformers": len(
                        [s for s in scores if s.overall_score <= 40]
                    ),
                },
                "grade_distribution": {},
                "common_strengths": {},
                "common_weaknesses": {},
                "component_analysis": {
                    "efficiency": np.mean([s.efficiency_score for s in scores]),
                    "volume": np.mean([s.volume_score for s in scores]),
                    "quality": np.mean([s.quality_score for s in scores]),
                    "trend": np.mean([s.trend_score for s in scores]),
                },
            }

            # Grade distribution
            for score in scores:
                grade = score.score_grade
                insights["grade_distribution"][grade] = (
                    insights["grade_distribution"].get(grade, 0) + 1
                )

            # Common strengths and weaknesses
            all_strengths = []
            all_weaknesses = []

            for score in scores:
                all_strengths.extend(score.strengths)
                all_weaknesses.extend(score.weaknesses)

            # Count frequency
            for strength in all_strengths:
                insights["common_strengths"][strength] = (
                    insights["common_strengths"].get(strength, 0) + 1
                )

            for weakness in all_weaknesses:
                insights["common_weaknesses"][weakness] = (
                    insights["common_weaknesses"].get(weakness, 0) + 1
                )

            return insights

        except Exception as e:
            logger.error(f"Performance insights generation failed: {e}")
            return {"error": str(e)}
