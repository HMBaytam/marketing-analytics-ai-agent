"""Campaign Analysis Agent for generating marketing performance insights."""

import logging
import statistics
from typing import Any

from langchain.schema import HumanMessage, SystemMessage
from langchain_anthropic import ChatAnthropic
from pydantic import BaseModel

from ..models.analytics import TrafficData
from ..models.campaign import Campaign, CampaignStatus
from .orchestrator import AgentState

logger = logging.getLogger(__name__)


class PerformanceInsight(BaseModel):
    """Performance insight model."""

    category: str
    insight: str
    confidence: float
    supporting_data: dict[str, Any]
    impact_level: str  # high, medium, low


class CampaignAnalyzerAgent:
    """Agent responsible for analyzing campaign performance and generating insights."""

    def __init__(self, llm: ChatAnthropic | None = None):
        """
        Initialize campaign analyzer agent.

        Args:
            llm: Language model for generating insights
        """
        self.llm = llm or ChatAnthropic(
            model="claude-3-sonnet-20240229", temperature=0.2, max_tokens=4000
        )

    async def analyze_campaigns(self, state: AgentState) -> AgentState:
        """
        Analyze campaign performance and generate insights.

        Args:
            state: Current workflow state

        Returns:
            Updated state with analysis results
        """
        try:
            logger.info("Starting campaign performance analysis")

            # Validate required data
            if not state.google_ads_data and not state.ga4_data:
                state.error_messages.append("No data available for analysis")
                return state

            # Perform different types of analysis
            insights = {}

            # Campaign efficiency analysis
            if state.google_ads_data:
                insights[
                    "campaign_efficiency"
                ] = await self._analyze_campaign_efficiency(state.google_ads_data)
                insights["spend_analysis"] = await self._analyze_spend_distribution(
                    state.google_ads_data
                )
                insights["performance_trends"] = await self._analyze_performance_trends(
                    state.google_ads_data
                )

            # Channel performance analysis
            if state.ga4_data:
                insights[
                    "channel_performance"
                ] = await self._analyze_channel_performance(state.ga4_data)
                insights["traffic_quality"] = await self._analyze_traffic_quality(
                    state.ga4_data
                )

            # Cross-channel attribution (if both data sources available)
            if state.google_ads_data and state.ga4_data:
                insights["attribution_analysis"] = await self._analyze_attribution(
                    state.google_ads_data, state.ga4_data
                )

            # Generate AI-powered insights
            ai_insights = await self._generate_ai_insights(state, insights)
            insights["ai_insights"] = ai_insights

            state.analysis_insights = insights

            logger.info(f"Analysis completed with {len(insights)} insight categories")
            return state

        except Exception as e:
            logger.error(f"Campaign analysis failed: {e}")
            state.error_messages.append(f"Campaign analysis error: {str(e)}")
            return state

    async def _analyze_campaign_efficiency(
        self, campaigns: list[Campaign]
    ) -> dict[str, Any]:
        """Analyze campaign cost efficiency metrics."""
        try:
            # Filter active campaigns with data
            active_campaigns = [
                c
                for c in campaigns
                if c.status == CampaignStatus.ENABLED and c.clicks and c.cost
            ]

            if not active_campaigns:
                return {"error": "No active campaigns with performance data"}

            # Calculate efficiency metrics
            cpcs = [float(c.cost) / c.clicks for c in active_campaigns if c.clicks > 0]
            ctrs = [c.ctr for c in active_campaigns if c.ctr is not None]
            conversion_rates = [
                c.conversion_rate
                for c in active_campaigns
                if c.conversion_rate is not None
            ]

            # Performance distribution
            efficiency_analysis = {
                "total_campaigns": len(campaigns),
                "active_campaigns": len(active_campaigns),
                "metrics": {
                    "average_cpc": statistics.mean(cpcs) if cpcs else 0,
                    "median_cpc": statistics.median(cpcs) if cpcs else 0,
                    "cpc_std": statistics.stdev(cpcs) if len(cpcs) > 1 else 0,
                    "average_ctr": statistics.mean(ctrs) if ctrs else 0,
                    "average_conversion_rate": statistics.mean(conversion_rates)
                    if conversion_rates
                    else 0,
                },
                "top_performers": [],
                "underperformers": [],
            }

            # Identify top performers and underperformers
            if cpcs:
                median_cpc = statistics.median(cpcs)

                for campaign in active_campaigns:
                    campaign_cpc = (
                        float(campaign.cost) / campaign.clicks
                        if campaign.clicks > 0
                        else float("inf")
                    )

                    if (
                        campaign_cpc <= median_cpc * 0.7
                        and campaign.conversions
                        and campaign.conversions > 0
                    ):
                        efficiency_analysis["top_performers"].append(
                            {
                                "name": campaign.name,
                                "cpc": campaign_cpc,
                                "ctr": campaign.ctr,
                                "conversions": campaign.conversions,
                            }
                        )
                    elif campaign_cpc >= median_cpc * 1.5:
                        efficiency_analysis["underperformers"].append(
                            {
                                "name": campaign.name,
                                "cpc": campaign_cpc,
                                "ctr": campaign.ctr,
                                "conversions": campaign.conversions or 0,
                            }
                        )

            return efficiency_analysis

        except Exception as e:
            logger.error(f"Campaign efficiency analysis failed: {e}")
            return {"error": str(e)}

    async def _analyze_spend_distribution(
        self, campaigns: list[Campaign]
    ) -> dict[str, Any]:
        """Analyze how budget is distributed across campaigns."""
        try:
            total_spend = sum(float(c.cost or 0) for c in campaigns)
            total_conversions = sum(c.conversions or 0 for c in campaigns)

            if total_spend == 0:
                return {"error": "No spend data available"}

            # Calculate spend distribution
            campaign_spend = []
            for campaign in campaigns:
                spend = float(campaign.cost or 0)
                if spend > 0:
                    spend_share = spend / total_spend * 100
                    conversion_share = (
                        (campaign.conversions or 0) / total_conversions * 100
                        if total_conversions > 0
                        else 0
                    )

                    campaign_spend.append(
                        {
                            "name": campaign.name,
                            "spend": spend,
                            "spend_share": spend_share,
                            "conversions": campaign.conversions or 0,
                            "conversion_share": conversion_share,
                            "efficiency_ratio": conversion_share / spend_share
                            if spend_share > 0
                            else 0,
                        }
                    )

            # Sort by spend
            campaign_spend.sort(key=lambda x: x["spend"], reverse=True)

            # Pareto analysis (80/20 rule)
            cumulative_spend = 0
            pareto_campaigns = []

            for camp in campaign_spend:
                cumulative_spend += camp["spend"]
                cumulative_share = cumulative_spend / total_spend * 100
                camp["cumulative_share"] = cumulative_share

                if cumulative_share <= 80:
                    pareto_campaigns.append(camp["name"])

            return {
                "total_spend": total_spend,
                "total_conversions": total_conversions,
                "campaign_distribution": campaign_spend[:10],  # Top 10 campaigns
                "pareto_analysis": {
                    "top_campaigns_80_percent_spend": pareto_campaigns,
                    "concentration_ratio": len(pareto_campaigns)
                    / len(campaign_spend)
                    * 100,
                },
            }

        except Exception as e:
            logger.error(f"Spend distribution analysis failed: {e}")
            return {"error": str(e)}

    async def _analyze_performance_trends(
        self, campaigns: list[Campaign]
    ) -> dict[str, Any]:
        """Analyze performance trends over time."""
        try:
            # This would typically require time-series data
            # For now, we'll analyze current performance distribution

            active_campaigns = [
                c for c in campaigns if c.status == CampaignStatus.ENABLED
            ]

            if not active_campaigns:
                return {"error": "No active campaigns for trend analysis"}

            # Performance distribution
            performance_buckets = {
                "high_performance": [],
                "medium_performance": [],
                "low_performance": [],
            }

            # Calculate performance scores
            for campaign in active_campaigns:
                score = self._calculate_performance_score(campaign)

                if score >= 0.7:
                    performance_buckets["high_performance"].append(
                        {"name": campaign.name, "score": score}
                    )
                elif score >= 0.4:
                    performance_buckets["medium_performance"].append(
                        {"name": campaign.name, "score": score}
                    )
                else:
                    performance_buckets["low_performance"].append(
                        {"name": campaign.name, "score": score}
                    )

            return {
                "performance_distribution": performance_buckets,
                "insights": {
                    "high_performers": len(performance_buckets["high_performance"]),
                    "improvement_candidates": len(
                        performance_buckets["low_performance"]
                    ),
                    "performance_ratio": len(performance_buckets["high_performance"])
                    / len(active_campaigns)
                    * 100,
                },
            }

        except Exception as e:
            logger.error(f"Performance trend analysis failed: {e}")
            return {"error": str(e)}

    async def _analyze_channel_performance(
        self, ga4_data: list[TrafficData]
    ) -> dict[str, Any]:
        """Analyze performance by traffic channel."""
        try:
            # Group by channel
            channel_metrics = {}

            for record in ga4_data:
                channel = record.channel_grouping or "Unknown"

                if channel not in channel_metrics:
                    channel_metrics[channel] = {
                        "sessions": 0,
                        "users": 0,
                        "pageviews": 0,
                        "conversions": 0,
                        "records": 0,
                    }

                channel_metrics[channel]["sessions"] += record.sessions
                channel_metrics[channel]["users"] += record.users
                channel_metrics[channel]["pageviews"] += record.pageviews or 0
                channel_metrics[channel]["conversions"] += (
                    getattr(record, "conversions", 0) or 0
                )
                channel_metrics[channel]["records"] += 1

            # Calculate channel performance metrics
            channel_analysis = {}
            total_sessions = sum(m["sessions"] for m in channel_metrics.values())

            for channel, metrics in channel_metrics.items():
                if metrics["sessions"] > 0:
                    channel_analysis[channel] = {
                        "sessions": metrics["sessions"],
                        "users": metrics["users"],
                        "session_share": metrics["sessions"] / total_sessions * 100,
                        "pages_per_session": metrics["pageviews"] / metrics["sessions"]
                        if metrics["sessions"] > 0
                        else 0,
                        "conversion_rate": metrics["conversions"]
                        / metrics["sessions"]
                        * 100
                        if metrics["sessions"] > 0
                        else 0,
                        "user_engagement": metrics["users"] / metrics["sessions"]
                        if metrics["sessions"] > 0
                        else 0,
                    }

            # Sort by session volume
            sorted_channels = sorted(
                channel_analysis.items(), key=lambda x: x[1]["sessions"], reverse=True
            )

            return {
                "total_sessions": total_sessions,
                "channel_performance": dict(sorted_channels),
                "top_channels": [channel for channel, _ in sorted_channels[:3]],
                "channel_count": len(channel_analysis),
            }

        except Exception as e:
            logger.error(f"Channel performance analysis failed: {e}")
            return {"error": str(e)}

    async def _analyze_traffic_quality(
        self, ga4_data: list[TrafficData]
    ) -> dict[str, Any]:
        """Analyze traffic quality metrics."""
        try:
            total_sessions = sum(record.sessions for record in ga4_data)
            total_users = sum(record.users for record in ga4_data)
            total_pageviews = sum(record.pageviews or 0 for record in ga4_data)

            if total_sessions == 0:
                return {"error": "No session data available"}

            # Quality metrics
            quality_metrics = {
                "pages_per_session": total_pageviews / total_sessions,
                "sessions_per_user": total_sessions / total_users
                if total_users > 0
                else 0,
                "total_sessions": total_sessions,
                "total_users": total_users,
                "total_pageviews": total_pageviews,
            }

            # Quality assessment
            quality_score = 0
            if quality_metrics["pages_per_session"] > 2:
                quality_score += 0.3
            if quality_metrics["sessions_per_user"] > 1.2:
                quality_score += 0.3
            if quality_metrics["sessions_per_user"] < 1.5:  # Good user retention
                quality_score += 0.4

            quality_assessment = (
                "high"
                if quality_score >= 0.7
                else "medium"
                if quality_score >= 0.4
                else "low"
            )

            return {
                "quality_metrics": quality_metrics,
                "quality_score": quality_score,
                "quality_assessment": quality_assessment,
            }

        except Exception as e:
            logger.error(f"Traffic quality analysis failed: {e}")
            return {"error": str(e)}

    async def _analyze_attribution(
        self, campaigns: list[Campaign], ga4_data: list[TrafficData]
    ) -> dict[str, Any]:
        """Analyze cross-channel attribution."""
        try:
            # Calculate paid search contribution
            paid_search_spend = sum(
                float(c.cost or 0)
                for c in campaigns
                if c.advertising_channel_type.value in ["SEARCH", "SHOPPING"]
            )

            paid_search_conversions = sum(
                c.conversions or 0
                for c in campaigns
                if c.advertising_channel_type.value in ["SEARCH", "SHOPPING"]
            )

            # GA4 channel data
            organic_sessions = sum(
                record.sessions
                for record in ga4_data
                if record.channel_grouping
                and "organic" in record.channel_grouping.lower()
            )

            paid_sessions = sum(
                record.sessions
                for record in ga4_data
                if record.channel_grouping and "paid" in record.channel_grouping.lower()
            )

            total_ga4_sessions = sum(record.sessions for record in ga4_data)

            return {
                "paid_search_metrics": {
                    "spend": paid_search_spend,
                    "conversions": paid_search_conversions,
                    "sessions": paid_sessions,
                },
                "organic_metrics": {
                    "sessions": organic_sessions,
                    "session_share": organic_sessions / total_ga4_sessions * 100
                    if total_ga4_sessions > 0
                    else 0,
                },
                "channel_mix": {
                    "paid_share": paid_sessions / total_ga4_sessions * 100
                    if total_ga4_sessions > 0
                    else 0,
                    "organic_share": organic_sessions / total_ga4_sessions * 100
                    if total_ga4_sessions > 0
                    else 0,
                },
            }

        except Exception as e:
            logger.error(f"Attribution analysis failed: {e}")
            return {"error": str(e)}

    async def _generate_ai_insights(
        self, state: AgentState, analysis_results: dict[str, Any]
    ) -> list[str]:
        """Generate AI-powered insights from analysis results."""
        try:
            insights_prompt = self._create_insights_prompt(
                state.user_query, analysis_results
            )

            response = await self.llm.ainvoke(
                [
                    SystemMessage(content=insights_prompt),
                    HumanMessage(content=f"Generate insights for: {state.user_query}"),
                ]
            )

            # Parse insights from response
            insights = self._parse_ai_insights(response.content)
            return insights

        except Exception as e:
            logger.error(f"AI insight generation failed: {e}")
            return [f"Unable to generate AI insights: {str(e)}"]

    def _calculate_performance_score(self, campaign: Campaign) -> float:
        """Calculate normalized performance score for a campaign."""
        score = 0.0

        # CTR component (0-0.3)
        if campaign.ctr:
            ctr_score = min(campaign.ctr / 5.0, 0.3)  # Cap at 5% CTR
            score += ctr_score

        # Conversion rate component (0-0.4)
        if campaign.conversion_rate:
            conv_score = min(
                campaign.conversion_rate / 10.0 * 0.4, 0.4
            )  # Cap at 10% conv rate
            score += conv_score

        # Cost efficiency component (0-0.3)
        if campaign.clicks and campaign.cost and float(campaign.cost) > 0:
            cpc = float(campaign.cost) / campaign.clicks
            # Lower CPC is better, normalize to 0-0.3
            cpc_score = max(0, 0.3 - (cpc / 10.0 * 0.3))  # Assume $10 CPC as benchmark
            score += cpc_score

        return min(score, 1.0)

    def _create_insights_prompt(
        self, user_query: str, analysis_results: dict[str, Any]
    ) -> str:
        """Create prompt for AI insight generation."""
        return f"""You are a marketing analytics expert. Based on the analysis results, generate 3-5 key insights
        that directly answer the user's question.

        User Question: {user_query}

        Analysis Results: {analysis_results}

        For each insight:
        1. State the finding clearly
        2. Provide supporting evidence from the data
        3. Explain the business implication

        Focus on actionable insights that can drive marketing decisions.
        """

    def _parse_ai_insights(self, response_text: str) -> list[str]:
        """Parse insights from AI response."""
        lines = response_text.strip().split("\n")
        insights = []

        current_insight = ""
        for line in lines:
            line = line.strip()
            if line and (
                line[0].isdigit() or line.startswith("-") or line.startswith("•")
            ):
                if current_insight:
                    insights.append(current_insight.strip())
                current_insight = line.split(".", 1)[-1].split("-", 1)[-1].strip()
            elif line and current_insight:
                current_insight += " " + line

        if current_insight:
            insights.append(current_insight.strip())

        return insights[:5]  # Limit to 5 insights
