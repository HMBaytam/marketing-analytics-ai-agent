"""Recommendation Generation Agent for creating actionable marketing insights."""

import logging
from typing import Any

from langchain.schema import HumanMessage, SystemMessage
from langchain_anthropic import ChatAnthropic
from pydantic import BaseModel, Field

from .orchestrator import AgentState

logger = logging.getLogger(__name__)


class Recommendation(BaseModel):
    """Recommendation model."""

    category: str = Field(..., description="Recommendation category")
    title: str = Field(..., description="Recommendation title")
    description: str = Field(..., description="Detailed recommendation")
    priority: str = Field(..., description="Priority level (high, medium, low)")
    impact_estimate: str = Field(..., description="Expected impact")
    effort_required: str = Field(..., description="Implementation effort")
    timeline: str = Field(..., description="Recommended timeline")
    supporting_data: dict[str, Any] = Field(
        default_factory=dict, description="Supporting data"
    )


class RecommendationAgent:
    """Agent responsible for generating actionable marketing recommendations."""

    def __init__(self, llm: ChatAnthropic | None = None):
        """
        Initialize recommendation generator agent.

        Args:
            llm: Language model for generating recommendations
        """
        self.llm = llm or ChatAnthropic(
            model="claude-3-sonnet-20240229",
            temperature=0.4,  # Slightly higher for creative recommendations
            max_tokens=4000,
        )

        # Recommendation templates and rules
        self.recommendation_rules = self._initialize_recommendation_rules()

    async def generate_recommendations(self, state: AgentState) -> AgentState:
        """
        Generate actionable recommendations based on analysis insights.

        Args:
            state: Current workflow state with analysis results

        Returns:
            Updated state with recommendations
        """
        try:
            logger.info("Starting recommendation generation")

            if not state.analysis_insights:
                state.error_messages.append(
                    "No analysis insights available for recommendation generation"
                )
                return state

            # Generate different types of recommendations
            recommendations = []

            # Campaign optimization recommendations
            if "campaign_efficiency" in state.analysis_insights:
                campaign_recs = await self._generate_campaign_recommendations(state)
                recommendations.extend(campaign_recs)

            # Budget allocation recommendations
            if "spend_analysis" in state.analysis_insights:
                budget_recs = await self._generate_budget_recommendations(state)
                recommendations.extend(budget_recs)

            # Channel optimization recommendations
            if "channel_performance" in state.analysis_insights:
                channel_recs = await self._generate_channel_recommendations(state)
                recommendations.extend(channel_recs)

            # Performance improvement recommendations
            if "performance_trends" in state.analysis_insights:
                performance_recs = await self._generate_performance_recommendations(
                    state
                )
                recommendations.extend(performance_recs)

            # AI-powered strategic recommendations
            ai_recs = await self._generate_ai_recommendations(state)
            recommendations.extend(ai_recs)

            # Filter and prioritize recommendations
            final_recommendations = self._prioritize_recommendations(recommendations)

            # Convert to string format for state
            state.recommendations = [
                self._format_recommendation(rec) for rec in final_recommendations
            ]

            logger.info(f"Generated {len(state.recommendations)} recommendations")
            return state

        except Exception as e:
            logger.error(f"Recommendation generation failed: {e}")
            state.error_messages.append(f"Recommendation generation error: {str(e)}")
            return state

    async def _generate_campaign_recommendations(
        self, state: AgentState
    ) -> list[Recommendation]:
        """Generate campaign-specific recommendations."""
        recommendations = []

        try:
            efficiency_data = state.analysis_insights.get("campaign_efficiency", {})

            if efficiency_data.get("error"):
                return recommendations

            # Underperforming campaign recommendations
            underperformers = efficiency_data.get("underperformers", [])
            if underperformers:
                for campaign in underperformers[:3]:  # Top 3 underperformers
                    recommendations.append(
                        Recommendation(
                            category="Campaign Optimization",
                            title=f"Optimize '{campaign['name']}' Campaign",
                            description=f"Campaign has high CPC of ${campaign['cpc']:.2f} and low conversion rate. "
                            f"Consider reviewing keywords, ad copy, and landing pages.",
                            priority="high"
                            if campaign["cpc"]
                            > efficiency_data.get("metrics", {}).get("median_cpc", 0)
                            * 2
                            else "medium",
                            impact_estimate="10-25% cost reduction",
                            effort_required="Medium - requires campaign audit and optimization",
                            timeline="1-2 weeks",
                            supporting_data=campaign,
                        )
                    )

            # Top performer scaling recommendations
            top_performers = efficiency_data.get("top_performers", [])
            if top_performers:
                for campaign in top_performers[:2]:  # Top 2 performers
                    recommendations.append(
                        Recommendation(
                            category="Budget Scaling",
                            title=f"Scale Budget for '{campaign['name']}'",
                            description=f"High-performing campaign with CPC of ${campaign['cpc']:.2f} and "
                            f"{campaign['conversions']} conversions. Consider increasing budget.",
                            priority="high",
                            impact_estimate="15-30% more conversions",
                            effort_required="Low - budget adjustment",
                            timeline="Immediate",
                            supporting_data=campaign,
                        )
                    )

        except Exception as e:
            logger.error(f"Campaign recommendation generation failed: {e}")

        return recommendations

    async def _generate_budget_recommendations(
        self, state: AgentState
    ) -> list[Recommendation]:
        """Generate budget allocation recommendations."""
        recommendations = []

        try:
            spend_data = state.analysis_insights.get("spend_analysis", {})

            if spend_data.get("error"):
                return recommendations

            pareto_data = spend_data.get("pareto_analysis", {})
            concentration_ratio = pareto_data.get("concentration_ratio", 0)

            # Budget concentration recommendations
            if (
                concentration_ratio < 20
            ):  # Less than 20% of campaigns driving 80% of spend
                recommendations.append(
                    Recommendation(
                        category="Budget Allocation",
                        title="Consolidate Budget on High-Performing Campaigns",
                        description=f"Only {concentration_ratio:.1f}% of campaigns drive 80% of spend. "
                        f"Consider reallocating budget from low-performing campaigns.",
                        priority="medium",
                        impact_estimate="5-15% efficiency improvement",
                        effort_required="Medium - requires performance analysis and reallocation",
                        timeline="2-3 weeks",
                        supporting_data={"concentration_ratio": concentration_ratio},
                    )
                )

            # Efficiency-based reallocation
            campaign_distribution = spend_data.get("campaign_distribution", [])
            if campaign_distribution:
                # Find campaigns with low efficiency ratios
                inefficient_campaigns = [
                    c
                    for c in campaign_distribution
                    if c.get("efficiency_ratio", 0) < 0.5
                    and c.get("spend_share", 0) > 5
                ]

                if inefficient_campaigns:
                    total_inefficient_spend = sum(
                        c["spend"] for c in inefficient_campaigns
                    )
                    recommendations.append(
                        Recommendation(
                            category="Budget Reallocation",
                            title="Reallocate Budget from Inefficient Campaigns",
                            description=f"${total_inefficient_spend:,.0f} is allocated to campaigns with low "
                            f"conversion efficiency. Consider redistributing to higher-performing campaigns.",
                            priority="high",
                            impact_estimate="20-40% improvement in conversion efficiency",
                            effort_required="High - requires detailed analysis and gradual reallocation",
                            timeline="3-4 weeks",
                            supporting_data={
                                "inefficient_campaigns": inefficient_campaigns[:3]
                            },
                        )
                    )

        except Exception as e:
            logger.error(f"Budget recommendation generation failed: {e}")

        return recommendations

    async def _generate_channel_recommendations(
        self, state: AgentState
    ) -> list[Recommendation]:
        """Generate channel optimization recommendations."""
        recommendations = []

        try:
            channel_data = state.analysis_insights.get("channel_performance", {})

            if channel_data.get("error"):
                return recommendations

            channel_performance = channel_data.get("channel_performance", {})
            top_channels = channel_data.get("top_channels", [])

            # Underperforming channel recommendations
            for channel, metrics in channel_performance.items():
                conversion_rate = metrics.get("conversion_rate", 0)
                pages_per_session = metrics.get("pages_per_session", 0)

                if (
                    conversion_rate < 1.0 and metrics.get("session_share", 0) > 10
                ):  # Low conv rate, high traffic
                    recommendations.append(
                        Recommendation(
                            category="Channel Optimization",
                            title=f"Improve {channel} Channel Performance",
                            description=f"{channel} has {metrics['session_share']:.1f}% of traffic but only "
                            f"{conversion_rate:.2f}% conversion rate. Focus on user experience improvements.",
                            priority="medium",
                            impact_estimate="2-5x conversion rate improvement",
                            effort_required="Medium - UX optimization and landing page improvements",
                            timeline="2-4 weeks",
                            supporting_data=metrics,
                        )
                    )

                if (
                    pages_per_session < 1.5 and metrics.get("sessions", 0) > 100
                ):  # Low engagement
                    recommendations.append(
                        Recommendation(
                            category="Content Strategy",
                            title=f"Improve {channel} User Engagement",
                            description=f"{channel} users view only {pages_per_session:.1f} pages per session. "
                            f"Improve content relevance and internal linking.",
                            priority="low",
                            impact_estimate="15-25% increase in pages per session",
                            effort_required="Medium - content optimization",
                            timeline="3-6 weeks",
                            supporting_data=metrics,
                        )
                    )

            # Top channel scaling recommendations
            if top_channels:
                top_channel = top_channels[0]
                top_metrics = channel_performance.get(top_channel, {})

                if (
                    top_metrics.get("conversion_rate", 0) > 2.0
                ):  # High-performing channel
                    recommendations.append(
                        Recommendation(
                            category="Channel Scaling",
                            title=f"Scale {top_channel} Channel Investment",
                            description=f"{top_channel} shows strong performance with {top_metrics.get('conversion_rate', 0):.2f}% "
                            f"conversion rate. Consider increasing investment in this channel.",
                            priority="high",
                            impact_estimate="20-50% more conversions",
                            effort_required="Low to Medium - increase channel investment",
                            timeline="1-2 weeks",
                            supporting_data=top_metrics,
                        )
                    )

        except Exception as e:
            logger.error(f"Channel recommendation generation failed: {e}")

        return recommendations

    async def _generate_performance_recommendations(
        self, state: AgentState
    ) -> list[Recommendation]:
        """Generate performance improvement recommendations."""
        recommendations = []

        try:
            performance_data = state.analysis_insights.get("performance_trends", {})

            if performance_data.get("error"):
                return recommendations

            insights = performance_data.get("insights", {})
            performance_ratio = insights.get("performance_ratio", 0)
            improvement_candidates = insights.get("improvement_candidates", 0)

            # Overall performance recommendations
            if performance_ratio < 30:  # Less than 30% high performers
                recommendations.append(
                    Recommendation(
                        category="Performance Improvement",
                        title="Improve Overall Campaign Performance",
                        description=f"Only {performance_ratio:.1f}% of campaigns are high-performing. "
                        f"Focus on optimizing the {improvement_candidates} underperforming campaigns.",
                        priority="high",
                        impact_estimate="25-40% overall performance improvement",
                        effort_required="High - systematic campaign optimization",
                        timeline="4-8 weeks",
                        supporting_data=insights,
                    )
                )

            # Specific improvement actions
            if improvement_candidates > 3:
                recommendations.append(
                    Recommendation(
                        category="Campaign Audit",
                        title="Conduct Comprehensive Campaign Audit",
                        description=f"With {improvement_candidates} campaigns needing improvement, "
                        f"conduct a systematic audit of keywords, ad copy, and landing pages.",
                        priority="medium",
                        impact_estimate="15-30% improvement in underperforming campaigns",
                        effort_required="High - detailed audit and optimization",
                        timeline="3-5 weeks",
                        supporting_data={
                            "improvement_candidates": improvement_candidates
                        },
                    )
                )

        except Exception as e:
            logger.error(f"Performance recommendation generation failed: {e}")

        return recommendations

    async def _generate_ai_recommendations(
        self, state: AgentState
    ) -> list[Recommendation]:
        """Generate AI-powered strategic recommendations."""
        try:
            # Create comprehensive prompt with all analysis data
            recommendation_prompt = self._create_recommendation_prompt(state)

            response = await self.llm.ainvoke(
                [
                    SystemMessage(content=recommendation_prompt),
                    HumanMessage(
                        content=f"Generate strategic recommendations based on the analysis for: {state.user_query}"
                    ),
                ]
            )

            # Parse AI recommendations
            ai_recommendations = self._parse_ai_recommendations(response.content)
            return ai_recommendations

        except Exception as e:
            logger.error(f"AI recommendation generation failed: {e}")
            return [
                Recommendation(
                    category="System Error",
                    title="Unable to Generate AI Recommendations",
                    description=f"AI recommendation generation failed: {str(e)}",
                    priority="low",
                    impact_estimate="N/A",
                    effort_required="N/A",
                    timeline="N/A",
                )
            ]

    def _prioritize_recommendations(
        self, recommendations: list[Recommendation]
    ) -> list[Recommendation]:
        """Prioritize recommendations based on impact and effort."""
        # Sort by priority: high -> medium -> low
        priority_order = {"high": 3, "medium": 2, "low": 1}

        sorted_recommendations = sorted(
            recommendations,
            key=lambda r: (priority_order.get(r.priority, 0), r.category),
            reverse=True,
        )

        # Limit to top 8 recommendations to avoid overwhelming users
        return sorted_recommendations[:8]

    def _format_recommendation(self, rec: Recommendation) -> str:
        """Format recommendation for display."""
        priority_emoji = {"high": "🔴", "medium": "🟡", "low": "🟢"}
        emoji = priority_emoji.get(rec.priority, "⚪")

        return (
            f"{emoji} **{rec.title}**\n"
            f"*{rec.category}*\n\n"
            f"{rec.description}\n\n"
            f"**Expected Impact:** {rec.impact_estimate}\n"
            f"**Effort Required:** {rec.effort_required}\n"
            f"**Timeline:** {rec.timeline}\n"
        )

    def _create_recommendation_prompt(self, state: AgentState) -> str:
        """Create comprehensive prompt for AI recommendations."""
        return f"""You are a senior marketing strategist. Based on the comprehensive analysis results,
        generate 2-3 high-level strategic recommendations that address the user's question.

        User Question: {state.user_query}
        Analysis Results: {state.analysis_insights}

        For each recommendation, provide:
        1. Clear title
        2. Strategic rationale
        3. Expected business impact
        4. Implementation approach
        5. Success metrics to track

        Focus on recommendations that:
        - Directly address the user's question
        - Have measurable business impact
        - Are actionable within 1-2 months
        - Leverage the specific insights from the data

        Format as: TITLE | DESCRIPTION | IMPACT | EFFORT | TIMELINE
        """

    def _parse_ai_recommendations(self, response_text: str) -> list[Recommendation]:
        """Parse AI-generated recommendations."""
        recommendations = []

        try:
            lines = response_text.strip().split("\n")

            for line in lines:
                line = line.strip()
                if "|" in line:  # Structured format
                    parts = [p.strip() for p in line.split("|")]
                    if len(parts) >= 5:
                        recommendations.append(
                            Recommendation(
                                category="Strategic",
                                title=parts[0],
                                description=parts[1],
                                priority="medium",  # Default for AI recommendations
                                impact_estimate=parts[2],
                                effort_required=parts[3],
                                timeline=parts[4],
                            )
                        )
                elif line.startswith(("1.", "2.", "3.", "-", "•")):  # List format
                    # Extract recommendation from list item
                    rec_text = line.split(".", 1)[-1].split("-", 1)[-1].strip()
                    if len(rec_text) > 20:  # Meaningful recommendation
                        recommendations.append(
                            Recommendation(
                                category="Strategic",
                                title=rec_text[:50] + "..."
                                if len(rec_text) > 50
                                else rec_text,
                                description=rec_text,
                                priority="medium",
                                impact_estimate="Moderate improvement expected",
                                effort_required="Medium",
                                timeline="2-4 weeks",
                            )
                        )

            return recommendations[:3]  # Limit to 3 AI recommendations

        except Exception as e:
            logger.error(f"AI recommendation parsing failed: {e}")
            return []

    def _initialize_recommendation_rules(self) -> dict[str, Any]:
        """Initialize recommendation generation rules and templates."""
        return {
            "efficiency_thresholds": {
                "cpc_high": 5.0,  # CPC above $5 is high
                "ctr_low": 1.0,  # CTR below 1% is low
                "conversion_rate_low": 0.5,  # Conv rate below 0.5% is low
            },
            "performance_benchmarks": {
                "pages_per_session_good": 2.0,
                "session_share_significant": 10.0,
                "conversion_rate_good": 2.0,
            },
            "priority_rules": {
                "high_impact_low_effort": "high",
                "high_impact_high_effort": "medium",
                "low_impact_low_effort": "low",
            },
        }
