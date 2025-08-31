"""ROI optimization engine for marketing campaigns."""

import logging
from datetime import datetime
from enum import Enum
from typing import Any

import numpy as np
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class OptimizationObjective(str, Enum):
    """ROI optimization objectives."""

    MAXIMIZE_TOTAL_ROI = "maximize_total_roi"
    MAXIMIZE_INCREMENTAL_ROI = "maximize_incremental_roi"
    MINIMIZE_PAYBACK_PERIOD = "minimize_payback_period"
    MAXIMIZE_LIFETIME_VALUE = "maximize_lifetime_value"
    OPTIMIZE_MARGINAL_ROI = "optimize_marginal_roi"
    MAXIMIZE_PROFIT_MARGIN = "maximize_profit_margin"


class ROIAnalysis(BaseModel):
    """ROI analysis results."""

    # Current performance
    current_roi: float = Field(description="Current ROI ratio")
    current_roas: float = Field(description="Current ROAS")
    current_profit_margin: float = Field(description="Current profit margin")

    # ROI components
    total_investment: float = Field(description="Total marketing investment")
    total_revenue: float = Field(description="Total revenue generated")
    total_profit: float = Field(description="Total profit")

    # Efficiency metrics
    cost_per_acquisition: float = Field(description="Cost per acquisition")
    customer_lifetime_value: float = Field(description="Customer lifetime value")
    payback_period_days: int = Field(description="Payback period in days")

    # Marginal analysis
    marginal_roi: float = Field(description="Marginal ROI from last investment")
    roi_trend: str = Field(description="ROI trend direction")
    efficiency_score: float = Field(description="Overall efficiency score")

    # Benchmarks
    industry_benchmark_roi: float = Field(description="Industry benchmark ROI")
    competitive_position: str = Field(description="Competitive position")


class ROIOptimization(BaseModel):
    """ROI optimization recommendation."""

    optimization_id: str = Field(description="Unique optimization ID")
    objective: OptimizationObjective = Field(description="Optimization objective")
    priority: str = Field(description="Optimization priority")

    # Current vs optimized
    current_metrics: ROIAnalysis = Field(description="Current ROI metrics")
    optimized_metrics: ROIAnalysis = Field(description="Projected optimized metrics")

    # Improvement potential
    roi_improvement: float = Field(description="Expected ROI improvement")
    revenue_uplift: float = Field(description="Expected revenue uplift")
    cost_efficiency_gain: float = Field(description="Cost efficiency improvement")

    # Optimization strategies
    recommended_actions: list[dict[str, Any]] = Field(
        description="Specific optimization actions"
    )
    budget_reallocation: dict[str, float] = Field(
        description="Budget reallocation plan"
    )
    timeline: str = Field(description="Implementation timeline")

    # Risk and validation
    confidence_level: float = Field(description="Confidence in optimization")
    risk_factors: list[str] = Field(description="Risk factors")
    success_metrics: list[str] = Field(description="Success measurement metrics")

    # Supporting analysis
    marginal_analysis: dict[str, Any] = Field(description="Marginal ROI analysis")
    sensitivity_analysis: dict[str, float] = Field(
        description="Sensitivity analysis results"
    )

    created_at: datetime = Field(default_factory=datetime.now)


class ROIOptimizer:
    """ROI optimization engine."""

    def __init__(self):
        self.optimization_history = []
        self.benchmark_data = {}

    def analyze_current_roi(
        self,
        campaign_data: list[dict[str, Any]],
        cost_data: list[dict[str, Any]],
        revenue_data: list[dict[str, Any]],
    ) -> ROIAnalysis:
        """Analyze current ROI performance."""

        logger.info("Analyzing current ROI performance")

        try:
            # Calculate basic ROI metrics
            total_cost = sum(record.get("spend", 0) for record in cost_data)
            total_revenue = sum(record.get("revenue", 0) for record in revenue_data)
            total_profit = total_revenue - total_cost

            # Calculate ratios
            current_roi = (total_profit / total_cost) if total_cost > 0 else 0
            current_roas = (total_revenue / total_cost) if total_cost > 0 else 0
            profit_margin = (total_profit / total_revenue) if total_revenue > 0 else 0

            # Calculate customer metrics
            total_acquisitions = sum(
                record.get("conversions", 0) for record in campaign_data
            )
            cpa = total_cost / total_acquisitions if total_acquisitions > 0 else 0

            # Estimate CLV (simplified)
            avg_order_value = (
                total_revenue / total_acquisitions if total_acquisitions > 0 else 0
            )
            clv = avg_order_value * 3  # Simplified 3x multiplier

            # Calculate payback period
            monthly_revenue_per_customer = (
                avg_order_value * 0.3
            )  # Simplified assumption
            payback_days = (
                int(cpa / (monthly_revenue_per_customer / 30))
                if monthly_revenue_per_customer > 0
                else 365
            )

            # Marginal ROI analysis
            marginal_roi = self._calculate_marginal_roi(cost_data, revenue_data)
            roi_trend = self._analyze_roi_trend(cost_data, revenue_data)

            # Efficiency score
            efficiency_score = self._calculate_efficiency_score(
                current_roi, current_roas, cpa, clv, payback_days
            )

            return ROIAnalysis(
                current_roi=current_roi,
                current_roas=current_roas,
                current_profit_margin=profit_margin,
                total_investment=total_cost,
                total_revenue=total_revenue,
                total_profit=total_profit,
                cost_per_acquisition=cpa,
                customer_lifetime_value=clv,
                payback_period_days=payback_days,
                marginal_roi=marginal_roi,
                roi_trend=roi_trend,
                efficiency_score=efficiency_score,
                industry_benchmark_roi=3.5,  # Placeholder benchmark
                competitive_position=self._assess_competitive_position(current_roi),
            )

        except Exception as e:
            logger.error(f"ROI analysis failed: {str(e)}")
            raise

    def optimize_roi(
        self,
        current_analysis: ROIAnalysis,
        campaign_performance: list[dict[str, Any]],
        objective: OptimizationObjective = OptimizationObjective.MAXIMIZE_TOTAL_ROI,
    ) -> ROIOptimization:
        """Generate comprehensive ROI optimization recommendations."""

        logger.info(f"Optimizing ROI with objective: {objective}")

        try:
            # Analyze optimization opportunities
            opportunities = self._identify_roi_opportunities(
                current_analysis, campaign_performance
            )

            # Generate optimization strategies based on objective
            strategies = self._generate_optimization_strategies(
                current_analysis, opportunities, objective
            )

            # Project optimized metrics
            optimized_metrics = self._project_optimized_metrics(
                current_analysis, strategies
            )

            # Calculate improvement potential
            improvements = self._calculate_improvements(
                current_analysis, optimized_metrics
            )

            # Generate specific actions
            actions = self._generate_optimization_actions(strategies, opportunities)

            # Create budget reallocation plan
            budget_plan = self._create_budget_reallocation_plan(
                strategies, current_analysis
            )

            # Risk assessment
            risks = self._assess_optimization_risks(strategies, current_analysis)

            # Marginal and sensitivity analysis
            marginal_analysis = self._perform_marginal_analysis(
                current_analysis, strategies
            )
            sensitivity_analysis = self._perform_sensitivity_analysis(
                current_analysis, strategies
            )

            return ROIOptimization(
                optimization_id=f"roi_opt_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                objective=objective,
                priority=self._determine_optimization_priority(improvements),
                current_metrics=current_analysis,
                optimized_metrics=optimized_metrics,
                roi_improvement=improvements["roi_improvement"],
                revenue_uplift=improvements["revenue_uplift"],
                cost_efficiency_gain=improvements["efficiency_gain"],
                recommended_actions=actions,
                budget_reallocation=budget_plan,
                timeline=self._estimate_optimization_timeline(strategies),
                confidence_level=self._calculate_optimization_confidence(
                    strategies, current_analysis
                ),
                risk_factors=risks,
                success_metrics=self._define_success_metrics(objective, improvements),
                marginal_analysis=marginal_analysis,
                sensitivity_analysis=sensitivity_analysis,
            )

        except Exception as e:
            logger.error(f"ROI optimization failed: {str(e)}")
            raise

    def _calculate_marginal_roi(
        self, cost_data: list[dict[str, Any]], revenue_data: list[dict[str, Any]]
    ) -> float:
        """Calculate marginal ROI from recent investments."""

        if len(cost_data) < 2 or len(revenue_data) < 2:
            return 0.0

        # Use last 30% of data as "marginal"
        split_point = int(len(cost_data) * 0.7)

        marginal_costs = cost_data[split_point:]
        marginal_revenues = revenue_data[split_point:]

        marginal_cost_total = sum(record.get("spend", 0) for record in marginal_costs)
        marginal_revenue_total = sum(
            record.get("revenue", 0) for record in marginal_revenues
        )

        marginal_profit = marginal_revenue_total - marginal_cost_total

        return (marginal_profit / marginal_cost_total) if marginal_cost_total > 0 else 0

    def _analyze_roi_trend(
        self, cost_data: list[dict[str, Any]], revenue_data: list[dict[str, Any]]
    ) -> str:
        """Analyze ROI trend over time."""

        if len(cost_data) < 7:
            return "insufficient_data"

        # Calculate ROI for recent periods
        roi_values = []
        window_size = max(1, len(cost_data) // 5)

        for i in range(0, len(cost_data), window_size):
            window_costs = cost_data[i : i + window_size]
            window_revenues = revenue_data[i : i + window_size]

            cost_sum = sum(r.get("spend", 0) for r in window_costs)
            revenue_sum = sum(r.get("revenue", 0) for r in window_revenues)

            if cost_sum > 0:
                roi = (revenue_sum - cost_sum) / cost_sum
                roi_values.append(roi)

        if len(roi_values) < 2:
            return "stable"

        # Simple trend analysis
        recent_roi = np.mean(roi_values[-2:])
        earlier_roi = np.mean(roi_values[:-2])

        if recent_roi > earlier_roi * 1.1:
            return "improving"
        elif recent_roi < earlier_roi * 0.9:
            return "declining"
        else:
            return "stable"

    def _calculate_efficiency_score(
        self, roi: float, roas: float, cpa: float, clv: float, payback_days: int
    ) -> float:
        """Calculate overall efficiency score."""

        # Normalize metrics (simplified scoring)
        roi_score = min(1.0, max(0, roi / 5.0))  # ROI of 5 = perfect score
        roas_score = min(1.0, max(0, roas / 8.0))  # ROAS of 8 = perfect score

        # CPA efficiency (lower is better)
        cpa_score = min(1.0, max(0, (100 - cpa) / 100)) if cpa <= 100 else 0

        # CLV efficiency
        clv_score = min(1.0, max(0, clv / 500)) if clv <= 500 else 1.0

        # Payback period efficiency (shorter is better)
        payback_score = (
            min(1.0, max(0, (365 - payback_days) / 365)) if payback_days <= 365 else 0
        )

        # Weighted average
        efficiency_score = (
            roi_score * 0.3
            + roas_score * 0.25
            + cpa_score * 0.2
            + clv_score * 0.15
            + payback_score * 0.1
        )

        return efficiency_score

    def _assess_competitive_position(self, current_roi: float) -> str:
        """Assess competitive position based on ROI."""

        if current_roi > 4.0:
            return "leader"
        elif current_roi > 2.5:
            return "above_average"
        elif current_roi > 1.0:
            return "average"
        elif current_roi > 0:
            return "below_average"
        else:
            return "poor"

    def _identify_roi_opportunities(
        self, current_analysis: ROIAnalysis, campaign_performance: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Identify ROI optimization opportunities."""

        opportunities = []

        # Low ROI opportunity
        if current_analysis.current_roi < 2.0:
            opportunities.append(
                {
                    "type": "low_roi",
                    "severity": "high",
                    "description": "Overall ROI below industry average",
                    "potential_impact": "high",
                    "strategies": [
                        "cost_reduction",
                        "revenue_optimization",
                        "efficiency_improvement",
                    ],
                }
            )

        # Long payback period
        if current_analysis.payback_period_days > 180:
            opportunities.append(
                {
                    "type": "long_payback",
                    "severity": "medium",
                    "description": "Payback period exceeds 6 months",
                    "potential_impact": "medium",
                    "strategies": ["accelerate_conversion", "improve_retention"],
                }
            )

        # Poor marginal ROI
        if current_analysis.marginal_roi < current_analysis.current_roi * 0.5:
            opportunities.append(
                {
                    "type": "declining_marginal_roi",
                    "severity": "high",
                    "description": "Marginal ROI significantly below average ROI",
                    "potential_impact": "high",
                    "strategies": ["budget_reallocation", "channel_optimization"],
                }
            )

        # Low efficiency score
        if current_analysis.efficiency_score < 0.6:
            opportunities.append(
                {
                    "type": "low_efficiency",
                    "severity": "medium",
                    "description": "Overall efficiency score below target",
                    "potential_impact": "medium",
                    "strategies": [
                        "process_optimization",
                        "automation",
                        "targeting_improvement",
                    ],
                }
            )

        # Channel-specific opportunities
        channel_analysis = self._analyze_channel_roi(campaign_performance)
        for channel, metrics in channel_analysis.items():
            if metrics["roi"] < current_analysis.current_roi * 0.7:
                opportunities.append(
                    {
                        "type": "underperforming_channel",
                        "channel": channel,
                        "severity": "medium",
                        "description": f"Channel {channel} underperforming vs portfolio average",
                        "potential_impact": "medium",
                        "strategies": ["channel_optimization", "budget_shift"],
                    }
                )

        return opportunities

    def _analyze_channel_roi(
        self, campaign_performance: list[dict[str, Any]]
    ) -> dict[str, dict[str, float]]:
        """Analyze ROI by channel."""

        channel_metrics = {}

        for record in campaign_performance:
            channel = record.get("channel", "unknown")
            spend = record.get("spend", 0)
            revenue = record.get("revenue", 0)

            if channel not in channel_metrics:
                channel_metrics[channel] = {"spend": 0, "revenue": 0}

            channel_metrics[channel]["spend"] += spend
            channel_metrics[channel]["revenue"] += revenue

        # Calculate ROI for each channel
        channel_roi = {}
        for channel, metrics in channel_metrics.items():
            if metrics["spend"] > 0:
                profit = metrics["revenue"] - metrics["spend"]
                roi = profit / metrics["spend"]
                channel_roi[channel] = {
                    "roi": roi,
                    "spend": metrics["spend"],
                    "revenue": metrics["revenue"],
                }

        return channel_roi

    def _generate_optimization_strategies(
        self,
        current_analysis: ROIAnalysis,
        opportunities: list[dict[str, Any]],
        objective: OptimizationObjective,
    ) -> list[dict[str, Any]]:
        """Generate optimization strategies based on objective and opportunities."""

        strategies = []

        if objective == OptimizationObjective.MAXIMIZE_TOTAL_ROI:
            # Focus on highest ROI activities
            strategies.extend(
                [
                    {
                        "strategy": "scale_high_roi_channels",
                        "description": "Increase investment in highest performing channels",
                        "impact": "high",
                        "implementation": "budget_reallocation",
                    },
                    {
                        "strategy": "optimize_underperformers",
                        "description": "Improve or reduce investment in low ROI channels",
                        "impact": "medium",
                        "implementation": "channel_optimization",
                    },
                ]
            )

        elif objective == OptimizationObjective.MINIMIZE_PAYBACK_PERIOD:
            strategies.extend(
                [
                    {
                        "strategy": "accelerate_conversions",
                        "description": "Focus on channels with fastest conversion cycles",
                        "impact": "high",
                        "implementation": "channel_prioritization",
                    },
                    {
                        "strategy": "improve_conversion_rate",
                        "description": "Optimize landing pages and conversion funnels",
                        "impact": "medium",
                        "implementation": "conversion_optimization",
                    },
                ]
            )

        elif objective == OptimizationObjective.MAXIMIZE_LIFETIME_VALUE:
            strategies.extend(
                [
                    {
                        "strategy": "focus_high_ltv_segments",
                        "description": "Target customer segments with highest lifetime value",
                        "impact": "high",
                        "implementation": "audience_optimization",
                    },
                    {
                        "strategy": "retention_optimization",
                        "description": "Invest in customer retention and repeat purchase campaigns",
                        "impact": "medium",
                        "implementation": "retention_programs",
                    },
                ]
            )

        elif objective == OptimizationObjective.OPTIMIZE_MARGINAL_ROI:
            strategies.extend(
                [
                    {
                        "strategy": "marginal_efficiency_optimization",
                        "description": "Optimize budget allocation based on marginal returns",
                        "impact": "high",
                        "implementation": "dynamic_budgeting",
                    },
                    {
                        "strategy": "diminishing_returns_management",
                        "description": "Identify and address diminishing returns in channels",
                        "impact": "medium",
                        "implementation": "spending_caps",
                    },
                ]
            )

        # Add opportunity-specific strategies
        for opp in opportunities:
            if opp["type"] == "low_roi":
                strategies.append(
                    {
                        "strategy": "comprehensive_roi_improvement",
                        "description": "Multi-faceted approach to improve overall ROI",
                        "impact": "high",
                        "implementation": "comprehensive_optimization",
                    }
                )

        return strategies

    def _project_optimized_metrics(
        self, current_analysis: ROIAnalysis, strategies: list[dict[str, Any]]
    ) -> ROIAnalysis:
        """Project metrics after optimization implementation."""

        # Calculate improvement factors based on strategies
        roi_multiplier = 1.0
        cost_reduction_factor = 1.0
        revenue_improvement_factor = 1.0

        for strategy in strategies:
            if strategy["impact"] == "high":
                roi_multiplier *= 1.25
                if "cost_reduction" in strategy.get("implementation", ""):
                    cost_reduction_factor *= 0.9
                if "revenue" in strategy.get("implementation", ""):
                    revenue_improvement_factor *= 1.15
            elif strategy["impact"] == "medium":
                roi_multiplier *= 1.15
                cost_reduction_factor *= 0.95
                revenue_improvement_factor *= 1.08

        # Cap improvements to realistic levels
        roi_multiplier = min(roi_multiplier, 2.0)
        cost_reduction_factor = max(cost_reduction_factor, 0.7)
        revenue_improvement_factor = min(revenue_improvement_factor, 1.5)

        # Project optimized metrics
        optimized_investment = current_analysis.total_investment * cost_reduction_factor
        optimized_revenue = current_analysis.total_revenue * revenue_improvement_factor
        optimized_profit = optimized_revenue - optimized_investment

        optimized_roi = (
            (optimized_profit / optimized_investment) if optimized_investment > 0 else 0
        )
        optimized_roas = (
            (optimized_revenue / optimized_investment)
            if optimized_investment > 0
            else 0
        )
        optimized_profit_margin = (
            (optimized_profit / optimized_revenue) if optimized_revenue > 0 else 0
        )

        # Estimate other optimized metrics
        optimized_cpa = current_analysis.cost_per_acquisition * cost_reduction_factor
        optimized_clv = (
            current_analysis.customer_lifetime_value * revenue_improvement_factor
        )
        optimized_payback_days = int(
            current_analysis.payback_period_days * cost_reduction_factor
        )

        optimized_efficiency = self._calculate_efficiency_score(
            optimized_roi,
            optimized_roas,
            optimized_cpa,
            optimized_clv,
            optimized_payback_days,
        )

        return ROIAnalysis(
            current_roi=optimized_roi,
            current_roas=optimized_roas,
            current_profit_margin=optimized_profit_margin,
            total_investment=optimized_investment,
            total_revenue=optimized_revenue,
            total_profit=optimized_profit,
            cost_per_acquisition=optimized_cpa,
            customer_lifetime_value=optimized_clv,
            payback_period_days=optimized_payback_days,
            marginal_roi=optimized_roi
            * 0.9,  # Assume marginal ROI improves but remains below average
            roi_trend="improving",
            efficiency_score=optimized_efficiency,
            industry_benchmark_roi=current_analysis.industry_benchmark_roi,
            competitive_position=self._assess_competitive_position(optimized_roi),
        )

    def _calculate_improvements(
        self, current_analysis: ROIAnalysis, optimized_metrics: ROIAnalysis
    ) -> dict[str, float]:
        """Calculate improvement metrics."""

        roi_improvement = (
            (
                (optimized_metrics.current_roi - current_analysis.current_roi)
                / current_analysis.current_roi
            )
            if current_analysis.current_roi != 0
            else 0
        )

        revenue_uplift = (
            (
                (optimized_metrics.total_revenue - current_analysis.total_revenue)
                / current_analysis.total_revenue
            )
            if current_analysis.total_revenue != 0
            else 0
        )

        efficiency_gain = (
            (
                (optimized_metrics.efficiency_score - current_analysis.efficiency_score)
                / current_analysis.efficiency_score
            )
            if current_analysis.efficiency_score != 0
            else 0
        )

        return {
            "roi_improvement": roi_improvement,
            "revenue_uplift": revenue_uplift,
            "efficiency_gain": efficiency_gain,
        }

    def _generate_optimization_actions(
        self, strategies: list[dict[str, Any]], opportunities: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Generate specific optimization actions."""

        actions = []

        for strategy in strategies:
            if strategy["strategy"] == "scale_high_roi_channels":
                actions.append(
                    {
                        "action": "Increase budget allocation to top 3 performing channels by 30%",
                        "priority": "high",
                        "timeline": "1-2 weeks",
                        "expected_impact": "ROI improvement of 20-25%",
                        "requirements": [
                            "Performance analysis",
                            "Budget approval",
                            "Campaign setup",
                        ],
                    }
                )

            elif strategy["strategy"] == "optimize_underperformers":
                actions.append(
                    {
                        "action": "Reduce budget for channels with ROI < 1.5 by 40%",
                        "priority": "high",
                        "timeline": "1 week",
                        "expected_impact": "Cost reduction of 15-20%",
                        "requirements": [
                            "Channel analysis",
                            "Budget reallocation plan",
                        ],
                    }
                )

            elif strategy["strategy"] == "accelerate_conversions":
                actions.append(
                    {
                        "action": "Optimize conversion funnels and reduce friction",
                        "priority": "medium",
                        "timeline": "3-4 weeks",
                        "expected_impact": "Payback period reduction of 20-30%",
                        "requirements": [
                            "UX analysis",
                            "A/B testing",
                            "Development resources",
                        ],
                    }
                )

            elif strategy["strategy"] == "focus_high_ltv_segments":
                actions.append(
                    {
                        "action": "Shift 25% of budget to high-LTV customer segments",
                        "priority": "medium",
                        "timeline": "2-3 weeks",
                        "expected_impact": "LTV improvement of 15-20%",
                        "requirements": [
                            "Customer segmentation",
                            "Audience analysis",
                            "Campaign restructure",
                        ],
                    }
                )

        return actions

    def _create_budget_reallocation_plan(
        self, strategies: list[dict[str, Any]], current_analysis: ROIAnalysis
    ) -> dict[str, float]:
        """Create budget reallocation plan."""

        # Simplified budget reallocation based on strategies
        reallocation_plan = {
            "high_roi_channels": 0.4,  # 40% to high ROI channels
            "medium_roi_channels": 0.35,  # 35% to medium ROI channels
            "testing_budget": 0.15,  # 15% for testing and optimization
            "emergency_reserve": 0.1,  # 10% reserve
        }

        # Adjust based on strategies
        for strategy in strategies:
            if "scale_high_roi" in strategy.get("strategy", ""):
                reallocation_plan["high_roi_channels"] += 0.1
                reallocation_plan["medium_roi_channels"] -= 0.05
                reallocation_plan["emergency_reserve"] -= 0.05

        return reallocation_plan

    def _assess_optimization_risks(
        self, strategies: list[dict[str, Any]], current_analysis: ROIAnalysis
    ) -> list[str]:
        """Assess risks associated with optimization strategies."""

        risks = []

        if current_analysis.current_roi < 1.0:
            risks.append("Current negative ROI increases risk of further losses")

        if len(strategies) > 3:
            risks.append(
                "Complex multi-strategy implementation may cause coordination issues"
            )

        for strategy in strategies:
            if "budget_reallocation" in strategy.get("implementation", ""):
                risks.append("Budget reallocation may temporarily disrupt performance")

            if strategy.get("impact") == "high":
                risks.append(
                    f"High-impact changes in {strategy['strategy']} carry execution risk"
                )

        return risks

    def _perform_marginal_analysis(
        self, current_analysis: ROIAnalysis, strategies: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Perform marginal ROI analysis."""

        return {
            "current_marginal_roi": current_analysis.marginal_roi,
            "optimal_marginal_roi": current_analysis.marginal_roi
            * 1.3,  # Projected improvement
            "marginal_efficiency_opportunities": [
                "Reallocate budget from saturated to unsaturated channels",
                "Implement dynamic bidding based on marginal returns",
                "Set spending caps at diminishing returns points",
            ],
            "marginal_roi_by_channel": {
                "search": current_analysis.marginal_roi * 1.2,
                "social": current_analysis.marginal_roi * 0.8,
                "display": current_analysis.marginal_roi * 0.9,
            },
        }

    def _perform_sensitivity_analysis(
        self, current_analysis: ROIAnalysis, strategies: list[dict[str, Any]]
    ) -> dict[str, float]:
        """Perform sensitivity analysis on key variables."""

        return {
            "roi_sensitivity_to_cost_reduction": 0.8,  # 10% cost reduction = 8% ROI improvement
            "roi_sensitivity_to_revenue_increase": 0.6,  # 10% revenue increase = 6% ROI improvement
            "payback_sensitivity_to_conversion_rate": -1.2,  # 10% CR increase = 12% payback reduction
            "ltv_sensitivity_to_retention": 1.5,  # 10% retention increase = 15% LTV improvement
            "overall_risk_sensitivity": 0.3,  # Overall sensitivity to market changes
        }

    def _determine_optimization_priority(self, improvements: dict[str, float]) -> str:
        """Determine optimization priority based on improvement potential."""

        roi_improvement = improvements.get("roi_improvement", 0)

        if roi_improvement > 0.5:  # >50% improvement
            return "critical"
        elif roi_improvement > 0.25:  # >25% improvement
            return "high"
        elif roi_improvement > 0.1:  # >10% improvement
            return "medium"
        else:
            return "low"

    def _estimate_optimization_timeline(self, strategies: list[dict[str, Any]]) -> str:
        """Estimate implementation timeline for optimization."""

        if len(strategies) <= 2:
            return "4-6 weeks"
        elif len(strategies) <= 4:
            return "6-8 weeks"
        else:
            return "8-12 weeks"

    def _calculate_optimization_confidence(
        self, strategies: list[dict[str, Any]], current_analysis: ROIAnalysis
    ) -> float:
        """Calculate confidence level in optimization recommendations."""

        # Base confidence
        confidence = 0.7

        # Adjust based on data quality
        if current_analysis.efficiency_score > 0.8:
            confidence += 0.1
        elif current_analysis.efficiency_score < 0.4:
            confidence -= 0.1

        # Adjust based on strategy complexity
        high_impact_strategies = sum(1 for s in strategies if s.get("impact") == "high")
        if high_impact_strategies <= 2:
            confidence += 0.1
        elif high_impact_strategies > 4:
            confidence -= 0.1

        return max(0.5, min(0.95, confidence))

    def _define_success_metrics(
        self, objective: OptimizationObjective, improvements: dict[str, float]
    ) -> list[str]:
        """Define success metrics based on optimization objective."""

        base_metrics = [
            f"ROI improvement of >{improvements['roi_improvement']*100:.1f}%",
            "Positive trend in efficiency score",
            "Maintained or increased conversion volume",
        ]

        if objective == OptimizationObjective.MAXIMIZE_TOTAL_ROI:
            base_metrics.extend(
                [
                    "Overall ROI above industry benchmark",
                    "Top 3 channels contributing >70% of profit",
                ]
            )
        elif objective == OptimizationObjective.MINIMIZE_PAYBACK_PERIOD:
            base_metrics.extend(
                [
                    "Payback period reduced by >20%",
                    "Faster time to profitability for new customers",
                ]
            )
        elif objective == OptimizationObjective.MAXIMIZE_LIFETIME_VALUE:
            base_metrics.extend(
                [
                    "Average customer LTV increased by >15%",
                    "Improved customer retention rates",
                ]
            )

        return base_metrics
