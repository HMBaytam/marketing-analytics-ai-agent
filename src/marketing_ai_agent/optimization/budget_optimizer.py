"""Budget allocation and optimization algorithms."""

import logging
from datetime import datetime
from enum import Enum
from typing import Any

import numpy as np
from pydantic import BaseModel, Field
from scipy.optimize import minimize

logger = logging.getLogger(__name__)


class AllocationStrategy(str, Enum):
    """Budget allocation strategies."""

    EQUAL_WEIGHT = "equal_weight"
    PERFORMANCE_BASED = "performance_based"
    ROI_MAXIMIZATION = "roi_maximization"
    RISK_ADJUSTED = "risk_adjusted"
    SEASONAL_ADJUSTED = "seasonal_adjusted"
    MARGINAL_EFFICIENCY = "marginal_efficiency"


class BudgetConstraints(BaseModel):
    """Budget allocation constraints."""

    # Total constraints
    total_budget: float = Field(description="Total available budget")
    min_budget_per_channel: float = Field(
        default=100, description="Minimum budget per channel"
    )
    max_budget_per_channel: float | None = Field(
        default=None, description="Maximum budget per channel"
    )

    # Relative constraints
    min_allocation_percentage: float = Field(
        default=0.05, description="Minimum allocation percentage"
    )
    max_allocation_percentage: float = Field(
        default=0.5, description="Maximum allocation percentage"
    )

    # Channel-specific constraints
    channel_constraints: dict[str, dict[str, float]] = Field(
        default_factory=dict, description="Channel-specific min/max constraints"
    )

    # Change constraints
    max_change_percentage: float = Field(
        default=0.3, description="Maximum change from current allocation"
    )
    preserve_top_performers: bool = Field(
        default=True, description="Protect top performing channels"
    )

    # Time constraints
    budget_period: str = Field(default="monthly", description="Budget period")
    allow_overspend: bool = Field(
        default=False, description="Allow temporary overspend"
    )
    overspend_limit: float = Field(
        default=0.1, description="Maximum overspend percentage"
    )


class ChannelAllocation(BaseModel):
    """Budget allocation for individual channel."""

    channel_id: str = Field(description="Channel identifier")
    channel_name: str = Field(description="Channel name")
    current_budget: float = Field(description="Current budget allocation")
    recommended_budget: float = Field(description="Recommended budget allocation")
    change_amount: float = Field(description="Budget change amount")
    change_percentage: float = Field(description="Budget change percentage")

    # Performance metrics
    current_roas: float = Field(description="Current ROAS")
    predicted_roas: float = Field(description="Predicted ROAS with new budget")
    efficiency_score: float = Field(description="Channel efficiency score")

    # Rationale
    allocation_rationale: str = Field(description="Reason for allocation change")
    risk_factors: list[str] = Field(description="Risk factors for this allocation")
    expected_outcomes: list[str] = Field(description="Expected outcomes")


class BudgetAllocation(BaseModel):
    """Complete budget allocation recommendation."""

    strategy: AllocationStrategy = Field(description="Allocation strategy used")
    total_budget: float = Field(description="Total budget being allocated")
    allocation_date: datetime = Field(default_factory=datetime.now)

    # Channel allocations
    channel_allocations: list[ChannelAllocation] = Field(
        description="Individual channel allocations"
    )

    # Overall metrics
    expected_total_roas: float = Field(description="Expected total ROAS")
    roas_improvement: float = Field(description="ROAS improvement vs current")
    budget_utilization: float = Field(description="Percentage of budget allocated")

    # Risk assessment
    overall_risk_score: float = Field(description="Overall allocation risk score")
    diversification_score: float = Field(description="Portfolio diversification score")

    # Implementation details
    implementation_phases: list[dict[str, Any]] = Field(
        description="Phased implementation plan"
    )
    monitoring_schedule: list[str] = Field(description="Monitoring checkpoints")
    reallocation_triggers: list[str] = Field(description="When to reallocate")


class BudgetOptimizer:
    """Budget allocation optimization engine."""

    def __init__(self, constraints: BudgetConstraints):
        self.constraints = constraints

    def optimize_budget_allocation(
        self,
        channels: list[dict[str, Any]],
        historical_performance: list[dict[str, Any]],
        strategy: AllocationStrategy = AllocationStrategy.ROI_MAXIMIZATION,
    ) -> BudgetAllocation:
        """Optimize budget allocation across channels."""

        logger.info(f"Optimizing budget allocation using {strategy} strategy")

        try:
            # Prepare channel data
            channel_data = self._prepare_channel_data(channels, historical_performance)

            # Apply optimization strategy
            if strategy == AllocationStrategy.EQUAL_WEIGHT:
                allocations = self._equal_weight_allocation(channel_data)
            elif strategy == AllocationStrategy.PERFORMANCE_BASED:
                allocations = self._performance_based_allocation(channel_data)
            elif strategy == AllocationStrategy.ROI_MAXIMIZATION:
                allocations = self._roi_maximization_allocation(channel_data)
            elif strategy == AllocationStrategy.RISK_ADJUSTED:
                allocations = self._risk_adjusted_allocation(channel_data)
            elif strategy == AllocationStrategy.SEASONAL_ADJUSTED:
                allocations = self._seasonal_adjusted_allocation(channel_data)
            elif strategy == AllocationStrategy.MARGINAL_EFFICIENCY:
                allocations = self._marginal_efficiency_allocation(channel_data)
            else:
                # Default to performance-based
                allocations = self._performance_based_allocation(channel_data)

            # Apply constraints
            constrained_allocations = self._apply_constraints(allocations, channel_data)

            # Create channel allocation objects
            channel_allocations = self._create_channel_allocations(
                constrained_allocations, channel_data
            )

            # Calculate overall metrics
            overall_metrics = self._calculate_overall_metrics(channel_allocations)

            # Generate implementation plan
            implementation_plan = self._generate_implementation_plan(
                channel_allocations
            )

            return BudgetAllocation(
                strategy=strategy,
                total_budget=self.constraints.total_budget,
                channel_allocations=channel_allocations,
                expected_total_roas=overall_metrics["expected_roas"],
                roas_improvement=overall_metrics["roas_improvement"],
                budget_utilization=overall_metrics["budget_utilization"],
                overall_risk_score=overall_metrics["risk_score"],
                diversification_score=overall_metrics["diversification_score"],
                implementation_phases=implementation_plan["phases"],
                monitoring_schedule=implementation_plan["monitoring"],
                reallocation_triggers=implementation_plan["triggers"],
            )

        except Exception as e:
            logger.error(f"Budget optimization failed: {str(e)}")
            raise

    def _prepare_channel_data(
        self,
        channels: list[dict[str, Any]],
        historical_performance: list[dict[str, Any]],
    ) -> dict[str, dict[str, Any]]:
        """Prepare and enrich channel data with performance metrics."""

        channel_data = {}

        for channel in channels:
            channel_id = channel.get("id", channel.get("name", "unknown"))

            # Get historical performance for this channel
            channel_performance = [
                perf
                for perf in historical_performance
                if perf.get("channel_id") == channel_id
                or perf.get("channel") == channel_id
            ]

            # Calculate performance metrics
            metrics = self._calculate_channel_metrics(channel_performance)

            channel_data[channel_id] = {
                "id": channel_id,
                "name": channel.get("name", channel_id),
                "current_budget": channel.get("budget", 0),
                "current_spend": channel.get("spend", 0),
                "performance_metrics": metrics,
                "channel_info": channel,
            }

        return channel_data

    def _calculate_channel_metrics(
        self, performance_data: list[dict[str, Any]]
    ) -> dict[str, float]:
        """Calculate channel performance metrics from historical data."""

        if not performance_data:
            return {
                "roas": 1.0,
                "cpa": 50.0,
                "conversion_rate": 0.01,
                "efficiency_score": 0.5,
                "volatility": 0.5,
                "trend_slope": 0.0,
            }

        # Calculate basic metrics
        total_spend = sum(record.get("spend", 0) for record in performance_data)
        total_revenue = sum(record.get("revenue", 0) for record in performance_data)
        total_conversions = sum(
            record.get("conversions", 0) for record in performance_data
        )
        total_clicks = sum(record.get("clicks", 0) for record in performance_data)

        # Calculate derived metrics
        roas = total_revenue / total_spend if total_spend > 0 else 1.0
        cpa = total_spend / total_conversions if total_conversions > 0 else 50.0
        conversion_rate = total_conversions / total_clicks if total_clicks > 0 else 0.01

        # Calculate efficiency score (normalized combination of metrics)
        efficiency_score = min(
            1.0, (roas / 4.0) * (conversion_rate / 0.02) * (50.0 / cpa)
        )

        # Calculate volatility (standard deviation of ROAS)
        roas_values = [
            record.get("revenue", 0) / record.get("spend", 1)
            for record in performance_data
            if record.get("spend", 0) > 0
        ]
        volatility = np.std(roas_values) if len(roas_values) > 1 else 0.5

        # Calculate trend (simplified linear trend)
        if len(performance_data) > 2:
            x = np.arange(len(performance_data))
            y = [record.get("roas", roas) for record in performance_data]
            trend_slope = np.polyfit(x, y, 1)[0] if len(y) > 1 else 0.0
        else:
            trend_slope = 0.0

        return {
            "roas": roas,
            "cpa": cpa,
            "conversion_rate": conversion_rate,
            "efficiency_score": efficiency_score,
            "volatility": volatility,
            "trend_slope": trend_slope,
        }

    def _equal_weight_allocation(
        self, channel_data: dict[str, dict[str, Any]]
    ) -> dict[str, float]:
        """Allocate budget equally across all channels."""

        n_channels = len(channel_data)
        if n_channels == 0:
            return {}

        allocation_per_channel = self.constraints.total_budget / n_channels

        return {
            channel_id: allocation_per_channel for channel_id in channel_data.keys()
        }

    def _performance_based_allocation(
        self, channel_data: dict[str, dict[str, Any]]
    ) -> dict[str, float]:
        """Allocate budget based on performance scores."""

        # Calculate performance weights
        performance_scores = {}
        total_score = 0

        for channel_id, data in channel_data.items():
            metrics = data["performance_metrics"]
            # Combine multiple performance factors
            score = (
                metrics["roas"] * 0.4
                + metrics["efficiency_score"] * 0.3
                + (1 / max(metrics["volatility"], 0.1)) * 0.2
                + max(0, metrics["trend_slope"]) * 0.1
            )
            performance_scores[channel_id] = score
            total_score += score

        # Allocate budget proportionally
        allocations = {}
        for channel_id, score in performance_scores.items():
            weight = score / total_score if total_score > 0 else 1 / len(channel_data)
            allocations[channel_id] = self.constraints.total_budget * weight

        return allocations

    def _roi_maximization_allocation(
        self, channel_data: dict[str, dict[str, Any]]
    ) -> dict[str, float]:
        """Optimize allocation to maximize total ROI using mathematical optimization."""

        channel_ids = list(channel_data.keys())
        n_channels = len(channel_ids)

        if n_channels == 0:
            return {}

        # Extract ROAS values for each channel
        roas_values = np.array(
            [
                channel_data[channel_id]["performance_metrics"]["roas"]
                for channel_id in channel_ids
            ]
        )

        # Define objective function (negative because we minimize)
        def objective(allocation):
            return -np.sum(allocation * roas_values)

        # Define constraints
        constraints = [
            {
                "type": "eq",
                "fun": lambda x: np.sum(x) - self.constraints.total_budget,
            }  # Budget constraint
        ]

        # Bounds for each channel
        bounds = []
        for _i, _channel_id in enumerate(channel_ids):
            min_budget = max(
                self.constraints.min_budget_per_channel,
                self.constraints.total_budget
                * self.constraints.min_allocation_percentage,
            )
            max_budget = min(
                self.constraints.max_budget_per_channel or float("inf"),
                self.constraints.total_budget
                * self.constraints.max_allocation_percentage,
            )
            bounds.append((min_budget, max_budget))

        # Initial guess (equal allocation)
        initial_allocation = np.full(
            n_channels, self.constraints.total_budget / n_channels
        )

        # Optimize
        result = minimize(
            objective,
            initial_allocation,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
        )

        if result.success:
            optimized_allocation = result.x
        else:
            logger.warning(
                "Optimization failed, falling back to performance-based allocation"
            )
            return self._performance_based_allocation(channel_data)

        return {channel_ids[i]: optimized_allocation[i] for i in range(n_channels)}

    def _risk_adjusted_allocation(
        self, channel_data: dict[str, dict[str, Any]]
    ) -> dict[str, float]:
        """Allocate budget with risk adjustment (Markowitz-style optimization)."""

        channel_ids = list(channel_data.keys())
        n_channels = len(channel_ids)

        # Extract returns (ROAS) and risks (volatility)
        returns = np.array(
            [
                channel_data[channel_id]["performance_metrics"]["roas"]
                for channel_id in channel_ids
            ]
        )

        risks = np.array(
            [
                channel_data[channel_id]["performance_metrics"]["volatility"]
                for channel_id in channel_ids
            ]
        )

        # Risk-adjusted scores
        risk_adjusted_scores = returns / (1 + risks)

        # Allocate based on risk-adjusted scores
        total_score = np.sum(risk_adjusted_scores)

        allocations = {}
        for i, channel_id in enumerate(channel_ids):
            weight = (
                risk_adjusted_scores[i] / total_score
                if total_score > 0
                else 1 / n_channels
            )
            allocations[channel_id] = self.constraints.total_budget * weight

        return allocations

    def _seasonal_adjusted_allocation(
        self, channel_data: dict[str, dict[str, Any]]
    ) -> dict[str, float]:
        """Allocate budget with seasonal adjustments."""

        # Get current season/month
        current_month = datetime.now().month
        current_quarter = (current_month - 1) // 3 + 1

        # Seasonal multipliers (simplified example)
        seasonal_multipliers = {
            1: {"search": 1.2, "social": 0.8, "display": 0.9, "email": 1.1},  # Q1
            2: {"search": 1.0, "social": 1.1, "display": 1.0, "email": 1.0},  # Q2
            3: {"search": 0.9, "social": 1.2, "display": 1.1, "email": 0.9},  # Q3
            4: {"search": 1.3, "social": 1.0, "display": 1.2, "email": 1.2},  # Q4
        }

        # Start with performance-based allocation
        base_allocations = self._performance_based_allocation(channel_data)

        # Apply seasonal adjustments
        adjusted_allocations = {}
        total_adjusted = 0

        for channel_id, base_allocation in base_allocations.items():
            channel_type = self._infer_channel_type(channel_id)
            multiplier = seasonal_multipliers[current_quarter].get(channel_type, 1.0)
            adjusted_allocations[channel_id] = base_allocation * multiplier
            total_adjusted += adjusted_allocations[channel_id]

        # Normalize to maintain total budget
        if total_adjusted > 0:
            normalization_factor = self.constraints.total_budget / total_adjusted
            for channel_id in adjusted_allocations:
                adjusted_allocations[channel_id] *= normalization_factor

        return adjusted_allocations

    def _marginal_efficiency_allocation(
        self, channel_data: dict[str, dict[str, Any]]
    ) -> dict[str, float]:
        """Allocate budget based on marginal efficiency curves."""

        # Simplified marginal efficiency allocation
        # In practice, this would use more sophisticated marginal return curves

        channel_ids = list(channel_data.keys())
        current_allocations = {
            channel_id: data["current_budget"]
            for channel_id, data in channel_data.items()
        }

        # Calculate marginal efficiency (simplified as diminishing returns)
        marginal_efficiencies = {}
        for channel_id, data in channel_data.items():
            base_roas = data["performance_metrics"]["roas"]
            current_budget = current_allocations[channel_id]

            # Simplified diminishing returns curve
            marginal_efficiency = base_roas * (1 - (current_budget / 10000) ** 0.5)
            marginal_efficiencies[channel_id] = max(0.1, marginal_efficiency)

        # Allocate to highest marginal efficiency first
        remaining_budget = self.constraints.total_budget
        allocations = {channel_id: 0 for channel_id in channel_ids}

        # Iterative allocation (simplified)
        allocation_increment = remaining_budget / 100  # Allocate in small increments

        while remaining_budget > allocation_increment:
            # Find channel with highest marginal efficiency
            best_channel = max(marginal_efficiencies.items(), key=lambda x: x[1])[0]

            # Allocate increment
            allocations[best_channel] += allocation_increment
            remaining_budget -= allocation_increment

            # Update marginal efficiency (diminishing returns)
            current_allocation = allocations[best_channel]
            base_roas = channel_data[best_channel]["performance_metrics"]["roas"]
            marginal_efficiencies[best_channel] = base_roas * (
                1 - (current_allocation / 10000) ** 0.5
            )

        return allocations

    def _apply_constraints(
        self,
        initial_allocations: dict[str, float],
        channel_data: dict[str, dict[str, Any]],
    ) -> dict[str, float]:
        """Apply budget constraints to allocations."""

        constrained_allocations = initial_allocations.copy()

        # Apply minimum budget constraints
        for channel_id, allocation in constrained_allocations.items():
            min_budget = max(
                self.constraints.min_budget_per_channel,
                self.constraints.total_budget
                * self.constraints.min_allocation_percentage,
            )
            constrained_allocations[channel_id] = max(allocation, min_budget)

        # Apply maximum budget constraints
        for channel_id, allocation in constrained_allocations.items():
            max_budget = min(
                self.constraints.max_budget_per_channel or float("inf"),
                self.constraints.total_budget
                * self.constraints.max_allocation_percentage,
            )
            constrained_allocations[channel_id] = min(allocation, max_budget)

        # Apply change constraints
        if self.constraints.max_change_percentage < 1.0:
            for channel_id, allocation in constrained_allocations.items():
                current_budget = channel_data[channel_id]["current_budget"]
                if current_budget > 0:
                    max_change = current_budget * self.constraints.max_change_percentage
                    min_allowed = current_budget - max_change
                    max_allowed = current_budget + max_change
                    constrained_allocations[channel_id] = np.clip(
                        allocation, min_allowed, max_allowed
                    )

        # Ensure total doesn't exceed budget (proportional scaling)
        total_allocated = sum(constrained_allocations.values())
        if total_allocated > self.constraints.total_budget:
            scale_factor = self.constraints.total_budget / total_allocated
            for channel_id in constrained_allocations:
                constrained_allocations[channel_id] *= scale_factor

        return constrained_allocations

    def _create_channel_allocations(
        self, allocations: dict[str, float], channel_data: dict[str, dict[str, Any]]
    ) -> list[ChannelAllocation]:
        """Create ChannelAllocation objects from allocation data."""

        channel_allocations = []

        for channel_id, recommended_budget in allocations.items():
            data = channel_data[channel_id]
            current_budget = data["current_budget"]
            change_amount = recommended_budget - current_budget
            change_percentage = (
                (change_amount / current_budget * 100) if current_budget > 0 else 0
            )

            # Predict new ROAS (simplified)
            current_roas = data["performance_metrics"]["roas"]
            budget_ratio = (
                recommended_budget / current_budget if current_budget > 0 else 1
            )

            # Diminishing returns effect
            predicted_roas = current_roas * (budget_ratio**0.8)

            # Generate rationale
            rationale = self._generate_allocation_rationale(
                change_percentage, data["performance_metrics"]
            )

            channel_allocations.append(
                ChannelAllocation(
                    channel_id=channel_id,
                    channel_name=data["name"],
                    current_budget=current_budget,
                    recommended_budget=recommended_budget,
                    change_amount=change_amount,
                    change_percentage=change_percentage,
                    current_roas=current_roas,
                    predicted_roas=predicted_roas,
                    efficiency_score=data["performance_metrics"]["efficiency_score"],
                    allocation_rationale=rationale,
                    risk_factors=self._identify_risk_factors(change_percentage, data),
                    expected_outcomes=self._predict_outcomes(
                        change_percentage, predicted_roas
                    ),
                )
            )

        return channel_allocations

    def _calculate_overall_metrics(
        self, channel_allocations: list[ChannelAllocation]
    ) -> dict[str, float]:
        """Calculate overall portfolio metrics."""

        total_current_budget = sum(ch.current_budget for ch in channel_allocations)
        total_recommended_budget = sum(
            ch.recommended_budget for ch in channel_allocations
        )

        # Weighted average ROAS
        current_weighted_roas = (
            sum(ch.current_roas * ch.current_budget for ch in channel_allocations)
            / total_current_budget
            if total_current_budget > 0
            else 0
        )

        expected_weighted_roas = (
            sum(ch.predicted_roas * ch.recommended_budget for ch in channel_allocations)
            / total_recommended_budget
            if total_recommended_budget > 0
            else 0
        )

        roas_improvement = (
            ((expected_weighted_roas - current_weighted_roas) / current_weighted_roas)
            if current_weighted_roas > 0
            else 0
        )

        budget_utilization = total_recommended_budget / self.constraints.total_budget

        # Risk score (based on allocation changes)
        risk_score = np.mean(
            [abs(ch.change_percentage) / 100 for ch in channel_allocations]
        )

        # Diversification score (inverse of concentration)
        allocations = [ch.recommended_budget for ch in channel_allocations]
        total = sum(allocations)
        proportions = (
            [a / total for a in allocations] if total > 0 else [0] * len(allocations)
        )
        herfindahl_index = sum(p**2 for p in proportions)
        diversification_score = 1 - herfindahl_index

        return {
            "expected_roas": expected_weighted_roas,
            "roas_improvement": roas_improvement,
            "budget_utilization": budget_utilization,
            "risk_score": risk_score,
            "diversification_score": diversification_score,
        }

    def _generate_implementation_plan(
        self, channel_allocations: list[ChannelAllocation]
    ) -> dict[str, Any]:
        """Generate implementation plan for budget reallocation."""

        # Sort by change magnitude
        sorted_channels = sorted(
            channel_allocations, key=lambda x: abs(x.change_percentage), reverse=True
        )

        # Create phases based on change magnitude
        phases = []
        if any(abs(ch.change_percentage) > 20 for ch in sorted_channels):
            # Phase 1: Large changes (gradual implementation)
            large_changes = [
                ch for ch in sorted_channels if abs(ch.change_percentage) > 20
            ]
            phases.append(
                {
                    "phase": 1,
                    "description": "Large budget adjustments (gradual rollout)",
                    "channels": [ch.channel_id for ch in large_changes],
                    "duration": "2 weeks",
                    "implementation": "Implement 50% of change in week 1, remaining 50% in week 2",
                }
            )

        # Phase 2: Medium changes
        medium_changes = [
            ch for ch in sorted_channels if 5 <= abs(ch.change_percentage) <= 20
        ]
        if medium_changes:
            phases.append(
                {
                    "phase": 2,
                    "description": "Medium budget adjustments",
                    "channels": [ch.channel_id for ch in medium_changes],
                    "duration": "1 week",
                    "implementation": "Full implementation",
                }
            )

        # Phase 3: Small changes
        small_changes = [ch for ch in sorted_channels if abs(ch.change_percentage) < 5]
        if small_changes:
            phases.append(
                {
                    "phase": 3,
                    "description": "Minor budget adjustments",
                    "channels": [ch.channel_id for ch in small_changes],
                    "duration": "3 days",
                    "implementation": "Immediate implementation",
                }
            )

        monitoring_schedule = [
            "Daily monitoring for first week",
            "Weekly monitoring for first month",
            "Bi-weekly monitoring thereafter",
        ]

        reallocation_triggers = [
            "ROAS drops below 80% of predicted",
            "Cost per acquisition increases by >25%",
            "Channel performance deviates >30% from forecast",
            "External market conditions change significantly",
        ]

        return {
            "phases": phases,
            "monitoring": monitoring_schedule,
            "triggers": reallocation_triggers,
        }

    def _infer_channel_type(self, channel_id: str) -> str:
        """Infer channel type from ID for seasonal adjustments."""
        channel_lower = channel_id.lower()
        if "search" in channel_lower or "google" in channel_lower:
            return "search"
        elif (
            "social" in channel_lower
            or "facebook" in channel_lower
            or "instagram" in channel_lower
        ):
            return "social"
        elif "display" in channel_lower or "banner" in channel_lower:
            return "display"
        elif "email" in channel_lower:
            return "email"
        else:
            return "other"

    def _generate_allocation_rationale(
        self, change_percentage: float, performance_metrics: dict[str, float]
    ) -> str:
        """Generate rationale for budget allocation change."""

        if change_percentage > 20:
            return f"Significant budget increase due to high ROAS ({performance_metrics['roas']:.2f}) and efficiency"
        elif change_percentage > 5:
            return "Budget increase to capitalize on strong performance trends"
        elif change_percentage < -20:
            return f"Budget reduction due to poor efficiency (ROAS: {performance_metrics['roas']:.2f})"
        elif change_percentage < -5:
            return "Minor budget reduction to optimize overall portfolio performance"
        else:
            return "Budget maintained due to stable performance"

    def _identify_risk_factors(
        self, change_percentage: float, channel_data: dict[str, Any]
    ) -> list[str]:
        """Identify risk factors for allocation change."""

        risks = []
        volatility = channel_data["performance_metrics"]["volatility"]

        if abs(change_percentage) > 30:
            risks.append("Large budget change may cause performance instability")

        if volatility > 0.5:
            risks.append("High historical volatility increases uncertainty")

        if channel_data["performance_metrics"]["trend_slope"] < -0.1:
            risks.append("Declining performance trend")

        if change_percentage > 20:
            risks.append("Potential for increased competition and higher costs")

        return risks

    def _predict_outcomes(
        self, change_percentage: float, predicted_roas: float
    ) -> list[str]:
        """Predict outcomes from budget allocation change."""

        outcomes = []

        if change_percentage > 15:
            outcomes.append(f"Expected ROAS: {predicted_roas:.2f}")
            outcomes.append("Increased conversion volume")
            outcomes.append("Potential for market share growth")
        elif change_percentage > 5:
            outcomes.append(f"Modest ROAS improvement to {predicted_roas:.2f}")
            outcomes.append("Stable to improved performance")
        elif change_percentage < -15:
            outcomes.append("Reduced spend and conversion volume")
            outcomes.append("Improved efficiency in other channels")
        else:
            outcomes.append("Maintained performance levels")

        return outcomes
