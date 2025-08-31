"""Tests for optimization modules."""


import numpy as np
import pandas as pd
import pytest
from marketing_ai_agent.optimization.ab_testing_optimizer import ABTestingOptimizer
from marketing_ai_agent.optimization.budget_optimizer import BudgetOptimizer
from marketing_ai_agent.optimization.ml_optimizer import MLOptimizer
from marketing_ai_agent.optimization.recommendations_engine import RecommendationsEngine
from marketing_ai_agent.optimization.roi_optimizer import ROIOptimizer
from marketing_ai_agent.optimization.rule_based_optimizer import RuleBasedOptimizer


@pytest.fixture
def sample_campaign_data():
    """Sample campaign data for optimization testing."""
    return pd.DataFrame(
        {
            "campaign_id": [f"camp_{i}" for i in range(10)],
            "campaign_name": [f"Campaign {i}" for i in range(10)],
            "budget": np.random.uniform(1000, 10000, 10),
            "impressions": np.random.randint(10000, 100000, 10),
            "clicks": np.random.randint(500, 5000, 10),
            "cost": np.random.uniform(500, 8000, 10),
            "conversions": np.random.randint(10, 100, 10),
            "revenue": np.random.uniform(200, 5000, 10),
            "status": ["ENABLED"] * 8 + ["PAUSED"] * 2,
            "bid_strategy": np.random.choice(
                ["TARGET_CPA", "TARGET_ROAS", "MAXIMIZE_CLICKS"], 10
            ),
            "target_cpa": np.random.uniform(10, 100, 10),
            "target_roas": np.random.uniform(200, 800, 10),
        }
    )


@pytest.fixture
def sample_keyword_data():
    """Sample keyword data for optimization testing."""
    return pd.DataFrame(
        {
            "campaign_id": np.random.choice([f"camp_{i}" for i in range(10)], 50),
            "keyword": [f"keyword {i}" for i in range(50)],
            "match_type": np.random.choice(["EXACT", "PHRASE", "BROAD"], 50),
            "impressions": np.random.randint(100, 5000, 50),
            "clicks": np.random.randint(10, 500, 50),
            "cost": np.random.uniform(10, 500, 50),
            "conversions": np.random.randint(0, 25, 50),
            "quality_score": np.random.randint(1, 10, 50),
            "avg_cpc": np.random.uniform(0.50, 5.00, 50),
            "search_impression_share": np.random.uniform(0.1, 0.9, 50),
        }
    )


@pytest.fixture
def optimization_constraints():
    """Sample optimization constraints."""
    return {
        "total_budget": 50000,
        "min_campaign_budget": 100,
        "max_campaign_budget": 15000,
        "target_roas": 400,
        "min_conversions": 5,
        "max_cpa": 50,
        "budget_change_limit": 0.3,  # Max 30% change
    }


class TestRecommendationsEngine:
    """Test recommendations engine functionality."""

    def test_recommendations_engine_initialization(self):
        """Test recommendations engine initialization."""
        engine = RecommendationsEngine()
        assert engine is not None
        assert hasattr(engine, "generate_recommendations")

    def test_generate_budget_recommendations(self, sample_campaign_data):
        """Test budget optimization recommendations."""
        engine = RecommendationsEngine()

        recommendations = engine.generate_recommendations(
            data=sample_campaign_data,
            recommendation_type="budget_optimization",
            max_recommendations=5,
        )

        assert isinstance(recommendations, list)
        assert len(recommendations) <= 5

        for rec in recommendations:
            assert "id" in rec
            assert "title" in rec
            assert "description" in rec
            assert "confidence_score" in rec
            assert "estimated_impact" in rec
            assert "priority" in rec
            assert 0 <= rec["confidence_score"] <= 1

    def test_generate_keyword_recommendations(self, sample_keyword_data):
        """Test keyword optimization recommendations."""
        engine = RecommendationsEngine()

        recommendations = engine.generate_recommendations(
            data=sample_keyword_data,
            recommendation_type="keyword_optimization",
            confidence_threshold=0.7,
        )

        assert isinstance(recommendations, list)

        for rec in recommendations:
            assert "keyword_specific" in rec or "campaign_id" in rec
            assert rec["confidence_score"] >= 0.7

    def test_generate_bid_recommendations(self, sample_campaign_data):
        """Test bid strategy recommendations."""
        engine = RecommendationsEngine()

        recommendations = engine.generate_recommendations(
            data=sample_campaign_data,
            recommendation_type="bid_optimization",
            context={"performance_goal": "maximize_conversions"},
        )

        assert isinstance(recommendations, list)

        for rec in recommendations:
            assert "current_strategy" in rec or "recommended_strategy" in rec

    def test_filter_recommendations_by_confidence(self, sample_campaign_data):
        """Test recommendation filtering by confidence."""
        engine = RecommendationsEngine()

        all_recommendations = engine.generate_recommendations(
            data=sample_campaign_data,
            recommendation_type="budget_optimization",
            confidence_threshold=0.5,
        )

        high_confidence = engine.generate_recommendations(
            data=sample_campaign_data,
            recommendation_type="budget_optimization",
            confidence_threshold=0.8,
        )

        assert len(high_confidence) <= len(all_recommendations)

        for rec in high_confidence:
            assert rec["confidence_score"] >= 0.8

    def test_recommendation_prioritization(self, sample_campaign_data):
        """Test recommendation prioritization."""
        engine = RecommendationsEngine()

        recommendations = engine.generate_recommendations(
            data=sample_campaign_data,
            recommendation_type="budget_optimization",
            prioritize_by="impact",
        )

        if len(recommendations) > 1:
            # Should be sorted by impact (descending)
            impacts = [rec["estimated_impact"] for rec in recommendations]
            assert impacts == sorted(impacts, reverse=True)


class TestBudgetOptimizer:
    """Test budget optimization functionality."""

    def test_budget_optimizer_initialization(self):
        """Test budget optimizer initialization."""
        optimizer = BudgetOptimizer()
        assert optimizer is not None
        assert hasattr(optimizer, "optimize_budgets")

    def test_maximize_conversions_optimization(
        self, sample_campaign_data, optimization_constraints
    ):
        """Test budget optimization for maximizing conversions."""
        optimizer = BudgetOptimizer()

        optimized = optimizer.optimize_budgets(
            campaign_data=sample_campaign_data,
            objective="maximize_conversions",
            constraints=optimization_constraints,
        )

        assert "optimized_budgets" in optimized
        assert "expected_performance" in optimized
        assert "optimization_summary" in optimized

        # Check budget constraints
        budgets = optimized["optimized_budgets"]
        total_budget = sum(budgets.values())

        assert total_budget <= optimization_constraints["total_budget"]

        for budget in budgets.values():
            assert (
                optimization_constraints["min_campaign_budget"]
                <= budget
                <= optimization_constraints["max_campaign_budget"]
            )

    def test_maximize_roas_optimization(
        self, sample_campaign_data, optimization_constraints
    ):
        """Test budget optimization for maximizing ROAS."""
        optimizer = BudgetOptimizer()

        optimized = optimizer.optimize_budgets(
            campaign_data=sample_campaign_data,
            objective="maximize_roas",
            constraints=optimization_constraints,
        )

        assert optimized is not None
        assert "optimized_budgets" in optimized

        # Check that expected ROAS meets target
        expected_perf = optimized["expected_performance"]
        if "total_roas" in expected_perf:
            assert (
                expected_perf["total_roas"]
                >= optimization_constraints.get("target_roas", 0) * 0.9
            )  # Allow 10% tolerance

    def test_budget_allocation_constraints(
        self, sample_campaign_data, optimization_constraints
    ):
        """Test budget allocation with strict constraints."""
        optimizer = BudgetOptimizer()

        # Add stricter constraints
        strict_constraints = optimization_constraints.copy()
        strict_constraints["budget_change_limit"] = 0.1  # Max 10% change

        current_budgets = dict(
            zip(
                sample_campaign_data["campaign_id"],
                sample_campaign_data["budget"],
                strict=False,
            )
        )

        optimized = optimizer.optimize_budgets(
            campaign_data=sample_campaign_data,
            objective="maximize_conversions",
            constraints=strict_constraints,
            current_budgets=current_budgets,
        )

        # Check that budget changes respect limits
        for campaign_id, new_budget in optimized["optimized_budgets"].items():
            if campaign_id in current_budgets:
                current_budget = current_budgets[campaign_id]
                change_ratio = abs(new_budget - current_budget) / current_budget
                assert (
                    change_ratio <= strict_constraints["budget_change_limit"] + 0.01
                )  # Small tolerance

    def test_performance_forecasting(
        self, sample_campaign_data, optimization_constraints
    ):
        """Test performance forecasting with optimized budgets."""
        optimizer = BudgetOptimizer()

        optimized = optimizer.optimize_budgets(
            campaign_data=sample_campaign_data,
            objective="maximize_conversions",
            constraints=optimization_constraints,
            forecast_period=30,
        )

        forecast = optimized.get("performance_forecast")
        if forecast:
            assert "projected_conversions" in forecast
            assert "projected_revenue" in forecast
            assert "projected_cost" in forecast
            assert forecast["projected_conversions"] > 0

    def test_budget_optimization_with_seasonality(
        self, sample_campaign_data, optimization_constraints
    ):
        """Test budget optimization considering seasonality."""
        optimizer = BudgetOptimizer()

        # Add seasonality data
        seasonality_factors = {
            "january": 0.8,
            "february": 0.9,
            "march": 1.1,
            "april": 1.0,
            "may": 1.2,
            "june": 1.3,
        }

        optimized = optimizer.optimize_budgets(
            campaign_data=sample_campaign_data,
            objective="maximize_conversions",
            constraints=optimization_constraints,
            seasonality_factors=seasonality_factors,
            target_month="may",
        )

        assert optimized is not None
        # Budgets should be adjusted for May seasonality (factor 1.2)


class TestROIOptimizer:
    """Test ROI optimization functionality."""

    def test_roi_optimizer_initialization(self):
        """Test ROI optimizer initialization."""
        optimizer = ROIOptimizer()
        assert optimizer is not None
        assert hasattr(optimizer, "optimize_roi")

    def test_campaign_roi_optimization(self, sample_campaign_data):
        """Test campaign-level ROI optimization."""
        optimizer = ROIOptimizer()

        optimization = optimizer.optimize_roi(
            campaign_data=sample_campaign_data,
            optimization_level="campaign",
            target_roi=3.0,
        )

        assert "optimized_campaigns" in optimization
        assert "roi_improvements" in optimization
        assert "recommended_actions" in optimization

        # Check ROI calculations
        for campaign_opt in optimization["optimized_campaigns"]:
            if "projected_roi" in campaign_opt:
                assert campaign_opt["projected_roi"] >= 0

    def test_keyword_roi_optimization(self, sample_keyword_data):
        """Test keyword-level ROI optimization."""
        optimizer = ROIOptimizer()

        # Add revenue data to keywords
        sample_keyword_data["revenue"] = sample_keyword_data[
            "conversions"
        ] * np.random.uniform(20, 200, len(sample_keyword_data))

        optimization = optimizer.optimize_roi(
            keyword_data=sample_keyword_data,
            optimization_level="keyword",
            min_roi_threshold=2.0,
        )

        assert optimization is not None

        if "underperforming_keywords" in optimization:
            for keyword in optimization["underperforming_keywords"]:
                assert keyword.get("current_roi", 0) < 2.0

    def test_portfolio_roi_optimization(self, sample_campaign_data):
        """Test portfolio-level ROI optimization."""
        optimizer = ROIOptimizer()

        optimization = optimizer.optimize_roi(
            campaign_data=sample_campaign_data,
            optimization_level="portfolio",
            rebalance_budget=True,
        )

        assert "portfolio_metrics" in optimization
        assert "rebalancing_recommendations" in optimization

        portfolio = optimization["portfolio_metrics"]
        assert "total_roi" in portfolio
        assert "weighted_avg_roi" in portfolio

    def test_roi_sensitivity_analysis(self, sample_campaign_data):
        """Test ROI sensitivity analysis."""
        optimizer = ROIOptimizer()

        sensitivity = optimizer.analyze_roi_sensitivity(
            campaign_data=sample_campaign_data,
            variables=["budget", "cpc", "conversion_rate"],
        )

        assert "sensitivity_scores" in sensitivity
        assert "impact_scenarios" in sensitivity

        for variable in ["budget", "cpc", "conversion_rate"]:
            assert variable in sensitivity["sensitivity_scores"]

    def test_roi_forecasting(self, sample_campaign_data):
        """Test ROI forecasting."""
        optimizer = ROIOptimizer()

        forecast = optimizer.forecast_roi(
            campaign_data=sample_campaign_data,
            forecast_horizon=90,
            confidence_interval=0.95,
        )

        assert "forecast_values" in forecast
        assert "confidence_bounds" in forecast
        assert "forecast_accuracy" in forecast


class TestMLOptimizer:
    """Test ML-based optimization functionality."""

    def test_ml_optimizer_initialization(self):
        """Test ML optimizer initialization."""
        optimizer = MLOptimizer()
        assert optimizer is not None
        assert hasattr(optimizer, "train_model")
        assert hasattr(optimizer, "optimize_with_ml")

    def test_model_training(self, sample_campaign_data):
        """Test ML model training for optimization."""
        optimizer = MLOptimizer()

        # Prepare training data
        features = ["budget", "impressions", "clicks", "cost"]
        target = "conversions"

        model_performance = optimizer.train_model(
            data=sample_campaign_data,
            features=features,
            target=target,
            model_type="gradient_boosting",
        )

        assert "model_score" in model_performance
        assert "feature_importance" in model_performance
        assert "cross_validation_score" in model_performance

        # Check feature importance
        importance = model_performance["feature_importance"]
        assert len(importance) == len(features)
        assert all(imp >= 0 for imp in importance.values())

    def test_ml_based_optimization(self, sample_campaign_data):
        """Test ML-based optimization recommendations."""
        optimizer = MLOptimizer()

        # Train model first
        features = ["budget", "impressions", "clicks", "cost"]
        target = "conversions"

        optimizer.train_model(
            data=sample_campaign_data, features=features, target=target
        )

        # Generate ML-based recommendations
        recommendations = optimizer.optimize_with_ml(
            campaign_data=sample_campaign_data, optimization_goal="maximize_conversions"
        )

        assert isinstance(recommendations, list)

        for rec in recommendations:
            assert "campaign_id" in rec
            assert "recommended_changes" in rec
            assert "predicted_impact" in rec
            assert "confidence" in rec

    def test_hyperparameter_optimization(self, sample_campaign_data):
        """Test hyperparameter optimization for ML models."""
        optimizer = MLOptimizer()

        features = ["budget", "impressions", "clicks", "cost"]
        target = "conversions"

        best_params = optimizer.optimize_hyperparameters(
            data=sample_campaign_data,
            features=features,
            target=target,
            model_type="random_forest",
            param_grid={"n_estimators": [10, 50, 100], "max_depth": [3, 5, 10]},
        )

        assert isinstance(best_params, dict)
        assert "n_estimators" in best_params
        assert "max_depth" in best_params
        assert "best_score" in best_params

    def test_ensemble_optimization(self, sample_campaign_data):
        """Test ensemble-based optimization."""
        optimizer = MLOptimizer()

        features = ["budget", "impressions", "clicks", "cost"]
        target = "conversions"

        ensemble_results = optimizer.train_ensemble(
            data=sample_campaign_data,
            features=features,
            target=target,
            models=["linear_regression", "random_forest", "gradient_boosting"],
        )

        assert "ensemble_score" in ensemble_results
        assert "model_weights" in ensemble_results
        assert "individual_scores" in ensemble_results

        # Ensemble should perform at least as well as best individual model
        best_individual = max(ensemble_results["individual_scores"].values())
        assert (
            ensemble_results["ensemble_score"] >= best_individual * 0.95
        )  # Allow small tolerance


class TestRuleBasedOptimizer:
    """Test rule-based optimization functionality."""

    def test_rule_based_optimizer_initialization(self):
        """Test rule-based optimizer initialization."""
        optimizer = RuleBasedOptimizer()
        assert optimizer is not None
        assert hasattr(optimizer, "apply_rules")

    def test_budget_rules_application(self, sample_campaign_data):
        """Test application of budget rules."""
        optimizer = RuleBasedOptimizer()

        rules = [
            {
                "name": "increase_high_performers",
                "condition": "roas > 400 and conversion_rate > 0.05",
                "action": "increase_budget",
                "adjustment": 0.2,  # Increase by 20%
            },
            {
                "name": "decrease_poor_performers",
                "condition": "roas < 200 and cost > 1000",
                "action": "decrease_budget",
                "adjustment": -0.15,  # Decrease by 15%
            },
        ]

        # Calculate derived metrics
        sample_campaign_data["roas"] = (
            sample_campaign_data["revenue"] / sample_campaign_data["cost"]
        ) * 100
        sample_campaign_data["conversion_rate"] = (
            sample_campaign_data["conversions"] / sample_campaign_data["clicks"]
        )

        recommendations = optimizer.apply_rules(data=sample_campaign_data, rules=rules)

        assert isinstance(recommendations, list)

        for rec in recommendations:
            assert "rule_name" in rec
            assert "campaign_id" in rec
            assert "action" in rec
            assert "justification" in rec

    def test_keyword_rules_application(self, sample_keyword_data):
        """Test application of keyword rules."""
        optimizer = RuleBasedOptimizer()

        rules = [
            {
                "name": "pause_low_quality_keywords",
                "condition": "quality_score < 3 and cost > 100",
                "action": "pause_keyword",
                "priority": "HIGH",
            },
            {
                "name": "increase_bids_high_impression_share",
                "condition": "search_impression_share > 0.8 and conversions > 5",
                "action": "increase_bid",
                "adjustment": 0.15,
            },
        ]

        recommendations = optimizer.apply_rules(data=sample_keyword_data, rules=rules)

        assert isinstance(recommendations, list)

        high_priority_recs = [r for r in recommendations if r.get("priority") == "HIGH"]
        if high_priority_recs:
            assert all(rec["action"] == "pause_keyword" for rec in high_priority_recs)

    def test_custom_rule_validation(self):
        """Test custom rule validation."""
        optimizer = RuleBasedOptimizer()

        valid_rule = {
            "name": "test_rule",
            "condition": "cost > 100",
            "action": "decrease_budget",
            "adjustment": -0.1,
        }

        invalid_rule = {
            "name": "invalid_rule",
            # Missing condition
            "action": "increase_budget",
        }

        assert optimizer.validate_rule(valid_rule) is True
        assert optimizer.validate_rule(invalid_rule) is False

    def test_rule_conflict_resolution(self, sample_campaign_data):
        """Test handling of conflicting rules."""
        optimizer = RuleBasedOptimizer()

        conflicting_rules = [
            {
                "name": "increase_for_high_roas",
                "condition": "revenue / cost > 3",
                "action": "increase_budget",
                "adjustment": 0.2,
                "priority": 1,
            },
            {
                "name": "decrease_for_high_cost",
                "condition": "cost > 5000",
                "action": "decrease_budget",
                "adjustment": -0.1,
                "priority": 2,
            },
        ]

        recommendations = optimizer.apply_rules(
            data=sample_campaign_data, rules=conflicting_rules, resolve_conflicts=True
        )

        # Should handle conflicts based on priority
        campaign_actions = {}
        for rec in recommendations:
            campaign_id = rec["campaign_id"]
            if campaign_id not in campaign_actions:
                campaign_actions[campaign_id] = []
            campaign_actions[campaign_id].append(rec)

        # No campaign should have conflicting actions
        for actions in campaign_actions.values():
            action_types = [a["action"] for a in actions]
            assert not (
                "increase_budget" in action_types and "decrease_budget" in action_types
            )


class TestABTestingOptimizer:
    """Test A/B testing optimization functionality."""

    def test_ab_testing_optimizer_initialization(self):
        """Test A/B testing optimizer initialization."""
        optimizer = ABTestingOptimizer()
        assert optimizer is not None
        assert hasattr(optimizer, "design_test")
        assert hasattr(optimizer, "analyze_results")

    def test_ab_test_design(self, sample_campaign_data):
        """Test A/B test design."""
        optimizer = ABTestingOptimizer()

        test_design = optimizer.design_test(
            campaign_data=sample_campaign_data,
            test_type="budget_split",
            test_parameters={
                "budget_increase": [0.1, 0.2, 0.3],  # Test 10%, 20%, 30% increases
                "duration_days": 14,
                "confidence_level": 0.95,
            },
        )

        assert "test_groups" in test_design
        assert "control_group" in test_design
        assert "sample_size_per_group" in test_design
        assert "expected_duration" in test_design

        # Check test groups
        assert len(test_design["test_groups"]) == 3  # Three budget increase levels

    def test_statistical_power_calculation(self, sample_campaign_data):
        """Test statistical power calculation for A/B tests."""
        optimizer = ABTestingOptimizer()

        power_analysis = optimizer.calculate_statistical_power(
            baseline_conversion_rate=0.03,
            minimum_detectable_effect=0.2,  # 20% relative change
            significance_level=0.05,
            sample_size=1000,
        )

        assert "statistical_power" in power_analysis
        assert "required_sample_size" in power_analysis
        assert "minimum_test_duration" in power_analysis
        assert 0 <= power_analysis["statistical_power"] <= 1

    def test_ab_test_results_analysis(self):
        """Test A/B test results analysis."""
        optimizer = ABTestingOptimizer()

        # Create mock test results
        test_results = {
            "control_group": {
                "impressions": 10000,
                "clicks": 300,
                "conversions": 30,
                "cost": 500,
            },
            "test_groups": [
                {
                    "group_id": "test_a",
                    "impressions": 10500,
                    "clicks": 340,
                    "conversions": 38,
                    "cost": 550,
                },
                {
                    "group_id": "test_b",
                    "impressions": 9800,
                    "clicks": 285,
                    "conversions": 25,
                    "cost": 475,
                },
            ],
        }

        analysis = optimizer.analyze_results(test_results)

        assert "statistical_significance" in analysis
        assert "confidence_intervals" in analysis
        assert "winner" in analysis
        assert "effect_sizes" in analysis

        # Check significance tests
        for group_analysis in analysis["statistical_significance"]:
            assert "p_value" in group_analysis
            assert "is_significant" in group_analysis

    def test_multivariate_test_design(self, sample_campaign_data):
        """Test multivariate test design."""
        optimizer = ABTestingOptimizer()

        mv_test_design = optimizer.design_multivariate_test(
            campaign_data=sample_campaign_data,
            test_variables={
                "budget_adjustment": [0.9, 1.0, 1.1],  # -10%, 0%, +10%
                "bid_adjustment": [0.85, 1.0, 1.15],  # -15%, 0%, +15%
            },
        )

        assert "test_matrix" in mv_test_design
        assert "total_combinations" in mv_test_design
        assert mv_test_design["total_combinations"] == 9  # 3x3 combinations

        # Check test matrix
        assert len(mv_test_design["test_matrix"]) == 9

    def test_sequential_testing(self):
        """Test sequential A/B testing (early stopping)."""
        optimizer = ABTestingOptimizer()

        # Simulate data collection over time
        daily_results = []
        for day in range(14):
            daily_results.append(
                {
                    "day": day + 1,
                    "control": {
                        "conversions": np.random.binomial(100, 0.03),
                        "trials": 100,
                    },
                    "test": {
                        "conversions": np.random.binomial(
                            100, 0.035
                        ),  # Slightly better
                        "trials": 100,
                    },
                }
            )

        sequential_analysis = optimizer.analyze_sequential_test(
            daily_results=daily_results, alpha=0.05, beta=0.2
        )

        assert "early_stopping_recommendation" in sequential_analysis
        assert "current_confidence" in sequential_analysis
        assert "days_to_significance" in sequential_analysis


@pytest.mark.integration
class TestOptimizationIntegration:
    """Integration tests for optimization modules."""

    def test_end_to_end_optimization_pipeline(
        self, sample_campaign_data, sample_keyword_data, optimization_constraints
    ):
        """Test complete optimization pipeline."""
        # Generate recommendations
        rec_engine = RecommendationsEngine()
        recommendations = rec_engine.generate_recommendations(
            data=sample_campaign_data, recommendation_type="comprehensive"
        )

        # Optimize budgets
        budget_optimizer = BudgetOptimizer()
        budget_optimization = budget_optimizer.optimize_budgets(
            campaign_data=sample_campaign_data,
            objective="maximize_conversions",
            constraints=optimization_constraints,
        )

        # Optimize ROI
        roi_optimizer = ROIOptimizer()
        roi_optimization = roi_optimizer.optimize_roi(
            campaign_data=sample_campaign_data, optimization_level="portfolio"
        )

        # Apply rules
        rule_optimizer = RuleBasedOptimizer()
        rules = [
            {
                "name": "test_rule",
                "condition": "cost > 1000",
                "action": "review_campaign",
            }
        ]
        rule_recommendations = rule_optimizer.apply_rules(
            data=sample_campaign_data, rules=rules
        )

        # Verify all components work together
        assert recommendations is not None
        assert budget_optimization is not None
        assert roi_optimization is not None
        assert rule_recommendations is not None

    def test_optimization_with_real_constraints(self, sample_campaign_data):
        """Test optimization with realistic business constraints."""
        real_constraints = {
            "total_budget": 25000,
            "min_campaign_budget": 500,
            "max_campaign_budget": 8000,
            "min_roas": 300,
            "max_cpa": 75,
            "budget_change_limit": 0.25,
            "excluded_campaigns": ["camp_9"],  # Cannot be optimized
            "priority_campaigns": ["camp_0", "camp_1"],  # Must maintain minimum budget
        }

        budget_optimizer = BudgetOptimizer()
        roi_optimizer = ROIOptimizer()

        # Budget optimization with constraints
        budget_result = budget_optimizer.optimize_budgets(
            campaign_data=sample_campaign_data,
            objective="maximize_conversions",
            constraints=real_constraints,
        )

        # ROI optimization with constraints
        roi_optimizer.optimize_roi(
            campaign_data=sample_campaign_data, constraints=real_constraints
        )

        # Verify constraints are respected
        optimized_budgets = budget_result["optimized_budgets"]

        # Check excluded campaigns
        if "camp_9" in optimized_budgets:
            original_budget = sample_campaign_data[
                sample_campaign_data["campaign_id"] == "camp_9"
            ]["budget"].iloc[0]
            assert optimized_budgets["camp_9"] == original_budget

        # Check priority campaigns have reasonable budgets
        for priority_camp in real_constraints["priority_campaigns"]:
            if priority_camp in optimized_budgets:
                assert (
                    optimized_budgets[priority_camp]
                    >= real_constraints["min_campaign_budget"]
                )


@pytest.mark.performance
class TestOptimizationPerformance:
    """Performance tests for optimization modules."""

    def test_budget_optimization_performance(self, benchmark):
        """Benchmark budget optimization performance."""
        # Create large dataset
        large_campaign_data = pd.DataFrame(
            {
                "campaign_id": [f"camp_{i}" for i in range(100)],
                "budget": np.random.uniform(1000, 10000, 100),
                "impressions": np.random.randint(10000, 100000, 100),
                "clicks": np.random.randint(500, 5000, 100),
                "cost": np.random.uniform(500, 8000, 100),
                "conversions": np.random.randint(10, 100, 100),
                "revenue": np.random.uniform(200, 5000, 100),
            }
        )

        constraints = {
            "total_budget": 500000,
            "min_campaign_budget": 100,
            "max_campaign_budget": 15000,
        }

        optimizer = BudgetOptimizer()

        def optimize_large_portfolio():
            return optimizer.optimize_budgets(
                campaign_data=large_campaign_data,
                objective="maximize_conversions",
                constraints=constraints,
            )

        result = benchmark(optimize_large_portfolio)
        assert result is not None

    def test_ml_optimization_performance(self, benchmark):
        """Benchmark ML optimization performance."""
        # Create large dataset
        large_data = pd.DataFrame(
            {
                "campaign_id": [f"camp_{i}" for i in range(500)],
                "budget": np.random.uniform(1000, 10000, 500),
                "impressions": np.random.randint(10000, 100000, 500),
                "clicks": np.random.randint(500, 5000, 500),
                "cost": np.random.uniform(500, 8000, 500),
                "conversions": np.random.randint(10, 100, 500),
                "revenue": np.random.uniform(200, 5000, 500),
            }
        )

        optimizer = MLOptimizer()

        def train_and_optimize():
            features = ["budget", "impressions", "clicks", "cost"]
            target = "conversions"

            optimizer.train_model(data=large_data, features=features, target=target)

            return optimizer.optimize_with_ml(
                campaign_data=large_data, optimization_goal="maximize_conversions"
            )

        result = benchmark(train_and_optimize)
        assert result is not None
