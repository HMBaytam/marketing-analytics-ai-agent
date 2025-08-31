"""ML-driven optimization system for marketing campaigns."""

import logging
from datetime import datetime
from enum import Enum
from typing import Any

import joblib
import numpy as np
from pydantic import BaseModel, Field
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class OptimizationType(str, Enum):
    """Types of ML-driven optimizations."""

    BID_OPTIMIZATION = "bid_optimization"
    BUDGET_ALLOCATION = "budget_allocation"
    AUDIENCE_TARGETING = "audience_targeting"
    CREATIVE_OPTIMIZATION = "creative_optimization"
    TIMING_OPTIMIZATION = "timing_optimization"
    CHANNEL_MIX = "channel_mix"


class MLOptimizationConfig(BaseModel):
    """Configuration for ML-driven optimization."""

    # Model parameters
    model_type: str = Field(default="random_forest", description="ML model type")
    optimization_objective: str = Field(
        default="roas", description="Optimization objective"
    )
    min_training_samples: int = Field(
        default=100, description="Minimum training samples"
    )
    test_size: float = Field(default=0.2, description="Test set proportion")

    # Optimization constraints
    max_bid_increase: float = Field(
        default=0.5, description="Maximum bid increase ratio"
    )
    max_budget_shift: float = Field(
        default=0.3, description="Maximum budget shift ratio"
    )
    confidence_threshold: float = Field(
        default=0.7, description="Minimum confidence for recommendations"
    )

    # Feature engineering
    include_temporal_features: bool = Field(
        default=True, description="Include time-based features"
    )
    include_interaction_features: bool = Field(
        default=True, description="Include feature interactions"
    )
    feature_selection_threshold: float = Field(
        default=0.01, description="Feature importance threshold"
    )

    # Clustering parameters
    n_clusters: int = Field(
        default=5, description="Number of clusters for segmentation"
    )
    cluster_random_state: int = Field(
        default=42, description="Random state for clustering"
    )


class MLRecommendation(BaseModel):
    """ML-generated optimization recommendation."""

    optimization_type: OptimizationType = Field(description="Type of optimization")
    confidence_score: float = Field(description="ML model confidence")
    expected_impact: float = Field(description="Expected performance impact")

    # Optimization parameters
    current_value: float = Field(description="Current parameter value")
    recommended_value: float = Field(description="Recommended new value")
    change_percentage: float = Field(description="Percentage change")

    # Supporting data
    feature_importance: dict[str, float] = Field(description="Contributing factors")
    similar_campaigns: list[str] = Field(
        description="Similar high-performing campaigns"
    )
    risk_assessment: dict[str, float] = Field(description="Risk factors")

    # Implementation details
    implementation_steps: list[str] = Field(description="Implementation steps")
    monitoring_metrics: list[str] = Field(description="Metrics to monitor")
    rollback_conditions: list[str] = Field(description="When to rollback changes")


class MLOptimizer:
    """ML-driven optimization engine."""

    def __init__(self, config: MLOptimizationConfig):
        self.config = config
        self.models = {}
        self.scalers = {}
        self.feature_names = []
        self.clusterer = None

    def generate_ml_recommendations(
        self,
        campaign_data: list[dict[str, Any]],
        historical_performance: list[dict[str, Any]],
        campaign_id: str | None = None,
    ) -> list[MLRecommendation]:
        """Generate ML-driven optimization recommendations."""

        logger.info(f"Generating ML recommendations for campaign: {campaign_id}")

        try:
            # Prepare training data
            X, y, feature_names = self._prepare_training_data(historical_performance)

            if len(X) < self.config.min_training_samples:
                logger.warning(
                    f"Insufficient data for ML optimization: {len(X)} samples"
                )
                return []

            # Train models for different optimization types
            trained_models = self._train_optimization_models(X, y, feature_names)

            # Cluster campaigns for similarity analysis
            clusters = self._cluster_campaigns(X)

            # Generate recommendations for each optimization type
            recommendations = []

            current_features = self._extract_current_features(campaign_data)

            for opt_type in OptimizationType:
                if opt_type.value in trained_models:
                    rec = self._generate_optimization_recommendation(
                        opt_type,
                        trained_models[opt_type.value],
                        current_features,
                        clusters,
                        campaign_id,
                    )
                    if rec:
                        recommendations.append(rec)

            # Filter by confidence threshold
            high_confidence_recs = [
                r
                for r in recommendations
                if r.confidence_score >= self.config.confidence_threshold
            ]

            # Sort by expected impact
            high_confidence_recs.sort(key=lambda x: x.expected_impact, reverse=True)

            logger.info(
                f"Generated {len(high_confidence_recs)} high-confidence ML recommendations"
            )
            return high_confidence_recs

        except Exception as e:
            logger.error(f"ML recommendation generation failed: {str(e)}")
            raise

    def _prepare_training_data(
        self, historical_data: list[dict[str, Any]]
    ) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """Prepare training data from historical performance."""

        features = []
        targets = []
        feature_names = []

        for record in historical_data:
            # Extract features
            feature_row = []

            # Performance metrics as features
            perf_metrics = [
                "ctr",
                "cpc",
                "cpa",
                "roas",
                "conversion_rate",
                "impressions",
                "clicks",
                "spend",
            ]
            for metric in perf_metrics:
                value = record.get(metric, 0)
                feature_row.append(float(value) if value is not None else 0.0)
                if len(feature_names) < len(perf_metrics):
                    feature_names.append(metric)

            # Campaign attributes
            campaign_attrs = [
                "campaign_type",
                "channel",
                "audience_size",
                "bid_strategy",
            ]
            for attr in campaign_attrs:
                value = record.get(attr, 0)
                if isinstance(value, str):
                    # Simple string encoding (in practice, use proper encoding)
                    feature_row.append(hash(value) % 1000)
                else:
                    feature_row.append(float(value) if value is not None else 0.0)
                if len(feature_names) < len(perf_metrics) + len(campaign_attrs):
                    feature_names.append(attr)

            # Temporal features
            if self.config.include_temporal_features and "date" in record:
                date = (
                    record["date"]
                    if isinstance(record["date"], datetime)
                    else datetime.now()
                )
                temporal_features = [
                    date.weekday(),
                    date.hour,
                    date.day,
                    date.month,
                    date.quarter,
                ]
                feature_row.extend(temporal_features)
                if len(feature_names) < len(perf_metrics) + len(campaign_attrs) + 5:
                    feature_names.extend(["weekday", "hour", "day", "month", "quarter"])

            # Interaction features
            if self.config.include_interaction_features and len(feature_row) >= 2:
                # Add some simple interactions
                feature_row.append(feature_row[0] * feature_row[1])  # ctr * cpc
                feature_row.append(feature_row[2] * feature_row[3])  # cpa * roas
                if len(feature_names) == len(feature_row) - 2:
                    feature_names.extend(
                        ["ctr_cpc_interaction", "cpa_roas_interaction"]
                    )

            features.append(feature_row)

            # Target variable (optimization objective)
            target = record.get(self.config.optimization_objective, 0)
            targets.append(float(target) if target is not None else 0.0)

        self.feature_names = feature_names
        return np.array(features), np.array(targets), feature_names

    def _train_optimization_models(
        self, X: np.ndarray, y: np.ndarray, feature_names: list[str]
    ) -> dict[str, Any]:
        """Train ML models for different optimization types."""

        models = {}

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.config.test_size, random_state=42
        )

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        self.scalers["main"] = scaler

        # Train regression model for continuous optimizations
        if self.config.model_type == "random_forest":
            regressor = RandomForestRegressor(n_estimators=100, random_state=42)
        else:
            # Default fallback
            regressor = RandomForestRegressor(n_estimators=100, random_state=42)

        regressor.fit(X_train_scaled, y_train)

        # Evaluate model
        y_pred = regressor.predict(X_test_scaled)
        mse = mean_squared_error(y_test, y_pred)

        logger.info(f"Regression model MSE: {mse:.4f}")

        # Store models for different optimization types
        models["bid_optimization"] = regressor
        models["budget_allocation"] = regressor
        models["audience_targeting"] = regressor
        models["creative_optimization"] = regressor
        models["timing_optimization"] = regressor
        models["channel_mix"] = regressor

        return models

    def _cluster_campaigns(self, X: np.ndarray) -> np.ndarray:
        """Cluster campaigns for similarity analysis."""

        try:
            # Scale features for clustering
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # Perform clustering
            clusterer = KMeans(
                n_clusters=self.config.n_clusters,
                random_state=self.config.cluster_random_state,
            )
            clusters = clusterer.fit_predict(X_scaled)

            self.clusterer = clusterer
            self.scalers["cluster"] = scaler

            return clusters

        except Exception as e:
            logger.warning(f"Clustering failed: {str(e)}")
            return np.zeros(len(X))

    def _extract_current_features(
        self, campaign_data: list[dict[str, Any]]
    ) -> np.ndarray:
        """Extract features from current campaign data."""

        if not campaign_data:
            return np.array([[0] * len(self.feature_names)])

        # Use most recent data point
        latest_data = campaign_data[-1] if campaign_data else {}

        # Extract features similar to training data preparation
        feature_row = []

        # Performance metrics
        perf_metrics = [
            "ctr",
            "cpc",
            "cpa",
            "roas",
            "conversion_rate",
            "impressions",
            "clicks",
            "spend",
        ]
        for metric in perf_metrics:
            value = latest_data.get(metric, 0)
            feature_row.append(float(value) if value is not None else 0.0)

        # Campaign attributes
        campaign_attrs = ["campaign_type", "channel", "audience_size", "bid_strategy"]
        for attr in campaign_attrs:
            value = latest_data.get(attr, 0)
            if isinstance(value, str):
                feature_row.append(hash(value) % 1000)
            else:
                feature_row.append(float(value) if value is not None else 0.0)

        # Temporal features
        if self.config.include_temporal_features:
            now = datetime.now()
            temporal_features = [
                now.weekday(),
                now.hour,
                now.day,
                now.month,
                now.quarter,
            ]
            feature_row.extend(temporal_features)

        # Interaction features
        if self.config.include_interaction_features and len(feature_row) >= 2:
            feature_row.append(feature_row[0] * feature_row[1])
            feature_row.append(feature_row[2] * feature_row[3])

        # Pad or trim to match training feature length
        while len(feature_row) < len(self.feature_names):
            feature_row.append(0.0)
        feature_row = feature_row[: len(self.feature_names)]

        return np.array([feature_row])

    def _generate_optimization_recommendation(
        self,
        opt_type: OptimizationType,
        model: Any,
        current_features: np.ndarray,
        clusters: np.ndarray,
        campaign_id: str | None,
    ) -> MLRecommendation | None:
        """Generate recommendation for specific optimization type."""

        try:
            # Scale current features
            current_scaled = self.scalers["main"].transform(current_features)

            # Get current prediction
            current_prediction = model.predict(current_scaled)[0]

            # Generate feature variations for optimization
            best_variation, best_prediction = self._find_optimal_variation(
                model, current_features[0], opt_type
            )

            if best_prediction <= current_prediction:
                return None  # No improvement found

            # Calculate confidence based on model performance and feature importance
            confidence = self._calculate_recommendation_confidence(
                model, current_features, opt_type
            )

            # Get feature importance
            importance_dict = self._get_feature_importance(model, opt_type)

            # Find similar campaigns
            similar_campaigns = self._find_similar_campaigns(current_features, clusters)

            # Generate implementation details
            (
                impl_steps,
                monitoring_metrics,
                rollback_conditions,
            ) = self._generate_implementation_details(
                opt_type, current_features[0], best_variation
            )

            # Calculate current and recommended values
            current_value, recommended_value = self._extract_optimization_values(
                opt_type, current_features[0], best_variation
            )

            change_percentage = (
                ((recommended_value - current_value) / current_value * 100)
                if current_value != 0
                else 0
            )

            return MLRecommendation(
                optimization_type=opt_type,
                confidence_score=confidence,
                expected_impact=(best_prediction - current_prediction)
                / current_prediction,
                current_value=current_value,
                recommended_value=recommended_value,
                change_percentage=change_percentage,
                feature_importance=importance_dict,
                similar_campaigns=similar_campaigns,
                risk_assessment=self._assess_optimization_risk(
                    opt_type, change_percentage
                ),
                implementation_steps=impl_steps,
                monitoring_metrics=monitoring_metrics,
                rollback_conditions=rollback_conditions,
            )

        except Exception as e:
            logger.error(f"Failed to generate {opt_type} recommendation: {str(e)}")
            return None

    def _find_optimal_variation(
        self, model: Any, current_features: np.ndarray, opt_type: OptimizationType
    ) -> tuple[np.ndarray, float]:
        """Find optimal feature variation for given optimization type."""

        best_features = current_features.copy()
        best_prediction = model.predict(
            self.scalers["main"].transform([current_features])
        )[0]

        # Define which features to optimize based on optimization type
        optimization_indices = self._get_optimization_feature_indices(opt_type)

        # Test variations
        for idx in optimization_indices:
            for multiplier in [0.8, 0.9, 1.1, 1.2, 1.5]:  # Test different values
                test_features = current_features.copy()
                test_features[idx] = test_features[idx] * multiplier

                # Ensure constraints
                test_features = self._apply_optimization_constraints(
                    test_features, current_features, opt_type
                )

                # Get prediction
                test_scaled = self.scalers["main"].transform([test_features])
                prediction = model.predict(test_scaled)[0]

                if prediction > best_prediction:
                    best_prediction = prediction
                    best_features = test_features.copy()

        return best_features, best_prediction

    def _get_optimization_feature_indices(
        self, opt_type: OptimizationType
    ) -> list[int]:
        """Get feature indices relevant to optimization type."""

        # Map optimization types to relevant feature indices
        # This is a simplified mapping - in practice, this would be more sophisticated

        if opt_type == OptimizationType.BID_OPTIMIZATION:
            return [1]  # CPC index
        elif opt_type == OptimizationType.BUDGET_ALLOCATION:
            return [7]  # Spend index
        elif opt_type == OptimizationType.AUDIENCE_TARGETING:
            return [10]  # Audience size index
        elif opt_type == OptimizationType.CREATIVE_OPTIMIZATION:
            return [0]  # CTR index (proxy for creative performance)
        elif opt_type == OptimizationType.TIMING_OPTIMIZATION:
            return [12, 13]  # Hour, day indices
        else:
            return [0, 1, 2]  # Default to main performance metrics

    def _apply_optimization_constraints(
        self,
        test_features: np.ndarray,
        original_features: np.ndarray,
        opt_type: OptimizationType,
    ) -> np.ndarray:
        """Apply constraints to optimization variations."""

        constrained_features = test_features.copy()

        # Apply maximum change constraints
        for i in range(len(test_features)):
            change_ratio = (
                test_features[i] / original_features[i]
                if original_features[i] != 0
                else 1
            )

            if opt_type == OptimizationType.BID_OPTIMIZATION:
                # Limit bid changes
                max_change = 1 + self.config.max_bid_increase
                constrained_features[i] = original_features[i] * min(
                    change_ratio, max_change
                )
            elif opt_type == OptimizationType.BUDGET_ALLOCATION:
                # Limit budget changes
                max_change = 1 + self.config.max_budget_shift
                constrained_features[i] = original_features[i] * min(
                    change_ratio, max_change
                )

        return constrained_features

    def _calculate_recommendation_confidence(
        self, model: Any, current_features: np.ndarray, opt_type: OptimizationType
    ) -> float:
        """Calculate confidence score for recommendation."""

        # Base confidence from model (simplified)
        base_confidence = 0.7  # Placeholder

        # Adjust based on feature importance
        if hasattr(model, "feature_importances_"):
            relevant_indices = self._get_optimization_feature_indices(opt_type)
            relevant_importance = sum(
                model.feature_importances_[i] for i in relevant_indices
            )
            importance_boost = min(relevant_importance * 2, 0.2)
            base_confidence += importance_boost

        return min(base_confidence, 1.0)

    def _get_feature_importance(
        self, model: Any, opt_type: OptimizationType
    ) -> dict[str, float]:
        """Get feature importance for the optimization type."""

        if not hasattr(model, "feature_importances_"):
            return {}

        importance_dict = {}
        relevant_indices = self._get_optimization_feature_indices(opt_type)

        for idx in relevant_indices:
            if idx < len(self.feature_names) and idx < len(model.feature_importances_):
                feature_name = self.feature_names[idx]
                importance = float(model.feature_importances_[idx])
                importance_dict[feature_name] = importance

        return importance_dict

    def _find_similar_campaigns(
        self, current_features: np.ndarray, clusters: np.ndarray
    ) -> list[str]:
        """Find similar high-performing campaigns."""

        # Simplified similarity - in practice, this would use actual campaign IDs
        # and performance data
        return [f"campaign_{i}" for i in range(3)]

    def _assess_optimization_risk(
        self, opt_type: OptimizationType, change_percentage: float
    ) -> dict[str, float]:
        """Assess risk factors for the optimization."""

        risk_factors = {}

        # Risk increases with larger changes
        magnitude_risk = min(abs(change_percentage) / 100, 1.0)
        risk_factors["magnitude_risk"] = magnitude_risk

        # Optimization-specific risks
        if opt_type == OptimizationType.BID_OPTIMIZATION:
            risk_factors["cost_increase_risk"] = magnitude_risk * 0.8
            risk_factors["volume_loss_risk"] = magnitude_risk * 0.6
        elif opt_type == OptimizationType.BUDGET_ALLOCATION:
            risk_factors["performance_disruption_risk"] = magnitude_risk * 0.7

        return risk_factors

    def _generate_implementation_details(
        self,
        opt_type: OptimizationType,
        current_features: np.ndarray,
        optimal_features: np.ndarray,
    ) -> tuple[list[str], list[str], list[str]]:
        """Generate implementation details for the optimization."""

        steps = []
        monitoring = []
        rollback = []

        if opt_type == OptimizationType.BID_OPTIMIZATION:
            steps = [
                "Gradually increase bids in 10% increments",
                "Monitor performance for 3 days between changes",
                "Apply changes during high-performance hours",
            ]
            monitoring = ["CPC", "CTR", "Conversion Rate", "ROAS"]
            rollback = ["CPA increase >20%", "ROAS decrease >15%", "Volume drop >30%"]

        elif opt_type == OptimizationType.BUDGET_ALLOCATION:
            steps = [
                "Reallocate budget in 20% increments",
                "Test reallocation with 30% of total budget",
                "Scale successful reallocations",
            ]
            monitoring = ["Spend Distribution", "ROAS by Channel", "Conversion Volume"]
            rollback = ["Overall ROAS decrease >10%", "Volume drop >25%"]

        # Add more optimization types as needed

        return steps, monitoring, rollback

    def _extract_optimization_values(
        self,
        opt_type: OptimizationType,
        current_features: np.ndarray,
        optimal_features: np.ndarray,
    ) -> tuple[float, float]:
        """Extract current and recommended values for the optimization."""

        if opt_type == OptimizationType.BID_OPTIMIZATION:
            # CPC is at index 1
            return float(current_features[1]), float(optimal_features[1])
        elif opt_type == OptimizationType.BUDGET_ALLOCATION:
            # Spend is at index 7
            return float(current_features[7]), float(optimal_features[7])
        else:
            # Default to first feature
            return float(current_features[0]), float(optimal_features[0])

    def save_models(self, model_path: str):
        """Save trained ML models."""
        joblib.dump(
            {
                "models": self.models,
                "scalers": self.scalers,
                "feature_names": self.feature_names,
                "clusterer": self.clusterer,
                "config": self.config,
            },
            model_path,
        )
        logger.info(f"ML models saved to: {model_path}")

    def load_models(self, model_path: str):
        """Load trained ML models."""
        saved_data = joblib.load(model_path)
        self.models = saved_data["models"]
        self.scalers = saved_data["scalers"]
        self.feature_names = saved_data["feature_names"]
        self.clusterer = saved_data["clusterer"]
        logger.info(f"ML models loaded from: {model_path}")
