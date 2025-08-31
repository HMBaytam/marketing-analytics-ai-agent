"""ML-based performance prediction system for marketing campaigns."""

import logging
from datetime import datetime, timedelta
from typing import Any

import joblib
import numpy as np
from pydantic import BaseModel, Field
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class PredictionConfig(BaseModel):
    """Configuration for ML-based performance prediction."""

    model_type: str = Field(default="random_forest", description="ML model type")
    test_size: float = Field(default=0.2, description="Test set proportion")
    cv_folds: int = Field(default=5, description="Cross-validation folds")
    random_state: int = Field(default=42, description="Random seed")

    # Feature engineering
    include_seasonal: bool = Field(
        default=True, description="Include seasonal features"
    )
    include_lag_features: bool = Field(default=True, description="Include lag features")
    lag_periods: list[int] = Field(
        default=[1, 7, 14, 30], description="Lag periods to include"
    )

    # Model-specific parameters
    rf_n_estimators: int = Field(default=100, description="Random Forest trees")
    rf_max_depth: int = Field(default=10, description="Random Forest max depth")
    gb_n_estimators: int = Field(default=100, description="Gradient Boosting trees")
    gb_learning_rate: float = Field(
        default=0.1, description="Gradient Boosting learning rate"
    )
    ridge_alpha: float = Field(default=1.0, description="Ridge regression alpha")

    # Prediction parameters
    forecast_horizon: int = Field(default=30, description="Days to forecast ahead")
    confidence_level: float = Field(
        default=0.95, description="Confidence level for intervals"
    )
    min_training_samples: int = Field(
        default=30, description="Minimum samples for training"
    )


class ModelMetrics(BaseModel):
    """Model performance metrics."""

    mse: float = Field(description="Mean squared error")
    rmse: float = Field(description="Root mean squared error")
    mae: float = Field(description="Mean absolute error")
    r2: float = Field(description="R-squared score")
    cv_score_mean: float = Field(description="Cross-validation mean score")
    cv_score_std: float = Field(description="Cross-validation score std")
    training_samples: int = Field(description="Number of training samples")


class FeatureImportance(BaseModel):
    """Feature importance for model interpretability."""

    feature_name: str = Field(description="Feature name")
    importance: float = Field(description="Feature importance score")
    rank: int = Field(description="Importance rank")


class PredictionResult(BaseModel):
    """ML model prediction result."""

    metric_name: str = Field(description="Predicted metric")
    model_type: str = Field(description="ML model used")
    prediction_date: datetime = Field(description="Date of prediction")

    # Predictions
    point_forecast: list[float] = Field(description="Point predictions")
    lower_bound: list[float] = Field(description="Lower confidence bound")
    upper_bound: list[float] = Field(description="Upper confidence bound")
    forecast_dates: list[datetime] = Field(description="Forecast dates")

    # Model performance
    metrics: ModelMetrics = Field(description="Model performance metrics")
    feature_importance: list[FeatureImportance] = Field(
        description="Feature importance"
    )

    # Insights
    trend_direction: str = Field(description="Overall trend direction")
    volatility_level: str = Field(description="Prediction volatility level")
    confidence_score: float = Field(description="Overall prediction confidence")
    recommendations: list[str] = Field(description="Model-based recommendations")


class PredictiveModel:
    """ML-based performance prediction engine."""

    def __init__(self, config: PredictionConfig):
        self.config = config
        self.models = {}
        self.scalers = {}
        self.feature_names = []

    def predict_performance(
        self,
        historical_data: list[dict[str, Any]],
        metric_name: str,
        campaign_id: str | None = None,
    ) -> PredictionResult:
        """Predict future performance using ML models."""

        logger.info(f"Starting ML prediction for metric: {metric_name}")

        try:
            # Prepare data
            features, targets, dates = self._prepare_data(historical_data, metric_name)

            if len(features) < self.config.min_training_samples:
                raise ValueError(
                    f"Insufficient data: {len(features)} < {self.config.min_training_samples}"
                )

            # Train model
            model, scaler, metrics = self._train_model(features, targets)

            # Generate predictions
            predictions = self._generate_predictions(model, scaler, features, dates)

            # Calculate feature importance
            importance = self._calculate_feature_importance(model)

            # Generate insights
            insights = self._generate_insights(predictions, targets, metrics)

            return PredictionResult(
                metric_name=metric_name,
                model_type=self.config.model_type,
                prediction_date=datetime.now(),
                point_forecast=predictions["point_forecast"],
                lower_bound=predictions["lower_bound"],
                upper_bound=predictions["upper_bound"],
                forecast_dates=predictions["forecast_dates"],
                metrics=metrics,
                feature_importance=importance,
                trend_direction=insights["trend_direction"],
                volatility_level=insights["volatility_level"],
                confidence_score=insights["confidence_score"],
                recommendations=insights["recommendations"],
            )

        except Exception as e:
            logger.error(f"ML prediction failed: {str(e)}")
            raise

    def _prepare_data(
        self, historical_data: list[dict[str, Any]], metric_name: str
    ) -> tuple[np.ndarray, np.ndarray, list[datetime]]:
        """Prepare data for ML training."""

        # Sort data by date
        sorted_data = sorted(
            historical_data, key=lambda x: x.get("date", datetime.now())
        )

        # Extract targets and dates
        targets = []
        dates = []
        raw_features = []

        for record in sorted_data:
            if metric_name in record:
                targets.append(float(record[metric_name]))
                dates.append(record.get("date", datetime.now()))
                raw_features.append(record)

        if len(targets) == 0:
            raise ValueError(f"No data found for metric: {metric_name}")

        # Create feature matrix
        features = self._create_features(raw_features, targets, dates)

        return np.array(features), np.array(targets), dates

    def _create_features(
        self,
        raw_features: list[dict[str, Any]],
        targets: list[float],
        dates: list[datetime],
    ) -> list[list[float]]:
        """Create feature matrix from raw data."""

        features = []
        self.feature_names = []

        for i, record in enumerate(raw_features):
            feature_row = []

            # Basic metrics features
            for key, value in record.items():
                if key != "date" and isinstance(value, int | float):
                    feature_row.append(float(value))
                    if i == 0:  # First row, collect feature names
                        self.feature_names.append(key)

            # Seasonal features
            if self.config.include_seasonal and i < len(dates):
                date = dates[i]
                feature_row.extend(
                    [
                        date.weekday(),  # Day of week
                        date.day,  # Day of month
                        date.month,  # Month
                        date.quarter,  # Quarter
                    ]
                )
                if i == 0:
                    self.feature_names.extend(
                        ["weekday", "day_of_month", "month", "quarter"]
                    )

            # Lag features
            if self.config.include_lag_features:
                for lag in self.config.lag_periods:
                    if i >= lag:
                        feature_row.append(targets[i - lag])
                        if i == 0:
                            self.feature_names.append(f"lag_{lag}")
                    else:
                        feature_row.append(0)  # Pad with zeros for early periods

            # Rolling statistics
            window_sizes = [7, 14, 30]
            for window in window_sizes:
                if i >= window - 1:
                    window_data = targets[max(0, i - window + 1) : i + 1]
                    feature_row.extend(
                        [
                            np.mean(window_data),
                            np.std(window_data) if len(window_data) > 1 else 0,
                        ]
                    )
                    if i == 0:
                        self.feature_names.extend(
                            [f"rolling_mean_{window}", f"rolling_std_{window}"]
                        )
                else:
                    feature_row.extend([0, 0])

            features.append(feature_row)

        return features

    def _train_model(
        self, features: np.ndarray, targets: np.ndarray
    ) -> tuple[Any, StandardScaler, ModelMetrics]:
        """Train ML model."""

        # Scale features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            features_scaled,
            targets,
            test_size=self.config.test_size,
            random_state=self.config.random_state,
        )

        # Select and train model
        model = self._get_model()
        model.fit(X_train, y_train)

        # Evaluate model
        y_pred = model.predict(X_test)

        # Cross-validation
        cv_scores = cross_val_score(
            model, features_scaled, targets, cv=self.config.cv_folds, scoring="r2"
        )

        metrics = ModelMetrics(
            mse=mean_squared_error(y_test, y_pred),
            rmse=np.sqrt(mean_squared_error(y_test, y_pred)),
            mae=mean_absolute_error(y_test, y_pred),
            r2=r2_score(y_test, y_pred),
            cv_score_mean=cv_scores.mean(),
            cv_score_std=cv_scores.std(),
            training_samples=len(X_train),
        )

        return model, scaler, metrics

    def _get_model(self):
        """Get ML model based on configuration."""

        if self.config.model_type == "random_forest":
            return RandomForestRegressor(
                n_estimators=self.config.rf_n_estimators,
                max_depth=self.config.rf_max_depth,
                random_state=self.config.random_state,
            )
        elif self.config.model_type == "gradient_boosting":
            return GradientBoostingRegressor(
                n_estimators=self.config.gb_n_estimators,
                learning_rate=self.config.gb_learning_rate,
                random_state=self.config.random_state,
            )
        elif self.config.model_type == "ridge":
            return Ridge(alpha=self.config.ridge_alpha)
        else:  # Default to linear regression
            return LinearRegression()

    def _generate_predictions(
        self,
        model: Any,
        scaler: StandardScaler,
        features: np.ndarray,
        dates: list[datetime],
    ) -> dict[str, list]:
        """Generate future predictions."""

        # Use last known features as base for prediction
        last_features = features[-1:].copy()

        predictions = []
        forecast_dates = []

        # Generate predictions for forecast horizon
        for i in range(self.config.forecast_horizon):
            # Scale features
            features_scaled = scaler.transform(last_features)

            # Make prediction
            pred = model.predict(features_scaled)[0]
            predictions.append(pred)

            # Generate forecast date
            last_date = dates[-1] if dates else datetime.now()
            forecast_date = last_date + timedelta(days=i + 1)
            forecast_dates.append(forecast_date)

            # Update features for next prediction (simple approach)
            # In practice, this would be more sophisticated
            if len(last_features[0]) > 0:
                last_features[0][0] = pred  # Update primary metric

        # Calculate confidence intervals (simplified)
        std_error = np.std(predictions) * 0.1  # Simplified error estimation
        z_score = 1.96 if self.config.confidence_level == 0.95 else 2.58

        lower_bound = [p - z_score * std_error for p in predictions]
        upper_bound = [p + z_score * std_error for p in predictions]

        return {
            "point_forecast": predictions,
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
            "forecast_dates": forecast_dates,
        }

    def _calculate_feature_importance(self, model: Any) -> list[FeatureImportance]:
        """Calculate feature importance."""

        importance_scores = []

        if hasattr(model, "feature_importances_"):
            importance_scores = model.feature_importances_
        elif hasattr(model, "coef_"):
            importance_scores = np.abs(model.coef_)
        else:
            # No importance available
            return []

        # Create importance objects
        importance_list = []
        for i, score in enumerate(importance_scores):
            if i < len(self.feature_names):
                importance_list.append(
                    FeatureImportance(
                        feature_name=self.feature_names[i],
                        importance=float(score),
                        rank=0,  # Will be set after sorting
                    )
                )

        # Sort by importance and set ranks
        importance_list.sort(key=lambda x: x.importance, reverse=True)
        for i, imp in enumerate(importance_list):
            imp.rank = i + 1

        return importance_list

    def _generate_insights(
        self, predictions: dict[str, list], targets: np.ndarray, metrics: ModelMetrics
    ) -> dict[str, Any]:
        """Generate insights from predictions."""

        point_forecast = predictions["point_forecast"]

        # Trend direction
        if len(point_forecast) > 1:
            trend_slope = np.polyfit(range(len(point_forecast)), point_forecast, 1)[0]
            if trend_slope > 0.01:
                trend_direction = "increasing"
            elif trend_slope < -0.01:
                trend_direction = "decreasing"
            else:
                trend_direction = "stable"
        else:
            trend_direction = "stable"

        # Volatility level
        forecast_std = np.std(point_forecast)
        historical_std = np.std(targets)
        volatility_ratio = forecast_std / historical_std if historical_std > 0 else 1

        if volatility_ratio > 1.5:
            volatility_level = "high"
        elif volatility_ratio > 1.2:
            volatility_level = "moderate"
        else:
            volatility_level = "low"

        # Confidence score
        confidence_score = (
            max(0, min(1, metrics.r2)) * 0.7 + (1 - min(1, metrics.cv_score_std)) * 0.3
        )

        # Recommendations
        recommendations = []
        if metrics.r2 < 0.5:
            recommendations.append(
                "Model accuracy is low - consider collecting more data"
            )
        if trend_direction == "decreasing":
            recommendations.append(
                "Declining trend detected - investigate potential causes"
            )
        if volatility_level == "high":
            recommendations.append(
                "High volatility predicted - increase monitoring frequency"
            )
        if confidence_score > 0.8:
            recommendations.append(
                "High confidence predictions - suitable for planning"
            )

        return {
            "trend_direction": trend_direction,
            "volatility_level": volatility_level,
            "confidence_score": confidence_score,
            "recommendations": recommendations,
        }

    def save_model(self, model_path: str, model: Any, scaler: StandardScaler):
        """Save trained model and scaler."""
        joblib.dump(
            {
                "model": model,
                "scaler": scaler,
                "feature_names": self.feature_names,
                "config": self.config,
            },
            model_path,
        )
        logger.info(f"Model saved to: {model_path}")

    def load_model(self, model_path: str):
        """Load trained model and scaler."""
        saved_data = joblib.load(model_path)
        self.feature_names = saved_data["feature_names"]
        self.config = saved_data["config"]
        logger.info(f"Model loaded from: {model_path}")
        return saved_data["model"], saved_data["scaler"]
