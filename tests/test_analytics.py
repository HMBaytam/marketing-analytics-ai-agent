"""Tests for analytics modules."""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta

from marketing_ai_agent.analytics.performance_scorer import PerformanceScorer
from marketing_ai_agent.analytics.trend_analyzer import TrendAnalyzer
from marketing_ai_agent.analytics.anomaly_detector import AnomalyDetector
from marketing_ai_agent.analytics.predictive_model import PredictiveModel
from marketing_ai_agent.analytics.benchmarking import BenchmarkAnalyzer
from marketing_ai_agent.core.exceptions import AnalyticsError, DataValidationError


@pytest.fixture
def sample_performance_data():
    """Sample performance data for testing."""
    return pd.DataFrame({
        'date': pd.date_range('2024-01-01', periods=30),
        'impressions': np.random.randint(1000, 10000, 30),
        'clicks': np.random.randint(100, 1000, 30),
        'cost': np.random.uniform(50, 500, 30),
        'conversions': np.random.randint(5, 50, 30),
        'revenue': np.random.uniform(100, 1000, 30)
    })


@pytest.fixture
def sample_time_series_data():
    """Sample time series data for testing."""
    dates = pd.date_range('2023-01-01', '2024-01-01', freq='D')
    values = np.random.normal(100, 20, len(dates)) + np.sin(np.arange(len(dates)) * 2 * np.pi / 365) * 10
    
    return pd.DataFrame({
        'date': dates,
        'metric_value': values,
        'secondary_metric': np.random.normal(50, 10, len(dates))
    })


class TestPerformanceScorer:
    """Test performance scoring functionality."""
    
    def test_performance_scorer_initialization(self):
        """Test performance scorer initialization."""
        scorer = PerformanceScorer()
        assert scorer is not None
        assert hasattr(scorer, 'calculate_score')
    
    def test_calculate_basic_metrics(self, sample_performance_data):
        """Test basic performance metrics calculation."""
        scorer = PerformanceScorer()
        
        metrics = scorer.calculate_basic_metrics(sample_performance_data)
        
        # Check required metrics are present
        assert 'ctr' in metrics
        assert 'cpc' in metrics
        assert 'conversion_rate' in metrics
        assert 'cpa' in metrics
        assert 'roas' in metrics
        
        # Check metric values are reasonable
        assert 0 <= metrics['ctr'] <= 100
        assert metrics['cpc'] >= 0
        assert 0 <= metrics['conversion_rate'] <= 100
        assert metrics['cpa'] >= 0
    
    def test_calculate_composite_score(self, sample_performance_data):
        """Test composite performance score calculation."""
        scorer = PerformanceScorer()
        
        score = scorer.calculate_score(sample_performance_data)
        
        assert isinstance(score, dict)
        assert 'overall_score' in score
        assert 'component_scores' in score
        assert 0 <= score['overall_score'] <= 100
        
        # Check component scores
        components = score['component_scores']
        assert 'efficiency' in components
        assert 'volume' in components
        assert 'profitability' in components
    
    def test_score_with_empty_data(self):
        """Test scoring with empty dataset."""
        scorer = PerformanceScorer()
        empty_data = pd.DataFrame()
        
        with pytest.raises(DataValidationError):
            scorer.calculate_score(empty_data)
    
    def test_score_with_missing_columns(self):
        """Test scoring with missing required columns."""
        scorer = PerformanceScorer()
        incomplete_data = pd.DataFrame({
            'impressions': [1000, 2000],
            'clicks': [100, 200]
            # Missing cost, conversions, revenue
        })
        
        with pytest.raises(DataValidationError):
            scorer.calculate_score(incomplete_data)
    
    def test_benchmark_comparison(self, sample_performance_data):
        """Test performance benchmarking."""
        scorer = PerformanceScorer()
        
        benchmarks = {
            'industry_avg_ctr': 2.5,
            'industry_avg_conversion_rate': 3.0,
            'competitor_avg_cpc': 1.50
        }
        
        comparison = scorer.compare_to_benchmarks(sample_performance_data, benchmarks)
        
        assert 'benchmark_scores' in comparison
        assert 'relative_performance' in comparison
        assert isinstance(comparison['benchmark_scores'], dict)


class TestTrendAnalyzer:
    """Test trend analysis functionality."""
    
    def test_trend_analyzer_initialization(self):
        """Test trend analyzer initialization."""
        analyzer = TrendAnalyzer()
        assert analyzer is not None
        assert hasattr(analyzer, 'analyze_trend')
    
    def test_basic_trend_detection(self, sample_time_series_data):
        """Test basic trend detection."""
        analyzer = TrendAnalyzer()
        
        trend = analyzer.analyze_trend(
            sample_time_series_data['date'],
            sample_time_series_data['metric_value']
        )
        
        assert 'direction' in trend
        assert 'strength' in trend
        assert 'significance' in trend
        assert trend['direction'] in ['increasing', 'decreasing', 'stable']
        assert 0 <= trend['strength'] <= 1
    
    def test_seasonal_decomposition(self, sample_time_series_data):
        """Test seasonal pattern detection."""
        analyzer = TrendAnalyzer()
        
        decomposition = analyzer.decompose_seasonality(
            sample_time_series_data['date'],
            sample_time_series_data['metric_value'],
            freq='monthly'
        )
        
        assert 'trend' in decomposition
        assert 'seasonal' in decomposition
        assert 'residual' in decomposition
        assert len(decomposition['trend']) == len(sample_time_series_data)
    
    def test_change_point_detection(self, sample_time_series_data):
        """Test change point detection in time series."""
        analyzer = TrendAnalyzer()
        
        # Add artificial change point
        data = sample_time_series_data.copy()
        midpoint = len(data) // 2
        data.loc[midpoint:, 'metric_value'] *= 1.5  # Increase values after midpoint
        
        change_points = analyzer.detect_change_points(
            data['date'],
            data['metric_value']
        )
        
        assert isinstance(change_points, list)
        if change_points:
            for cp in change_points:
                assert 'date' in cp
                assert 'confidence' in cp
                assert 0 <= cp['confidence'] <= 1
    
    def test_trend_forecasting(self, sample_time_series_data):
        """Test trend-based forecasting."""
        analyzer = TrendAnalyzer()
        
        forecast = analyzer.forecast_trend(
            sample_time_series_data['date'],
            sample_time_series_data['metric_value'],
            periods=30
        )
        
        assert 'forecast' in forecast
        assert 'confidence_intervals' in forecast
        assert len(forecast['forecast']) == 30
    
    def test_correlation_analysis(self, sample_time_series_data):
        """Test correlation analysis between metrics."""
        analyzer = TrendAnalyzer()
        
        correlation = analyzer.analyze_correlation(
            sample_time_series_data['metric_value'],
            sample_time_series_data['secondary_metric']
        )
        
        assert 'correlation_coefficient' in correlation
        assert 'significance' in correlation
        assert 'strength' in correlation
        assert -1 <= correlation['correlation_coefficient'] <= 1


class TestAnomalyDetector:
    """Test anomaly detection functionality."""
    
    def test_anomaly_detector_initialization(self):
        """Test anomaly detector initialization."""
        detector = AnomalyDetector()
        assert detector is not None
        assert hasattr(detector, 'detect_anomalies')
    
    def test_statistical_anomaly_detection(self, sample_time_series_data):
        """Test statistical anomaly detection."""
        detector = AnomalyDetector()
        
        anomalies = detector.detect_anomalies(
            sample_time_series_data['metric_value'],
            method='statistical',
            threshold=2.0
        )
        
        assert isinstance(anomalies, dict)
        assert 'anomaly_indices' in anomalies
        assert 'anomaly_scores' in anomalies
        assert 'threshold_used' in anomalies
        
        # Check anomaly indices are valid
        if anomalies['anomaly_indices']:
            assert all(0 <= idx < len(sample_time_series_data) for idx in anomalies['anomaly_indices'])
    
    def test_isolation_forest_detection(self, sample_time_series_data):
        """Test isolation forest anomaly detection."""
        detector = AnomalyDetector()
        
        # Create feature matrix
        features = sample_time_series_data[['metric_value', 'secondary_metric']]
        
        anomalies = detector.detect_anomalies(
            features,
            method='isolation_forest',
            contamination=0.1
        )
        
        assert isinstance(anomalies, dict)
        assert 'anomaly_indices' in anomalies
        assert 'anomaly_scores' in anomalies
    
    def test_anomaly_with_artificial_outliers(self):
        """Test anomaly detection with known outliers."""
        detector = AnomalyDetector()
        
        # Create data with artificial outliers
        normal_data = np.random.normal(100, 10, 100)
        normal_data[25] = 500  # Outlier
        normal_data[75] = -200  # Outlier
        
        anomalies = detector.detect_anomalies(
            normal_data,
            method='statistical',
            threshold=3.0
        )
        
        # Should detect the outliers
        assert len(anomalies['anomaly_indices']) >= 1
        assert 25 in anomalies['anomaly_indices'] or 75 in anomalies['anomaly_indices']
    
    def test_time_series_anomaly_detection(self, sample_time_series_data):
        """Test time series specific anomaly detection."""
        detector = AnomalyDetector()
        
        anomalies = detector.detect_time_series_anomalies(
            sample_time_series_data['date'],
            sample_time_series_data['metric_value'],
            window_size=7
        )
        
        assert 'anomalies' in anomalies
        assert 'seasonal_anomalies' in anomalies
        assert 'trend_anomalies' in anomalies
    
    def test_anomaly_explanation(self, sample_time_series_data):
        """Test anomaly explanation generation."""
        detector = AnomalyDetector()
        
        # Detect anomalies first
        anomalies = detector.detect_anomalies(
            sample_time_series_data['metric_value'],
            method='statistical'
        )
        
        if anomalies['anomaly_indices']:
            explanations = detector.explain_anomalies(
                sample_time_series_data,
                anomalies['anomaly_indices']
            )
            
            assert isinstance(explanations, list)
            for explanation in explanations:
                assert 'index' in explanation
                assert 'severity' in explanation
                assert 'possible_causes' in explanation


class TestPredictiveModel:
    """Test predictive modeling functionality."""
    
    def test_predictive_model_initialization(self):
        """Test predictive model initialization."""
        model = PredictiveModel()
        assert model is not None
        assert hasattr(model, 'train')
        assert hasattr(model, 'predict')
    
    def test_linear_regression_training(self, sample_performance_data):
        """Test linear regression model training."""
        model = PredictiveModel(model_type='linear_regression')
        
        # Prepare features and target
        features = sample_performance_data[['impressions', 'clicks', 'cost']]
        target = sample_performance_data['conversions']
        
        model.train(features, target)
        
        assert model.is_trained()
        assert hasattr(model, '_model')
    
    def test_model_prediction(self, sample_performance_data):
        """Test model prediction."""
        model = PredictiveModel(model_type='linear_regression')
        
        # Train model
        features = sample_performance_data[['impressions', 'clicks', 'cost']]
        target = sample_performance_data['conversions']
        model.train(features, target)
        
        # Make predictions
        predictions = model.predict(features)
        
        assert len(predictions) == len(features)
        assert all(isinstance(pred, (int, float)) for pred in predictions)
    
    def test_model_evaluation(self, sample_performance_data):
        """Test model evaluation metrics."""
        model = PredictiveModel(model_type='linear_regression')
        
        # Train model
        features = sample_performance_data[['impressions', 'clicks', 'cost']]
        target = sample_performance_data['conversions']
        model.train(features, target)
        
        # Evaluate model
        evaluation = model.evaluate(features, target)
        
        assert 'mse' in evaluation
        assert 'r2_score' in evaluation
        assert 'mae' in evaluation
        assert evaluation['mse'] >= 0
        assert -1 <= evaluation['r2_score'] <= 1
    
    def test_feature_importance(self, sample_performance_data):
        """Test feature importance analysis."""
        model = PredictiveModel(model_type='random_forest')
        
        features = sample_performance_data[['impressions', 'clicks', 'cost']]
        target = sample_performance_data['conversions']
        model.train(features, target)
        
        importance = model.get_feature_importance()
        
        assert len(importance) == len(features.columns)
        assert sum(importance.values()) == pytest.approx(1.0, rel=0.1)
    
    def test_time_series_prediction(self, sample_time_series_data):
        """Test time series prediction."""
        model = PredictiveModel(model_type='time_series')
        
        # Train on time series data
        model.train_time_series(
            sample_time_series_data['date'],
            sample_time_series_data['metric_value']
        )
        
        # Predict future values
        future_dates = pd.date_range(
            start=sample_time_series_data['date'].max() + timedelta(days=1),
            periods=30
        )
        
        predictions = model.predict_time_series(future_dates)
        
        assert len(predictions) == len(future_dates)
    
    def test_model_cross_validation(self, sample_performance_data):
        """Test model cross-validation."""
        model = PredictiveModel(model_type='linear_regression')
        
        features = sample_performance_data[['impressions', 'clicks', 'cost']]
        target = sample_performance_data['conversions']
        
        cv_scores = model.cross_validate(features, target, cv=3)
        
        assert 'scores' in cv_scores
        assert 'mean_score' in cv_scores
        assert 'std_score' in cv_scores
        assert len(cv_scores['scores']) == 3


class TestBenchmarkAnalyzer:
    """Test benchmarking functionality."""
    
    def test_benchmark_analyzer_initialization(self):
        """Test benchmark analyzer initialization."""
        analyzer = BenchmarkAnalyzer()
        assert analyzer is not None
        assert hasattr(analyzer, 'compare_to_benchmark')
    
    def test_industry_benchmark_comparison(self, sample_performance_data):
        """Test comparison against industry benchmarks."""
        analyzer = BenchmarkAnalyzer()
        
        industry_benchmarks = {
            'avg_ctr': 2.5,
            'avg_conversion_rate': 3.0,
            'avg_cpc': 1.50,
            'avg_roas': 400
        }
        
        comparison = analyzer.compare_to_benchmark(
            sample_performance_data,
            industry_benchmarks,
            benchmark_type='industry'
        )
        
        assert 'performance_vs_benchmark' in comparison
        assert 'percentile_ranking' in comparison
        assert 'improvement_opportunities' in comparison
    
    def test_competitor_benchmark_comparison(self, sample_performance_data):
        """Test comparison against competitor benchmarks."""
        analyzer = BenchmarkAnalyzer()
        
        # Create mock competitor data
        competitor_data = pd.DataFrame({
            'competitor': ['Competitor A', 'Competitor B', 'Competitor C'],
            'avg_ctr': [2.8, 3.1, 2.3],
            'avg_conversion_rate': [3.5, 2.9, 4.1],
            'avg_cpc': [1.20, 1.80, 1.35]
        })
        
        comparison = analyzer.compare_to_competitors(
            sample_performance_data,
            competitor_data
        )
        
        assert 'competitive_position' in comparison
        assert 'outperforming_metrics' in comparison
        assert 'underperforming_metrics' in comparison
    
    def test_historical_benchmark_comparison(self, sample_performance_data):
        """Test comparison against historical performance."""
        analyzer = BenchmarkAnalyzer()
        
        # Create historical data
        historical_data = sample_performance_data.copy()
        historical_data['period'] = 'historical'
        
        current_data = sample_performance_data.copy()
        current_data['period'] = 'current'
        
        comparison = analyzer.compare_to_historical(
            current_data,
            historical_data
        )
        
        assert 'performance_change' in comparison
        assert 'trend_analysis' in comparison
        assert 'improvement_metrics' in comparison
    
    def test_benchmark_scoring(self, sample_performance_data):
        """Test benchmark scoring calculation."""
        analyzer = BenchmarkAnalyzer()
        
        benchmarks = {
            'excellent': {'ctr': 5.0, 'conversion_rate': 6.0},
            'good': {'ctr': 3.0, 'conversion_rate': 4.0},
            'average': {'ctr': 2.0, 'conversion_rate': 2.5},
            'poor': {'ctr': 1.0, 'conversion_rate': 1.0}
        }
        
        score = analyzer.calculate_benchmark_score(
            sample_performance_data,
            benchmarks
        )
        
        assert 'overall_score' in score
        assert 'category_scores' in score
        assert 'performance_grade' in score
        assert 0 <= score['overall_score'] <= 100


@pytest.mark.integration
class TestAnalyticsIntegration:
    """Integration tests for analytics modules."""
    
    def test_end_to_end_analysis_pipeline(self, sample_performance_data):
        """Test complete analytics pipeline."""
        # Performance scoring
        scorer = PerformanceScorer()
        performance_score = scorer.calculate_score(sample_performance_data)
        
        # Trend analysis
        analyzer = TrendAnalyzer()
        trend = analyzer.analyze_trend(
            sample_performance_data['date'],
            sample_performance_data['conversions']
        )
        
        # Anomaly detection
        detector = AnomalyDetector()
        anomalies = detector.detect_anomalies(
            sample_performance_data['conversions'],
            method='statistical'
        )
        
        # Benchmarking
        benchmark_analyzer = BenchmarkAnalyzer()
        benchmarks = {'avg_conversion_rate': 3.0}
        benchmark_comparison = benchmark_analyzer.compare_to_benchmark(
            sample_performance_data,
            benchmarks
        )
        
        # Verify all components work together
        assert performance_score is not None
        assert trend is not None
        assert anomalies is not None
        assert benchmark_comparison is not None
    
    def test_analytics_with_missing_data(self):
        """Test analytics handling of missing data."""
        # Create data with missing values
        data_with_na = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=10),
            'impressions': [1000, np.nan, 1200, 1100, np.nan, 1300, 1250, 1400, np.nan, 1350],
            'clicks': [100, 120, np.nan, 110, 125, np.nan, 130, 140, 135, np.nan],
            'conversions': [10, 12, 11, np.nan, 13, 14, np.nan, 15, 14, 13]
        })
        
        # Test various components handle missing data gracefully
        scorer = PerformanceScorer()
        detector = AnomalyDetector()
        
        # Should handle missing data without crashing
        try:
            # This might raise DataValidationError, which is acceptable
            scorer.calculate_basic_metrics(data_with_na)
        except DataValidationError:
            pass  # Expected for insufficient data
        
        # Anomaly detection with missing values
        clean_values = data_with_na['conversions'].dropna()
        if len(clean_values) > 5:  # Ensure enough data points
            anomalies = detector.detect_anomalies(clean_values, method='statistical')
            assert anomalies is not None


@pytest.mark.performance
class TestAnalyticsPerformance:
    """Performance tests for analytics modules."""
    
    def test_performance_scoring_large_dataset(self, benchmark):
        """Benchmark performance scoring with large dataset."""
        # Create large dataset
        large_data = pd.DataFrame({
            'date': pd.date_range('2020-01-01', periods=1000),
            'impressions': np.random.randint(1000, 10000, 1000),
            'clicks': np.random.randint(100, 1000, 1000),
            'cost': np.random.uniform(50, 500, 1000),
            'conversions': np.random.randint(5, 50, 1000),
            'revenue': np.random.uniform(100, 1000, 1000)
        })
        
        scorer = PerformanceScorer()
        
        def score_large_dataset():
            return scorer.calculate_score(large_data)
        
        result = benchmark(score_large_dataset)
        assert result is not None
    
    def test_anomaly_detection_performance(self, benchmark):
        """Benchmark anomaly detection performance."""
        # Create large time series
        large_timeseries = np.random.normal(100, 20, 5000)
        
        detector = AnomalyDetector()
        
        def detect_anomalies_large():
            return detector.detect_anomalies(large_timeseries, method='statistical')
        
        result = benchmark(detect_anomalies_large)
        assert result is not None