"""Advanced analytics and scoring engines for marketing performance analysis."""

from .anomaly_detector import AnomalyConfig, AnomalyDetector, AnomalyResult
from .benchmarking import BenchmarkConfig, BenchmarkingEngine, BenchmarkResult
from .performance_scorer import PerformanceScore, PerformanceScorer, ScoringConfig
from .predictive_model import PredictionConfig, PredictionResult, PredictiveModel
from .trend_analyzer import TrendAnalysis, TrendAnalyzer, TrendConfig

__all__ = [
    "PerformanceScorer",
    "ScoringConfig",
    "PerformanceScore",
    "TrendAnalyzer",
    "TrendConfig",
    "TrendAnalysis",
    "AnomalyDetector",
    "AnomalyConfig",
    "AnomalyResult",
    "BenchmarkingEngine",
    "BenchmarkConfig",
    "BenchmarkResult",
    "PredictiveModel",
    "PredictionConfig",
    "PredictionResult",
]
