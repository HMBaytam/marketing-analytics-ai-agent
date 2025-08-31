"""Advanced analytics and scoring engines for marketing performance analysis."""

from .performance_scorer import PerformanceScorer, ScoringConfig, PerformanceScore
from .trend_analyzer import TrendAnalyzer, TrendConfig, TrendAnalysis
from .anomaly_detector import AnomalyDetector, AnomalyConfig, AnomalyResult
from .benchmarking import BenchmarkingEngine, BenchmarkConfig, BenchmarkResult
from .predictive_model import PredictiveModel, PredictionConfig, PredictionResult

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
    "PredictionResult"
]