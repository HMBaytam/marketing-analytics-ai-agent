"""Core modules for the Marketing AI Agent."""

from .config import Config, load_config, save_config
from .logging import initialize_logging, get_logger, PerformanceLogger, AuditLogger
from .exceptions import (
    MarketingAIAgentError, ConfigurationError, APIError, GA4APIError, 
    GoogleAdsAPIError, DataValidationError, ModelError, AnalyticsError, 
    OptimizationError, ReportGenerationError, FileOperationError,
    RetryableError, RateLimitError, TemporaryServiceError
)
from .error_handlers import (
    RetryConfig, retry_on_error, circuit_breaker, handle_errors, 
    validate_input, error_reporter
)
from .monitoring import (
    HealthCheck, PerformanceMetrics, MetricsCollector, HealthChecker, 
    SystemMonitor, system_monitor
)

__all__ = [
    # Config
    "Config", "load_config", "save_config",
    
    # Logging
    "initialize_logging", "get_logger", "PerformanceLogger", "AuditLogger",
    
    # Exceptions
    "MarketingAIAgentError", "ConfigurationError", "APIError", "GA4APIError",
    "GoogleAdsAPIError", "DataValidationError", "ModelError", "AnalyticsError",
    "OptimizationError", "ReportGenerationError", "FileOperationError",
    "RetryableError", "RateLimitError", "TemporaryServiceError",
    
    # Error Handlers
    "RetryConfig", "retry_on_error", "circuit_breaker", "handle_errors",
    "validate_input", "error_reporter",
    
    # Monitoring
    "HealthCheck", "PerformanceMetrics", "MetricsCollector", "HealthChecker",
    "SystemMonitor", "system_monitor"
]