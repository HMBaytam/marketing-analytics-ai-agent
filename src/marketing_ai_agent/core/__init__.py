"""Core modules for the Marketing AI Agent."""

from .config import Config, load_config, save_config
from .error_handlers import (
    RetryConfig,
    circuit_breaker,
    error_reporter,
    handle_errors,
    retry_on_error,
    validate_input,
)
from .exceptions import (
    AnalyticsError,
    APIError,
    ConfigurationError,
    DataValidationError,
    FileOperationError,
    GA4APIError,
    GoogleAdsAPIError,
    MarketingAIAgentError,
    ModelError,
    OptimizationError,
    RateLimitError,
    ReportGenerationError,
    RetryableError,
    TemporaryServiceError,
)
from .logging import AuditLogger, PerformanceLogger, get_logger, initialize_logging
from .monitoring import (
    HealthCheck,
    HealthChecker,
    MetricsCollector,
    PerformanceMetrics,
    SystemMonitor,
    system_monitor,
)

__all__ = [
    # Config
    "Config",
    "load_config",
    "save_config",
    # Logging
    "initialize_logging",
    "get_logger",
    "PerformanceLogger",
    "AuditLogger",
    # Exceptions
    "MarketingAIAgentError",
    "ConfigurationError",
    "APIError",
    "GA4APIError",
    "GoogleAdsAPIError",
    "DataValidationError",
    "ModelError",
    "AnalyticsError",
    "OptimizationError",
    "ReportGenerationError",
    "FileOperationError",
    "RetryableError",
    "RateLimitError",
    "TemporaryServiceError",
    # Error Handlers
    "RetryConfig",
    "retry_on_error",
    "circuit_breaker",
    "handle_errors",
    "validate_input",
    "error_reporter",
    # Monitoring
    "HealthCheck",
    "PerformanceMetrics",
    "MetricsCollector",
    "HealthChecker",
    "SystemMonitor",
    "system_monitor",
]
