"""Error handling utilities and decorators."""

import functools
import time
import random
from typing import Callable, Type, Tuple, Optional, Any, Dict
from datetime import datetime, timedelta

from .exceptions import (
    MarketingAIAgentError, APIError, RateLimitError, TemporaryServiceError,
    RetryableError, ConfigurationError, DataValidationError
)
from .logging import get_logger, AuditLogger


class RetryConfig:
    """Configuration for retry behavior."""
    
    def __init__(
        self,
        max_attempts: int = 3,
        initial_delay: float = 1.0,
        max_delay: float = 60.0,
        backoff_factor: float = 2.0,
        jitter: bool = True,
        retryable_exceptions: Tuple[Type[Exception], ...] = (RetryableError,)
    ):
        self.max_attempts = max_attempts
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.backoff_factor = backoff_factor
        self.jitter = jitter
        self.retryable_exceptions = retryable_exceptions


def retry_on_error(config: RetryConfig = None):
    """Decorator for automatic retry with exponential backoff."""
    if config is None:
        config = RetryConfig()
    
    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            logger = get_logger(func.__module__)
            last_exception = None
            
            for attempt in range(config.max_attempts):
                try:
                    if attempt > 0:
                        delay = min(
                            config.initial_delay * (config.backoff_factor ** (attempt - 1)),
                            config.max_delay
                        )
                        
                        if config.jitter:
                            delay *= (0.5 + random.random())
                        
                        logger.info(f"Retrying {func.__name__} in {delay:.2f}s (attempt {attempt + 1}/{config.max_attempts})")
                        time.sleep(delay)
                    
                    result = func(*args, **kwargs)
                    
                    if attempt > 0:
                        logger.info(f"Successfully completed {func.__name__} after {attempt + 1} attempts")
                    
                    return result
                
                except config.retryable_exceptions as e:
                    last_exception = e
                    logger.warning(f"Attempt {attempt + 1} failed for {func.__name__}: {str(e)}")
                    
                    # Handle specific retry delays
                    if isinstance(e, RateLimitError) and e.retry_after:
                        logger.info(f"Rate limited, waiting {e.retry_after}s before retry")
                        time.sleep(e.retry_after)
                    
                    if attempt == config.max_attempts - 1:
                        logger.error(f"All {config.max_attempts} attempts failed for {func.__name__}")
                        raise
                
                except Exception as e:
                    logger.error(f"Non-retryable error in {func.__name__}: {str(e)}")
                    raise
            
            # Should never reach here, but just in case
            raise last_exception
        
        return wrapper
    return decorator


class CircuitBreaker:
    """Circuit breaker pattern implementation."""
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
        expected_exception: Type[Exception] = Exception
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        self.failure_count = 0
        self.last_failure_time = None
        self.state = 'CLOSED'  # CLOSED, OPEN, HALF_OPEN
        self.logger = get_logger(__name__)
    
    def __call__(self, func: Callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if self.state == 'OPEN':
                if self._should_attempt_reset():
                    self.state = 'HALF_OPEN'
                    self.logger.info(f"Circuit breaker for {func.__name__} transitioning to HALF_OPEN")
                else:
                    raise MarketingAIAgentError(
                        f"Circuit breaker OPEN for {func.__name__}. "
                        f"Will retry after {self.recovery_timeout}s",
                        error_code="CIRCUIT_BREAKER_OPEN"
                    )
            
            try:
                result = func(*args, **kwargs)
                self._on_success(func.__name__)
                return result
            except self.expected_exception as e:
                self._on_failure(func.__name__)
                raise
        
        return wrapper
    
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt a reset."""
        return (
            self.last_failure_time and
            datetime.now() - self.last_failure_time > timedelta(seconds=self.recovery_timeout)
        )
    
    def _on_success(self, func_name: str):
        """Handle successful function call."""
        if self.state == 'HALF_OPEN':
            self.state = 'CLOSED'
            self.failure_count = 0
            self.logger.info(f"Circuit breaker for {func_name} reset to CLOSED")
    
    def _on_failure(self, func_name: str):
        """Handle failed function call."""
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        
        if self.failure_count >= self.failure_threshold:
            self.state = 'OPEN'
            self.logger.warning(f"Circuit breaker for {func_name} OPENED after {self.failure_count} failures")


def circuit_breaker(
    failure_threshold: int = 5,
    recovery_timeout: int = 60,
    expected_exception: Type[Exception] = Exception
):
    """Decorator for circuit breaker pattern."""
    breaker = CircuitBreaker(failure_threshold, recovery_timeout, expected_exception)
    return breaker


class ErrorReporter:
    """Centralized error reporting and monitoring."""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.audit_logger = AuditLogger("errors")
        self.error_counts = {}
        self.recent_errors = []
        self.max_recent_errors = 100
    
    def report_error(
        self,
        error: Exception,
        context: Dict[str, Any] = None,
        severity: str = "ERROR",
        notify: bool = False
    ):
        """Report an error with context and tracking."""
        error_type = type(error).__name__
        error_message = str(error)
        
        # Update error counts
        self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1
        
        # Add to recent errors
        error_info = {
            "timestamp": datetime.now().isoformat(),
            "error_type": error_type,
            "message": error_message,
            "severity": severity,
            "context": context or {}
        }
        
        self.recent_errors.append(error_info)
        if len(self.recent_errors) > self.max_recent_errors:
            self.recent_errors.pop(0)
        
        # Log the error
        self.logger.error(f"Error reported: {error_message}", extra={
            "error_type": error_type,
            "severity": severity,
            "context": context,
            "error_count": self.error_counts[error_type]
        })
        
        # Audit log
        self.audit_logger.logger.error("Error reported", extra={
            "event_type": "error_reported",
            "error_type": error_type,
            "message": error_message,
            "severity": severity,
            "context": context
        })
        
        # Handle specific error types
        if isinstance(error, MarketingAIAgentError):
            self._handle_marketing_ai_error(error)
        
        # Notification logic (placeholder)
        if notify:
            self._send_notification(error_info)
    
    def _handle_marketing_ai_error(self, error: MarketingAIAgentError):
        """Handle Marketing AI specific errors."""
        if isinstance(error, APIError):
            self._handle_api_error(error)
        elif isinstance(error, ConfigurationError):
            self._handle_config_error(error)
        elif isinstance(error, DataValidationError):
            self._handle_data_validation_error(error)
    
    def _handle_api_error(self, error: APIError):
        """Handle API errors with specific logic."""
        if error.status_code == 429:  # Rate limit
            self.logger.warning(f"Rate limit exceeded for {error.api_name} API")
        elif error.status_code and error.status_code >= 500:
            self.logger.warning(f"Server error from {error.api_name} API: {error.status_code}")
    
    def _handle_config_error(self, error: ConfigurationError):
        """Handle configuration errors."""
        self.logger.critical(f"Configuration error for key '{error.config_key}': {error.message}")
    
    def _handle_data_validation_error(self, error: DataValidationError):
        """Handle data validation errors."""
        self.logger.warning(f"Data validation failed for field '{error.field_name}': {error.message}")
    
    def _send_notification(self, error_info: Dict[str, Any]):
        """Send error notification (placeholder for future implementation)."""
        # This would integrate with email, Slack, PagerDuty, etc.
        self.logger.info("Error notification sent", extra=error_info)
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get summary of recent errors."""
        return {
            "total_error_types": len(self.error_counts),
            "error_counts": self.error_counts.copy(),
            "recent_errors": self.recent_errors[-10:],  # Last 10 errors
            "most_common_error": max(self.error_counts.items(), key=lambda x: x[1]) if self.error_counts else None
        }


# Global error reporter instance
error_reporter = ErrorReporter()


def handle_errors(
    reraise: bool = True,
    report: bool = True,
    severity: str = "ERROR",
    notify: bool = False
):
    """Decorator for comprehensive error handling."""
    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if report:
                    context = {
                        "function": func.__name__,
                        "module": func.__module__,
                        "args_count": len(args),
                        "kwargs_keys": list(kwargs.keys()) if kwargs else []
                    }
                    error_reporter.report_error(e, context, severity, notify)
                
                if reraise:
                    raise
                
                return None
        return wrapper
    return decorator


def validate_input(validation_func: Callable, error_message: str = None):
    """Decorator for input validation."""
    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                if not validation_func(*args, **kwargs):
                    message = error_message or f"Input validation failed for {func.__name__}"
                    raise DataValidationError(message, context={
                        "function": func.__name__,
                        "args_count": len(args),
                        "kwargs": kwargs
                    })
                return func(*args, **kwargs)
            except DataValidationError:
                raise
            except Exception as e:
                raise DataValidationError(f"Validation error in {func.__name__}: {str(e)}")
        return wrapper
    return decorator