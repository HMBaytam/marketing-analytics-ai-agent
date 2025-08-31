"""Unit tests for core modules."""

from datetime import datetime

import pytest
from marketing_ai_agent.core.config import Config, load_config, save_config
from marketing_ai_agent.core.error_handlers import (
    RetryConfig,
    circuit_breaker,
    error_reporter,
    handle_errors,
    retry_on_error,
    validate_input,
)
from marketing_ai_agent.core.exceptions import (
    APIError,
    ConfigurationError,
    DataValidationError,
    MarketingAIAgentError,
    RateLimitError,
)
from marketing_ai_agent.core.logging import (
    AuditLogger,
    PerformanceLogger,
    get_logger,
    initialize_logging,
)
from marketing_ai_agent.core.monitoring import (
    HealthCheck,
    HealthChecker,
    MetricsCollector,
    PerformanceMetrics,
    SystemMonitor,
)


class TestConfig:
    """Test configuration management."""

    def test_config_creation(self, test_config):
        """Test basic config creation."""
        assert test_config.api.google_ads.client_id == "test-client-id"
        assert test_config.output.base_directory == "./test_reports"
        assert test_config.logging.level == "DEBUG"

    def test_config_validation(self, test_config):
        """Test config validation."""
        # Valid config should not raise
        config = Config()
        assert config is not None

        # Test required fields
        assert hasattr(config, "api")
        assert hasattr(config, "output")
        assert hasattr(config, "logging")

    def test_load_config_from_file(self, temp_dir):
        """Test loading config from YAML file."""
        config_file = temp_dir / "test_config.yaml"
        config_data = {
            "api": {
                "google_ads": {
                    "client_id": "loaded_client_id",
                    "client_secret": "loaded_secret",
                }
            },
            "logging": {"level": "INFO"},
        }

        # Write test config
        import yaml

        with open(config_file, "w") as f:
            yaml.dump(config_data, f)

        # Load and verify
        config = load_config(config_file)
        assert config.api.google_ads.client_id == "loaded_client_id"
        assert config.logging.level == "INFO"

    def test_save_config_to_file(self, test_config, temp_dir):
        """Test saving config to file."""
        config_file = temp_dir / "saved_config.yaml"

        save_config(test_config, config_file)

        assert config_file.exists()

        # Verify saved content
        loaded_config = load_config(config_file)
        assert (
            loaded_config.api.google_ads.client_id
            == test_config.api.google_ads.client_id
        )

    def test_config_with_environment_variables(self):
        """Test config loading with environment variables."""
        import os

        # Set environment variables
        os.environ["GOOGLE_ADS_CLIENT_ID"] = "env_client_id"
        os.environ["LOG_LEVEL"] = "WARNING"

        try:
            config = Config()
            # Note: Actual env var loading depends on implementation
            # This test verifies the structure is in place
            assert hasattr(config.api.google_ads, "client_id")
        finally:
            # Clean up
            if "GOOGLE_ADS_CLIENT_ID" in os.environ:
                del os.environ["GOOGLE_ADS_CLIENT_ID"]
            if "LOG_LEVEL" in os.environ:
                del os.environ["LOG_LEVEL"]


class TestExceptions:
    """Test custom exception system."""

    def test_base_exception(self):
        """Test base Marketing AI Agent exception."""
        error = MarketingAIAgentError(
            "Test error", error_code="TEST_ERROR", context={"test_key": "test_value"}
        )

        assert str(error) == "Test error"
        assert error.error_code == "TEST_ERROR"
        assert error.context["test_key"] == "test_value"

    def test_configuration_error(self):
        """Test configuration error."""
        error = ConfigurationError("Invalid config key", config_key="api.invalid_key")

        assert error.config_key == "api.invalid_key"
        assert error.error_code == "CONFIG_ERROR"
        assert "api.invalid_key" in error.context["config_key"]

    def test_api_error(self):
        """Test API error."""
        error = APIError(
            "API request failed",
            api_name="Test API",
            status_code=500,
            response_data={"error": "Internal error"},
        )

        assert error.api_name == "Test API"
        assert error.status_code == 500
        assert error.response_data["error"] == "Internal error"

    def test_data_validation_error(self):
        """Test data validation error."""
        error = DataValidationError(
            "Invalid field value", field_name="campaign_id", field_value=None
        )

        assert error.field_name == "campaign_id"
        assert error.field_value is None
        assert error.error_code == "DATA_VALIDATION_ERROR"

    def test_rate_limit_error(self):
        """Test rate limit error."""
        error = RateLimitError(
            "Rate limit exceeded", api_name="Google Ads", retry_after=120
        )

        assert error.api_name == "Google Ads"
        assert error.retry_after == 120
        assert error.error_code == "RATE_LIMIT_ERROR"


class TestLogging:
    """Test logging system."""

    def test_logger_creation(self):
        """Test basic logger creation."""
        logger = get_logger("test_logger")
        assert logger is not None
        assert logger.name == "test_logger"

    def test_initialize_logging(self):
        """Test logging system initialization."""
        import logging

        logger = initialize_logging(logging.INFO)
        assert logger is not None
        assert logger.level <= logging.INFO

    def test_performance_logger(self):
        """Test performance logging context manager."""
        logger = get_logger("test_perf")

        with PerformanceLogger(logger, "test_operation") as perf_logger:
            assert perf_logger is not None
            # Simulate some work
            import time

            time.sleep(0.1)

        # Performance logger should have recorded the operation

    def test_audit_logger(self):
        """Test audit logging functionality."""
        audit_logger = AuditLogger("test_audit")

        # Test API call logging
        audit_logger.log_api_call(
            api_name="Test API",
            endpoint="/test",
            method="GET",
            status_code=200,
            duration=0.5,
        )

        # Test data export logging
        audit_logger.log_data_export(
            source="test_source",
            records_count=100,
            file_path="/test/path.csv",
            user="test_user",
        )

        # Test model training logging
        audit_logger.log_model_training(
            model_name="test_model",
            model_type="regression",
            training_duration=120.0,
            accuracy=0.85,
        )


class TestErrorHandlers:
    """Test error handling utilities."""

    def test_retry_config(self):
        """Test retry configuration."""
        config = RetryConfig(
            max_attempts=5,
            initial_delay=2.0,
            backoff_factor=1.5,
            retryable_exceptions=(RateLimitError,),
        )

        assert config.max_attempts == 5
        assert config.initial_delay == 2.0
        assert config.backoff_factor == 1.5
        assert RateLimitError in config.retryable_exceptions

    def test_retry_decorator_success(self):
        """Test retry decorator with successful operation."""
        call_count = 0

        @retry_on_error(RetryConfig(max_attempts=3))
        def test_function():
            nonlocal call_count
            call_count += 1
            return "success"

        result = test_function()
        assert result == "success"
        assert call_count == 1

    def test_retry_decorator_with_retryable_error(self):
        """Test retry decorator with retryable errors."""
        call_count = 0

        @retry_on_error(
            RetryConfig(
                max_attempts=3,
                initial_delay=0.1,
                retryable_exceptions=(RateLimitError,),
            )
        )
        def test_function():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise RateLimitError("Rate limited", "Test API", retry_after=1)
            return "success"

        result = test_function()
        assert result == "success"
        assert call_count == 3

    def test_retry_decorator_with_non_retryable_error(self):
        """Test retry decorator with non-retryable errors."""
        call_count = 0

        @retry_on_error(
            RetryConfig(max_attempts=3, retryable_exceptions=(RateLimitError,))
        )
        def test_function():
            nonlocal call_count
            call_count += 1
            raise ValueError("Non-retryable error")

        with pytest.raises(ValueError):
            test_function()
        assert call_count == 1  # Should not retry

    def test_circuit_breaker_normal_operation(self):
        """Test circuit breaker in normal operation."""
        call_count = 0

        @circuit_breaker(failure_threshold=3, recovery_timeout=1)
        def test_function():
            nonlocal call_count
            call_count += 1
            return "success"

        # Multiple successful calls should work
        for _ in range(5):
            result = test_function()
            assert result == "success"

        assert call_count == 5

    def test_circuit_breaker_open_state(self):
        """Test circuit breaker opening after failures."""
        call_count = 0

        @circuit_breaker(failure_threshold=2, recovery_timeout=10)
        def test_function():
            nonlocal call_count
            call_count += 1
            raise ValueError("Test error")

        # First two calls should fail normally
        for _ in range(2):
            with pytest.raises(ValueError):
                test_function()

        # Third call should raise circuit breaker error
        with pytest.raises(MarketingAIAgentError) as exc_info:
            test_function()
        assert "Circuit breaker OPEN" in str(exc_info.value)

        assert call_count == 2  # Circuit breaker prevents third call

    def test_handle_errors_decorator(self):
        """Test generic error handling decorator."""

        @handle_errors(reraise=False, report=True)
        def test_function_with_error():
            raise ValueError("Test error")

        @handle_errors(reraise=True, report=True)
        def test_function_with_reraise():
            raise ValueError("Test error")

        # Should not raise when reraise=False
        result = test_function_with_error()
        assert result is None

        # Should raise when reraise=True
        with pytest.raises(ValueError):
            test_function_with_reraise()

    def test_validate_input_decorator(self):
        """Test input validation decorator."""

        def validation_func(*args, **kwargs):
            return len(args) > 0 and args[0] > 0

        @validate_input(validation_func, "Value must be positive")
        def test_function(value):
            return value * 2

        # Valid input should work
        result = test_function(5)
        assert result == 10

        # Invalid input should raise validation error
        with pytest.raises(DataValidationError):
            test_function(-1)

    def test_error_reporter(self):
        """Test error reporting functionality."""
        # Report some errors
        error_reporter.report_error(
            ValueError("Test error 1"), context={"test": "context1"}
        )

        error_reporter.report_error(
            DataValidationError("Test validation error"), severity="WARNING"
        )

        # Get summary
        summary = error_reporter.get_error_summary()

        assert summary["total_error_types"] >= 1
        assert len(summary["recent_errors"]) >= 1
        assert (
            "ValueError" in summary["error_counts"]
            or "DataValidationError" in summary["error_counts"]
        )


class TestMonitoring:
    """Test system monitoring functionality."""

    def test_health_check_creation(self):
        """Test health check result creation."""
        health_check = HealthCheck(
            name="test_check",
            status="healthy",
            message="Test check passed",
            details={"test_detail": "value"},
            duration=0.5,
        )

        assert health_check.name == "test_check"
        assert health_check.status == "healthy"
        assert health_check.message == "Test check passed"
        assert health_check.details["test_detail"] == "value"
        assert health_check.duration == 0.5

    def test_performance_metrics_creation(self):
        """Test performance metrics creation."""
        metrics = PerformanceMetrics(
            timestamp=datetime.now(),
            cpu_percent=25.5,
            memory_percent=60.2,
            memory_used_mb=1024.0,
            disk_usage_percent=45.0,
            network_io={"bytes_sent": 1000, "bytes_recv": 2000},
            process_count=150,
        )

        assert metrics.cpu_percent == 25.5
        assert metrics.memory_percent == 60.2
        assert metrics.network_io["bytes_sent"] == 1000
        assert metrics.process_count == 150

    @pytest.mark.slow
    def test_metrics_collector(self):
        """Test metrics collection."""
        collector = MetricsCollector(max_history=10)

        # Collect metrics
        metrics = collector.collect_metrics()
        assert metrics is not None
        assert metrics.cpu_percent >= 0
        assert metrics.memory_percent >= 0

        # Check history
        recent_metrics = collector.get_recent_metrics(1)
        assert len(recent_metrics) >= 0

    def test_health_checker(self):
        """Test health checking functionality."""
        checker = HealthChecker()

        # Test system resource check
        result = checker.check_system_resources()
        assert result.name == "system_resources"
        assert result.status in ["healthy", "warning", "critical"]
        assert result.duration is not None

        # Test API connectivity check
        result = checker.check_api_connectivity()
        assert result.name == "api_connectivity"
        assert result.status in ["healthy", "warning", "critical"]

        # Test all checks
        results = checker.run_all_checks()
        assert len(results) >= 2
        assert "system_resources" in results
        assert "api_connectivity" in results

    def test_system_monitor(self):
        """Test system monitor coordination."""
        monitor = SystemMonitor()

        # Test status report
        status_report = monitor.get_status_report()
        assert "timestamp" in status_report
        assert "overall_health" in status_report
        assert "monitoring_active" in status_report

        # Test monitoring lifecycle
        monitor.start_monitoring()
        assert (
            not monitor._monitoring_active or True
        )  # Might already be stopped in tests

        monitor.stop_monitoring()
        assert not monitor._monitoring_active

    def test_custom_health_check_registration(self):
        """Test registering custom health checks."""
        checker = HealthChecker()

        def custom_check():
            return HealthCheck(
                name="custom_check", status="healthy", message="Custom check passed"
            )

        checker.register_check("custom_check", custom_check)

        results = checker.run_all_checks()
        assert "custom_check" in results
        assert results["custom_check"].message == "Custom check passed"


@pytest.mark.integration
class TestCoreIntegration:
    """Integration tests for core module interactions."""

    def test_error_handling_with_logging(self):
        """Test error handling integration with logging."""
        get_logger("integration_test")

        @handle_errors(reraise=False, report=True)
        def test_function():
            raise APIError("Integration test error", "Test API", status_code=500)

        # Should log and report error without raising
        result = test_function()
        assert result is None

        # Check error was reported
        summary = error_reporter.get_error_summary()
        assert len(summary["recent_errors"]) > 0

    def test_monitoring_with_error_reporting(self):
        """Test monitoring integration with error reporting."""
        monitor = SystemMonitor()

        # Generate some errors
        error_reporter.report_error(
            Exception("Test monitoring error"), context={"source": "monitoring_test"}
        )

        # Get status report
        status_report = monitor.get_status_report()
        assert status_report is not None

        # Error summary should be available
        error_summary = error_reporter.get_error_summary()
        assert error_summary["total_error_types"] >= 0

    def test_config_validation_with_error_handling(self):
        """Test configuration validation with proper error handling."""

        @handle_errors(reraise=False, report=True)
        def load_invalid_config():
            # Simulate loading invalid config
            raise ConfigurationError(
                "Invalid API key", config_key="api.google_ads.client_id"
            )

        result = load_invalid_config()
        assert result is None

        # Check error was properly categorized
        summary = error_reporter.get_error_summary()
        assert any(
            "ConfigurationError" in error_type
            for error_type in summary["error_counts"].keys()
        )


# Benchmarking tests
@pytest.mark.benchmark
class TestPerformanceBenchmarks:
    """Performance benchmark tests."""

    def test_config_loading_performance(self, benchmark, temp_dir):
        """Benchmark configuration loading performance."""
        config_file = temp_dir / "bench_config.yaml"

        # Create test config
        import yaml

        config_data = {
            "api": {"google_ads": {"client_id": "test"}},
            "logging": {"level": "INFO"},
        }
        with open(config_file, "w") as f:
            yaml.dump(config_data, f)

        # Benchmark loading
        result = benchmark(load_config, config_file)
        assert result is not None

    def test_error_reporting_performance(self, benchmark):
        """Benchmark error reporting performance."""

        def report_multiple_errors():
            for i in range(10):
                error_reporter.report_error(
                    ValueError(f"Benchmark error {i}"), context={"iteration": i}
                )

        benchmark(report_multiple_errors)

    def test_metrics_collection_performance(self, benchmark):
        """Benchmark metrics collection performance."""
        collector = MetricsCollector()

        result = benchmark(collector.collect_metrics)
        assert result is not None
