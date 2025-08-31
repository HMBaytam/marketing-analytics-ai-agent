# Error Handling and Logging System

The Marketing AI Agent includes a comprehensive error handling and logging system designed for production-ready applications with robust monitoring, structured logging, and intelligent error recovery.

## Overview

The system provides:
- **Custom exception hierarchy** with contextual information
- **Structured logging** with JSON output and multiple handlers
- **Automatic retry mechanisms** with exponential backoff
- **Circuit breaker pattern** for fault tolerance
- **System monitoring** with health checks and metrics
- **Error reporting and analytics** for operational insights
- **Performance tracking** with automatic instrumentation

## Architecture

### Core Components

#### 1. Exception System (`exceptions.py`)
Custom exception hierarchy providing detailed error context:

```python
from marketing_ai_agent.core.exceptions import (
    MarketingAIAgentError,    # Base exception
    ConfigurationError,       # Configuration issues
    APIError,                # API-related failures
    GA4APIError,             # Google Analytics specific
    GoogleAdsAPIError,       # Google Ads specific
    DataValidationError,     # Data validation failures
    ModelError,              # ML model issues
    OptimizationError,       # Optimization failures
    RetryableError,          # Errors that can be retried
    RateLimitError,          # API rate limiting
    TemporaryServiceError    # Temporary service issues
)
```

**Key Features:**
- Structured error context with metadata
- Error codes for categorization
- Hierarchical inheritance for specific handling
- Integration with logging and monitoring

#### 2. Logging System (`logging.py`)
Multi-level logging with structured output:

```python
from marketing_ai_agent.core.logging import (
    initialize_logging,      # System initialization
    get_logger,             # Logger factory
    PerformanceLogger,      # Context manager for timing
    AuditLogger,            # Specialized audit logging
    log_performance,        # Decorator for performance tracking
    log_errors,             # Decorator for error logging
)
```

**Features:**
- Console output with colors
- File logging with rotation
- JSON structured logs for analysis
- Performance timing
- Audit trail logging
- Automatic error context capture

#### 3. Error Handlers (`error_handlers.py`)
Intelligent error recovery and handling:

```python
from marketing_ai_agent.core.error_handlers import (
    retry_on_error,         # Retry decorator
    circuit_breaker,        # Circuit breaker pattern
    handle_errors,          # Generic error handling
    validate_input,         # Input validation
    error_reporter,         # Global error tracking
    RetryConfig,           # Retry configuration
    CircuitBreaker,        # Circuit breaker class
    ErrorReporter          # Error analytics
)
```

**Capabilities:**
- Exponential backoff retry logic
- Circuit breaker for failing services
- Error aggregation and reporting
- Input validation decorators
- Configurable retry policies

#### 4. System Monitoring (`monitoring.py`)
Comprehensive system health monitoring:

```python
from marketing_ai_agent.core.monitoring import (
    system_monitor,         # Global monitor instance
    HealthChecker,          # Health check coordinator
    MetricsCollector,       # Performance metrics
    PerformanceMetrics,     # Metrics data structure
    HealthCheck            # Health check results
)
```

**Monitoring Features:**
- Real-time system metrics (CPU, memory, disk)
- Health checks for APIs and services
- Metrics history and trending
- Automatic alerting thresholds
- Export capabilities for analysis

## Usage Examples

### Basic Error Handling

```python
from marketing_ai_agent.core.exceptions import APIError
from marketing_ai_agent.core.logging import get_logger
from marketing_ai_agent.core.error_handlers import handle_errors

logger = get_logger(__name__)

@handle_errors(reraise=True, report=True, severity="ERROR")
def call_external_api():
    """Function with automatic error handling."""
    if api_unavailable:
        raise APIError("Service unavailable", "External API", status_code=503)
    return {"success": True}
```

### Retry Logic with Backoff

```python
from marketing_ai_agent.core.error_handlers import retry_on_error, RetryConfig

@retry_on_error(RetryConfig(
    max_attempts=5,
    initial_delay=1.0,
    backoff_factor=2.0,
    retryable_exceptions=(RateLimitError, TemporaryServiceError)
))
def resilient_api_call():
    """API call with automatic retry on transient failures."""
    return external_service.get_data()
```

### Circuit Breaker Protection

```python
from marketing_ai_agent.core.error_handlers import circuit_breaker

@circuit_breaker(failure_threshold=5, recovery_timeout=60)
def protected_service_call():
    """Service call protected by circuit breaker."""
    return unreliable_service.process_request()
```

### Performance Tracking

```python
from marketing_ai_agent.core.logging import PerformanceLogger, log_performance

# Context manager approach
def process_large_dataset():
    with PerformanceLogger(logger, "Dataset processing"):
        # Time-intensive operation
        return analyze_data(dataset)

# Decorator approach
@log_performance("Model training")
def train_ml_model():
    return model.fit(training_data)
```

### System Monitoring

```python
from marketing_ai_agent.core.monitoring import system_monitor

# Start monitoring
system_monitor.start_monitoring(
    metrics_interval=60,    # Collect metrics every minute
    health_check_interval=300  # Health checks every 5 minutes
)

# Get current status
status_report = system_monitor.get_status_report()
print(f"System health: {status_report['overall_health']}")

# Export metrics
system_monitor.export_metrics("system_metrics.json")
```

## CLI Integration

The error handling system is fully integrated into the CLI:

### System Status
```bash
# Basic status check
ai-agent status

# Detailed status with health checks
ai-agent status --detailed

# Export status report
ai-agent status --export status_report.json
```

### Monitoring Management
```bash
# Start system monitoring
ai-agent monitor --start

# Stop monitoring  
ai-agent monitor --stop

# View monitoring status
ai-agent monitor

# Export metrics
ai-agent monitor --export-metrics metrics.json
```

### Error Management
```bash
# View recent errors
ai-agent errors --recent 10

# Show error summary
ai-agent errors --summary

# Export error report
ai-agent errors --export error_report.json
```

### Enable Monitoring at Startup
```bash
# Start with monitoring enabled
ai-agent --monitor data export --source ga4
```

## Configuration

### Logging Configuration
```yaml
logging:
  level: "INFO"           # DEBUG, INFO, WARNING, ERROR, CRITICAL
  file: "./logs/agent.log"
  max_file_size: "10MB"
  backup_count: 5
  json_logging: true
  console_colors: true
```

### Monitoring Thresholds
```yaml
monitoring:
  health_checks:
    cpu_warning: 70       # CPU usage warning threshold (%)
    cpu_critical: 90      # CPU usage critical threshold (%)
    memory_warning: 80    # Memory usage warning threshold (%)
    memory_critical: 95   # Memory usage critical threshold (%)
    disk_warning: 80      # Disk usage warning threshold (%)
    disk_critical: 95     # Disk usage critical threshold (%)
    response_time_warning: 5.0   # API response warning (seconds)
    response_time_critical: 10.0 # API response critical (seconds)
```

### Retry Configuration
```yaml
error_handling:
  retry:
    max_attempts: 3       # Maximum retry attempts
    initial_delay: 1.0    # Initial delay between retries (seconds)
    max_delay: 60.0       # Maximum delay between retries (seconds)
    backoff_factor: 2.0   # Exponential backoff multiplier
    jitter: true          # Add randomization to delays
  
  circuit_breaker:
    failure_threshold: 5   # Failures before opening circuit
    recovery_timeout: 60   # Time before attempting reset (seconds)
```

## Log Output Examples

### Console Output
```
2024-01-15 10:30:45 - marketing_ai_agent - INFO - Marketing AI Agent starting
2024-01-15 10:30:45 - data_exporter - ERROR - GA4 API rate limit exceeded
2024-01-15 10:30:46 - data_exporter - INFO - Retrying GA4 API call in 2.3s (attempt 2/5)
2024-01-15 10:30:49 - data_exporter - INFO - Successfully completed GA4 data export after 2 attempts
```

### JSON Log Output
```json
{
  "timestamp": "2024-01-15T10:30:45.123",
  "level": "ERROR",
  "logger": "data_exporter",
  "message": "GA4 API rate limit exceeded",
  "module": "ga4_client",
  "function": "fetch_analytics_data",
  "line": 156,
  "error_code": "RATE_LIMIT_ERROR",
  "context": {
    "api_name": "GA4",
    "retry_after": 120,
    "request_id": "req_123456"
  }
}
```

### Audit Logs
```json
{
  "timestamp": "2024-01-15T10:30:45.123",
  "level": "INFO",
  "logger": "audit.data_export",
  "message": "Data export completed",
  "event_type": "data_export",
  "source": "ga4",
  "records_count": 15420,
  "file_path": "./exports/ga4_data_2024-01-15.csv",
  "user": "system"
}
```

## Error Recovery Strategies

### API Rate Limiting
- Automatic retry with exponential backoff
- Respect rate limit headers (`retry_after`)
- Circuit breaker protection for consistently failing endpoints

### Transient Service Failures
- Configurable retry attempts
- Jittered delays to avoid thundering herd
- Graceful degradation when possible

### Data Validation Errors
- Detailed field-level error reporting
- Context preservation for debugging
- Automatic sanitization where applicable

### System Resource Issues
- Memory usage monitoring
- Disk space warnings
- CPU utilization tracking
- Automatic cleanup procedures

## Best Practices

### Exception Handling
1. **Use specific exceptions** instead of generic `Exception`
2. **Provide context** in error messages and metadata
3. **Log errors** at appropriate levels (WARNING for retryable, ERROR for failures)
4. **Handle errors close to source** but report at application level

### Logging
1. **Use structured logging** with consistent field names
2. **Include correlation IDs** for request tracing
3. **Log performance metrics** for optimization opportunities
4. **Avoid logging sensitive data** (API keys, user data)

### Monitoring
1. **Set appropriate thresholds** based on application requirements
2. **Monitor trends** not just current values
3. **Create actionable alerts** that require human intervention
4. **Regular review** of error patterns and system performance

## Troubleshooting

### Common Issues

#### High Error Rates
```bash
# Check error summary
ai-agent errors --summary

# Review detailed error log
tail -f logs/marketing_ai_agent_errors.log

# Check system health
ai-agent status --detailed
```

#### Performance Problems
```bash
# Start monitoring if not already running
ai-agent monitor --start

# Check current metrics
ai-agent monitor

# Export metrics for analysis
ai-agent monitor --export-metrics perf_analysis.json
```

#### Configuration Issues
```bash
# Validate current configuration
ai-agent config --validate

# Check system status
ai-agent status

# Review startup logs
grep "Configuration" logs/marketing_ai_agent.json
```

## Testing Error Handling

A comprehensive demo script is available to test all error handling features:

```bash
# Run the error handling demonstration
python -m marketing_ai_agent.utils.error_demo
```

This demo covers:
- Basic error handling and reporting
- Automatic retry mechanisms
- Circuit breaker patterns
- Data validation errors
- System monitoring integration
- Error reporting and analytics

## Integration with External Systems

### Alerting Systems
The error reporting system can be extended to integrate with:
- **Slack notifications** for critical errors
- **Email alerts** for system health issues
- **PagerDuty** for incident management
- **Custom webhooks** for specialized integrations

### Log Management
Structured JSON logs can be forwarded to:
- **Elasticsearch/Kibana** for search and analysis
- **Splunk** for enterprise log management
- **DataDog** for application monitoring
- **CloudWatch** for AWS-based deployments

### Metrics Export
System metrics can be exported to:
- **Prometheus** for time-series monitoring
- **Grafana** for visualization dashboards
- **InfluxDB** for high-performance metrics storage
- **Custom analytics platforms** via JSON export

This comprehensive error handling and logging system provides a solid foundation for production deployment of the Marketing AI Agent with enterprise-grade reliability and observability.