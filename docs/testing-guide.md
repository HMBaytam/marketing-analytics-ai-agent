# Testing Guide

Comprehensive testing framework for the Marketing Analytics AI Agent with unit tests, integration tests, performance benchmarks, and automated test execution.

## Overview

The testing framework provides:
- **Unit tests** for individual components and functions
- **Integration tests** for component interactions and workflows
- **CLI tests** for command-line interface functionality
- **Performance tests** for benchmarking and optimization
- **Error handling tests** for robustness and reliability
- **Mock data and fixtures** for consistent test scenarios
- **Automated test execution** with coverage reporting

## Test Structure

### Directory Organization

```
tests/
├── conftest.py              # Shared fixtures and configuration
├── factories.py             # Test data factories using Factory Boy
├── test_runner.py           # Test execution and management
├── test_core.py             # Core module unit tests
├── test_cli.py              # CLI integration tests
├── test_analytics.py        # Analytics module tests
├── test_optimization.py     # Optimization module tests
├── test_error_handling.py   # Error handling tests
└── __init__.py              # Test package initialization
```

### Test Categories

Tests are organized by markers for flexible execution:

- `unit`: Unit tests for individual functions/classes
- `integration`: Tests for component interactions
- `cli`: Command-line interface tests
- `slow`: Long-running tests (>1 second)
- `performance`: Benchmark and performance tests
- `api`: Tests requiring external API access
- `error_handling`: Error scenarios and recovery tests
- `analytics`: Analytics engine functionality
- `optimization`: Optimization algorithm tests
- `data`: Data processing and validation tests
- `reporting`: Report generation tests
- `monitoring`: System monitoring tests

## Running Tests

### Quick Start

```bash
# Install test dependencies
poetry install --with dev

# Run all tests
python tests/test_runner.py all

# Run quick tests (development)
python tests/test_runner.py quick

# Run specific test suite
python tests/test_runner.py unit
python tests/test_runner.py integration
python tests/test_runner.py cli
```

### Using pytest directly

```bash
# Run all tests with coverage
pytest --cov=src/marketing_ai_agent --cov-report=html --cov-report=term

# Run specific test categories
pytest -m unit                    # Unit tests only
pytest -m "not slow"             # Exclude slow tests
pytest -m "cli or integration"   # Multiple markers

# Run specific test files
pytest tests/test_core.py         # Core module tests
pytest tests/test_cli.py          # CLI tests

# Run with different verbosity
pytest -v                        # Verbose output
pytest -q                        # Quiet output
pytest -s                        # Show print statements

# Run specific test functions
pytest tests/test_core.py::TestConfig::test_config_creation
pytest -k "test_config"          # Run tests matching pattern
```

### Performance Testing

```bash
# Run benchmark tests (requires pytest-benchmark)
python tests/test_runner.py performance

# With pytest directly
pytest -m performance --benchmark-only --benchmark-sort=mean

# Generate benchmark reports
pytest --benchmark-only --benchmark-html=reports/benchmark.html
```

### Coverage Analysis

```bash
# Generate HTML coverage report
python tests/test_runner.py --coverage-report

# View coverage in browser
open htmlcov/index.html

# Coverage with different formats
pytest --cov=src --cov-report=xml --cov-report=json
```

## Test Fixtures and Factories

### Core Fixtures (`conftest.py`)

```python
# Configuration and environment
test_config()              # Test configuration object
temp_dir()                 # Temporary directory for test files
setup_test_environment()   # Automated test environment setup

# Mock clients and services  
mock_ga4_client()          # Mocked Google Analytics client
mock_google_ads_client()   # Mocked Google Ads client
mock_claude_client()       # Mocked Claude API client
mock_system_monitor()      # Mocked system monitoring

# Test data
sample_campaign_data()     # Campaign performance data
sample_analytics_data()    # Analytics time series data
sample_optimization_results() # Optimization recommendations
benchmark_data()           # Large datasets for performance testing

# Testing utilities
cli_runner()               # Typer CLI test runner
isolated_filesystem()     # Isolated filesystem for file operations
memory_limit()             # Memory constraints for performance tests
```

### Data Factories (`factories.py`)

Factory classes generate realistic test data:

```python
# Campaign and advertising data
CampaignDataFactory.build()        # Single campaign
CampaignDataFactory.build_batch(n) # Multiple campaigns
KeywordDataFactory.build()         # Keyword performance data

# Analytics data
AnalyticsDataFactory.build()       # GA4 analytics data
PerformanceMetricsFactory.build()  # System performance metrics

# Optimization data
OptimizationResultFactory.build()  # Optimization recommendations
ReportDataFactory.build()          # Report generation data

# Error and monitoring data
ErrorDataFactory.build()           # Error tracking data
PerformanceMetricsFactory.build()  # System metrics

# Utility functions
create_campaign_with_keywords(n)   # Campaign with associated keywords
create_analytics_funnel()          # Complete conversion funnel
create_optimization_scenario()     # Before/after optimization scenario
```

## Writing Tests

### Unit Test Example

```python
import pytest
from marketing_ai_agent.core.config import Config
from marketing_ai_agent.core.exceptions import ConfigurationError

class TestConfig:
    """Test configuration management."""
    
    def test_config_creation(self, test_config):
        """Test basic config creation."""
        assert test_config.api.google_ads.client_id == "test-client-id"
        assert test_config.output.base_directory == "./test_reports"
    
    def test_invalid_config_raises_error(self):
        """Test invalid configuration handling."""
        with pytest.raises(ConfigurationError):
            Config(api={"invalid_key": "invalid_value"})
    
    @pytest.mark.parametrize("log_level", ["DEBUG", "INFO", "WARNING", "ERROR"])
    def test_config_log_levels(self, log_level):
        """Test different log level configurations."""
        config = Config(logging={"level": log_level})
        assert config.logging.level == log_level
```

### Integration Test Example

```python
@pytest.mark.integration
class TestCLIIntegration:
    """Integration tests for CLI functionality."""
    
    def test_data_export_workflow(self, cli_runner, mock_ga4_client):
        """Test complete data export workflow."""
        # Mock API client
        with patch('marketing_ai_agent.api_clients.ga4_client.GA4Client') as mock_client:
            mock_client.return_value = mock_ga4_client
            
            # Run CLI command
            result = cli_runner.invoke(app, [
                "data", "export", "ga4", 
                "--date-range", "7d",
                "--format", "json"
            ])
            
            # Verify results
            assert result.exit_code == 0
            assert "Export completed" in result.stdout
```

### Performance Test Example

```python
@pytest.mark.performance
class TestPerformanceBenchmarks:
    """Performance benchmark tests."""
    
    def test_analytics_processing_performance(self, benchmark, benchmark_data):
        """Benchmark analytics processing with large dataset."""
        from marketing_ai_agent.analytics.performance_scorer import PerformanceScorer
        
        scorer = PerformanceScorer()
        large_dataset = benchmark_data['analytics']
        
        def process_analytics():
            return scorer.calculate_score(large_dataset)
        
        result = benchmark(process_analytics)
        assert result is not None
    
    def test_memory_usage_optimization(self, memory_limit):
        """Test memory-constrained optimization."""
        import psutil
        process = psutil.Process()
        
        initial_memory = process.memory_info().rss
        
        # Run memory-intensive operation
        large_data = generate_large_dataset(size=10000)
        result = process_data(large_data)
        
        final_memory = process.memory_info().rss
        memory_used = final_memory - initial_memory
        
        assert memory_used < memory_limit
        assert result is not None
```

### Error Handling Test Example

```python
@pytest.mark.error_handling
class TestErrorRecovery:
    """Test error handling and recovery scenarios."""
    
    def test_api_timeout_recovery(self, mock_ga4_client):
        """Test recovery from API timeout errors."""
        # Configure mock to raise timeout
        mock_ga4_client.get_report.side_effect = [
            TimeoutError("Request timeout"),
            {"data": "success"}  # Succeeds on retry
        ]
        
        from marketing_ai_agent.core.error_handlers import retry_on_error, RetryConfig
        
        @retry_on_error(RetryConfig(max_attempts=3, initial_delay=0.1))
        def fetch_data():
            return mock_ga4_client.get_report()
        
        # Should succeed after retry
        result = fetch_data()
        assert result == {"data": "success"}
        assert mock_ga4_client.get_report.call_count == 2
    
    def test_circuit_breaker_activation(self):
        """Test circuit breaker pattern."""
        from marketing_ai_agent.core.error_handlers import circuit_breaker
        
        call_count = 0
        
        @circuit_breaker(failure_threshold=2, recovery_timeout=1)
        def failing_function():
            nonlocal call_count
            call_count += 1
            raise Exception("Service unavailable")
        
        # First two calls should raise original exception
        for _ in range(2):
            with pytest.raises(Exception, match="Service unavailable"):
                failing_function()
        
        # Third call should raise circuit breaker exception
        with pytest.raises(Exception, match="Circuit breaker OPEN"):
            failing_function()
        
        assert call_count == 2  # Circuit prevented third call
```

## Mock Data and Testing Scenarios

### Campaign Performance Testing

```python
def test_campaign_optimization_scenarios():
    """Test various campaign optimization scenarios."""
    
    # High-performing campaigns
    high_performers = CampaignDataFactory.build_batch(
        3, 
        conversions=50, 
        cost=1000, 
        revenue=5000
    )
    
    # Poor-performing campaigns
    poor_performers = CampaignDataFactory.build_batch(
        2,
        conversions=5,
        cost=2000,
        revenue=500
    )
    
    # Mixed performance portfolio
    mixed_campaigns = high_performers + poor_performers
    
    optimizer = BudgetOptimizer()
    result = optimizer.optimize_budgets(mixed_campaigns)
    
    # Verify optimization logic
    for campaign in result['optimized_campaigns']:
        if campaign['id'] in [c['id'] for c in high_performers]:
            assert campaign['budget_change'] > 0  # Increase budget
        else:
            assert campaign['budget_change'] <= 0  # Maintain or decrease
```

### Time Series Analytics Testing

```python
def test_trend_analysis_scenarios():
    """Test trend analysis with different patterns."""
    
    # Create data with known patterns
    dates = pd.date_range('2024-01-01', periods=90)
    
    # Upward trend
    upward_trend = np.linspace(100, 200, 90) + np.random.normal(0, 5, 90)
    
    # Seasonal pattern
    seasonal_pattern = 100 + 20 * np.sin(np.arange(90) * 2 * np.pi / 30)
    
    # Test trend detection
    analyzer = TrendAnalyzer()
    
    trend_result = analyzer.analyze_trend(dates, upward_trend)
    assert trend_result['direction'] == 'increasing'
    assert trend_result['strength'] > 0.7
    
    seasonal_result = analyzer.decompose_seasonality(dates, seasonal_pattern)
    assert 'seasonal' in seasonal_result
    assert np.std(seasonal_result['seasonal']) > 5  # Significant seasonality
```

## Continuous Integration

### GitHub Actions Workflow

```yaml
name: Test Suite

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.8, 3.9, 3.10, 3.11]
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        pip install poetry
        poetry install --with dev
    
    - name: Run tests
      run: |
        poetry run python tests/test_runner.py all
    
    - name: Upload coverage
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
```

### Pre-commit Hooks

```yaml
# .pre-commit-config.yaml
repos:
  - repo: local
    hooks:
      - id: pytest
        name: pytest
        entry: python tests/test_runner.py quick
        language: python
        pass_filenames: false
        always_run: true
      
      - id: pytest-coverage
        name: pytest-coverage
        entry: pytest --cov=src --cov-fail-under=80
        language: python
        pass_filenames: false
        stages: [pre-push]
```

## Test Data Management

### Environment Variables

```bash
# Test configuration
TESTING=1
LOG_LEVEL=DEBUG
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1

# Mock API responses
MOCK_GA4_API=1
MOCK_GOOGLE_ADS_API=1
MOCK_CLAUDE_API=1

# Performance testing
PERFORMANCE_TESTS_ENABLED=1
BENCHMARK_TIMEOUT=300
```

### Test Database Setup

```python
@pytest.fixture
def test_database():
    """Create isolated test database."""
    import sqlite3
    import tempfile
    
    with tempfile.NamedTemporaryFile(suffix='.db') as temp_db:
        conn = sqlite3.connect(temp_db.name)
        
        # Create test schema
        conn.execute("""
            CREATE TABLE campaigns (
                id TEXT PRIMARY KEY,
                name TEXT,
                budget REAL,
                performance_data JSON
            )
        """)
        
        yield conn
        conn.close()
```

## Debugging Tests

### Common Issues

1. **Import Errors**
   ```bash
   # Fix Python path issues
   export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
   
   # Or use development installation
   pip install -e .
   ```

2. **Mock Configuration**
   ```python
   # Ensure mocks are properly configured
   @patch('marketing_ai_agent.api_clients.ga4_client.GA4Client')
   def test_function(self, mock_client):
       mock_client.return_value.method.return_value = expected_result
       # Test implementation
   ```

3. **Async Test Issues**
   ```python
   # Use pytest-asyncio for async tests
   @pytest.mark.asyncio
   async def test_async_function():
       result = await async_function()
       assert result is not None
   ```

### Test Debugging Commands

```bash
# Run tests with debugging
pytest -s -vv tests/test_specific.py::test_function

# Drop into debugger on failure
pytest --pdb tests/

# Run with coverage debugging
pytest --cov-report=term-missing --cov-branch

# Profile test execution
pytest --profile --profile-svg

# Run specific test with maximum output
pytest -s -vv --tb=long tests/test_file.py::TestClass::test_method
```

## Best Practices

### Test Organization
- Group related tests in classes
- Use descriptive test names that explain the scenario
- Follow AAA pattern: Arrange, Act, Assert
- One assertion per test when possible

### Mock Usage
- Mock external dependencies (APIs, file system, network)
- Use factory classes for consistent test data
- Reset mocks between tests
- Verify mock interactions when relevant

### Performance Testing
- Use realistic data sizes
- Set appropriate timeouts
- Monitor memory usage
- Compare against baselines

### Error Testing
- Test both error conditions and recovery
- Verify error messages and context
- Test edge cases and boundary conditions
- Ensure proper cleanup after errors

This comprehensive testing framework ensures the Marketing Analytics AI Agent is reliable, performant, and maintainable across all components and use cases.