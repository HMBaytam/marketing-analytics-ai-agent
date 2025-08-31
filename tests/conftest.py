"""Test configuration and shared fixtures for Marketing AI Agent tests."""

import os
import tempfile
import pytest
from pathlib import Path
from typing import Dict, Any, Generator
from unittest.mock import Mock, MagicMock

from marketing_ai_agent.core.config import Config
from marketing_ai_agent.core.logging import initialize_logging
from marketing_ai_agent.core.monitoring import system_monitor
from .factories import (
    CampaignDataFactory,
    AnalyticsDataFactory,
    OptimizationResultFactory,
    ReportDataFactory
)


@pytest.fixture(scope="session")
def test_config() -> Config:
    """Create a test configuration."""
    return Config(
        api={
            "google_ads": {
                "client_id": "test-client-id",
                "client_secret": "test-client-secret",
                "refresh_token": "test-refresh-token",
                "developer_token": "test-developer-token"
            },
            "google_analytics": {
                "property_id": "test-property-id",
                "credentials_file": "test-credentials.json"
            }
        },
        output={
            "base_directory": "./test_reports",
            "format": "markdown",
            "include_charts": True
        },
        logging={
            "level": "DEBUG",
            "file": "./test_logs/agent.log"
        },
        analytics={
            "default_date_range": "30d",
            "significance_threshold": 0.05
        },
        optimization={
            "confidence_threshold": 0.7,
            "max_recommendations": 10,
            "include_experimental": True
        }
    )


@pytest.fixture(scope="session")
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def test_logger():
    """Create a test logger with debug level."""
    return initialize_logging(level=10)  # DEBUG level


@pytest.fixture
def mock_ga4_client():
    """Mock GA4 API client."""
    mock_client = Mock()
    mock_client.get_report = Mock(return_value={
        "data": [
            {"date": "2024-01-01", "sessions": 1000, "conversions": 50},
            {"date": "2024-01-02", "sessions": 1200, "conversions": 60}
        ],
        "totals": {"sessions": 2200, "conversions": 110}
    })
    mock_client.get_realtime_report = Mock(return_value={
        "active_users": 45,
        "new_users": 12,
        "current_conversions": 3
    })
    return mock_client


@pytest.fixture
def mock_google_ads_client():
    """Mock Google Ads API client."""
    mock_client = Mock()
    mock_client.get_campaigns = Mock(return_value=[
        {
            "id": "12345",
            "name": "Test Campaign 1",
            "status": "ENABLED",
            "budget": 1000.0,
            "impressions": 50000,
            "clicks": 2500,
            "cost": 750.0,
            "conversions": 45
        },
        {
            "id": "12346", 
            "name": "Test Campaign 2",
            "status": "ENABLED",
            "budget": 2000.0,
            "impressions": 75000,
            "clicks": 3750,
            "cost": 1200.0,
            "conversions": 80
        }
    ])
    
    mock_client.get_keywords = Mock(return_value=[
        {
            "campaign_id": "12345",
            "ad_group_id": "67890",
            "keyword": "marketing analytics",
            "match_type": "BROAD",
            "impressions": 10000,
            "clicks": 500,
            "cost": 150.0,
            "conversions": 10
        }
    ])
    
    mock_client.update_campaign_budget = Mock(return_value={"success": True})
    mock_client.pause_campaign = Mock(return_value={"success": True})
    
    return mock_client


@pytest.fixture
def sample_campaign_data():
    """Generate sample campaign data using factory."""
    return CampaignDataFactory.build_batch(5)


@pytest.fixture
def sample_analytics_data():
    """Generate sample analytics data using factory."""
    return AnalyticsDataFactory.build_batch(10)


@pytest.fixture
def sample_optimization_results():
    """Generate sample optimization results using factory."""
    return OptimizationResultFactory.build_batch(3)


@pytest.fixture
def sample_report_data():
    """Generate sample report data using factory."""
    return ReportDataFactory.build()


@pytest.fixture
def mock_claude_client():
    """Mock Claude/Anthropic API client."""
    mock_client = Mock()
    mock_client.messages.create = Mock(return_value=Mock(
        content=[Mock(text="This is a mock Claude response for testing.")]
    ))
    return mock_client


@pytest.fixture
def mock_openai_client():
    """Mock OpenAI API client."""
    mock_client = Mock()
    mock_client.chat.completions.create = Mock(return_value=Mock(
        choices=[Mock(message=Mock(content="This is a mock OpenAI response for testing."))]
    ))
    return mock_client


@pytest.fixture(autouse=True)
def setup_test_environment(temp_dir, test_config):
    """Setup test environment before each test."""
    # Set test directories
    test_config.output.base_directory = str(temp_dir / "reports")
    test_config.logging.file = str(temp_dir / "logs" / "test.log")
    
    # Create directories
    (temp_dir / "reports").mkdir(exist_ok=True)
    (temp_dir / "logs").mkdir(exist_ok=True)
    (temp_dir / "data").mkdir(exist_ok=True)
    
    # Set environment variables for testing
    os.environ["TESTING"] = "1"
    os.environ["LOG_LEVEL"] = "DEBUG"
    
    yield
    
    # Cleanup after test
    if "TESTING" in os.environ:
        del os.environ["TESTING"]


@pytest.fixture
def mock_system_monitor():
    """Mock system monitor for testing."""
    mock_monitor = Mock()
    mock_monitor.get_status_report = Mock(return_value={
        "timestamp": "2024-01-15T10:30:00",
        "overall_health": "healthy",
        "monitoring_active": True,
        "current_metrics": {
            "cpu_percent": 25.5,
            "memory_percent": 45.2,
            "disk_usage_percent": 60.1,
            "memory_used_mb": 2048.5
        },
        "health_checks": {
            "system_resources": {
                "status": "healthy",
                "message": "System resources OK",
                "duration": 0.1
            },
            "api_connectivity": {
                "status": "healthy", 
                "message": "All APIs accessible",
                "duration": 0.5
            }
        },
        "average_metrics_10min": {
            "avg_cpu_percent": 22.3,
            "avg_memory_percent": 42.8,
            "avg_disk_usage_percent": 60.0
        },
        "metrics_history_count": 150
    })
    
    mock_monitor.start_monitoring = Mock()
    mock_monitor.stop_monitoring = Mock()
    mock_monitor.export_metrics = Mock()
    
    return mock_monitor


@pytest.fixture
def mock_error_reporter():
    """Mock error reporter for testing."""
    mock_reporter = Mock()
    mock_reporter.get_error_summary = Mock(return_value={
        "total_error_types": 2,
        "error_counts": {
            "APIError": 3,
            "DataValidationError": 2
        },
        "recent_errors": [
            {
                "timestamp": "2024-01-15T10:25:00",
                "error_type": "APIError",
                "message": "Test API error",
                "severity": "ERROR",
                "context": {"api_name": "Test API"}
            }
        ],
        "most_common_error": ("APIError", 3)
    })
    
    mock_reporter.report_error = Mock()
    return mock_reporter


@pytest.fixture
def cli_runner():
    """Typer CLI test runner."""
    from typer.testing import CliRunner
    return CliRunner()


@pytest.fixture
def isolated_filesystem():
    """Create isolated filesystem for CLI tests."""
    with tempfile.TemporaryDirectory() as temp_dir:
        original_cwd = os.getcwd()
        os.chdir(temp_dir)
        try:
            yield Path(temp_dir)
        finally:
            os.chdir(original_cwd)


class MockResponse:
    """Mock HTTP response for API testing."""
    
    def __init__(self, json_data: Dict[Any, Any], status_code: int = 200):
        self.json_data = json_data
        self.status_code = status_code
        self.headers = {"content-type": "application/json"}
        self.text = str(json_data)
    
    def json(self):
        return self.json_data
    
    def raise_for_status(self):
        if self.status_code >= 400:
            raise Exception(f"HTTP {self.status_code}")


@pytest.fixture
def mock_requests_session():
    """Mock requests session for API testing."""
    session = Mock()
    session.get = Mock(return_value=MockResponse({"success": True}))
    session.post = Mock(return_value=MockResponse({"success": True}))
    session.put = Mock(return_value=MockResponse({"success": True}))
    session.delete = Mock(return_value=MockResponse({"success": True}))
    return session


# Parametrized fixtures for testing different scenarios
@pytest.fixture(params=["ga4", "google_ads", "both"])
def data_source(request):
    """Parametrized fixture for different data sources."""
    return request.param


@pytest.fixture(params=["csv", "json", "excel"])
def export_format(request):
    """Parametrized fixture for different export formats."""
    return request.param


@pytest.fixture(params=["7d", "30d", "90d", "1y"])
def date_range(request):
    """Parametrized fixture for different date ranges."""
    return request.param


@pytest.fixture(params=["executive", "detailed", "technical"])
def report_template(request):
    """Parametrized fixture for different report templates."""
    return request.param


# Performance testing fixtures
@pytest.fixture
def benchmark_data():
    """Large dataset for performance testing."""
    return {
        "campaigns": CampaignDataFactory.build_batch(100),
        "analytics": AnalyticsDataFactory.build_batch(1000),
        "keywords": [
            {
                "keyword": f"test keyword {i}",
                "impressions": 1000 + i * 10,
                "clicks": 50 + i,
                "cost": 25.0 + i * 0.5
            }
            for i in range(500)
        ]
    }


@pytest.fixture
def memory_limit():
    """Memory limit for performance tests."""
    return 100 * 1024 * 1024  # 100MB


# Database/storage fixtures for integration tests
@pytest.fixture
def temp_database():
    """Temporary SQLite database for testing."""
    import sqlite3
    
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as temp_file:
        db_path = temp_file.name
    
    conn = sqlite3.connect(db_path)
    
    # Create test tables
    conn.execute("""
        CREATE TABLE campaigns (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            budget REAL,
            status TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    conn.execute("""
        CREATE TABLE metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            campaign_id TEXT,
            date TEXT,
            impressions INTEGER,
            clicks INTEGER,
            cost REAL,
            conversions INTEGER,
            FOREIGN KEY (campaign_id) REFERENCES campaigns (id)
        )
    """)
    
    conn.commit()
    
    yield conn
    
    conn.close()
    os.unlink(db_path)


# Error testing fixtures
@pytest.fixture
def api_error_scenarios():
    """Different API error scenarios for testing."""
    return {
        "rate_limit": MockResponse(
            {"error": "Rate limit exceeded"}, 
            status_code=429
        ),
        "unauthorized": MockResponse(
            {"error": "Unauthorized"}, 
            status_code=401
        ),
        "server_error": MockResponse(
            {"error": "Internal server error"}, 
            status_code=500
        ),
        "timeout": Exception("Request timeout"),
        "connection_error": Exception("Connection failed")
    }