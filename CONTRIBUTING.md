# Contributing to Marketing Analytics AI Agent

Thank you for your interest in contributing to the Marketing Analytics AI Agent! This document provides comprehensive guidelines for contributors to ensure high-quality, consistent contributions.

## 📋 Table of Contents

- [Getting Started](#getting-started)
- [Development Environment](#development-environment)
- [Project Architecture](#project-architecture)
- [Code Standards](#code-standards)
- [Testing Requirements](#testing-requirements)
- [Pull Request Process](#pull-request-process)
- [Issue Guidelines](#issue-guidelines)
- [Release Process](#release-process)

## 🚀 Getting Started

### Prerequisites

- **Python 3.11+** (3.12 recommended for optimal performance)
- **Poetry** for dependency management
- **Git** for version control
- **API Access**: Google Ads API and/or Google Analytics 4 credentials (for testing)

### Development Setup

1. **Fork and Clone**
   ```bash
   # Fork the repository on GitHub, then clone your fork
   git clone https://github.com/YOUR_USERNAME/marketing-ai-agent.git
   cd marketing-ai-agent
   
   # Add upstream remote
   git remote add upstream https://github.com/original-org/marketing-ai-agent.git
   ```

2. **Environment Setup**
   ```bash
   # Install dependencies (includes dev dependencies)
   poetry install --with dev,test,docs
   
   # Activate virtual environment
   poetry shell
   
   # Install pre-commit hooks
   poetry run pre-commit install
   ```

3. **Configuration**
   ```bash
   # Copy example environment file
   cp .env.example .env
   
   # Edit .env with your API credentials (optional for most development)
   # Note: You can develop most features without real API credentials
   ```

4. **Verify Setup**
   ```bash
   # Run tests to ensure everything works
   poetry run pytest tests/test_core.py
   
   # Check code quality tools
   poetry run ruff check src/
   poetry run black --check src/
   poetry run mypy src/
   ```

## 🏗️ Development Environment

### Required Tools

All tools are configured in `pyproject.toml` and installed via Poetry:

- **Code Formatting**: `black` (line length: 88)
- **Linting**: `ruff` (configured with comprehensive rules)
- **Type Checking**: `mypy` (strict mode enabled)
- **Testing**: `pytest` with plugins for async, coverage, and benchmarking
- **Pre-commit**: Automated quality checks on commit

### VS Code Setup (Recommended)

```json
// .vscode/settings.json
{
    "python.defaultInterpreterPath": "./.venv/bin/python",
    "python.formatting.provider": "black",
    "python.linting.enabled": true,
    "python.linting.ruffEnabled": true,
    "python.typeChecking": "mypy",
    "editor.formatOnSave": true,
    "editor.codeActionsOnSave": {
        "source.organizeImports": true
    }
}
```

### Development Commands

```bash
# Code quality (run before committing)
make format        # or: poetry run black src/ tests/
make lint          # or: poetry run ruff check src/ tests/
make typecheck     # or: poetry run mypy src/
make test          # or: poetry run pytest

# Alternative: use pre-commit to run all checks
poetry run pre-commit run --all-files
```

## 🏛️ Project Architecture

### Directory Structure

```
src/marketing_ai_agent/
├── agents/                    # AI agent implementations
│   ├── orchestrator.py        # 🎯 Main coordination agent
│   ├── data_ingestion.py     # 📤 Data retrieval agent
│   ├── campaign_analyzer.py  # 📊 Analysis agent
│   ├── recommendation_generator.py # 📋 Reporting agent
│   └── workflow_manager.py   # 🤖 ML and workflow agent
├── api_clients/              # External API integrations
│   ├── ga4_client.py         # Google Analytics 4 client
│   ├── google_ads_client.py  # Google Ads API client
│   └── __init__.py           # Client exports
├── analytics/                # Analytics engines
│   ├── performance_scorer.py # Performance analysis
│   ├── trend_analyzer.py     # Trend detection
│   ├── anomaly_detector.py   # Anomaly detection
│   ├── benchmarking.py       # Competitive analysis
│   └── predictive_model.py   # ML forecasting
├── optimization/             # Optimization algorithms
│   ├── budget_optimizer.py   # Budget allocation
│   ├── recommendations_engine.py # Optimization recommendations
│   ├── ml_optimizer.py       # ML-driven optimization
│   ├── roi_optimizer.py      # ROI maximization
│   └── ab_testing_optimizer.py # A/B testing
├── cli/                      # Command-line interface
│   ├── main.py               # Main CLI application
│   └── commands/             # Command implementations
├── core/                     # Core infrastructure
│   ├── config.py             # Configuration management
│   ├── exceptions.py         # Custom exceptions
│   ├── logging.py            # Structured logging
│   ├── error_handlers.py     # Error handling & retry logic
│   └── monitoring.py         # System monitoring
├── models/                   # Data models and schemas
│   ├── campaign.py           # Campaign data models
│   ├── metrics.py            # Performance metrics
│   ├── analytics.py          # Analytics data models
│   ├── transformers.py       # Data transformation
│   ├── cache.py              # Caching models
│   └── exporters.py          # Export utilities
├── auth/                     # Authentication
│   ├── oauth2_manager.py     # OAuth2 flow management
│   └── config_manager.py     # API configuration
└── utils/                    # Utility functions
    └── error_demo.py         # Error handling demonstrations
```

### Agent Architecture Principles

1. **Single Responsibility**: Each agent has a clear, focused purpose
2. **Loose Coupling**: Agents communicate through well-defined interfaces
3. **Error Resilience**: Graceful degradation when sub-agents fail
4. **Stateless Design**: Agents don't maintain persistent state between queries
5. **Composability**: Agents can be combined for complex workflows

## 📝 Code Standards

### Python Code Style

We follow **PEP 8** with specific modifications defined in `pyproject.toml`:

#### Formatting Standards
```python
# Line length: 88 characters (Black default)
# Use double quotes for strings
# Use f-strings for string formatting
# Use type hints for all function parameters and returns

# Example function signature
def analyze_campaign(
    campaign_id: str,
    date_range: tuple[datetime, datetime],
    metrics: list[str] | None = None,
) -> CampaignAnalysis:
    """Analyze campaign performance for specified date range.
    
    Args:
        campaign_id: Unique campaign identifier
        date_range: Start and end dates for analysis period
        metrics: Optional list of metrics to include
        
    Returns:
        CampaignAnalysis object with performance data and insights
        
    Raises:
        CampaignNotFoundError: When campaign ID is invalid
        APIConnectionError: When unable to fetch data
    """
```

#### Type Annotations
- **Required**: All function parameters and return types
- **Modern Syntax**: Use `X | Y` instead of `Union[X, Y]` (Python 3.10+)
- **Specific Types**: Use specific types over `Any` when possible
- **Generic Types**: Use `list[str]` instead of `List[str]`

#### Import Organization
```python
# Standard library imports
import os
import sys
from datetime import datetime, timedelta
from typing import Any, Dict, List

# Third-party imports
import pandas as pd
import typer
from pydantic import BaseModel

# Local imports
from ..core.config import Config
from ..models.campaign import Campaign
```

#### Documentation Standards
- **Docstrings**: Google-style docstrings for all public functions and classes
- **Type Information**: Include parameter and return type documentation
- **Examples**: Provide usage examples for complex functions
- **Error Handling**: Document all raised exceptions

### Error Handling Patterns

#### Exception Hierarchy
```python
# Use project-specific exceptions
from ..core.exceptions import (
    MarketingAIAgentError,      # Base exception
    ConfigurationError,         # Configuration issues
    APIConnectionError,         # API communication failures
    DataValidationError,        # Data quality issues
    AnalysisError,             # Analytics processing errors
)

# Exception chaining for context
try:
    result = api_call()
except APIError as e:
    raise APIConnectionError("Failed to fetch campaign data") from e
```

#### Retry Logic and Circuit Breakers
```python
# Use the provided decorators
from ..core.error_handlers import with_retry, with_circuit_breaker

@with_retry(max_attempts=3, backoff_factor=2.0)
@with_circuit_breaker(failure_threshold=5, recovery_timeout=60)
def fetch_campaign_data(campaign_id: str) -> CampaignData:
    """Fetch campaign data with automatic retry and circuit breaker."""
```

### Logging Standards

#### Structured Logging
```python
from ..core.logging import get_logger

logger = get_logger(__name__)

# Use structured logging with context
logger.info(
    "Campaign analysis completed",
    extra={
        "campaign_id": campaign_id,
        "metrics_count": len(metrics),
        "duration_ms": duration,
        "status": "success"
    }
)

# Performance logging for expensive operations
from ..core.logging import log_performance

@log_performance("campaign_analysis")
def analyze_campaign(campaign_id: str) -> CampaignAnalysis:
    """Analyze campaign with automatic performance logging."""
```

#### Log Levels and Usage
- **DEBUG**: Detailed diagnostic information
- **INFO**: General operational messages and successful operations
- **WARNING**: Potentially harmful situations that don't stop execution
- **ERROR**: Error events that might still allow execution to continue
- **CRITICAL**: Serious errors that might cause the program to abort

### Data Model Standards

#### Pydantic Models
```python
from pydantic import BaseModel, Field, validator
from datetime import datetime
from typing import Optional

class CampaignMetrics(BaseModel):
    """Campaign performance metrics with validation."""
    
    campaign_id: str = Field(..., description="Unique campaign identifier")
    impressions: int = Field(ge=0, description="Number of impressions")
    clicks: int = Field(ge=0, description="Number of clicks")
    conversions: float = Field(ge=0, description="Number of conversions")
    cost: float = Field(ge=0, description="Total cost in currency units")
    revenue: Optional[float] = Field(None, ge=0, description="Total revenue")
    date: datetime = Field(..., description="Date of metrics")
    
    @validator('clicks')
    def clicks_not_exceed_impressions(cls, v, values):
        """Validate that clicks don't exceed impressions."""
        if 'impressions' in values and v > values['impressions']:
            raise ValueError('Clicks cannot exceed impressions')
        return v
    
    @property
    def ctr(self) -> float:
        """Calculate click-through rate."""
        return self.clicks / self.impressions if self.impressions > 0 else 0.0
    
    @property
    def cpa(self) -> Optional[float]:
        """Calculate cost per acquisition."""
        return self.cost / self.conversions if self.conversions > 0 else None
```

## 🧪 Testing Requirements

### Test Coverage Requirements

- **Minimum Coverage**: 80% overall, 90% for core modules
- **Critical Paths**: 100% coverage for error handling and API clients
- **New Features**: Must include comprehensive tests before merge

### Test Structure and Organization

#### Test Categories
```python
import pytest

# Unit tests - fast, isolated
@pytest.mark.unit
def test_campaign_metrics_calculation():
    """Test campaign metrics calculations in isolation."""

# Integration tests - test component interactions
@pytest.mark.integration  
def test_ga4_client_integration():
    """Test GA4 client with real API (requires credentials)."""

# End-to-end tests - full workflow testing
@pytest.mark.e2e
def test_complete_analysis_workflow():
    """Test complete analysis workflow from query to report."""

# Performance tests - benchmarking and load testing
@pytest.mark.slow
@pytest.mark.performance
def test_large_dataset_performance():
    """Test performance with large datasets."""
```

#### Test Fixtures and Factories
```python
# Use factories for test data generation
from tests.factories import CampaignFactory, MetricsFactory

def test_campaign_analysis():
    """Test campaign analysis with factory-generated data."""
    campaign = CampaignFactory.create(
        campaign_id="test_campaign",
        status="ENABLED"
    )
    metrics = MetricsFactory.create_batch(size=30)
    
    result = analyze_campaign(campaign, metrics)
    assert result.performance_score > 0
```

#### Mocking External Dependencies
```python
import pytest
from unittest.mock import patch, MagicMock

@patch('marketing_ai_agent.api_clients.ga4_client.BetaAnalyticsDataClient')
def test_ga4_data_fetch(mock_client):
    """Test GA4 data fetching with mocked API responses."""
    mock_client.return_value.run_report.return_value = create_mock_response()
    
    client = GA4APIClient(credentials=mock_credentials)
    result = client.fetch_campaign_data("2024-01-01", "2024-01-31")
    
    assert len(result) > 0
    mock_client.return_value.run_report.assert_called_once()
```

#### Test Commands
```bash
# Run all tests with coverage
poetry run pytest --cov=src/marketing_ai_agent --cov-report=html

# Run specific test categories
poetry run pytest -m unit                    # Unit tests only
poetry run pytest -m "integration and not slow"  # Fast integration tests
poetry run pytest -m e2e                     # End-to-end tests

# Run tests for specific modules
poetry run pytest tests/test_analytics.py    # Analytics tests
poetry run pytest tests/test_agents.py       # Agent tests

# Performance and benchmark tests
poetry run pytest -m performance --benchmark-only
```

## 🎯 Pull Request Process

### Before Submitting a PR

1. **Create Feature Branch**
   ```bash
   git checkout -b feature/your-feature-name
   # or
   git checkout -b fix/issue-number-description
   ```

2. **Development Checklist**
   - [ ] Code follows project style guidelines
   - [ ] All tests pass locally
   - [ ] Code coverage meets requirements (80%+ overall)
   - [ ] Type checking passes without errors
   - [ ] Documentation updated for user-facing changes
   - [ ] CHANGELOG.md updated (if applicable)

3. **Quality Checks**
   ```bash
   # Automated checks (pre-commit will run these)
   poetry run black src/ tests/
   poetry run ruff check --fix src/ tests/
   poetry run mypy src/
   poetry run pytest --cov=src/marketing_ai_agent
   ```

### PR Requirements

#### PR Title Format
Use [Conventional Commits](https://conventionalcommits.org/) format:

```
feat: add budget optimization algorithm
fix: resolve GA4 API timeout issue
docs: update installation instructions
test: add integration tests for reporting agent
refactor: improve error handling consistency
perf: optimize campaign data processing
ci: update GitHub Actions workflow
```

#### PR Description Template

```markdown
## Description
Brief description of changes and motivation.

## Type of Change
- [ ] Bug fix (non-breaking change that fixes an issue)
- [ ] New feature (non-breaking change that adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Code refactoring

## Testing
- [ ] Unit tests added/updated
- [ ] Integration tests added/updated
- [ ] All tests pass locally
- [ ] Manual testing completed

## Documentation
- [ ] Code comments added/updated
- [ ] API documentation updated
- [ ] README updated (if user-facing changes)
- [ ] CONTRIBUTING.md updated (if development process changes)

## Screenshots (if applicable)
Add screenshots for UI changes or CLI output examples.

## Related Issues
Closes #123
Addresses #456

## Checklist
- [ ] Code follows project style guidelines
- [ ] Self-review completed
- [ ] Code coverage maintained/improved
- [ ] Breaking changes documented
- [ ] Backward compatibility considered
```

### Review Process

1. **Automated Checks**: All CI checks must pass
2. **Code Review**: At least one maintainer review required
3. **Testing**: Manual testing for user-facing changes
4. **Documentation**: Verify docs are updated and accurate

### Merge Requirements

- ✅ All CI checks passing
- ✅ At least one approving review from maintainer
- ✅ No merge conflicts
- ✅ Branch is up-to-date with main
- ✅ All conversations resolved

## 🧪 Testing Standards

### Test Writing Guidelines

#### Test Structure (AAA Pattern)
```python
def test_campaign_analysis_with_valid_data():
    """Test campaign analysis with valid input data."""
    # Arrange
    campaign = CampaignFactory.create(status="ENABLED")
    metrics = MetricsFactory.create_batch(size=10)
    analyzer = CampaignAnalyzer()
    
    # Act
    result = analyzer.analyze(campaign, metrics)
    
    # Assert
    assert result.performance_score > 0
    assert result.recommendations is not None
    assert len(result.insights) > 0
```

#### Test Naming Convention
- **Function names**: `test_[component]_[scenario]_[expected_outcome]`
- **Clear descriptions**: Test docstrings explain the test purpose
- **Edge cases**: Test boundary conditions and error scenarios

#### Mock Usage Guidelines
```python
# Good: Mock external dependencies, not internal logic
@patch('marketing_ai_agent.api_clients.ga4_client.BetaAnalyticsDataClient')
def test_data_fetch_with_api_error(mock_client):
    mock_client.side_effect = GoogleAPIError("API quota exceeded")
    # Test error handling

# Bad: Don't mock the code you're testing
@patch('marketing_ai_agent.analytics.performance_scorer.calculate_score')
def test_performance_scoring(mock_calculate):
    # This doesn't test the actual logic
```

### Performance Testing

#### Benchmark Tests
```python
import pytest

@pytest.mark.performance
def test_campaign_analysis_performance(benchmark):
    """Benchmark campaign analysis performance."""
    campaign = CampaignFactory.create()
    metrics = MetricsFactory.create_batch(size=1000)
    
    result = benchmark(analyze_campaign, campaign, metrics)
    
    # Performance assertions
    assert result.execution_time_ms < 500
    assert result.memory_usage_mb < 100
```

#### Load Testing
```python
@pytest.mark.slow
@pytest.mark.performance
def test_concurrent_analysis_load():
    """Test system behavior under concurrent load."""
    import concurrent.futures
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = [
            executor.submit(analyze_campaign, f"campaign_{i}")
            for i in range(100)
        ]
        
        results = [f.result() for f in futures]
        assert all(r.success for r in results)
```

## 🔍 Code Review Guidelines

### For Authors

#### Before Requesting Review
- [ ] **Self-review**: Review your own code for obvious issues
- [ ] **Testing**: Ensure all tests pass and coverage is adequate
- [ ] **Documentation**: Update docs for user-facing changes
- [ ] **Scope**: Keep PRs focused on a single concern
- [ ] **Size**: Aim for <500 lines changed (split large changes)

#### Responding to Feedback
- **Be responsive**: Address feedback promptly
- **Be open**: Consider alternative approaches suggested by reviewers
- **Ask questions**: Seek clarification when feedback is unclear
- **Update tests**: Modify tests based on code changes

### For Reviewers

#### Review Checklist
- [ ] **Functionality**: Does the code solve the intended problem?
- [ ] **Architecture**: Is the solution well-designed and maintainable?
- [ ] **Performance**: Are there any obvious performance issues?
- [ ] **Security**: Are there any security concerns?
- [ ] **Testing**: Is the code adequately tested?
- [ ] **Documentation**: Is the code well-documented?
- [ ] **Standards**: Does the code follow project conventions?

#### Review Comments
- **Be constructive**: Suggest improvements rather than just pointing out issues
- **Be specific**: Provide concrete examples and alternatives
- **Be respectful**: Maintain a positive, collaborative tone
- **Explain reasoning**: Help the author understand the why behind suggestions

## 🐛 Issue Guidelines

### Bug Reports

Use the bug report template and include:

```markdown
## Bug Description
Clear description of the issue.

## Steps to Reproduce
1. Run command: `ai-agent analyze --campaign "Test"`
2. Observe error message
3. Check log file for details

## Expected Behavior
Should generate campaign analysis report.

## Actual Behavior
Receives API timeout error after 30 seconds.

## Environment
- OS: macOS 14.0
- Python: 3.12.0
- Poetry: 1.6.1
- Marketing AI Agent: 0.1.0

## Error Messages
```
ERROR: GoogleAPIError: Request timeout after 30s
Traceback (most recent call last):
  ...
```

## Additional Context
- Issue occurs only with specific campaign ID
- Other campaigns work correctly
- Started happening after recent Google Ads API update
```

### Feature Requests

```markdown
## Feature Description
Clear description of the proposed feature.

## Use Case
Why is this feature needed? What problem does it solve?

## Proposed Solution
How should this feature work? Include examples if possible.

## Alternatives Considered
Other approaches you've considered.

## Implementation Notes
Technical considerations or suggestions for implementation.
```

### Issue Labels

- **Type**: `bug`, `feature`, `enhancement`, `documentation`
- **Priority**: `critical`, `high`, `medium`, `low`
- **Complexity**: `good-first-issue`, `help-wanted`, `complex`
- **Component**: `agents`, `analytics`, `api-clients`, `cli`, `core`

## 🚀 Release Process

### Versioning

We follow [Semantic Versioning](https://semver.org/):
- **MAJOR**: Breaking changes
- **MINOR**: New features (backward compatible)
- **PATCH**: Bug fixes (backward compatible)

### Release Workflow

1. **Version Bump**
   ```bash
   # Update version in pyproject.toml
   poetry version patch  # or minor/major
   ```

2. **Update Changelog**
   ```markdown
   ## [0.2.0] - 2024-01-15
   
   ### Added
   - New budget optimization algorithms
   - Support for TikTok Ads API
   
   ### Changed
   - Improved error handling for API timeouts
   
   ### Fixed
   - Fixed campaign analysis edge case
   
   ### Deprecated
   - Legacy reporting format (will be removed in v0.3.0)
   ```

3. **Release Checklist**
   - [ ] All tests pass
   - [ ] Documentation updated
   - [ ] CHANGELOG.md updated
   - [ ] Version bumped in pyproject.toml
   - [ ] Release notes prepared

## 🎯 Contribution Areas

### High Priority

#### New API Integrations
- **Facebook Ads API**: Campaign and performance data
- **Microsoft Advertising API**: Bing Ads integration
- **TikTok Ads API**: Social media advertising platform
- **LinkedIn Ads API**: B2B marketing campaigns

**Requirements**:
- Follow existing client patterns in `api_clients/`
- Include comprehensive error handling
- Add OAuth2 authentication support
- Provide rate limiting and caching
- Include integration tests with mock responses

#### Advanced Analytics Features
- **Attribution Modeling**: Multi-touch attribution algorithms
- **Incrementality Testing**: Statistical significance testing
- **Cohort Analysis**: User behavior tracking over time
- **Competitive Intelligence**: Market share and competitor analysis

**Requirements**:
- Use existing analytics patterns in `analytics/`
- Include statistical validation
- Provide confidence intervals and p-values
- Add visualization support
- Include benchmark tests

### Medium Priority

#### Performance Optimization
- **Query Optimization**: Improve API query efficiency
- **Caching Layer**: Advanced caching strategies
- **Parallel Processing**: Concurrent data fetching
- **Memory Management**: Optimize large dataset handling

#### CLI Enhancements
- **Interactive Mode**: Step-by-step guided workflows
- **Configuration Wizard**: Automated API setup
- **Progress Indicators**: Better feedback for long operations
- **Output Formatting**: Additional export formats

### Getting Started Areas

Perfect for first-time contributors:

- **Documentation**: Improve examples, fix typos, add clarifications
- **Testing**: Add test cases for edge cases and error scenarios
- **Code Quality**: Fix linting issues, improve type annotations
- **Examples**: Create example scripts and use case demonstrations

## 📞 Getting Help

### Development Questions
- **GitHub Discussions**: General questions and ideas
- **Issues**: Specific bugs or feature requests
- **Code Review**: Ask for feedback on draft PRs

### Communication Guidelines
- **Be respectful**: Maintain a welcoming, inclusive environment
- **Be patient**: Allow time for responses, especially for complex questions
- **Be specific**: Provide context and examples when asking for help
- **Search first**: Check existing issues and discussions before posting

---

## 🙏 Acknowledgments

We appreciate all contributors who help make this project better. Your contributions, whether code, documentation, bug reports, or feature suggestions, help the entire marketing analytics community.

**Special thanks to:**
- All contributors who have submitted code, documentation, and bug reports
- The LangChain community for the foundational AI agent framework
- Google for providing comprehensive APIs for marketing data
- The open source Python ecosystem that makes this project possible

---

**Happy Contributing! 🎉**

For questions about contributing, please open a discussion or reach out to the maintainers.