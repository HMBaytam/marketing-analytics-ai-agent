# Marketing Analytics AI Agent

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Poetry](https://img.shields.io/badge/dependency%20manager-poetry-blue.svg)](https://python-poetry.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Tests](https://img.shields.io/badge/tests-pytest-green.svg)](https://pytest.org/)
[![Coverage](https://img.shields.io/badge/coverage-80%25+-brightgreen.svg)](https://pytest-cov.readthedocs.io/)
[![Type Checking](https://img.shields.io/badge/type%20checking-mypy-blue.svg)](http://mypy-lang.org/)

**Enterprise-grade AI-powered marketing analytics and optimization platform** built with LangChain, featuring intelligent agent orchestration, advanced analytics engines, ML-driven optimization algorithms, comprehensive error handling, and enterprise-ready monitoring systems.

## 🚀 Features

### 🤖 AI Agent Architecture
- **Intelligent Orchestrator**: Main coordination agent that interprets queries and delegates tasks
- **Specialized Sub-Agents**: 
  - **Data Exporter Agent**: API integrations and data retrieval
  - **Analysis Agent**: Campaign performance scoring and trend analysis  
  - **Reporting Agent**: Automated report generation with visualizations
  - **ML Agent**: Machine learning models for insights and predictions
- **LangChain Integration**: Advanced agent workflows with tool orchestration

### 📊 Analytics & Intelligence
- **Performance Scoring**: Multi-dimensional campaign performance analysis
- **Trend Analysis**: Time-series analysis with seasonal decomposition
- **Anomaly Detection**: Statistical outlier identification and alerting
- **Predictive Modeling**: ML-driven forecasting and trend prediction
- **Attribution Analysis**: Multi-touch attribution modeling and incrementality testing
- **Competitive Intelligence**: Market share analysis and competitive benchmarking

### ⚡ Optimization Engines  
- **Budget Optimization**: ML-driven budget allocation across channels and campaigns
- **ROI Maximization**: Automated bidding strategies and spend optimization
- **A/B Testing**: Statistical testing framework with confidence intervals
- **Audience Optimization**: Segment performance analysis and targeting recommendations
- **Creative Optimization**: Asset performance analysis and creative recommendations

### 📈 Data Integration & APIs
- **Google Analytics 4 (GA4)**: Complete GA4 Data API integration
- **Google Ads API**: Campaign, keyword, and performance data
- **Multi-Channel Support**: Social media, email, and affiliate integrations
- **Real-Time Data**: Stream processing and live dashboard updates
- **Data Quality**: Automated validation, cleansing, and enrichment

### 📋 Reporting & Visualization
- **Interactive Reports**: Dynamic Markdown reports with embedded charts
- **Multi-Format Export**: PDF, Excel, JSON, and CSV outputs
- **Scheduled Reports**: Automated report generation and distribution
- **Custom Dashboards**: Configurable KPI dashboards and alerts
- **Data Storytelling**: Natural language insights and recommendations

### 💻 Enterprise CLI Interface
- **Comprehensive Commands**: Full-featured command-line interface
- **Interactive Mode**: Rich formatting and progress indicators  
- **Batch Processing**: Bulk operations and automated workflows
- **Configuration Management**: Environment-specific settings and secrets
- **Plugin System**: Extensible architecture for custom commands

### 🛡️ Enterprise-Grade Infrastructure
- **Error Handling**: Circuit breaker patterns, retry logic, and graceful degradation
- **System Monitoring**: Real-time health checks, performance metrics, and alerting
- **Audit Logging**: Comprehensive audit trails and compliance reporting
- **Security**: API key management, data encryption, and access controls
- **Scalability**: Horizontal scaling support and load balancing
- **High Availability**: Fault tolerance and disaster recovery capabilities

### 🧪 Quality Assurance
- **Comprehensive Testing**: Unit, integration, and end-to-end tests (80%+ coverage)
- **Performance Testing**: Benchmarking and load testing capabilities
- **Code Quality**: Static analysis, type checking, and automated formatting
- **Continuous Integration**: Automated testing and deployment pipelines
- **Documentation**: Comprehensive API docs and user guides

## 🏗️ Architecture

### System Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   CLI Interface │────│ Orchestrator     │────│ Sub-Agents      │
│   - Commands    │    │ Agent            │    │ - Data Export   │
│   - Interactive │    │ - Query Parser   │    │ - Analysis      │  
│   - Batch       │    │ - Task Router    │    │ - Reporting     │
└─────────────────┘    │ - Agent Manager  │    │ - ML Models     │
                       └──────────────────┘    └─────────────────┘
                                │                        │
                       ┌────────┴────────┐              │
                       │                 │              │
                ┌──────▼──────┐   ┌─────▼─────┐   ┌────▼────┐
                │ Data Layer  │   │ Analytics │   │ Output  │
                │ - GA4 API   │   │ Engine    │   │ Engine  │
                │ - Ads API   │   │ - Scoring │   │ - Reports│
                │ - Storage   │   │ - ML      │   │ - Alerts│
                └─────────────┘   └───────────┘   └─────────┘
```

### Agent Architecture

#### 🎯 Orchestrator Agent
- **Query Intelligence**: Natural language understanding and intent recognition
- **Task Decomposition**: Breaks complex queries into executable steps
- **Agent Coordination**: Manages sub-agent workflows and dependencies
- **Result Synthesis**: Combines outputs into coherent final reports
- **Error Recovery**: Handles failures and provides graceful degradation

#### 📤 Data Exporter Agent  
- **API Management**: Handles authentication and rate limiting
- **Data Retrieval**: Fetches campaign, audience, and performance data
- **Data Validation**: Ensures data quality and completeness
- **Format Conversion**: Transforms data into analysis-ready formats
- **Caching Strategy**: Implements intelligent caching for performance

#### 📊 Analysis Agent
- **Performance Scoring**: Multi-dimensional campaign evaluation
- **Statistical Analysis**: Trend analysis and significance testing  
- **Anomaly Detection**: Identifies unusual patterns and outliers
- **Comparative Analysis**: Cross-channel and time-period comparisons
- **Insight Generation**: Produces actionable recommendations

#### 📋 Reporting Agent
- **Dynamic Reports**: Context-aware report generation
- **Visualization Engine**: Creates charts, graphs, and interactive elements
- **Multi-Format Output**: Supports Markdown, PDF, Excel, and JSON
- **Template System**: Customizable report templates and themes
- **Distribution**: Automated report delivery and scheduling

#### 🤖 ML Agent
- **Model Library**: Pre-built models for common marketing use cases
- **Attribution Modeling**: Multi-touch attribution and incrementality
- **Forecasting**: Time series prediction and scenario planning
- **Optimization**: Budget allocation and bid optimization algorithms
- **A/B Testing**: Statistical significance testing and confidence intervals

### Technology Stack

#### 🧠 AI & Machine Learning
- **LangChain**: Agent orchestration and tool integration
- **LangGraph**: Complex workflow management
- **Anthropic Claude**: Primary language model for analysis
- **scikit-learn**: Machine learning algorithms and statistical models
- **TensorFlow/PyTorch**: Deep learning for advanced analytics

#### 📊 Data Processing & Analytics  
- **Pandas**: Data manipulation and analysis
- **Polars**: High-performance data processing
- **NumPy**: Numerical computing and statistics
- **Plotly**: Interactive visualizations and charts
- **Statsmodels**: Statistical modeling and econometrics

#### 🌐 API Integration & Data Sources
- **Google Analytics Data API**: GA4 reporting and real-time data
- **Google Ads API**: Campaign management and performance metrics
- **httpx**: Async HTTP client for API calls
- **Pydantic**: Data validation and serialization

#### 💻 Interface & CLI
- **Typer**: Modern CLI framework with rich formatting
- **Rich**: Advanced terminal formatting and progress indicators
- **Click**: Additional CLI utilities and extensions

#### 🛡️ Infrastructure & Operations
- **Poetry**: Dependency management and packaging
- **pytest**: Testing framework with fixtures and mocking
- **mypy**: Static type checking and code analysis
- **Black**: Code formatting and style consistency
- **Ruff**: Fast Python linter and code quality

## 🚀 Installation

### System Requirements

- **Python**: 3.11 or 3.12 (recommended for optimal performance)
- **Memory**: Minimum 4GB RAM, 8GB+ recommended for large datasets
- **Storage**: 500MB for installation, additional space for data and reports
- **Network**: Internet connection for API access and model downloads

### API Prerequisites

- **Google Ads API**: Developer token, client credentials, and refresh token
- **Google Analytics 4**: Service account JSON file or OAuth2 credentials  
- **Anthropic API**: API key for Claude model access (optional for advanced features)

### Quick Start

1. **Clone and Setup**:
   ```bash
   git clone <repository-url>
   cd ai-agent
   
   # Install Poetry (if not already installed)
   curl -sSL https://install.python-poetry.org | python3 -
   
   # Install dependencies
   poetry install
   
   # Activate virtual environment
   poetry shell
   ```

2. **Configuration**:
   ```bash
   # Initialize configuration
   ai-agent init
   
   # Configure APIs interactively
   ai-agent config setup
   ```

3. **Verify Installation**:
   ```bash
   # Check system status
   ai-agent status
   
   # Run test query
   ai-agent "Show me campaign performance for the last 7 days"
   ```

### Manual Configuration

Create a `.env` file in the project root:

```env
# Google Ads API
GOOGLE_ADS_DEVELOPER_TOKEN=your_developer_token
GOOGLE_ADS_CLIENT_ID=your_client_id
GOOGLE_ADS_CLIENT_SECRET=your_client_secret
GOOGLE_ADS_REFRESH_TOKEN=your_refresh_token
GOOGLE_ADS_CUSTOMER_ID=your_customer_id

# Google Analytics 4
GOOGLE_ANALYTICS_PROPERTY_ID=your_property_id
GOOGLE_ANALYTICS_CREDENTIALS_PATH=path/to/service-account.json
# OR for OAuth2
GOOGLE_ANALYTICS_CLIENT_ID=your_oauth_client_id
GOOGLE_ANALYTICS_CLIENT_SECRET=your_oauth_client_secret
GOOGLE_ANALYTICS_REFRESH_TOKEN=your_oauth_refresh_token

# AI Models (Optional)
ANTHROPIC_API_KEY=your_anthropic_api_key
OPENAI_API_KEY=your_openai_api_key

# Application Settings
OUTPUT_DIRECTORY=./reports
LOG_LEVEL=INFO
CACHE_ENABLED=true
CACHE_TTL=3600
```

### Development Setup

For development and testing:

```bash
# Install with development dependencies
poetry install --with dev,test

# Install pre-commit hooks
poetry run pre-commit install

# Run tests
poetry run pytest

# Type checking
poetry run mypy src/

# Code formatting
poetry run black src/ tests/
poetry run ruff check src/ tests/
```

### Docker Installation (Alternative)

```dockerfile
# Build Docker image
docker build -t marketing-ai-agent .

# Run container
docker run -it --env-file .env marketing-ai-agent
```

## 💻 Usage

### Command Overview

The AI agent provides a comprehensive CLI with natural language queries and structured commands:

```bash
# Natural Language Queries (Recommended)
ai-agent "How did Paid Search contribute to conversions last quarter?"
ai-agent "What channels drove the most efficient conversions in Q2?"
ai-agent "Analyze campaign performance for the last 30 days"
ai-agent "Generate a weekly performance report for all active campaigns"

# Structured Commands
ai-agent analyze --campaign "Search Campaign" --period "last_7_days" 
ai-agent report --account-id "123456789" --format "markdown"
ai-agent optimize --budget 10000 --objective "maximize_conversions"
```

### Core Commands

#### 📊 Analysis Commands
```bash
# Campaign performance analysis
ai-agent analyze --campaign "Campaign Name" --metrics "conversions,cost,revenue"

# Multi-channel attribution analysis  
ai-agent analyze attribution --model "data_driven" --lookback_days 30

# Anomaly detection
ai-agent analyze anomalies --metric "conversions" --sensitivity 0.05

# Trend analysis with forecasting
ai-agent analyze trends --period "last_90_days" --forecast_days 30
```

#### 📈 Data & Export Commands
```bash
# Export campaign data
ai-agent data export ga4 --date-range "7d" --metrics "sessions,conversions" 

# Export Google Ads data
ai-agent data export ads --campaigns "all" --date-range "30d"

# Real-time data monitoring
ai-agent data monitor --interval 60 --threshold "conversions<10"
```

#### 🎯 Optimization Commands
```bash
# Budget optimization
ai-agent optimize budget --total 50000 --objective "maximize_roas"

# Bid optimization
ai-agent optimize bids --campaigns "Search*" --target_roas 4.0

# A/B testing
ai-agent optimize test --variant_split 0.5 --test_duration 14
```

#### 📋 Reporting Commands
```bash
# Automated report generation
ai-agent report generate --template "weekly" --format "markdown" 

# Custom dashboard
ai-agent report dashboard --kpis "roas,cpa,conversion_rate"

# Scheduled reports
ai-agent report schedule --frequency "weekly" --recipients "team@company.com"
```

#### ⚙️ System & Configuration Commands
```bash
# System status and health
ai-agent status --detailed

# Configuration management
ai-agent config show
ai-agent config set google_ads.customer_id "123456789"

# Error monitoring
ai-agent errors show --last 24h
ai-agent errors export --format json --output errors.json
```

### Natural Language Examples

#### Performance Analysis
```bash
ai-agent "Show me the top performing campaigns by ROAS for the last month"
ai-agent "Which keywords are driving the most conversions at the lowest cost?"
ai-agent "Compare performance between Display and Search campaigns"
```

#### Optimization & Insights  
```bash
ai-agent "How should I reallocate budget to maximize conversions?"
ai-agent "What's the optimal bid strategy for my Search campaigns?"
ai-agent "Identify underperforming campaigns and suggest improvements"
```

#### Reporting & Monitoring
```bash
ai-agent "Generate a comprehensive monthly performance report"
ai-agent "Create a weekly executive summary with key insights"
ai-agent "Set up alerts for campaigns with declining performance"
```

### Advanced Workflows

#### Multi-Step Analysis
```bash
# Complex attribution analysis with reporting
ai-agent "Analyze the incrementality of Paid Search on conversions for Q3, 
         include statistical significance testing, and generate a detailed 
         report with recommendations for Q4 budget allocation"
```

#### Automated Optimization
```bash  
# End-to-end optimization workflow
ai-agent "Review current campaign performance, identify optimization 
         opportunities, simulate budget reallocation scenarios, and 
         provide implementation recommendations with expected impact"
```

#### Custom Analysis
```bash
# Business-specific analysis
ai-agent "Analyze seasonal trends in our retail campaigns, identify 
         peak performance periods, and create a seasonal bidding strategy 
         for the upcoming holiday season"
```

### Output Examples

#### Performance Report Output
```markdown
## Campaign Performance Analysis - Last 30 Days

### Executive Summary
- **Total Conversions**: 2,547 (+12% vs. previous period)
- **Total Revenue**: $127,350 (+18% vs. previous period)  
- **Average ROAS**: 4.2x (+0.3x vs. previous period)
- **Cost Per Acquisition**: $15.20 (-8% vs. previous period)

### Top Performing Campaigns
| Campaign Name          | Conversions | Revenue   | ROAS | CPA   |
|------------------------|-------------|-----------|------|-------|
| Search - Brand Terms   | 1,205       | $65,400   | 8.1x | $12.50|
| Search - Generic Terms | 847         | $41,200   | 3.2x | $18.75|
| Display - Remarketing  | 495         | $20,750   | 2.8x | $22.10|

### ML Insights & Attribution
- **Paid Search** contributed +35% incremental conversions (p<0.01)
- **Display campaigns** show strong assisted conversion impact (+22% view-through)
- **Brand campaigns** demonstrate highest efficiency but limited scale opportunity

### Recommendations
1. **Increase Budget**: Allocate additional 15% budget to Brand Search campaigns
2. **Optimization**: Implement automated bidding for Generic Search terms  
3. **Testing**: Launch expanded remarketing audience tests for Display
4. **Monitoring**: Set up performance alerts for campaigns below 3.0x ROAS

*Report generated on 2024-01-15 at 09:30 UTC*
*Data sources: Google Ads API, Google Analytics 4*
```

## ⚙️ Configuration

### Configuration Hierarchy

The system uses a hierarchical configuration approach:

1. **Environment Variables** (`.env` file)
2. **Configuration Files** (`config/config.yaml`)  
3. **CLI Arguments** (command-line overrides)
4. **Default Values** (built-in fallbacks)

### Environment Configuration

#### Google Ads API Setup
```env
GOOGLE_ADS_DEVELOPER_TOKEN=your_developer_token
GOOGLE_ADS_CLIENT_ID=your_client_id
GOOGLE_ADS_CLIENT_SECRET=your_client_secret
GOOGLE_ADS_REFRESH_TOKEN=your_refresh_token
GOOGLE_ADS_CUSTOMER_ID=your_customer_id
GOOGLE_ADS_LOGIN_CUSTOMER_ID=your_login_customer_id
```

#### Google Analytics 4 Configuration
```env
# Service Account (Recommended)
GOOGLE_ANALYTICS_PROPERTY_ID=your_property_id
GOOGLE_ANALYTICS_CREDENTIALS_PATH=path/to/service-account.json

# OAuth2 Alternative  
GOOGLE_ANALYTICS_CLIENT_ID=your_oauth_client_id
GOOGLE_ANALYTICS_CLIENT_SECRET=your_oauth_client_secret
GOOGLE_ANALYTICS_REFRESH_TOKEN=your_oauth_refresh_token
```

#### AI Model Configuration
```env
# Primary AI model
ANTHROPIC_API_KEY=your_anthropic_api_key
ANTHROPIC_MODEL=claude-3-sonnet-20240229

# Alternative models
OPENAI_API_KEY=your_openai_api_key
OPENAI_MODEL=gpt-4

# Model preferences
DEFAULT_MODEL=anthropic
FALLBACK_MODEL=openai
```

#### Application Settings
```env
# Output and reporting
OUTPUT_DIRECTORY=./reports
REPORT_TEMPLATES_PATH=./templates
DEFAULT_REPORT_FORMAT=markdown

# Performance and caching
CACHE_ENABLED=true
CACHE_DIRECTORY=./.cache
CACHE_TTL=3600
MAX_CONCURRENT_REQUESTS=10

# Logging and monitoring
LOG_LEVEL=INFO
LOG_FORMAT=json
LOG_FILE=logs/marketing-ai-agent.log
ENABLE_AUDIT_LOGGING=true

# Security
API_RATE_LIMIT_ENABLED=true
MAX_REQUESTS_PER_MINUTE=60
ENCRYPT_CACHE=true
```

### Advanced Configuration

#### config/config.yaml
```yaml
# API Settings
apis:
  google_ads:
    api_version: v14
    timeout: 30
    retry_attempts: 3
    retry_delay: 1.0
  
  google_analytics:
    api_version: v1beta
    timeout: 30
    batch_size: 100000
    max_dimensions: 10

# Analytics Engine
analytics:
  performance_scoring:
    weights:
      conversions: 0.4
      revenue: 0.3
      efficiency: 0.2
      trend: 0.1
  
  anomaly_detection:
    sensitivity: 0.05
    min_data_points: 14
    algorithms: ['isolation_forest', 'statistical']

# ML Models
machine_learning:
  attribution:
    model_type: 'shapley'
    lookback_days: 30
    conversion_lag: 7
  
  forecasting:
    model_type: 'arima'
    seasonal_periods: [7, 30, 365]
    confidence_intervals: [0.8, 0.95]

# Optimization
optimization:
  budget:
    algorithms: ['linear_programming', 'genetic_algorithm']
    constraints: ['min_budget', 'max_budget', 'roas_threshold']
  
  bidding:
    strategies: ['target_cpa', 'target_roas', 'maximize_conversions']
    adjustment_limits: [-50, 200]  # percentage
```

### CLI Configuration Commands

```bash
# View current configuration
ai-agent config show

# Set configuration values
ai-agent config set google_ads.customer_id "123456789"
ai-agent config set output.format "json"

# Validate configuration
ai-agent config validate

# Reset to defaults
ai-agent config reset

# Export configuration
ai-agent config export --output config-backup.yaml
```

## 🧪 Development

### Project Structure

```
marketing-ai-agent/
├── src/marketing_ai_agent/
│   ├── agents/                 # AI agent implementations
│   │   ├── orchestrator.py     # Main coordination agent
│   │   ├── data_exporter.py    # Data retrieval agent
│   │   ├── analyzer.py         # Analysis agent
│   │   ├── reporter.py         # Reporting agent
│   │   └── ml_agent.py         # ML modeling agent
│   ├── api_clients/            # External API integrations
│   │   ├── ga4_client.py       # Google Analytics 4 client
│   │   ├── google_ads_client.py # Google Ads API client
│   │   └── base_client.py      # Base API client class
│   ├── analytics/              # Analytics engines
│   │   ├── performance_scorer.py # Performance analysis
│   │   ├── trend_analyzer.py   # Trend detection
│   │   ├── anomaly_detector.py # Anomaly detection
│   │   └── attribution.py     # Attribution modeling
│   ├── optimization/           # Optimization algorithms
│   │   ├── budget_optimizer.py # Budget allocation
│   │   ├── bid_optimizer.py    # Bidding strategies
│   │   └── testing_engine.py   # A/B testing
│   ├── cli/                    # Command-line interface
│   │   ├── main.py             # Main CLI application
│   │   └── commands/           # Command implementations
│   ├── core/                   # Core infrastructure
│   │   ├── config.py           # Configuration management
│   │   ├── exceptions.py       # Custom exceptions
│   │   ├── logging.py          # Logging system
│   │   ├── error_handlers.py   # Error handling
│   │   └── monitoring.py       # System monitoring
│   ├── models/                 # Data models and schemas
│   ├── reports/                # Report generation
│   └── utils/                  # Utility functions
├── tests/                      # Comprehensive test suite
│   ├── conftest.py             # Test configuration
│   ├── factories.py            # Test data factories
│   ├── test_*.py               # Test modules
│   └── fixtures/               # Test data fixtures
├── docs/                       # Documentation
├── config/                     # Configuration files
├── templates/                  # Report templates
└── scripts/                    # Utility scripts
```

### Development Workflow

#### Environment Setup
```bash
# Clone repository
git clone <repository-url>
cd marketing-ai-agent

# Setup development environment
poetry install --with dev,test,docs

# Install pre-commit hooks
poetry run pre-commit install

# Activate environment
poetry shell
```

#### Testing

```bash
# Run all tests
poetry run pytest

# Run with coverage
poetry run pytest --cov=src/marketing_ai_agent --cov-report=html

# Run specific test categories
poetry run pytest -m "unit"              # Unit tests only
poetry run pytest -m "integration"       # Integration tests
poetry run pytest -m "not slow"          # Skip slow tests

# Run performance tests
poetry run pytest -m "performance" --benchmark-only

# Test runner with reporting
python tests/test_runner.py all
```

#### Code Quality

```bash
# Format code
poetry run black src/ tests/

# Lint code
poetry run ruff check src/ tests/

# Type checking
poetry run mypy src/

# Security scanning
poetry run bandit -r src/

# Import sorting
poetry run isort src/ tests/
```

#### Documentation

```bash
# Generate API documentation
poetry run sphinx-build -b html docs/ docs/_build/

# Serve documentation locally
poetry run python -m http.server 8000 --directory docs/_build/
```

### Debugging and Profiling

```bash
# Debug mode with verbose logging
ai-agent --debug analyze --campaign "Test Campaign"

# Performance profiling
python -m cProfile -o profile.stats src/marketing_ai_agent/cli/main.py

# Memory profiling
python -m memory_profiler src/marketing_ai_agent/cli/main.py

# Line profiling (requires line_profiler)
kernprof -l -v src/marketing_ai_agent/analytics/performance_scorer.py
```

## 🤝 Contributing

We welcome contributions to the Marketing Analytics AI Agent! Please follow our contribution guidelines to ensure a smooth process.

### Getting Started

1. **Fork the repository** and create your feature branch:
   ```bash
   git checkout -b feature/amazing-feature
   ```

2. **Set up development environment**:
   ```bash
   poetry install --with dev,test,docs
   poetry run pre-commit install
   ```

3. **Make your changes** following our coding standards:
   - Follow PEP 8 style guidelines
   - Add type hints for all functions
   - Include comprehensive docstrings
   - Write tests for new functionality

4. **Run quality checks**:
   ```bash
   # Format and lint
   poetry run black src/ tests/
   poetry run ruff check src/ tests/
   
   # Type checking
   poetry run mypy src/
   
   # Run tests
   poetry run pytest --cov=src/marketing_ai_agent
   ```

5. **Commit and push** your changes:
   ```bash
   git add .
   git commit -m "feat: add amazing feature"
   git push origin feature/amazing-feature
   ```

6. **Submit a pull request** with a clear description of your changes.

### Development Guidelines

#### Code Standards
- **Type Hints**: All functions must have complete type annotations
- **Docstrings**: Use Google-style docstrings for all public functions
- **Error Handling**: Use custom exceptions with proper context
- **Testing**: Maintain 80%+ test coverage for new code
- **Documentation**: Update README and docs for user-facing changes

#### Commit Messages
Follow the [Conventional Commits](https://conventionalcommits.org/) specification:
```
feat: add new optimization algorithm
fix: resolve API rate limiting issue
docs: update installation instructions
test: add integration tests for reporting
refactor: improve error handling consistency
```

#### Pull Request Process
1. Ensure all tests pass and coverage meets requirements
2. Update documentation for any user-facing changes
3. Add appropriate labels to your PR
4. Request review from maintainers
5. Address feedback and iterate as needed

### Areas for Contribution

#### 🚀 High Priority
- **New API Integrations**: Facebook Ads, Microsoft Ads, TikTok Ads
- **Advanced ML Models**: Deep learning attribution models, advanced forecasting
- **Performance Optimization**: Query optimization, caching improvements
- **Testing**: Additional test coverage, performance benchmarks

#### 🔧 Medium Priority  
- **Visualization**: New chart types, interactive dashboards
- **Export Formats**: PowerBI, Tableau, Data Studio connectors
- **CLI Enhancements**: Interactive modes, configuration wizards
- **Documentation**: Video tutorials, use case examples

#### 💡 Nice to Have
- **Mobile Support**: React Native app, mobile-optimized reports
- **Internationalization**: Multi-language support, currency handling
- **Plugin System**: Custom analytics engines, third-party integrations
- **Cloud Deployment**: Docker containers, Kubernetes manifests

### Bug Reports

When reporting bugs, please include:
- **Environment**: OS, Python version, dependency versions
- **Steps to Reproduce**: Detailed steps to recreate the issue
- **Expected Behavior**: What should have happened
- **Actual Behavior**: What actually happened
- **Error Messages**: Full error logs and stack traces
- **Additional Context**: Screenshots, configuration files

### Feature Requests

For new features, please provide:
- **Use Case**: Why is this feature needed?
- **Proposed Solution**: How should it work?
- **Alternatives**: Other approaches considered
- **Implementation**: Technical approach if known

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### Third-Party Licenses
- **LangChain**: MIT License
- **Google APIs**: Apache License 2.0  
- **Anthropic SDK**: MIT License
- **Other dependencies**: Various open-source licenses

## 🆘 Support

### Getting Help

1. **Documentation**: Check our comprehensive docs in the `/docs` directory
2. **Issues**: Search existing issues before creating new ones
3. **Discussions**: Join our community discussions for general questions
4. **Stack Overflow**: Tag questions with `marketing-ai-agent`

### Contact

- **GitHub Issues**: Bug reports and feature requests
- **Email**: [maintainers@marketing-ai-agent.com](mailto:maintainers@marketing-ai-agent.com)
- **Community**: Join our [Discord server](https://discord.gg/marketing-ai-agent)

### Enterprise Support

For enterprise users requiring:
- **Priority Support**: SLA-backed response times
- **Custom Development**: Tailored features and integrations
- **Training & Consulting**: Implementation guidance and best practices
- **Deployment Assistance**: Production deployment and monitoring

Contact us at [enterprise@marketing-ai-agent.com](mailto:enterprise@marketing-ai-agent.com)

---

**Built with ❤️ by the Marketing AI Community**

[![Made with Python](https://img.shields.io/badge/Made%20with-Python-blue?logo=python&logoColor=white)](https://python.org)
[![Powered by LangChain](https://img.shields.io/badge/Powered%20by-LangChain-green)](https://langchain.com)
[![AI Enabled](https://img.shields.io/badge/AI-Enabled-purple)](https://anthropic.com)
