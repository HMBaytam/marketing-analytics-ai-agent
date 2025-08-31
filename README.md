# Marketing Analytics AI Agent

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Poetry](https://img.shields.io/badge/dependency%20manager-poetry-blue.svg)](https://python-poetry.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Tests](https://img.shields.io/badge/tests-pytest-green.svg)](https://pytest.org/)

**AI-powered marketing analytics agent** that uses LangChain to orchestrate specialized sub-agents for campaign analysis, optimization, and reporting.

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/your-org/marketing-ai-agent.git
cd marketing-ai-agent

# Install with Poetry
poetry install
poetry shell

# Configure your APIs
ai-agent config setup

# Verify installation
ai-agent status
```

### Basic Usage

```bash
# Natural language queries
ai-agent "How did Paid Search contribute to conversions last quarter?"
ai-agent "What channels drove the most efficient conversions in Q2?"

# Structured commands
ai-agent analyze --campaign "Search Campaign" --period "last_7_days"
ai-agent report generate --template "weekly" --format "markdown"
```

## 🏗️ Architecture

The agent uses a **multi-agent architecture** with specialized components:

- **🎯 Orchestrator Agent**: Interprets queries and coordinates sub-agents
- **📤 Data Exporter Agent**: Fetches data from GA4 and Google Ads APIs
- **📊 Analysis Agent**: Performs campaign scoring and trend analysis
- **📋 Reporting Agent**: Generates Markdown reports with charts
- **🤖 ML Agent**: Builds regression models and attribution analysis

## 🔧 Configuration

Create a `.env` file with your API credentials:

```env
# Google Ads API
GOOGLE_ADS_DEVELOPER_TOKEN=your_token
GOOGLE_ADS_CLIENT_ID=your_client_id
GOOGLE_ADS_CLIENT_SECRET=your_secret
GOOGLE_ADS_REFRESH_TOKEN=your_refresh_token
GOOGLE_ADS_CUSTOMER_ID=your_customer_id

# Google Analytics 4
GOOGLE_ANALYTICS_PROPERTY_ID=your_property_id
GOOGLE_ANALYTICS_CREDENTIALS_PATH=path/to/service-account.json

# AI Models (Optional)
ANTHROPIC_API_KEY=your_anthropic_key
```

## 📊 Features

- **Multi-Channel Analytics**: GA4 and Google Ads integration
- **AI-Powered Insights**: Natural language analysis and recommendations
- **ML-Driven Optimization**: Budget allocation and bidding strategies
- **Automated Reporting**: Dynamic Markdown reports with visualizations
- **Enterprise Ready**: Error handling, monitoring, and audit logging

## 🧪 Development

```bash
# Setup development environment
poetry install --with dev,test
poetry run pre-commit install

# Run tests
poetry run pytest --cov=src/marketing_ai_agent

# Code quality
poetry run black src/ tests/
poetry run ruff check src/ tests/
poetry run mypy src/
```

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines on:

- Development setup and workflow
- Code standards and quality requirements
- Testing procedures and coverage expectations
- Pull request process and review criteria
- Issue reporting and feature requests

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

- **Documentation**: Check the `/docs` directory for detailed guides
- **Issues**: Report bugs or request features via GitHub Issues
- **Discussions**: Join community discussions for questions and ideas

---

**Built with ❤️ by the Marketing AI Community**