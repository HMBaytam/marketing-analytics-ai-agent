"""Main CLI application for Marketing AI Agent."""

import os
import sys
from pathlib import Path

import typer

from ..core.config import Config, load_config
from ..core.error_handlers import error_reporter, handle_errors
from ..core.exceptions import ConfigurationError
from ..core.logging import initialize_logging
from ..core.monitoring import system_monitor
from .commands import analytics, data, optimization, report

# Add the parent directory to the Python path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

app = typer.Typer(
    name="ai-agent",
    help="🤖 Marketing Analytics AI Agent - Intelligent campaign analysis and optimization",
    rich_markup_mode="rich",
)

# Add command groups
app.add_typer(data.app, name="data", help="📊 Data export and management commands")
app.add_typer(analytics.app, name="analytics", help="📈 Advanced analytics and scoring")
app.add_typer(optimization.app, name="optimize", help="⚡ Optimization recommendations")
app.add_typer(report.app, name="report", help="📋 Report generation and export")

# Global state for configuration
config_state = {"config": None}


@app.callback()
@handle_errors(reraise=True, report=True, severity="CRITICAL")
def main(
    config_file: Path | None = typer.Option(
        None, "--config", "-c", help="Path to configuration file"
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable verbose logging"
    ),
    output_dir: Path | None = typer.Option(
        None, "--output-dir", "-o", help="Output directory for reports and exports"
    ),
    monitor: bool = typer.Option(False, "--monitor", help="Enable system monitoring"),
):
    """
    🤖 Marketing Analytics AI Agent

    Intelligent campaign analysis, optimization, and reporting powered by AI.

    Features:
    • Data export from GA4 and Google Ads
    • Advanced analytics and performance scoring
    • ML-driven optimization recommendations
    • Automated report generation
    • Comprehensive error handling and monitoring

    Examples:
        ai-agent data export --source ga4 --date-range 30d
        ai-agent analytics score --campaign-id ABC123
        ai-agent optimize roi --objective maximize_total_roi
        ai-agent report generate --template executive --output reports/
    """

    try:
        # Load configuration
        if config_file and config_file.exists():
            config = load_config(config_file)
        else:
            config = Config()

        # Apply CLI overrides
        if verbose:
            config.logging.level = "DEBUG"
        if output_dir:
            config.output.base_directory = str(output_dir)

        # Store config in global state
        config_state["config"] = config

        # Initialize comprehensive logging system
        import logging

        log_level = getattr(logging, config.logging.level)
        logger = initialize_logging(log_level)

        logger.info(
            "Marketing AI Agent starting",
            extra={
                "version": "1.0.0",
                "config_file": str(config_file) if config_file else "default",
                "verbose": verbose,
                "output_dir": str(output_dir)
                if output_dir
                else config.output.base_directory,
                "monitoring_enabled": monitor,
            },
        )

        # Start system monitoring if requested
        if monitor:
            system_monitor.start_monitoring()
            logger.info("System monitoring enabled")

    except Exception as e:
        # If logging isn't set up yet, fall back to basic error handling
        if isinstance(e, ConfigurationError):
            typer.secho(f"❌ Configuration Error: {e.message}", fg=typer.colors.RED)
            if e.config_key:
                typer.echo(f"   Problem with config key: {e.config_key}")
        else:
            typer.secho(f"❌ Startup Error: {str(e)}", fg=typer.colors.RED)

        raise typer.Exit(1)


@app.command()
def version():
    """Show version information."""
    typer.echo("Marketing AI Agent v1.0.0")
    typer.echo("Built with ❤️ using Claude 3, LangChain, and Typer")


@app.command()
@handle_errors(reraise=False, report=True)
def monitor(
    start: bool = typer.Option(False, "--start", help="Start system monitoring"),
    stop: bool = typer.Option(False, "--stop", help="Stop system monitoring"),
    metrics_interval: int = typer.Option(
        60, "--metrics-interval", help="Metrics collection interval (seconds)"
    ),
    health_interval: int = typer.Option(
        300, "--health-interval", help="Health check interval (seconds)"
    ),
    export_metrics: Path | None = typer.Option(
        None, "--export-metrics", help="Export metrics to file"
    ),
):
    """Manage system monitoring."""

    if start and stop:
        typer.secho(
            "❌ Cannot start and stop monitoring at the same time", fg=typer.colors.RED
        )
        raise typer.Exit(1)

    if start:
        try:
            system_monitor.start_monitoring(metrics_interval, health_interval)
            typer.secho("✅ System monitoring started", fg=typer.colors.GREEN)
            typer.echo(f"   📊 Metrics interval: {metrics_interval}s")
            typer.echo(f"   🏥 Health check interval: {health_interval}s")
        except Exception as e:
            typer.secho(f"❌ Failed to start monitoring: {str(e)}", fg=typer.colors.RED)
            raise typer.Exit(1)

    elif stop:
        try:
            system_monitor.stop_monitoring()
            typer.secho("⏹️  System monitoring stopped", fg=typer.colors.YELLOW)
        except Exception as e:
            typer.secho(f"❌ Failed to stop monitoring: {str(e)}", fg=typer.colors.RED)
            raise typer.Exit(1)

    elif export_metrics:
        try:
            system_monitor.export_metrics(str(export_metrics))
            typer.secho(
                f"📊 Metrics exported to {export_metrics}", fg=typer.colors.GREEN
            )
        except Exception as e:
            typer.secho(f"❌ Failed to export metrics: {str(e)}", fg=typer.colors.RED)
            raise typer.Exit(1)

    else:
        # Show monitoring status
        status_report = system_monitor.get_status_report()

        typer.echo("📊 System Monitoring Status")
        typer.echo("=" * 30)

        if status_report["monitoring_active"]:
            typer.secho("✅ Status: Active", fg=typer.colors.GREEN)
        else:
            typer.secho("⏸️  Status: Inactive", fg=typer.colors.YELLOW)

        if status_report["current_metrics"]:
            metrics = status_report["current_metrics"]
            avg_metrics = status_report["average_metrics_10min"]

            typer.echo("\n📈 Current Metrics:")
            typer.echo(f"   CPU: {metrics['cpu_percent']:.1f}%")
            typer.echo(f"   Memory: {metrics['memory_percent']:.1f}%")
            typer.echo(f"   Disk: {metrics['disk_usage_percent']:.1f}%")

            if avg_metrics:
                typer.echo("\n📊 10-Min Averages:")
                typer.echo(f"   CPU: {avg_metrics.get('avg_cpu_percent', 0):.1f}%")
                typer.echo(
                    f"   Memory: {avg_metrics.get('avg_memory_percent', 0):.1f}%"
                )
                typer.echo(
                    f"   Disk: {avg_metrics.get('avg_disk_usage_percent', 0):.1f}%"
                )

        typer.echo(
            f"\n💾 Metrics History: {status_report['metrics_history_count']} records"
        )


@app.command()
@handle_errors(reraise=False, report=True)
def errors(
    show_recent: int = typer.Option(
        10, "--recent", "-r", help="Number of recent errors to show"
    ),
    show_summary: bool = typer.Option(
        False, "--summary", "-s", help="Show error summary"
    ),
    export: Path | None = typer.Option(
        None, "--export", help="Export error report to file"
    ),
):
    """Manage and view error reports."""

    try:
        error_summary = error_reporter.get_error_summary()

        typer.echo("🚨 Error Management")
        typer.echo("=" * 25)

        if show_summary or not error_summary["recent_errors"]:
            typer.echo("📊 Error Summary:")
            typer.echo(f"   Total error types: {error_summary['total_error_types']}")

            if error_summary["error_counts"]:
                typer.echo("   Error counts by type:")
                for error_type, count in error_summary["error_counts"].items():
                    typer.echo(f"     • {error_type}: {count}")

                if error_summary["most_common_error"]:
                    most_common = error_summary["most_common_error"]
                    typer.echo(
                        f"   Most common: {most_common[0]} ({most_common[1]} occurrences)"
                    )
            else:
                typer.secho("✅ No errors recorded", fg=typer.colors.GREEN)

        if error_summary["recent_errors"] and not show_summary:
            typer.echo(
                f"\n🕐 Recent Errors (last {min(show_recent, len(error_summary['recent_errors']))}):"
            )

            recent_errors = error_summary["recent_errors"][-show_recent:]
            for error in recent_errors:
                severity_color = {
                    "ERROR": typer.colors.RED,
                    "WARNING": typer.colors.YELLOW,
                    "CRITICAL": typer.colors.MAGENTA,
                }.get(error["severity"], typer.colors.WHITE)

                typer.secho(
                    f"   • [{error['timestamp'][:19]}] {error['error_type']}: {error['message']}",
                    fg=severity_color,
                )

                if error.get("context") and any(error["context"].values()):
                    typer.echo(f"     Context: {error['context']}")

        # Export if requested
        if export:
            import json

            export.parent.mkdir(parents=True, exist_ok=True)

            export_data = {
                "timestamp": typer.get_app_name(),
                "summary": error_summary,
                "detailed_errors": error_summary["recent_errors"],
            }

            with open(export, "w") as f:
                json.dump(export_data, f, indent=2, default=str)

            typer.secho(f"📄 Error report exported to {export}", fg=typer.colors.GREEN)

    except Exception as e:
        typer.secho(f"❌ Error accessing error reports: {str(e)}", fg=typer.colors.RED)


@app.command()
def init(
    directory: Path = typer.Argument(
        ".", help="Directory to initialize (default: current directory)"
    ),
    template: str = typer.Option("basic", help="Configuration template to use"),
):
    """
    Initialize a new project directory with configuration files.

    Creates:
    • config.yaml - Main configuration file
    • .env.example - Environment variables template
    • reports/ - Output directory for reports
    • data/ - Directory for exported data
    """

    if directory.exists() and any(directory.iterdir()):
        if not typer.confirm(f"Directory {directory} is not empty. Continue?"):
            raise typer.Abort()

    directory.mkdir(parents=True, exist_ok=True)

    # Create directory structure
    (directory / "reports").mkdir(exist_ok=True)
    (directory / "data").mkdir(exist_ok=True)
    (directory / "configs").mkdir(exist_ok=True)

    # Create config file
    config_content = """# Marketing AI Agent Configuration
api:
  google_ads:
    client_id: "your-client-id"
    client_secret: "your-client-secret"
    refresh_token: "your-refresh-token"
    developer_token: "your-developer-token"

  google_analytics:
    property_id: "your-property-id"
    credentials_file: "path/to/credentials.json"

output:
  base_directory: "./reports"
  format: "markdown"
  include_charts: true

logging:
  level: "INFO"
  file: "./logs/agent.log"

optimization:
  confidence_threshold: 0.7
  max_recommendations: 10
  include_experimental: true

analytics:
  default_date_range: "30d"
  significance_threshold: 0.05
"""

    (directory / "config.yaml").write_text(config_content)

    # Create .env.example
    env_content = """# Environment Variables for Marketing AI Agent

# Google Ads API
GOOGLE_ADS_CLIENT_ID=your_client_id
GOOGLE_ADS_CLIENT_SECRET=your_client_secret
GOOGLE_ADS_REFRESH_TOKEN=your_refresh_token
GOOGLE_ADS_DEVELOPER_TOKEN=your_developer_token

# Google Analytics
GA_PROPERTY_ID=your_property_id
GOOGLE_APPLICATION_CREDENTIALS=path/to/credentials.json

# OpenAI/Anthropic (for advanced analysis)
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key

# Output Configuration
OUTPUT_DIRECTORY=./reports
LOG_LEVEL=INFO
"""

    (directory / ".env.example").write_text(env_content)

    # Create sample configuration files
    typer.secho(f"✅ Initialized project in {directory}", fg=typer.colors.GREEN)
    typer.echo(f"📝 Edit {directory}/config.yaml to configure your API connections")
    typer.echo("🔐 Copy .env.example to .env and add your credentials")


@app.command()
def config(
    show: bool = typer.Option(False, "--show", help="Show current configuration"),
    validate: bool = typer.Option(False, "--validate", help="Validate configuration"),
):
    """Manage configuration settings."""

    current_config = config_state.get("config")

    if show:
        if current_config:
            typer.echo("Current Configuration:")
            typer.echo(f"Output Directory: {current_config.output.base_directory}")
            typer.echo(f"Log Level: {current_config.logging.level}")
            typer.echo(
                f"Default Date Range: {getattr(current_config.analytics, 'default_date_range', '30d')}"
            )
        else:
            typer.echo("No configuration loaded")

    if validate:
        if current_config:
            # Validate API connections
            typer.echo("🔍 Validating configuration...")

            # Check API credentials (placeholder)
            if hasattr(current_config, "api"):
                typer.secho("✅ API configuration found", fg=typer.colors.GREEN)
            else:
                typer.secho("❌ API configuration missing", fg=typer.colors.RED)

            # Check output directory
            output_dir = Path(current_config.output.base_directory)
            if output_dir.exists():
                typer.secho(
                    f"✅ Output directory exists: {output_dir}", fg=typer.colors.GREEN
                )
            else:
                typer.secho(
                    f"⚠️  Output directory will be created: {output_dir}",
                    fg=typer.colors.YELLOW,
                )
        else:
            typer.secho("❌ No configuration to validate", fg=typer.colors.RED)


@app.command()
@handle_errors(reraise=False, report=True)
def status(
    detailed: bool = typer.Option(
        False, "--detailed", "-d", help="Show detailed system information"
    ),
    export: Path | None = typer.Option(
        None, "--export", help="Export status report to file"
    ),
):
    """Show comprehensive system status and health checks."""

    typer.echo("🤖 Marketing AI Agent Status")
    typer.echo("=" * 50)

    try:
        # Get comprehensive status report
        status_report = system_monitor.get_status_report()

        # Configuration status
        current_config = config_state.get("config")
        if current_config:
            typer.secho("✅ Configuration: Loaded", fg=typer.colors.GREEN)
        else:
            typer.secho("❌ Configuration: Not loaded", fg=typer.colors.RED)

        # Output directory status
        if current_config:
            output_dir = Path(current_config.output.base_directory)
            if output_dir.exists():
                typer.secho(f"✅ Output Directory: {output_dir}", fg=typer.colors.GREEN)
            else:
                typer.secho(
                    f"⚠️  Output Directory: {output_dir} (will be created)",
                    fg=typer.colors.YELLOW,
                )

        # Overall health status
        health_status = status_report["overall_health"]
        if health_status == "healthy":
            typer.secho("✅ System Health: Healthy", fg=typer.colors.GREEN)
        elif health_status == "warning":
            typer.secho("⚠️  System Health: Warning", fg=typer.colors.YELLOW)
        else:
            typer.secho("❌ System Health: Critical", fg=typer.colors.RED)

        # Monitoring status
        if status_report["monitoring_active"]:
            typer.secho("✅ Monitoring: Active", fg=typer.colors.GREEN)
        else:
            typer.secho("⏸️  Monitoring: Inactive", fg=typer.colors.YELLOW)

        # Current metrics
        if status_report["current_metrics"]:
            metrics = status_report["current_metrics"]
            typer.echo("\n📊 Current System Metrics:")
            typer.echo(f"   CPU Usage: {metrics['cpu_percent']:.1f}%")
            typer.echo(f"   Memory Usage: {metrics['memory_percent']:.1f}%")
            typer.echo(f"   Disk Usage: {metrics['disk_usage_percent']:.1f}%")

        # Detailed information
        if detailed:
            typer.echo("\n🔍 Detailed Health Checks:")
            for check_name, check_result in status_report["health_checks"].items():
                status_icon = {"healthy": "✅", "warning": "⚠️", "critical": "❌"}.get(
                    check_result["status"], "❓"
                )
                typer.echo(f"   {status_icon} {check_name}: {check_result['message']}")

            # Error summary
            error_summary = error_reporter.get_error_summary()
            if error_summary["total_error_types"] > 0:
                typer.echo(
                    f"\n⚠️  Recent Errors ({error_summary['total_error_types']} types):"
                )
                for error_type, count in error_summary["error_counts"].items():
                    typer.echo(f"   • {error_type}: {count} occurrences")

        # System dependencies
        typer.echo("\n🔧 Dependencies:")
        try:
            import pandas

            typer.secho("   ✅ Pandas: Available", fg=typer.colors.GREEN)
        except ImportError:
            typer.secho("   ❌ Pandas: Missing", fg=typer.colors.RED)

        try:
            import numpy

            typer.secho("   ✅ NumPy: Available", fg=typer.colors.GREEN)
        except ImportError:
            typer.secho("   ❌ NumPy: Missing", fg=typer.colors.RED)

        try:
            import psutil

            typer.secho("   ✅ psutil: Available", fg=typer.colors.GREEN)
        except ImportError:
            typer.secho("   ❌ psutil: Missing", fg=typer.colors.RED)

        # Export report if requested
        if export:
            import json

            export.parent.mkdir(parents=True, exist_ok=True)
            with open(export, "w") as f:
                json.dump(status_report, f, indent=2, default=str)
            typer.secho(f"📄 Status report exported to {export}", fg=typer.colors.GREEN)

    except Exception as e:
        typer.secho(f"❌ Error getting system status: {str(e)}", fg=typer.colors.RED)


def get_config() -> Config | None:
    """Get the current configuration."""
    return config_state.get("config")


if __name__ == "__main__":
    app()
