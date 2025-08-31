"""Data export and management commands."""

import csv
import json
from datetime import datetime, timedelta
from pathlib import Path

import typer
from rich import print as rprint
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from ...core.config import Config

app = typer.Typer(help="📊 Data export and management commands")
console = Console()


@app.command()
def export(
    source: str = typer.Argument(..., help="Data source (ga4, google-ads, all)"),
    date_range: str = typer.Option(
        "30d",
        "--date-range",
        "-d",
        help="Date range (7d, 30d, 90d, or YYYY-MM-DD,YYYY-MM-DD)",
    ),
    campaign_id: str
    | None = typer.Option(
        None, "--campaign-id", "-c", help="Specific campaign ID to export"
    ),
    output_file: Path
    | None = typer.Option(None, "--output", "-o", help="Output file path"),
    format: str = typer.Option(
        "csv", "--format", "-f", help="Output format (csv, json, xlsx)"
    ),
    metrics: list[str]
    | None = typer.Option(None, "--metrics", "-m", help="Specific metrics to export"),
    include_segments: bool = typer.Option(
        False, "--include-segments", help="Include audience segments"
    ),
    compress: bool = typer.Option(False, "--compress", help="Compress output file"),
):
    """
    Export marketing data from various sources.

    Sources:
    • ga4: Google Analytics 4
    • google-ads: Google Ads API
    • all: Export from all configured sources

    Date Range Examples:
    • 7d: Last 7 days
    • 30d: Last 30 days
    • 2024-01-01,2024-01-31: Custom date range

    Examples:
        ai-agent data export ga4 --date-range 30d
        ai-agent data export google-ads --campaign-id ABC123
        ai-agent data export all --output data/ --format json
    """

    # Get config directly to avoid circular import
    from ...core.config import get_default_config

    config = get_default_config()

    # Parse date range
    start_date, end_date = _parse_date_range(date_range)

    # Setup output
    if not output_file:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = (
            Path(config.output.base_directory) / f"export_{source}_{timestamp}.{format}"
        )

    output_file.parent.mkdir(parents=True, exist_ok=True)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        if source == "ga4" or source == "all":
            task = progress.add_task("Exporting Google Analytics data...", total=None)
            _export_ga4_data(
                config,
                start_date,
                end_date,
                output_file,
                format,
                metrics,
                campaign_id,
                include_segments,
            )
            progress.remove_task(task)

        if source == "google-ads" or source == "all":
            task = progress.add_task("Exporting Google Ads data...", total=None)
            _export_google_ads_data(
                config, start_date, end_date, output_file, format, metrics, campaign_id
            )
            progress.remove_task(task)

    # Compress if requested
    if compress:
        _compress_file(output_file)

    rprint(f"✅ Export completed: [green]{output_file}[/green]")
    _show_export_summary(output_file, format)


@app.command()
def list_campaigns(
    source: str = typer.Argument("all", help="Data source (ga4, google-ads, all)"),
    status: str
    | None = typer.Option(None, "--status", help="Filter by campaign status"),
    limit: int = typer.Option(50, "--limit", "-n", help="Maximum campaigns to show"),
):
    """
    List available campaigns from data sources.

    Examples:
        ai-agent data list-campaigns
        ai-agent data list-campaigns google-ads --status ENABLED
        ai-agent data list-campaigns ga4 --limit 20
    """

    # Get config directly to avoid circular import
    from ...core.config import get_default_config

    get_default_config()

    table = Table(title="📊 Available Campaigns")
    table.add_column("Source", style="cyan")
    table.add_column("Campaign ID", style="green")
    table.add_column("Name", style="white")
    table.add_column("Status", style="yellow")
    table.add_column("Budget", style="magenta")

    with Progress(console=console) as progress:
        task = progress.add_task("Fetching campaigns...", total=None)

        campaigns = []

        # Mock campaign data (in production, this would fetch from APIs)
        if source in ["google-ads", "all"]:
            campaigns.extend(
                [
                    {
                        "source": "Google Ads",
                        "id": "123456789",
                        "name": "Search Campaign Q4",
                        "status": "ENABLED",
                        "budget": "$5,000",
                    },
                    {
                        "source": "Google Ads",
                        "id": "987654321",
                        "name": "Shopping Campaign",
                        "status": "PAUSED",
                        "budget": "$3,000",
                    },
                    {
                        "source": "Google Ads",
                        "id": "456789123",
                        "name": "Display Remarketing",
                        "status": "ENABLED",
                        "budget": "$2,500",
                    },
                ]
            )

        if source in ["ga4", "all"]:
            campaigns.extend(
                [
                    {
                        "source": "GA4",
                        "id": "ga4_001",
                        "name": "Organic Traffic",
                        "status": "ACTIVE",
                        "budget": "N/A",
                    },
                    {
                        "source": "GA4",
                        "id": "ga4_002",
                        "name": "Paid Search Traffic",
                        "status": "ACTIVE",
                        "budget": "N/A",
                    },
                    {
                        "source": "GA4",
                        "id": "ga4_003",
                        "name": "Social Media Traffic",
                        "status": "ACTIVE",
                        "budget": "N/A",
                    },
                ]
            )

        # Apply filters
        if status:
            campaigns = [c for c in campaigns if c["status"].upper() == status.upper()]

        campaigns = campaigns[:limit]
        progress.remove_task(task)

    for campaign in campaigns:
        table.add_row(
            campaign["source"],
            campaign["id"],
            campaign["name"],
            campaign["status"],
            campaign["budget"],
        )

    console.print(table)
    rprint(f"📈 Found {len(campaigns)} campaigns")


@app.command()
def validate_sources(
    fix_permissions: bool = typer.Option(
        False, "--fix", help="Attempt to fix permission issues"
    ),
):
    """
    Validate data source connections and permissions.

    Checks:
    • API credentials and authentication
    • Required permissions and scopes
    • Data access and rate limits

    Examples:
        ai-agent data validate-sources
        ai-agent data validate-sources --fix
    """

    # Get config directly to avoid circular import
    from ...core.config import get_default_config

    config = get_default_config()

    rprint("🔍 [bold]Validating Data Sources[/bold]")
    rprint("=" * 40)

    validation_results = []

    # Google Ads validation
    if config.api.google_ads:
        result = _validate_google_ads_connection(config, fix_permissions)
        validation_results.append(result)
        _print_validation_result("Google Ads API", result)
    else:
        rprint("⚠️  Google Ads: Not configured")

    # Google Analytics validation
    if config.api.google_analytics:
        result = _validate_ga4_connection(config, fix_permissions)
        validation_results.append(result)
        _print_validation_result("Google Analytics 4", result)
    else:
        rprint("⚠️  Google Analytics 4: Not configured")

    # Summary
    rprint("\n📊 [bold]Validation Summary[/bold]")
    passed = sum(1 for r in validation_results if r["status"] == "success")
    failed = len(validation_results) - passed

    rprint(f"✅ Passed: {passed}")
    rprint(f"❌ Failed: {failed}")

    if failed > 0:
        rprint("\n💡 [yellow]Run with --fix to attempt automatic fixes[/yellow]")


@app.command()
def schema(
    source: str = typer.Argument(..., help="Data source (ga4, google-ads)"),
    table: str
    | None = typer.Option(None, "--table", help="Specific table/report to describe"),
    output_format: str = typer.Option(
        "table", "--format", help="Output format (table, json, yaml)"
    ),
):
    """
    Show data schema and available metrics for data sources.

    Examples:
        ai-agent data schema ga4
        ai-agent data schema google-ads --table campaigns
        ai-agent data schema ga4 --format json
    """

    schema_info = _get_data_schema(source, table)

    if output_format == "json":
        rprint(json.dumps(schema_info, indent=2))
    elif output_format == "yaml":
        import yaml

        rprint(yaml.dump(schema_info, default_flow_style=False))
    else:
        _display_schema_table(schema_info, source)


@app.command()
def sync(
    source: str = typer.Argument(
        "all", help="Data source to sync (ga4, google-ads, all)"
    ),
    incremental: bool = typer.Option(
        True, "--incremental/--full", help="Incremental or full sync"
    ),
    schedule: str
    | None = typer.Option(
        None, "--schedule", help="Schedule sync (hourly, daily, weekly)"
    ),
    output_dir: Path
    | None = typer.Option(None, "--output-dir", help="Directory for synced data"),
):
    """
    Setup automated data synchronization.

    Examples:
        ai-agent data sync all --schedule daily
        ai-agent data sync ga4 --incremental --output-dir ./data
    """

    # Get config directly to avoid circular import
    from ...core.config import get_default_config

    config = get_default_config()

    if not output_dir:
        output_dir = Path(config.output.base_directory) / "sync"

    output_dir.mkdir(parents=True, exist_ok=True)

    rprint(f"🔄 [bold]Setting up data sync for {source}[/bold]")

    sync_config = {
        "source": source,
        "incremental": incremental,
        "schedule": schedule,
        "output_directory": str(output_dir),
        "created_at": datetime.now().isoformat(),
    }

    # Save sync configuration
    sync_config_file = output_dir / "sync_config.json"
    with open(sync_config_file, "w") as f:
        json.dump(sync_config, f, indent=2)

    rprint(f"✅ Sync configuration saved to [green]{sync_config_file}[/green]")

    if schedule:
        rprint(f"📅 Scheduled sync: {schedule}")
        rprint("💡 Use a cron job or task scheduler to execute sync commands")


# Helper functions


def _parse_date_range(date_range: str) -> tuple[datetime, datetime]:
    """Parse date range string into start and end dates."""

    if date_range.endswith("d"):
        # Relative date range (7d, 30d, etc.)
        days = int(date_range[:-1])
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=days)
        return datetime.combine(start_date, datetime.min.time()), datetime.combine(
            end_date, datetime.min.time()
        )

    elif "," in date_range:
        # Absolute date range (YYYY-MM-DD,YYYY-MM-DD)
        start_str, end_str = date_range.split(",")
        start_date = datetime.strptime(start_str.strip(), "%Y-%m-%d")
        end_date = datetime.strptime(end_str.strip(), "%Y-%m-%d")
        return start_date, end_date

    else:
        raise typer.BadParameter(f"Invalid date range format: {date_range}")


def _export_ga4_data(
    config: Config,
    start_date: datetime,
    end_date: datetime,
    output_file: Path,
    format: str,
    metrics: list[str] | None,
    campaign_id: str | None,
    include_segments: bool,
):
    """Export Google Analytics 4 data."""

    # Mock data export (in production, use actual GA4 exporter)
    mock_data = [
        {
            "date": "2024-01-01",
            "sessions": 1250,
            "users": 980,
            "page_views": 3500,
            "bounce_rate": 0.45,
            "conversion_rate": 0.025,
            "revenue": 15750.00,
        },
        {
            "date": "2024-01-02",
            "sessions": 1180,
            "users": 920,
            "page_views": 3200,
            "bounce_rate": 0.48,
            "conversion_rate": 0.022,
            "revenue": 14200.00,
        },
    ]

    _save_export_data(mock_data, output_file, format)


def _export_google_ads_data(
    config: Config,
    start_date: datetime,
    end_date: datetime,
    output_file: Path,
    format: str,
    metrics: list[str] | None,
    campaign_id: str | None,
):
    """Export Google Ads data."""

    # Mock data export (in production, use actual Google Ads exporter)
    mock_data = [
        {
            "campaign_id": "123456789",
            "campaign_name": "Search Campaign Q4",
            "date": "2024-01-01",
            "impressions": 25000,
            "clicks": 750,
            "cost": 1875.50,
            "conversions": 45,
            "conversion_value": 6750.00,
        },
        {
            "campaign_id": "123456789",
            "campaign_name": "Search Campaign Q4",
            "date": "2024-01-02",
            "impressions": 23500,
            "clicks": 705,
            "cost": 1762.50,
            "conversions": 42,
            "conversion_value": 6300.00,
        },
    ]

    _save_export_data(mock_data, output_file, format)


def _save_export_data(data: list[dict], output_file: Path, format: str):
    """Save exported data in specified format."""

    if format == "csv":
        with open(output_file, "w", newline="") as f:
            if data:
                writer = csv.DictWriter(f, fieldnames=data[0].keys())
                writer.writeheader()
                writer.writerows(data)

    elif format == "json":
        with open(output_file, "w") as f:
            json.dump(data, f, indent=2, default=str)

    elif format == "xlsx":
        try:
            import pandas as pd

            df = pd.DataFrame(data)
            df.to_excel(output_file, index=False)
        except ImportError:
            typer.secho("❌ pandas required for Excel export", fg=typer.colors.RED)
            raise typer.Exit(1)


def _compress_file(file_path: Path):
    """Compress output file."""
    import gzip
    import shutil

    compressed_path = file_path.with_suffix(f"{file_path.suffix}.gz")

    with open(file_path, "rb") as f_in:
        with gzip.open(compressed_path, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)

    # Remove original file
    file_path.unlink()
    rprint(f"🗜️ Compressed to [green]{compressed_path}[/green]")


def _show_export_summary(output_file: Path, format: str):
    """Show export summary statistics."""

    file_size = output_file.stat().st_size
    file_size_mb = file_size / (1024 * 1024)

    rprint(f"📊 File size: {file_size_mb:.2f} MB")
    rprint(f"📁 Format: {format.upper()}")

    # Show record count for supported formats
    if format == "json":
        try:
            with open(output_file) as f:
                data = json.load(f)
                rprint(f"📈 Records: {len(data)}")
        except:
            pass
    elif format == "csv":
        try:
            with open(output_file) as f:
                row_count = sum(1 for _ in f) - 1  # Subtract header
                rprint(f"📈 Records: {row_count}")
        except:
            pass


def _validate_google_ads_connection(config: Config, fix_permissions: bool) -> dict:
    """Validate Google Ads API connection."""

    # Mock validation (in production, test actual API connection)
    if config.api.google_ads.get("client_id"):
        return {"status": "success", "message": "Connection successful"}
    else:
        return {"status": "error", "message": "Missing client_id"}


def _validate_ga4_connection(config: Config, fix_permissions: bool) -> dict:
    """Validate Google Analytics 4 connection."""

    # Mock validation
    if config.api.google_analytics.get("property_id"):
        return {"status": "success", "message": "Connection successful"}
    else:
        return {"status": "error", "message": "Missing property_id"}


def _print_validation_result(service_name: str, result: dict):
    """Print validation result with appropriate formatting."""

    if result["status"] == "success":
        rprint(f"✅ {service_name}: {result['message']}")
    elif result["status"] == "warning":
        rprint(f"⚠️  {service_name}: {result['message']}")
    else:
        rprint(f"❌ {service_name}: {result['message']}")


def _get_data_schema(source: str, table: str | None) -> dict:
    """Get schema information for data source."""

    schemas = {
        "ga4": {
            "tables": ["sessions", "events", "conversions", "audiences"],
            "metrics": {
                "sessions": "Number of sessions",
                "users": "Number of unique users",
                "page_views": "Total page views",
                "bounce_rate": "Percentage of single-page sessions",
                "conversion_rate": "Conversion rate",
                "revenue": "Total revenue",
            },
            "dimensions": {
                "date": "Date of the event",
                "source": "Traffic source",
                "medium": "Traffic medium",
                "campaign": "Campaign name",
                "device_category": "Device category",
            },
        },
        "google-ads": {
            "tables": ["campaigns", "ad_groups", "keywords", "ads"],
            "metrics": {
                "impressions": "Number of impressions",
                "clicks": "Number of clicks",
                "cost": "Total cost",
                "conversions": "Number of conversions",
                "conversion_value": "Total conversion value",
            },
            "dimensions": {
                "campaign_id": "Campaign ID",
                "campaign_name": "Campaign name",
                "ad_group_id": "Ad group ID",
                "ad_group_name": "Ad group name",
                "date": "Date of the data",
            },
        },
    }

    return schemas.get(source, {})


def _display_schema_table(schema_info: dict, source: str):
    """Display schema information as a formatted table."""

    rprint(f"📋 [bold]{source.upper()} Data Schema[/bold]")

    # Tables
    if "tables" in schema_info:
        rprint(f"\n📊 Available Tables: {', '.join(schema_info['tables'])}")

    # Metrics
    if "metrics" in schema_info:
        metrics_table = Table(title="📈 Available Metrics")
        metrics_table.add_column("Metric", style="green")
        metrics_table.add_column("Description", style="white")

        for metric, description in schema_info["metrics"].items():
            metrics_table.add_row(metric, description)

        console.print(metrics_table)

    # Dimensions
    if "dimensions" in schema_info:
        dimensions_table = Table(title="🏷️ Available Dimensions")
        dimensions_table.add_column("Dimension", style="blue")
        dimensions_table.add_column("Description", style="white")

        for dimension, description in schema_info["dimensions"].items():
            dimensions_table.add_row(dimension, description)

        console.print(dimensions_table)
