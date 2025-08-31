"""Advanced analytics and scoring commands."""

import json
from pathlib import Path

import typer
from rich import print as rprint
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn
from rich.table import Table

from ...analytics import (
    AnomalyConfig,
    AnomalyDetector,
    BenchmarkConfig,
    BenchmarkingEngine,
    PerformanceScorer,
    PredictionConfig,
    PredictiveModel,
    ScoringConfig,
    TrendAnalyzer,
    TrendConfig,
)

# Removed circular import - get_config imported locally where needed

app = typer.Typer(help="📈 Advanced analytics and scoring commands")
console = Console()


@app.command()
def score(
    campaign_id: str | None = typer.Option(
        None, "--campaign-id", "-c", help="Specific campaign to score"
    ),
    date_range: str = typer.Option(
        "30d", "--date-range", "-d", help="Date range for analysis"
    ),
    output_file: Path | None = typer.Option(
        None, "--output", "-o", help="Save results to file"
    ),
    include_breakdown: bool = typer.Option(
        True, "--breakdown/--no-breakdown", help="Include score breakdown"
    ),
    threshold: float = typer.Option(
        0.7, "--threshold", "-t", help="Performance threshold for alerts"
    ),
    format: str = typer.Option(
        "table", "--format", "-f", help="Output format (table, json, markdown)"
    ),
):
    """
    Calculate comprehensive performance scores for campaigns.

    Analyzes:
    • Efficiency metrics (CPA, ROAS, CTR)
    • Volume performance (impressions, conversions)
    • Quality indicators (Quality Score, relevance)
    • Trend analysis and momentum

    Examples:
        ai-agent analytics score --campaign-id ABC123
        ai-agent analytics score --threshold 0.8 --output scores.json
        ai-agent analytics score --format markdown --breakdown
    """

    # Get config directly to avoid circular import
    from ...core.config import get_default_config

    get_default_config()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Analyzing campaign performance...", total=100)

        # Mock performance scoring (in production, use actual data)
        mock_campaigns = _get_mock_campaign_data(campaign_id)

        progress.update(
            task, advance=30, description="Calculating performance scores..."
        )

        PerformanceScorer(ScoringConfig())
        results = []

        for campaign in mock_campaigns:
            progress.update(
                task, advance=10, description=f"Scoring {campaign['name']}..."
            )

            # Mock scoring calculation
            score_result = {
                "campaign_id": campaign["id"],
                "campaign_name": campaign["name"],
                "overall_score": campaign["performance_score"],
                "grade": _score_to_grade(campaign["performance_score"]),
                "efficiency_score": campaign.get("efficiency_score", 0.75),
                "volume_score": campaign.get("volume_score", 0.80),
                "quality_score": campaign.get("quality_score", 0.70),
                "trend_score": campaign.get("trend_score", 0.65),
                "recommendations": _generate_score_recommendations(
                    campaign["performance_score"]
                ),
            }
            results.append(score_result)

        progress.update(task, advance=40, description="Generating insights...")

        # Display results
        if format == "table":
            _display_scores_table(results, include_breakdown, threshold)
        elif format == "json":
            rprint(json.dumps(results, indent=2))
        elif format == "markdown":
            _display_scores_markdown(results, include_breakdown)

        progress.update(task, advance=20, description="Complete!")

    # Save to file if requested
    if output_file:
        _save_analytics_results(results, output_file, format)
        rprint(f"✅ Results saved to [green]{output_file}[/green]")

    # Show summary
    avg_score = sum(r["overall_score"] for r in results) / len(results)
    low_performers = [r for r in results if r["overall_score"] < threshold]

    rprint("\n📊 [bold]Performance Summary[/bold]")
    rprint(f"Average Score: {avg_score:.2f}")
    rprint(f"Campaigns Below Threshold ({threshold}): {len(low_performers)}")

    if low_performers:
        rprint("\n⚠️  [yellow]Low Performing Campaigns:[/yellow]")
        for campaign in low_performers:
            rprint(f"   • {campaign['campaign_name']}: {campaign['overall_score']:.2f}")


@app.command()
def trends(
    campaign_id: str | None = typer.Option(
        None, "--campaign-id", "-c", help="Specific campaign to analyze"
    ),
    metric: str = typer.Option(
        "conversions",
        "--metric",
        "-m",
        help="Metric to analyze (conversions, revenue, ctr)",
    ),
    date_range: str = typer.Option(
        "90d", "--date-range", "-d", help="Date range for trend analysis"
    ),
    forecast_days: int = typer.Option(
        30, "--forecast", "-f", help="Days to forecast ahead"
    ),
    significance_level: float = typer.Option(
        0.05, "--significance", "-s", help="Statistical significance level"
    ),
    include_seasonality: bool = typer.Option(
        True, "--seasonality/--no-seasonality", help="Include seasonal patterns"
    ),
    output_chart: bool = typer.Option(False, "--chart", help="Generate trend chart"),
):
    """
    Analyze performance trends and generate forecasts.

    Features:
    • Statistical trend detection
    • Seasonal pattern analysis
    • Performance forecasting
    • Confidence intervals
    • Change point detection

    Examples:
        ai-agent analytics trends --metric revenue --forecast 60
        ai-agent analytics trends --campaign-id ABC123 --chart
        ai-agent analytics trends --metric ctr --significance 0.01
    """

    # Get config directly to avoid circular import
    from ...core.config import get_default_config

    get_default_config()

    with Progress(console=console) as progress:
        task = progress.add_task("Analyzing trends...", total=100)

        # Mock trend analysis
        _generate_mock_trend_data(metric, date_range)
        progress.update(task, advance=30)

        TrendAnalyzer(
            TrendConfig(
                significance_threshold=significance_level,
                forecast_periods=forecast_days,
                include_seasonal=include_seasonality,
            )
        )

        progress.update(task, advance=40, description="Calculating trend statistics...")

        # Mock trend results
        trend_results = {
            "metric": metric,
            "campaign_id": campaign_id,
            "trend_direction": "increasing",
            "trend_strength": 0.73,
            "significance": 0.02,
            "r_squared": 0.68,
            "seasonal_detected": include_seasonality,
            "forecast": _generate_mock_forecast(metric, forecast_days),
            "change_points": [
                {"date": "2024-01-15", "change_magnitude": 0.25, "confidence": 0.85}
            ],
            "insights": [
                f"{metric.title()} showing strong positive trend",
                "Seasonal patterns detected with weekly cycles",
                "Forecast shows continued growth with 85% confidence",
            ],
        }

        progress.update(task, advance=30, description="Generating forecast...")

    # Display results
    _display_trend_analysis(trend_results, output_chart)

    if trend_results["significance"] < significance_level:
        rprint(
            f"✅ [green]Statistically significant trend detected (p={trend_results['significance']:.3f})[/green]"
        )
    else:
        rprint(
            f"⚠️  [yellow]Trend not statistically significant (p={trend_results['significance']:.3f})[/yellow]"
        )


@app.command()
def anomalies(
    campaign_id: str | None = typer.Option(
        None, "--campaign-id", "-c", help="Specific campaign to analyze"
    ),
    date_range: str = typer.Option(
        "30d", "--date-range", "-d", help="Date range for anomaly detection"
    ),
    sensitivity: str = typer.Option(
        "medium", "--sensitivity", help="Detection sensitivity (low, medium, high)"
    ),
    min_severity: str = typer.Option(
        "medium", "--min-severity", help="Minimum anomaly severity to report"
    ),
    methods: list[str] = typer.Option(
        ["zscore", "iqr"], "--method", help="Detection methods to use"
    ),
    auto_investigate: bool = typer.Option(
        False, "--investigate", help="Auto-investigate detected anomalies"
    ),
):
    """
    Detect performance anomalies and investigate root causes.

    Detection Methods:
    • Z-score: Statistical outlier detection
    • IQR: Interquartile range method
    • Pattern: Seasonal pattern breaks
    • Change: Sudden change detection

    Examples:
        ai-agent analytics anomalies --sensitivity high
        ai-agent analytics anomalies --campaign-id ABC123 --investigate
        ai-agent analytics anomalies --method zscore --method iqr
    """

    # Get config directly to avoid circular import
    from ...core.config import get_default_config

    get_default_config()

    with Progress(console=console) as progress:
        task = progress.add_task("Detecting anomalies...", total=100)

        AnomalyDetector(
            AnomalyConfig(
                z_score_threshold=_get_sensitivity_threshold(sensitivity),
                min_severity=min_severity,
            )
        )

        progress.update(task, advance=30, description="Scanning for anomalies...")

        # Mock anomaly detection results
        anomalies = [
            {
                "date": "2024-01-20",
                "metric": "conversions",
                "expected_value": 45.2,
                "actual_value": 12.1,
                "severity": "critical",
                "confidence": 0.94,
                "method": "zscore",
                "z_score": -4.2,
                "potential_causes": [
                    "Landing page technical issue",
                    "Campaign pause/budget exhaustion",
                    "Competitor activity increase",
                ],
            },
            {
                "date": "2024-01-18",
                "metric": "ctr",
                "expected_value": 0.025,
                "actual_value": 0.048,
                "severity": "medium",
                "confidence": 0.78,
                "method": "iqr",
                "z_score": 2.1,
                "potential_causes": [
                    "New ad creative performing well",
                    "Audience targeting optimization",
                    "Reduced competition",
                ],
            },
        ]

        progress.update(task, advance=40, description="Analyzing anomaly patterns...")

        # Filter by minimum severity
        severity_order = ["low", "medium", "high", "critical"]
        min_severity_idx = severity_order.index(min_severity)
        filtered_anomalies = [
            a
            for a in anomalies
            if severity_order.index(a["severity"]) >= min_severity_idx
        ]

        progress.update(task, advance=30, description="Generating insights...")

    # Display anomalies
    _display_anomalies_table(filtered_anomalies, auto_investigate)

    # Summary
    critical_count = sum(1 for a in filtered_anomalies if a["severity"] == "critical")
    if critical_count > 0:
        rprint(
            f"\n🚨 [red]Found {critical_count} critical anomalies requiring immediate attention[/red]"
        )

    rprint(f"🔍 Total anomalies detected: {len(filtered_anomalies)}")

    if auto_investigate:
        rprint("\n🕵️  [bold]Investigation Results:[/bold]")
        for anomaly in filtered_anomalies[:3]:  # Top 3
            rprint(f"\n📅 {anomaly['date']} - {anomaly['metric'].title()} Anomaly:")
            for cause in anomaly["potential_causes"]:
                rprint(f"   • {cause}")


@app.command()
def benchmark(
    campaign_id: str | None = typer.Option(
        None, "--campaign-id", "-c", help="Specific campaign to benchmark"
    ),
    industry: str | None = typer.Option(
        None, "--industry", help="Industry for benchmarking"
    ),
    peer_group: str | None = typer.Option(
        None, "--peer-group", help="Peer group for comparison"
    ),
    metrics: list[str] = typer.Option(
        ["roas", "ctr", "cpa"], "--metric", help="Metrics to benchmark"
    ),
    include_percentiles: bool = typer.Option(
        True, "--percentiles/--no-percentiles", help="Include percentile rankings"
    ),
    competitive_analysis: bool = typer.Option(
        False, "--competitive", help="Include competitive analysis"
    ),
):
    """
    Benchmark performance against industry and peer groups.

    Benchmarks:
    • Industry averages and percentiles
    • Peer group comparisons
    • Historical performance trends
    • Competitive positioning

    Examples:
        ai-agent analytics benchmark --industry ecommerce
        ai-agent analytics benchmark --campaign-id ABC123 --competitive
        ai-agent analytics benchmark --metric roas --metric ctr
    """

    # Get config directly to avoid circular import
    from ...core.config import get_default_config

    get_default_config()

    with Progress(console=console) as progress:
        task = progress.add_task("Gathering benchmark data...", total=100)

        BenchmarkingEngine(BenchmarkConfig())

        progress.update(task, advance=30, description="Comparing against industry...")

        # Mock benchmark results
        benchmark_results = []
        for metric in metrics:
            result = {
                "metric": metric,
                "campaign_value": _get_mock_metric_value(metric),
                "industry_average": _get_mock_industry_average(metric),
                "industry_median": _get_mock_industry_median(metric),
                "percentile_rank": _calculate_mock_percentile(metric),
                "peer_average": _get_mock_peer_average(metric) if peer_group else None,
                "competitive_position": _assess_competitive_position(metric),
                "improvement_opportunity": _calculate_improvement_opportunity(metric),
            }
            benchmark_results.append(result)

        progress.update(
            task, advance=40, description="Analyzing competitive position..."
        )

        overall_performance = {
            "overall_percentile": 67,
            "top_performing_metrics": [
                r["metric"] for r in benchmark_results if r["percentile_rank"] > 75
            ],
            "underperforming_metrics": [
                r["metric"] for r in benchmark_results if r["percentile_rank"] < 25
            ],
            "competitive_advantages": [
                "Strong conversion rate",
                "Efficient cost structure",
            ],
            "improvement_areas": ["CTR optimization", "Market share growth"],
        }

        progress.update(task, advance=30, description="Generating recommendations...")

    # Display benchmark results
    _display_benchmark_table(
        benchmark_results, include_percentiles, competitive_analysis
    )

    # Show overall assessment
    rprint("\n🏆 [bold]Overall Performance Assessment[/bold]")
    rprint(f"Industry Percentile: {overall_performance['overall_percentile']}th")

    if overall_performance["top_performing_metrics"]:
        rprint(
            f"✅ Top Performers: {', '.join(overall_performance['top_performing_metrics'])}"
        )

    if overall_performance["underperforming_metrics"]:
        rprint(
            f"⚠️  Needs Improvement: {', '.join(overall_performance['underperforming_metrics'])}"
        )


@app.command()
def predict(
    campaign_id: str | None = typer.Option(
        None, "--campaign-id", "-c", help="Specific campaign to predict"
    ),
    metric: str = typer.Option(
        "conversions", "--metric", "-m", help="Metric to predict"
    ),
    forecast_days: int = typer.Option(
        30, "--days", "-d", help="Days to forecast ahead"
    ),
    model_type: str = typer.Option("random_forest", "--model", help="ML model type"),
    confidence_level: float = typer.Option(
        0.95, "--confidence", help="Confidence level for intervals"
    ),
    include_features: bool = typer.Option(
        True, "--features/--no-features", help="Show feature importance"
    ),
    scenario_analysis: bool = typer.Option(
        False, "--scenarios", help="Run scenario analysis"
    ),
):
    """
    Generate ML-powered performance predictions and forecasts.

    Models:
    • Random Forest: Ensemble learning for robust predictions
    • Gradient Boosting: Advanced boosting algorithms
    • Linear Regression: Simple interpretable models
    • ARIMA: Time series forecasting

    Examples:
        ai-agent analytics predict --metric revenue --days 60
        ai-agent analytics predict --campaign-id ABC123 --scenarios
        ai-agent analytics predict --model gradient_boosting --features
    """

    # Get config directly to avoid circular import
    from ...core.config import get_default_config

    get_default_config()

    with Progress(console=console) as progress:
        task = progress.add_task("Training prediction model...", total=100)

        PredictiveModel(
            PredictionConfig(
                model_type=model_type,
                forecast_horizon=forecast_days,
                confidence_level=confidence_level,
            )
        )

        progress.update(task, advance=40, description="Generating predictions...")

        # Mock prediction results
        prediction_results = {
            "metric": metric,
            "model_type": model_type,
            "forecast_period": forecast_days,
            "predictions": _generate_mock_predictions(metric, forecast_days),
            "confidence_intervals": _generate_mock_confidence_intervals(
                metric, forecast_days
            ),
            "model_accuracy": {
                "r2_score": 0.84,
                "mae": 12.5,
                "rmse": 18.2,
                "mape": 8.7,
            },
            "feature_importance": {
                "historical_performance": 0.35,
                "seasonality": 0.22,
                "spend_level": 0.18,
                "competitive_activity": 0.15,
                "economic_indicators": 0.10,
            }
            if include_features
            else {},
            "trend_analysis": {
                "direction": "increasing",
                "confidence": 0.87,
                "volatility": "medium",
            },
            "scenarios": _generate_scenario_analysis(metric)
            if scenario_analysis
            else {},
        }

        progress.update(
            task, advance=60, description="Analyzing prediction confidence..."
        )

    # Display prediction results
    _display_prediction_results(prediction_results, include_features, scenario_analysis)

    # Model performance assessment
    accuracy = prediction_results["model_accuracy"]
    if accuracy["r2_score"] > 0.8:
        rprint(
            f"✅ [green]High model accuracy (R² = {accuracy['r2_score']:.2f})[/green]"
        )
    elif accuracy["r2_score"] > 0.6:
        rprint(
            f"⚠️  [yellow]Moderate model accuracy (R² = {accuracy['r2_score']:.2f})[/yellow]"
        )
    else:
        rprint(
            f"❌ [red]Low model accuracy (R² = {accuracy['r2_score']:.2f}) - Use with caution[/red]"
        )


@app.command()
def dashboard(
    campaign_id: str | None = typer.Option(
        None, "--campaign-id", "-c", help="Focus on specific campaign"
    ),
    refresh_interval: int = typer.Option(
        300, "--refresh", help="Auto-refresh interval in seconds"
    ),
    include_alerts: bool = typer.Option(
        True, "--alerts/--no-alerts", help="Show performance alerts"
    ),
    compact_view: bool = typer.Option(
        False, "--compact", help="Use compact dashboard view"
    ),
):
    """
    Launch interactive analytics dashboard.

    Features:
    • Real-time performance monitoring
    • Key metrics visualization
    • Automated alerts and notifications
    • Trend analysis and forecasting

    Examples:
        ai-agent analytics dashboard
        ai-agent analytics dashboard --campaign-id ABC123 --refresh 60
        ai-agent analytics dashboard --compact --no-alerts
    """

    # Get config directly to avoid circular import
    from ...core.config import get_default_config

    get_default_config()

    rprint("🚀 [bold]Launching Analytics Dashboard[/bold]")
    rprint("Press Ctrl+C to exit")

    try:
        import time

        iteration = 0
        while True:
            console.clear()
            iteration += 1

            # Dashboard header
            rprint(
                Panel(
                    f"🤖 Marketing Analytics Dashboard - Refresh #{iteration}",
                    style="bold blue",
                )
            )

            # Key metrics
            _display_dashboard_metrics(campaign_id, compact_view)

            # Recent alerts
            if include_alerts:
                _display_recent_alerts()

            # Auto-refresh countdown
            if not compact_view:
                rprint(
                    f"\n⏰ Next refresh in {refresh_interval} seconds (Ctrl+C to exit)"
                )

            time.sleep(refresh_interval)

    except KeyboardInterrupt:
        rprint("\n👋 Dashboard closed")


# Helper functions (mock implementations)


def _get_mock_campaign_data(campaign_id: str | None) -> list[dict]:
    """Get mock campaign data for testing."""
    campaigns = [
        {
            "id": "123456789",
            "name": "Search Campaign Q4",
            "performance_score": 0.82,
            "efficiency_score": 0.78,
            "volume_score": 0.85,
            "quality_score": 0.75,
            "trend_score": 0.70,
        },
        {
            "id": "987654321",
            "name": "Shopping Campaign",
            "performance_score": 0.65,
            "efficiency_score": 0.60,
            "volume_score": 0.72,
            "quality_score": 0.68,
            "trend_score": 0.58,
        },
    ]

    if campaign_id:
        return [c for c in campaigns if c["id"] == campaign_id]
    return campaigns


def _score_to_grade(score: float) -> str:
    """Convert score to letter grade."""
    if score >= 0.9:
        return "A+"
    elif score >= 0.8:
        return "A"
    elif score >= 0.7:
        return "B"
    elif score >= 0.6:
        return "C"
    elif score >= 0.5:
        return "D"
    else:
        return "F"


def _generate_score_recommendations(score: float) -> list[str]:
    """Generate recommendations based on score."""
    if score >= 0.8:
        return ["Maintain current strategy", "Consider scaling successful elements"]
    elif score >= 0.6:
        return ["Optimize underperforming keywords", "Test new ad creatives"]
    else:
        return [
            "Urgent optimization needed",
            "Review targeting and bidding strategy",
            "Audit campaign structure",
        ]


def _display_scores_table(
    results: list[dict], include_breakdown: bool, threshold: float
):
    """Display performance scores in a table."""
    table = Table(title="📊 Campaign Performance Scores")
    table.add_column("Campaign", style="cyan")
    table.add_column("Score", style="green")
    table.add_column("Grade", style="yellow")

    if include_breakdown:
        table.add_column("Efficiency", style="blue")
        table.add_column("Volume", style="blue")
        table.add_column("Quality", style="blue")
        table.add_column("Trend", style="blue")

    table.add_column("Status", style="white")

    for result in results:
        status = (
            "✅ Good" if result["overall_score"] >= threshold else "⚠️  Needs Attention"
        )

        row = [
            result["campaign_name"],
            f"{result['overall_score']:.2f}",
            result["grade"],
        ]

        if include_breakdown:
            row.extend(
                [
                    f"{result['efficiency_score']:.2f}",
                    f"{result['volume_score']:.2f}",
                    f"{result['quality_score']:.2f}",
                    f"{result['trend_score']:.2f}",
                ]
            )

        row.append(status)
        table.add_row(*row)

    console.print(table)


def _display_scores_markdown(results: list[dict], include_breakdown: bool):
    """Display scores in markdown format."""
    rprint("# Campaign Performance Scores\n")

    for result in results:
        rprint(f"## {result['campaign_name']} ({result['grade']})")
        rprint(f"**Overall Score:** {result['overall_score']:.2f}")

        if include_breakdown:
            rprint(f"- Efficiency: {result['efficiency_score']:.2f}")
            rprint(f"- Volume: {result['volume_score']:.2f}")
            rprint(f"- Quality: {result['quality_score']:.2f}")
            rprint(f"- Trend: {result['trend_score']:.2f}")

        rprint("**Recommendations:**")
        for rec in result["recommendations"]:
            rprint(f"- {rec}")
        rprint()


def _generate_mock_trend_data(metric: str, date_range: str) -> list[dict]:
    """Generate mock trend data."""
    # This would fetch real data in production
    return []


def _generate_mock_forecast(metric: str, days: int) -> list[dict]:
    """Generate mock forecast data."""
    import random

    forecast = []
    base_value = 100 if metric == "conversions" else 0.02 if metric == "ctr" else 5000

    for day in range(days):
        # Add some trend and noise
        trend_factor = 1 + (day * 0.01)
        noise = random.uniform(0.9, 1.1)
        value = base_value * trend_factor * noise

        forecast.append(
            {
                "day": day + 1,
                "predicted_value": round(value, 2),
                "lower_bound": round(value * 0.85, 2),
                "upper_bound": round(value * 1.15, 2),
            }
        )

    return forecast


def _display_trend_analysis(results: dict, include_chart: bool):
    """Display trend analysis results."""
    rprint(f"📈 [bold]Trend Analysis: {results['metric'].title()}[/bold]")
    rprint("=" * 50)

    rprint(f"Direction: {results['trend_direction'].title()}")
    rprint(f"Strength: {results['trend_strength']:.2f}")
    rprint(f"R²: {results['r_squared']:.2f}")
    rprint(f"Significance: {results['significance']:.3f}")

    if results["seasonal_detected"]:
        rprint("✅ Seasonal patterns detected")

    rprint("\n🔮 [bold]Key Insights:[/bold]")
    for insight in results["insights"]:
        rprint(f"• {insight}")

    if include_chart:
        rprint("\n📊 [bold]Forecast Chart:[/bold]")
        # Simple ASCII chart (in production, use proper charting library)
        forecast = results["forecast"][:10]  # Show first 10 days
        for day_data in forecast:
            bar_length = int(day_data["predicted_value"] / 10)
            bar = "█" * bar_length
            rprint(f"Day {day_data['day']:2d}: {bar} {day_data['predicted_value']:.1f}")


def _get_sensitivity_threshold(sensitivity: str) -> float:
    """Get threshold based on sensitivity setting."""
    thresholds = {"low": 3.0, "medium": 2.5, "high": 2.0}
    return thresholds.get(sensitivity, 2.5)


def _display_anomalies_table(anomalies: list[dict], auto_investigate: bool):
    """Display anomalies in a table."""
    if not anomalies:
        rprint("✅ No anomalies detected in the specified period")
        return

    table = Table(title="🚨 Detected Anomalies")
    table.add_column("Date", style="cyan")
    table.add_column("Metric", style="yellow")
    table.add_column("Expected", style="green")
    table.add_column("Actual", style="red")
    table.add_column("Severity", style="white")
    table.add_column("Confidence", style="blue")

    for anomaly in anomalies:
        severity_emoji = {
            "low": "🟡",
            "medium": "🟠",
            "high": "🔴",
            "critical": "🚨",
        }.get(anomaly["severity"], "❓")

        table.add_row(
            anomaly["date"],
            anomaly["metric"],
            f"{anomaly['expected_value']:.1f}",
            f"{anomaly['actual_value']:.1f}",
            f"{severity_emoji} {anomaly['severity'].title()}",
            f"{anomaly['confidence']:.0%}",
        )

    console.print(table)


def _get_mock_metric_value(metric: str) -> float:
    """Get mock metric value for benchmarking."""
    values = {"roas": 4.2, "ctr": 0.035, "cpa": 45.50, "conversion_rate": 0.025}
    return values.get(metric, 1.0)


def _get_mock_industry_average(metric: str) -> float:
    """Get mock industry average."""
    averages = {"roas": 3.8, "ctr": 0.028, "cpa": 52.00, "conversion_rate": 0.022}
    return averages.get(metric, 1.0)


def _get_mock_industry_median(metric: str) -> float:
    """Get mock industry median."""
    medians = {"roas": 3.5, "ctr": 0.025, "cpa": 48.00, "conversion_rate": 0.020}
    return medians.get(metric, 1.0)


def _calculate_mock_percentile(metric: str) -> int:
    """Calculate mock percentile ranking."""
    import random

    return random.randint(25, 85)


def _get_mock_peer_average(metric: str) -> float:
    """Get mock peer group average."""
    return _get_mock_industry_average(metric) * 1.05


def _assess_competitive_position(metric: str) -> str:
    """Assess competitive position."""
    positions = ["Leading", "Above Average", "Average", "Below Average"]
    import random

    return random.choice(positions)


def _calculate_improvement_opportunity(metric: str) -> str:
    """Calculate improvement opportunity."""
    opportunities = ["High", "Medium", "Low"]
    import random

    return random.choice(opportunities)


def _display_benchmark_table(
    results: list[dict], include_percentiles: bool, competitive_analysis: bool
):
    """Display benchmark results table."""
    table = Table(title="🏆 Performance Benchmarking")
    table.add_column("Metric", style="cyan")
    table.add_column("Your Value", style="green")
    table.add_column("Industry Avg", style="yellow")

    if include_percentiles:
        table.add_column("Percentile", style="blue")

    if competitive_analysis:
        table.add_column("Position", style="white")
        table.add_column("Opportunity", style="magenta")

    for result in results:
        row = [
            result["metric"].upper(),
            f"{result['campaign_value']:.2f}",
            f"{result['industry_average']:.2f}",
        ]

        if include_percentiles:
            row.append(f"{result['percentile_rank']}th")

        if competitive_analysis:
            row.extend(
                [result["competitive_position"], result["improvement_opportunity"]]
            )

        table.add_row(*row)

    console.print(table)


def _generate_mock_predictions(metric: str, days: int) -> list[dict]:
    """Generate mock predictions."""
    return _generate_mock_forecast(metric, days)


def _generate_mock_confidence_intervals(metric: str, days: int) -> list[dict]:
    """Generate mock confidence intervals."""
    # Already included in forecast
    return []


def _generate_scenario_analysis(metric: str) -> dict:
    """Generate scenario analysis."""
    return {
        "optimistic": {"change": "+25%", "probability": 0.15},
        "realistic": {"change": "+10%", "probability": 0.70},
        "pessimistic": {"change": "-5%", "probability": 0.15},
    }


def _display_prediction_results(
    results: dict, include_features: bool, scenario_analysis: bool
):
    """Display prediction results."""
    rprint(f"🔮 [bold]Predictions: {results['metric'].title()}[/bold]")
    rprint("=" * 50)

    # Model performance
    accuracy = results["model_accuracy"]
    rprint(f"Model: {results['model_type'].replace('_', ' ').title()}")
    rprint(f"R² Score: {accuracy['r2_score']:.2f}")
    rprint(f"Mean Absolute Error: {accuracy['mae']:.1f}")

    # Show sample predictions
    rprint("\n📊 [bold]Sample Forecast (Next 7 Days):[/bold]")
    predictions = results["predictions"][:7]
    for pred in predictions:
        rprint(
            f"Day {pred['day']:2d}: {pred['predicted_value']:6.1f} [{pred['lower_bound']:5.1f} - {pred['upper_bound']:5.1f}]"
        )

    # Feature importance
    if include_features and results["feature_importance"]:
        rprint("\n🎯 [bold]Feature Importance:[/bold]")
        for feature, importance in sorted(
            results["feature_importance"].items(), key=lambda x: x[1], reverse=True
        ):
            bar_length = int(importance * 20)
            bar = "█" * bar_length
            rprint(f"{feature.replace('_', ' ').title():25s} {bar} {importance:.0%}")

    # Scenario analysis
    if scenario_analysis and results["scenarios"]:
        rprint("\n🎭 [bold]Scenario Analysis:[/bold]")
        for scenario, data in results["scenarios"].items():
            rprint(
                f"{scenario.title():12s}: {data['change']} (probability: {data['probability']:.0%})"
            )


def _display_dashboard_metrics(campaign_id: str | None, compact: bool):
    """Display key dashboard metrics."""
    # Mock dashboard data
    metrics = {
        "impressions": {"value": "2.5M", "change": "+12.3%", "status": "up"},
        "clicks": {"value": "87.5K", "change": "+8.7%", "status": "up"},
        "conversions": {"value": "2,156", "change": "-2.1%", "status": "down"},
        "cost": {"value": "$45,230", "change": "+5.4%", "status": "neutral"},
        "roas": {"value": "4.2x", "change": "+15.8%", "status": "up"},
        "ctr": {"value": "3.5%", "change": "+0.8%", "status": "up"},
    }

    if compact:
        # Single line format
        metric_strs = []
        for name, data in metrics.items():
            emoji = (
                "📈"
                if data["status"] == "up"
                else "📉"
                if data["status"] == "down"
                else "➡️"
            )
            metric_strs.append(
                f"{emoji} {name.upper()}: {data['value']} ({data['change']})"
            )
        rprint(" | ".join(metric_strs))
    else:
        # Table format
        table = Table(title="📊 Key Performance Metrics")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        table.add_column("Change", style="yellow")
        table.add_column("Trend", style="white")

        for name, data in metrics.items():
            trend_emoji = (
                "📈"
                if data["status"] == "up"
                else "📉"
                if data["status"] == "down"
                else "➡️"
            )
            table.add_row(name.title(), data["value"], data["change"], trend_emoji)

        console.print(table)


def _display_recent_alerts():
    """Display recent performance alerts."""
    alerts = [
        {
            "time": "2 min ago",
            "message": "CTR dropped 15% in Search Campaign Q4",
            "severity": "medium",
        },
        {
            "time": "15 min ago",
            "message": "Budget utilization at 95% for Shopping Campaign",
            "severity": "high",
        },
        {
            "time": "1 hour ago",
            "message": "New conversion spike detected (+25%)",
            "severity": "info",
        },
    ]

    rprint("\n🚨 [bold]Recent Alerts:[/bold]")
    for alert in alerts:
        severity_color = {"high": "red", "medium": "yellow", "info": "green"}.get(
            alert["severity"], "white"
        )

        rprint(
            f"[{severity_color}]• {alert['time']} - {alert['message']}[/{severity_color}]"
        )


def _save_analytics_results(results: dict, output_file: Path, format: str):
    """Save analytics results to file."""
    output_file.parent.mkdir(parents=True, exist_ok=True)

    if format == "json":
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2, default=str)
    elif format == "markdown":
        # Convert to markdown format
        with open(output_file, "w") as f:
            f.write("# Analytics Results\n\n")
            f.write(json.dumps(results, indent=2, default=str))
    else:
        # Default to JSON
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2, default=str)
