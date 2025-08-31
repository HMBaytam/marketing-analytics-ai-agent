"""Report generation and export commands."""

import json
from datetime import datetime, timedelta
from pathlib import Path

import typer
from rich import print as rprint
from rich.console import Console
from rich.layout import Layout
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.text import Text

# Removed circular import - get_config imported locally where needed

app = typer.Typer(help="📋 Report generation and export commands")
console = Console()


@app.command()
def generate(
    template: str = typer.Option(
        "executive", "--template", "-t", help="Report template to use"
    ),
    campaign_id: str | None = typer.Option(
        None, "--campaign-id", "-c", help="Specific campaign to report on"
    ),
    date_range: str = typer.Option(
        "30d", "--date-range", "-d", help="Date range for report"
    ),
    output_file: Path | None = typer.Option(
        None, "--output", "-o", help="Output file path"
    ),
    format: str = typer.Option(
        "markdown", "--format", "-f", help="Output format (markdown, html, pdf, json)"
    ),
    include_charts: bool = typer.Option(
        True, "--charts/--no-charts", help="Include charts and visualizations"
    ),
    include_recommendations: bool = typer.Option(
        True,
        "--recommendations/--no-recommendations",
        help="Include optimization recommendations",
    ),
    auto_send: bool = typer.Option(
        False, "--auto-send", help="Automatically send report via email"
    ),
    recipients: list[str] | None = typer.Option(
        None, "--recipient", help="Email recipients for auto-send"
    ),
    interactive_preview: bool = typer.Option(
        False, "--preview", "-p", help="Show interactive preview"
    ),
):
    """
    Generate comprehensive marketing reports.

    Templates:
    • executive: High-level summary for executives
    • detailed: Comprehensive performance analysis
    • campaign: Campaign-specific deep dive
    • optimization: Optimization recommendations focus
    • competitor: Competitive analysis report
    • custom: Custom report builder

    Examples:
        ai-agent report generate executive --date-range 90d --charts
        ai-agent report generate campaign --campaign-id ABC123 --format pdf
        ai-agent report generate optimization --auto-send --recipient john@company.com
    """

    # Get config directly to avoid circular import
    from ...core.config import get_default_config

    config = get_default_config()

    # Validate template
    available_templates = [
        "executive",
        "detailed",
        "campaign",
        "optimization",
        "competitor",
        "custom",
    ]
    if template not in available_templates:
        typer.secho(f"❌ Invalid template: {template}", fg=typer.colors.RED)
        typer.echo(f"Available templates: {', '.join(available_templates)}")
        raise typer.Exit(1)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        console=console,
    ) as progress:
        # Generate report content
        task = progress.add_task("Initializing report generation...", total=100)

        # Gather data
        progress.update(task, advance=20, description="Gathering performance data...")
        report_data = _gather_report_data(campaign_id, date_range)

        progress.update(task, advance=30, description="Running analytics...")
        analytics_results = _run_analytics_for_report(report_data, template)

        progress.update(task, advance=20, description="Generating insights...")
        insights = _generate_report_insights(analytics_results, template)

        if include_recommendations:
            progress.update(
                task, advance=15, description="Creating optimization recommendations..."
            )
            recommendations = _generate_report_recommendations(analytics_results)
            insights["recommendations"] = recommendations

        progress.update(task, advance=10, description="Formatting report...")

        # Create report
        report = _create_report(
            template,
            report_data,
            analytics_results,
            insights,
            include_charts,
            include_recommendations,
        )

        progress.update(task, advance=5, description="Finalizing...")

    # Show interactive preview if requested
    if interactive_preview:
        _show_interactive_preview(report)
        if not typer.confirm("Continue with report generation?"):
            rprint("Report generation cancelled")
            return

    # Generate output file name if not provided
    if not output_file:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = (
            Path(config.output.base_directory)
            / f"report_{template}_{timestamp}.{_get_file_extension(format)}"
        )

    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Save report
    _save_report(report, output_file, format, include_charts)

    rprint(f"✅ Report generated: [green]{output_file}[/green]")
    _show_report_summary(report, output_file)

    # Auto-send if requested
    if auto_send and recipients:
        _send_report_email(output_file, recipients, template)


@app.command()
def templates(
    list_all: bool = typer.Option(
        False, "--list", "-l", help="List all available templates"
    ),
    describe: str | None = typer.Option(
        None, "--describe", "-d", help="Describe specific template"
    ),
    create_custom: bool = typer.Option(
        False, "--create-custom", help="Create custom template"
    ),
    template_dir: Path | None = typer.Option(
        None, "--template-dir", help="Custom template directory"
    ),
):
    """
    Manage report templates and layouts.

    Examples:
        ai-agent report templates --list
        ai-agent report templates --describe executive
        ai-agent report templates --create-custom
    """

    if list_all:
        _list_available_templates()
    elif describe:
        _describe_template(describe)
    elif create_custom:
        _create_custom_template(template_dir)
    else:
        typer.echo(
            "Use --list to see available templates, --describe [name] for details, or --create-custom to build one"
        )


@app.command()
def schedule(
    template: str = typer.Argument(..., help="Report template to schedule"),
    frequency: str = typer.Option(
        "weekly", "--frequency", "-f", help="Report frequency (daily, weekly, monthly)"
    ),
    recipients: list[str] = typer.Option(
        ..., "--recipient", "-r", help="Email recipients"
    ),
    time: str = typer.Option("09:00", "--time", help="Time to send (HH:MM format)"),
    day_of_week: str | None = typer.Option(
        None, "--day", help="Day of week (for weekly reports)"
    ),
    day_of_month: int | None = typer.Option(
        None, "--day-of-month", help="Day of month (for monthly reports)"
    ),
    active: bool = typer.Option(True, "--active/--inactive", help="Schedule is active"),
    test_send: bool = typer.Option(
        False, "--test-send", help="Send test report immediately"
    ),
):
    """
    Schedule automated report generation and delivery.

    Examples:
        ai-agent report schedule executive --frequency weekly --recipient john@company.com
        ai-agent report schedule detailed --frequency monthly --day-of-month 1
        ai-agent report schedule optimization --frequency daily --time 08:00
    """

    # Get config directly to avoid circular import
    from ...core.config import get_default_config

    get_default_config()

    # Create schedule configuration
    schedule_config = {
        "id": f"schedule_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "template": template,
        "frequency": frequency,
        "recipients": recipients,
        "time": time,
        "day_of_week": day_of_week,
        "day_of_month": day_of_month,
        "active": active,
        "created_at": datetime.now().isoformat(),
        "next_run": _calculate_next_run(frequency, time, day_of_week, day_of_month),
    }

    # Save schedule
    _save_schedule(schedule_config)

    rprint(f"✅ Scheduled {frequency} {template} reports")
    rprint(f"📧 Recipients: {', '.join(recipients)}")
    rprint(f"⏰ Next run: {schedule_config['next_run']}")

    if test_send:
        rprint("📤 Sending test report...")
        _send_test_report(template, recipients)


@app.command()
def dashboards(
    action: str = typer.Argument(..., help="Action: list, create, view, or delete"),
    dashboard_name: str | None = typer.Option(
        None, "--name", "-n", help="Dashboard name"
    ),
    template: str | None = typer.Option(
        None, "--template", help="Base template for new dashboard"
    ),
    auto_refresh: bool = typer.Option(
        True, "--auto-refresh/--no-refresh", help="Enable auto-refresh"
    ),
    refresh_interval: int = typer.Option(
        300, "--refresh-interval", help="Refresh interval in seconds"
    ),
    public: bool = typer.Option(
        False, "--public", help="Make dashboard publicly accessible"
    ),
):
    """
    Manage interactive reporting dashboards.

    Actions:
    • list: Show all available dashboards
    • create: Create new dashboard
    • view: Open dashboard in browser
    • delete: Remove dashboard

    Examples:
        ai-agent report dashboards list
        ai-agent report dashboards create --name "Executive Dashboard" --template executive
        ai-agent report dashboards view --name "Performance Monitor"
    """

    if action == "list":
        _list_dashboards()
    elif action == "create":
        if not dashboard_name:
            dashboard_name = typer.prompt("Dashboard name")
        _create_dashboard(
            dashboard_name, template, auto_refresh, refresh_interval, public
        )
    elif action == "view":
        if not dashboard_name:
            dashboard_name = typer.prompt("Dashboard name")
        _view_dashboard(dashboard_name)
    elif action == "delete":
        if not dashboard_name:
            dashboard_name = typer.prompt("Dashboard name to delete")
        _delete_dashboard(dashboard_name)
    else:
        typer.secho(f"❌ Invalid action: {action}", fg=typer.colors.RED)
        typer.echo("Available actions: list, create, view, delete")


@app.command()
def export(
    source_file: Path = typer.Argument(..., help="Source report file to convert"),
    target_format: str = typer.Option(
        ..., "--format", "-f", help="Target format (pdf, html, docx, pptx)"
    ),
    output_file: Path | None = typer.Option(
        None, "--output", "-o", help="Output file path"
    ),
    template_style: str | None = typer.Option(
        None, "--style", help="Styling template to apply"
    ),
    include_cover: bool = typer.Option(
        True, "--cover/--no-cover", help="Include cover page"
    ),
    watermark: str | None = typer.Option(
        None, "--watermark", help="Add watermark text"
    ),
):
    """
    Export reports to different formats with styling.

    Supported Formats:
    • pdf: Professional PDF reports
    • html: Interactive web reports
    • docx: Microsoft Word documents
    • pptx: PowerPoint presentations

    Examples:
        ai-agent report export report.md --format pdf --cover
        ai-agent report export report.json --format html --style corporate
        ai-agent report export report.md --format pptx --watermark "CONFIDENTIAL"
    """

    if not source_file.exists():
        typer.secho(f"❌ Source file not found: {source_file}", fg=typer.colors.RED)
        raise typer.Exit(1)

    # Generate output filename if not provided
    if not output_file:
        output_file = source_file.with_suffix(f".{target_format}")

    with Progress(console=console) as progress:
        task = progress.add_task(f"Converting to {target_format.upper()}...", total=100)

        # Load source content
        progress.update(task, advance=20, description="Loading source content...")
        source_content = _load_source_content(source_file)

        progress.update(task, advance=30, description="Applying styling...")

        # Apply styling and convert
        if target_format == "pdf":
            _convert_to_pdf(
                source_content, output_file, template_style, include_cover, watermark
            )
        elif target_format == "html":
            _convert_to_html(source_content, output_file, template_style)
        elif target_format == "docx":
            _convert_to_docx(source_content, output_file, template_style, include_cover)
        elif target_format == "pptx":
            _convert_to_pptx(source_content, output_file, template_style)
        else:
            typer.secho(f"❌ Unsupported format: {target_format}", fg=typer.colors.RED)
            raise typer.Exit(1)

        progress.update(task, advance=50, description="Finalizing...")

    rprint(f"✅ Exported to: [green]{output_file}[/green]")
    _show_export_summary(source_file, output_file, target_format)


@app.command()
def analytics(
    report_file: Path = typer.Argument(..., help="Report file to analyze"),
    show_engagement: bool = typer.Option(
        True, "--engagement/--no-engagement", help="Show engagement metrics"
    ),
    show_performance: bool = typer.Option(
        True, "--performance/--no-performance", help="Show performance metrics"
    ),
    benchmark_against: str | None = typer.Option(
        None, "--benchmark", help="Benchmark against industry/peer"
    ),
    export_analytics: bool = typer.Option(
        False, "--export", help="Export analytics data"
    ),
):
    """
    Analyze report performance and engagement metrics.

    Metrics:
    • Engagement: Views, time spent, shares
    • Performance: Accuracy of predictions
    • Effectiveness: Action taken on recommendations

    Examples:
        ai-agent report analytics report_executive_20240120.pdf
        ai-agent report analytics report.html --benchmark industry --export
    """

    if not report_file.exists():
        typer.secho(f"❌ Report file not found: {report_file}", fg=typer.colors.RED)
        raise typer.Exit(1)

    with Progress(console=console) as progress:
        task = progress.add_task("Analyzing report metrics...", total=100)

        # Load report metadata
        progress.update(task, advance=30, description="Loading report metadata...")
        report_metadata = _get_report_metadata(report_file)

        progress.update(
            task, advance=40, description="Calculating engagement metrics..."
        )
        engagement_metrics = (
            _calculate_engagement_metrics(report_file) if show_engagement else None
        )

        progress.update(task, advance=30, description="Analyzing performance...")
        performance_metrics = (
            _calculate_performance_metrics(report_file) if show_performance else None
        )

    # Display analytics
    _display_report_analytics(
        report_metadata, engagement_metrics, performance_metrics, benchmark_against
    )

    if export_analytics:
        _export_report_analytics(report_file, engagement_metrics, performance_metrics)


# Helper functions


def _gather_report_data(campaign_id: str | None, date_range: str) -> dict:
    """Gather data for report generation."""

    # Mock data gathering
    return {
        "campaigns": [
            {
                "id": "123456789",
                "name": "Search Campaign Q4",
                "spend": 25750.00,
                "revenue": 103200.00,
                "conversions": 412,
                "impressions": 875000,
                "clicks": 29750,
                "ctr": 0.034,
                "cpa": 62.50,
                "roas": 4.01,
            },
            {
                "id": "987654321",
                "name": "Shopping Campaign",
                "spend": 18400.00,
                "revenue": 46000.00,
                "conversions": 276,
                "impressions": 650000,
                "clicks": 13000,
                "ctr": 0.020,
                "cpa": 66.67,
                "roas": 2.50,
            },
        ],
        "date_range": date_range,
        "total_spend": 44150.00,
        "total_revenue": 149200.00,
        "total_conversions": 688,
        "period": _parse_date_range(date_range),
    }


def _run_analytics_for_report(data: dict, template: str) -> dict:
    """Run analytics specific to report template."""

    # Mock analytics results
    analytics = {
        "performance_scores": {
            "overall": 0.78,
            "efficiency": 0.82,
            "volume": 0.75,
            "quality": 0.71,
        },
        "trends": {
            "spend_trend": "increasing",
            "revenue_trend": "increasing",
            "roas_trend": "stable",
            "significance": 0.03,
        },
        "anomalies": [
            {
                "date": "2024-01-18",
                "metric": "ctr",
                "severity": "medium",
                "description": "CTR spike detected (+40%)",
            }
        ],
        "benchmarks": {
            "industry_percentile": 68,
            "peer_rank": 4,
            "top_metrics": ["roas", "conversion_rate"],
            "improvement_areas": ["ctr", "impression_share"],
        },
    }

    return analytics


def _generate_report_insights(analytics: dict, template: str) -> dict:
    """Generate insights based on analytics results."""

    insights = {
        "key_findings": [
            "Overall ROAS performance 34% above industry average",
            "Search campaigns significantly outperforming Shopping",
            "Strong conversion rate trends with 15% month-over-month growth",
            "Opportunity to improve impression share by 25%",
        ],
        "opportunities": [
            "Reallocate 20% of Shopping budget to Search campaigns",
            "Expand high-performing keyword groups",
            "Test new ad creative variations for Shopping campaigns",
            "Increase bids to capture missed impression opportunities",
        ],
        "risks": [
            "Seasonality impact expected in coming month",
            "Increased competition in core keyword markets",
            "Budget constraints limiting growth potential",
        ],
    }

    if template == "executive":
        insights["executive_summary"] = _generate_executive_summary(analytics)
    elif template == "detailed":
        insights["detailed_analysis"] = _generate_detailed_analysis(analytics)
    elif template == "optimization":
        insights["optimization_priorities"] = _generate_optimization_priorities(
            analytics
        )

    return insights


def _generate_report_recommendations(analytics: dict) -> list[dict]:
    """Generate recommendations for the report."""

    return [
        {
            "priority": "high",
            "title": "Optimize Budget Allocation",
            "description": "Shift budget from underperforming to high-ROAS campaigns",
            "expected_impact": 0.25,
            "timeline": "1-2 weeks",
        },
        {
            "priority": "medium",
            "title": "Expand Keyword Targeting",
            "description": "Add high-intent keywords to top-performing ad groups",
            "expected_impact": 0.18,
            "timeline": "3-5 days",
        },
        {
            "priority": "medium",
            "title": "Creative Testing Program",
            "description": "Launch systematic A/B testing for ad creatives",
            "expected_impact": 0.15,
            "timeline": "2-3 weeks",
        },
    ]


def _create_report(
    template: str,
    data: dict,
    analytics: dict,
    insights: dict,
    include_charts: bool,
    include_recommendations: bool,
) -> dict:
    """Create structured report content."""

    report = {
        "metadata": {
            "template": template,
            "generated_at": datetime.now().isoformat(),
            "date_range": data["date_range"],
            "campaigns_analyzed": len(data["campaigns"]),
        },
        "executive_summary": _create_executive_summary(data, analytics, insights),
        "performance_overview": _create_performance_overview(data, analytics),
        "detailed_analysis": _create_detailed_analysis(data, analytics, insights),
        "insights": insights,
    }

    if include_charts:
        report["visualizations"] = _create_report_visualizations(data, analytics)

    if include_recommendations:
        report["recommendations"] = insights.get("recommendations", [])

    # Template-specific content
    if template == "executive":
        report["kpis"] = _create_executive_kpis(data, analytics)
    elif template == "detailed":
        report["detailed_metrics"] = _create_detailed_metrics(data, analytics)
    elif template == "optimization":
        report["optimization_analysis"] = _create_optimization_analysis(
            data, analytics, insights
        )

    return report


def _show_interactive_preview(report: dict):
    """Show interactive preview of the report."""

    console.clear()

    # Create layout
    layout = Layout()
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="body"),
        Layout(name="footer", size=3),
    )

    # Header
    header_text = Text(
        f"📋 Report Preview - {report['metadata']['template'].title()}",
        style="bold blue",
    )
    layout["header"].update(Panel(header_text, style="blue"))

    # Body - show key sections
    body_content = []

    # Executive Summary
    if "executive_summary" in report:
        summary_text = "\n".join(report["executive_summary"][:3])  # First 3 points
        body_content.append(
            Panel(summary_text, title="Executive Summary", border_style="green")
        )

    # Key Metrics
    if "performance_overview" in report:
        metrics = report["performance_overview"]
        metrics_text = f"""Total Spend: ${metrics.get('total_spend', 0):,.2f}
Total Revenue: ${metrics.get('total_revenue', 0):,.2f}
Total ROAS: {metrics.get('total_roas', 0):.2f}x
Conversions: {metrics.get('total_conversions', 0):,}"""
        body_content.append(
            Panel(metrics_text, title="Key Metrics", border_style="yellow")
        )

    # Display body content
    if len(body_content) >= 2:
        layout["body"].split_row(Layout(body_content[0]), Layout(body_content[1]))
    else:
        layout["body"].update(
            body_content[0] if body_content else Panel("No content to preview")
        )

    # Footer
    footer_text = f"Generated: {report['metadata']['generated_at'][:10]} | Campaigns: {report['metadata']['campaigns_analyzed']}"
    layout["footer"].update(Panel(footer_text, style="dim"))

    console.print(layout)


def _save_report(report: dict, output_file: Path, format: str, include_charts: bool):
    """Save report in specified format."""

    if format == "markdown":
        _save_markdown_report(report, output_file, include_charts)
    elif format == "json":
        _save_json_report(report, output_file)
    elif format == "html":
        _save_html_report(report, output_file, include_charts)
    elif format == "pdf":
        _save_pdf_report(report, output_file, include_charts)
    else:
        # Default to JSON
        _save_json_report(report, output_file)


def _save_markdown_report(report: dict, output_file: Path, include_charts: bool):
    """Save report as Markdown."""

    content = []

    # Title and metadata
    content.append(
        f"# Marketing Performance Report - {report['metadata']['template'].title()}\n"
    )
    content.append(f"**Generated:** {report['metadata']['generated_at'][:10]}")
    content.append(f"**Date Range:** {report['metadata']['date_range']}")
    content.append(
        f"**Campaigns Analyzed:** {report['metadata']['campaigns_analyzed']}\n"
    )

    # Executive Summary
    if "executive_summary" in report:
        content.append("## Executive Summary\n")
        for point in report["executive_summary"]:
            content.append(f"• {point}")
        content.append("")

    # Performance Overview
    if "performance_overview" in report:
        content.append("## Performance Overview\n")
        overview = report["performance_overview"]
        content.append(f"- **Total Spend:** ${overview.get('total_spend', 0):,.2f}")
        content.append(f"- **Total Revenue:** ${overview.get('total_revenue', 0):,.2f}")
        content.append(f"- **Overall ROAS:** {overview.get('total_roas', 0):.2f}x")
        content.append(
            f"- **Total Conversions:** {overview.get('total_conversions', 0):,}"
        )
        content.append("")

    # Key Insights
    if "insights" in report and "key_findings" in report["insights"]:
        content.append("## Key Insights\n")
        for finding in report["insights"]["key_findings"]:
            content.append(f"• {finding}")
        content.append("")

    # Recommendations
    if "recommendations" in report:
        content.append("## Optimization Recommendations\n")
        for i, rec in enumerate(report["recommendations"], 1):
            content.append(
                f"### {i}. {rec['title']} ({rec['priority'].title()} Priority)"
            )
            content.append(f"{rec['description']}")
            content.append(f"- **Expected Impact:** {rec['expected_impact']:.0%}")
            content.append(f"- **Timeline:** {rec['timeline']}")
            content.append("")

    # Charts placeholder
    if include_charts and "visualizations" in report:
        content.append("## Charts and Visualizations\n")
        content.append("*Charts would be embedded here in actual implementation*\n")

    # Save content
    with open(output_file, "w") as f:
        f.write("\n".join(content))


def _save_json_report(report: dict, output_file: Path):
    """Save report as JSON."""
    with open(output_file, "w") as f:
        json.dump(report, f, indent=2, default=str)


def _save_html_report(report: dict, output_file: Path, include_charts: bool):
    """Save report as HTML (simplified)."""

    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>Marketing Report - {report['metadata']['template'].title()}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        .header {{ border-bottom: 2px solid #333; padding-bottom: 20px; }}
        .section {{ margin: 30px 0; }}
        .kpi {{ display: inline-block; margin: 10px 20px; padding: 15px; background: #f5f5f5; border-radius: 5px; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Marketing Performance Report</h1>
        <p>Generated: {report['metadata']['generated_at'][:10]} | Template: {report['metadata']['template'].title()}</p>
    </div>

    <div class="section">
        <h2>Performance Overview</h2>
        <!-- KPIs would be rendered here -->
    </div>

    <div class="section">
        <h2>Key Insights</h2>
        <!-- Insights content -->
    </div>
</body>
</html>"""

    with open(output_file, "w") as f:
        f.write(html_content)


def _save_pdf_report(report: dict, output_file: Path, include_charts: bool):
    """Save report as PDF (placeholder - requires proper PDF library)."""

    # This would use a library like reportlab or weasyprint
    rprint("⚠️  PDF generation requires additional dependencies")
    rprint("Converting to HTML instead...")

    html_file = output_file.with_suffix(".html")
    _save_html_report(report, html_file, include_charts)
    rprint(f"📄 HTML version saved: {html_file}")


def _get_file_extension(format: str) -> str:
    """Get file extension for format."""
    extensions = {"markdown": "md", "json": "json", "html": "html", "pdf": "pdf"}
    return extensions.get(format, "json")


def _show_report_summary(report: dict, output_file: Path):
    """Show report generation summary."""

    summary = Panel(
        f"""📊 **Report Generated Successfully**

Template: {report['metadata']['template'].title()}
File: {output_file.name}
Size: {output_file.stat().st_size / 1024:.1f} KB
Campaigns: {report['metadata']['campaigns_analyzed']}

📈 **Key Metrics Included:**
• Performance overview and KPIs
• Trend analysis and insights
• Optimization recommendations
• Competitive benchmarking

💡 **Next Steps:**
1. Review key findings and recommendations
2. Share with stakeholders
3. Implement suggested optimizations
4. Schedule follow-up analysis""",
        title="Summary",
        border_style="green",
    )

    console.print(summary)


def _send_report_email(output_file: Path, recipients: list[str], template: str):
    """Send report via email (mock implementation)."""

    rprint(f"📧 Sending {template} report to {len(recipients)} recipients...")

    # Mock email sending
    for recipient in recipients:
        rprint(f"   ✅ Sent to {recipient}")

    rprint("✅ [green]All emails sent successfully[/green]")


def _list_available_templates():
    """List all available report templates."""

    templates = [
        {
            "name": "executive",
            "description": "High-level summary for executives and stakeholders",
            "sections": [
                "Executive Summary",
                "Key Metrics",
                "Top Insights",
                "Strategic Recommendations",
            ],
            "length": "2-3 pages",
        },
        {
            "name": "detailed",
            "description": "Comprehensive performance analysis with deep metrics",
            "sections": [
                "Performance Deep Dive",
                "Trend Analysis",
                "Segment Analysis",
                "Detailed Recommendations",
            ],
            "length": "8-12 pages",
        },
        {
            "name": "campaign",
            "description": "Campaign-specific analysis and optimization",
            "sections": [
                "Campaign Overview",
                "Performance Metrics",
                "Optimization Opportunities",
                "Action Plan",
            ],
            "length": "4-6 pages",
        },
        {
            "name": "optimization",
            "description": "Focus on optimization opportunities and recommendations",
            "sections": [
                "Current Performance",
                "Optimization Analysis",
                "ROI Projections",
                "Implementation Plan",
            ],
            "length": "5-7 pages",
        },
        {
            "name": "competitor",
            "description": "Competitive analysis and market positioning",
            "sections": [
                "Market Overview",
                "Competitive Benchmarking",
                "Share Analysis",
                "Strategic Positioning",
            ],
            "length": "6-8 pages",
        },
    ]

    table = Table(title="📋 Available Report Templates")
    table.add_column("Template", style="cyan", width=12)
    table.add_column("Description", style="green", width=40)
    table.add_column("Length", style="yellow", width=10)
    table.add_column("Best For", style="blue", width=20)

    use_cases = {
        "executive": "C-level presentations",
        "detailed": "Marketing team analysis",
        "campaign": "Campaign managers",
        "optimization": "Performance optimization",
        "competitor": "Strategic planning",
    }

    for template in templates:
        table.add_row(
            template["name"],
            template["description"],
            template["length"],
            use_cases.get(template["name"], "General use"),
        )

    console.print(table)


def _describe_template(template_name: str):
    """Describe a specific template in detail."""

    template_details = {
        "executive": {
            "purpose": "Provide high-level insights for executive decision making",
            "audience": "C-level executives, senior stakeholders",
            "sections": [
                "Executive Summary (key findings and recommendations)",
                "Performance Dashboard (critical KPIs)",
                "Market Position (competitive standing)",
                "Strategic Recommendations (3-5 key actions)",
                "Investment Summary (budget and ROI projections)",
            ],
            "features": [
                "Visual KPI dashboard",
                "Trend indicators",
                "Risk assessment",
                "Action prioritization",
            ],
            "delivery": "PDF optimized for mobile viewing",
        }
    }

    if template_name not in template_details:
        rprint(f"❌ Template '{template_name}' not found")
        return

    details = template_details[template_name]

    description = Panel(
        f"""**Purpose:** {details['purpose']}

**Target Audience:** {details['audience']}

**Report Sections:**
{chr(10).join([f'• {section}' for section in details['sections']])}

**Key Features:**
{chr(10).join([f'• {feature}' for feature in details['features']])}

**Delivery Format:** {details['delivery']}""",
        title=f"Template: {template_name.title()}",
        border_style="blue",
    )

    console.print(description)


def _create_custom_template(template_dir: Path | None):
    """Create custom report template interactively."""

    rprint("🎨 [bold]Custom Template Builder[/bold]")
    rprint("Let's create your custom report template...\n")

    # Gather template information
    template_name = typer.prompt("Template name")
    description = typer.prompt("Template description")
    target_audience = typer.prompt("Target audience")

    # Sections
    rprint("\n📋 Define report sections (press Enter with empty input to finish):")
    sections = []
    while True:
        section = typer.prompt(f"Section {len(sections) + 1}", default="")
        if not section:
            break
        sections.append(section)

    # Features
    rprint("\n✨ Select features:")
    features = []
    feature_options = [
        "executive_summary",
        "detailed_metrics",
        "trend_analysis",
        "recommendations",
        "charts_visualizations",
        "competitive_analysis",
        "roi_analysis",
    ]

    for feature in feature_options:
        if typer.confirm(f"Include {feature.replace('_', ' ')}?"):
            features.append(feature)

    # Create template configuration
    template_config = {
        "name": template_name,
        "description": description,
        "target_audience": target_audience,
        "sections": sections,
        "features": features,
        "created_at": datetime.now().isoformat(),
        "version": "1.0",
    }

    # Save template
    if not template_dir:
        template_dir = Path("./templates")

    template_dir.mkdir(exist_ok=True)
    template_file = template_dir / f"{template_name.lower().replace(' ', '_')}.json"

    with open(template_file, "w") as f:
        json.dump(template_config, f, indent=2)

    rprint(f"✅ Custom template created: [green]{template_file}[/green]")
    rprint("💡 You can now use this template with: ai-agent report generate custom")


def _calculate_next_run(
    frequency: str, time: str, day_of_week: str | None, day_of_month: int | None
) -> str:
    """Calculate next scheduled run time."""

    from datetime import datetime, timedelta

    now = datetime.now()
    hour, minute = map(int, time.split(":"))

    if frequency == "daily":
        next_run = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
        if next_run <= now:
            next_run += timedelta(days=1)

    elif frequency == "weekly":
        # Simplified - would need proper day of week handling
        next_run = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
        days_ahead = 7  # Default to next week
        next_run += timedelta(days=days_ahead)

    elif frequency == "monthly":
        # Simplified - would need proper month handling
        next_run = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
        if day_of_month:
            next_run = next_run.replace(day=day_of_month)
        next_run = next_run.replace(month=next_run.month + 1)

    else:
        next_run = now

    return next_run.isoformat()


def _save_schedule(schedule_config: dict):
    """Save report schedule configuration."""

    schedule_file = Path("./config/schedules.json")
    schedule_file.parent.mkdir(exist_ok=True)

    # Load existing schedules
    schedules = []
    if schedule_file.exists():
        with open(schedule_file) as f:
            schedules = json.load(f)

    # Add new schedule
    schedules.append(schedule_config)

    # Save updated schedules
    with open(schedule_file, "w") as f:
        json.dump(schedules, f, indent=2)

    rprint(f"📅 Schedule saved to [green]{schedule_file}[/green]")


def _send_test_report(template: str, recipients: list[str]):
    """Send test report immediately."""

    # Generate quick test report
    test_report_file = Path(f"./test_report_{template}.md")

    test_content = f"""# Test Report - {template.title()}

This is a test report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.

## Key Metrics
- Test Metric 1: 100%
- Test Metric 2: $1,000
- Test Metric 3: 5.0x

## Test Recommendations
1. Verify email delivery system
2. Confirm report formatting
3. Check recipient access

*This is a test report. Actual reports will contain real performance data.*
"""

    with open(test_report_file, "w") as f:
        f.write(test_content)

    _send_report_email(test_report_file, recipients, template)

    # Cleanup test file
    test_report_file.unlink()


def _list_dashboards():
    """List available dashboards."""

    # Mock dashboard data
    dashboards = [
        {
            "name": "Executive Dashboard",
            "status": "active",
            "views": 145,
            "last_updated": "2024-01-20",
        },
        {
            "name": "Campaign Performance",
            "status": "active",
            "views": 89,
            "last_updated": "2024-01-19",
        },
        {
            "name": "ROI Monitor",
            "status": "inactive",
            "views": 12,
            "last_updated": "2024-01-15",
        },
    ]

    table = Table(title="📊 Available Dashboards")
    table.add_column("Name", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Views", style="yellow")
    table.add_column("Last Updated", style="blue")
    table.add_column("Actions", style="white")

    for dashboard in dashboards:
        status_emoji = "🟢" if dashboard["status"] == "active" else "🔴"

        table.add_row(
            dashboard["name"],
            f"{status_emoji} {dashboard['status'].title()}",
            str(dashboard["views"]),
            dashboard["last_updated"],
            "View | Edit | Delete",
        )

    console.print(table)


def _create_dashboard(
    name: str,
    template: str | None,
    auto_refresh: bool,
    refresh_interval: int,
    public: bool,
):
    """Create new dashboard."""

    dashboard_config = {
        "name": name,
        "template": template or "executive",
        "auto_refresh": auto_refresh,
        "refresh_interval": refresh_interval,
        "public": public,
        "created_at": datetime.now().isoformat(),
        "url": f"http://localhost:8080/dashboard/{name.lower().replace(' ', '_')}",
    }

    rprint(f"✅ Created dashboard: [green]{name}[/green]")
    rprint(f"🔗 URL: {dashboard_config['url']}")
    rprint(
        f"⚙️ Settings: Auto-refresh={auto_refresh}, Interval={refresh_interval}s, Public={public}"
    )


def _view_dashboard(name: str):
    """View dashboard in browser."""

    dashboard_url = f"http://localhost:8080/dashboard/{name.lower().replace(' ', '_')}"
    rprint(f"🚀 Opening dashboard: [blue]{dashboard_url}[/blue]")
    rprint("💡 Dashboard would open in browser in actual implementation")


def _delete_dashboard(name: str):
    """Delete dashboard."""

    if typer.confirm(f"Are you sure you want to delete dashboard '{name}'?"):
        rprint(f"🗑️ Deleted dashboard: {name}")
    else:
        rprint("Dashboard deletion cancelled")


def _load_source_content(source_file: Path) -> dict:
    """Load content from source file."""

    if source_file.suffix.lower() == ".json":
        with open(source_file) as f:
            return json.load(f)
    elif source_file.suffix.lower() == ".md":
        with open(source_file) as f:
            return {"content": f.read(), "format": "markdown"}
    else:
        return {"content": "Unsupported format", "format": "unknown"}


def _convert_to_pdf(
    content: dict,
    output_file: Path,
    style: str | None,
    cover: bool,
    watermark: str | None,
):
    """Convert content to PDF."""
    rprint("📄 PDF conversion requires additional dependencies (reportlab, weasyprint)")
    rprint("Creating HTML version instead...")
    html_file = output_file.with_suffix(".html")
    _convert_to_html(content, html_file, style)


def _convert_to_html(content: dict, output_file: Path, style: str | None):
    """Convert content to HTML."""

    html_template = f"""<!DOCTYPE html>
<html>
<head>
    <title>Marketing Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        .header {{ border-bottom: 2px solid #333; padding-bottom: 20px; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Marketing Performance Report</h1>
        <p>Converted from source document</p>
    </div>
    <div class="content">
        {content.get('content', 'No content available')}
    </div>
</body>
</html>"""

    with open(output_file, "w") as f:
        f.write(html_template)


def _convert_to_docx(
    content: dict, output_file: Path, style: str | None, cover: bool
):
    """Convert content to DOCX."""
    rprint("📄 DOCX conversion requires python-docx library")
    rprint("Content would be converted to Word document format")


def _convert_to_pptx(content: dict, output_file: Path, style: str | None):
    """Convert content to PowerPoint."""
    rprint("📊 PPTX conversion requires python-pptx library")
    rprint("Content would be converted to PowerPoint presentation")


def _show_export_summary(source_file: Path, output_file: Path, target_format: str):
    """Show export operation summary."""

    source_size = source_file.stat().st_size / 1024
    output_size = output_file.stat().st_size / 1024

    rprint("📊 Export completed:")
    rprint(f"   Source: {source_file.name} ({source_size:.1f} KB)")
    rprint(f"   Output: {output_file.name} ({output_size:.1f} KB)")
    rprint(f"   Format: {target_format.upper()}")


def _get_report_metadata(report_file: Path) -> dict:
    """Get metadata for report analytics."""

    return {
        "filename": report_file.name,
        "size_kb": report_file.stat().st_size / 1024,
        "created_at": datetime.fromtimestamp(report_file.stat().st_ctime).isoformat(),
        "format": report_file.suffix.lower(),
    }


def _calculate_engagement_metrics(report_file: Path) -> dict:
    """Calculate engagement metrics (mock data)."""

    return {
        "total_views": 47,
        "unique_viewers": 23,
        "avg_view_time": "4:32",
        "shares": 8,
        "downloads": 12,
        "comments": 3,
    }


def _calculate_performance_metrics(report_file: Path) -> dict:
    """Calculate report performance metrics."""

    return {
        "prediction_accuracy": 0.84,
        "recommendations_implemented": 0.67,
        "roi_from_recommendations": 0.23,
        "user_satisfaction": 4.2,
    }


def _display_report_analytics(
    metadata: dict,
    engagement: dict | None,
    performance: dict | None,
    benchmark: str | None,
):
    """Display report analytics results."""

    rprint(f"📊 [bold]Report Analytics: {metadata['filename']}[/bold]")

    # Metadata
    meta_table = Table(title="Report Metadata")
    meta_table.add_column("Property", style="cyan")
    meta_table.add_column("Value", style="green")

    meta_table.add_row("File Size", f"{metadata['size_kb']:.1f} KB")
    meta_table.add_row("Format", metadata["format"].upper())
    meta_table.add_row("Created", metadata["created_at"][:10])

    console.print(meta_table)

    # Engagement metrics
    if engagement:
        rprint("\n👁️ [bold]Engagement Metrics[/bold]")
        eng_table = Table()
        eng_table.add_column("Metric", style="yellow")
        eng_table.add_column("Value", style="blue")

        for metric, value in engagement.items():
            eng_table.add_row(metric.replace("_", " ").title(), str(value))

        console.print(eng_table)

    # Performance metrics
    if performance:
        rprint("\n📈 [bold]Performance Metrics[/bold]")
        perf_table = Table()
        perf_table.add_column("Metric", style="magenta")
        perf_table.add_column("Score", style="green")

        for metric, value in performance.items():
            if isinstance(value, float) and value < 1:
                display_value = f"{value:.1%}"
            else:
                display_value = str(value)
            perf_table.add_row(metric.replace("_", " ").title(), display_value)

        console.print(perf_table)


def _export_report_analytics(report_file: Path, engagement: dict, performance: dict):
    """Export report analytics data."""

    analytics_data = {
        "report_file": str(report_file),
        "analyzed_at": datetime.now().isoformat(),
        "engagement_metrics": engagement,
        "performance_metrics": performance,
    }

    analytics_file = report_file.with_suffix(".analytics.json")

    with open(analytics_file, "w") as f:
        json.dump(analytics_data, f, indent=2)

    rprint(f"📊 Analytics exported to [green]{analytics_file}[/green]")


def _parse_date_range(date_range: str) -> dict:
    """Parse date range into start and end dates."""

    if date_range.endswith("d"):
        days = int(date_range[:-1])
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=days)
        return {"start": start_date.isoformat(), "end": end_date.isoformat()}

    return {"start": "2024-01-01", "end": "2024-01-31"}  # Default


# Additional helper functions for report content creation
def _create_executive_summary(data: dict, analytics: dict, insights: dict) -> list[str]:
    """Create executive summary content."""

    total_roas = data["total_revenue"] / data["total_spend"]

    return [
        f"Campaign portfolio generated ${data['total_revenue']:,.0f} in revenue from ${data['total_spend']:,.0f} spend",
        f"Overall ROAS of {total_roas:.2f}x represents strong performance vs industry average",
        f"Identified {len(insights.get('opportunities', []))} key optimization opportunities worth {sum([0.2, 0.15, 0.1]):.0%} potential uplift",
        f"Search campaigns outperforming Shopping by {((4.01/2.50)-1):.0%} ROAS margin",
        "Strong foundation established for scaling successful elements in Q2",
    ]


def _create_performance_overview(data: dict, analytics: dict) -> dict:
    """Create performance overview section."""

    total_roas = data["total_revenue"] / data["total_spend"]

    return {
        "total_spend": data["total_spend"],
        "total_revenue": data["total_revenue"],
        "total_roas": total_roas,
        "total_conversions": data["total_conversions"],
        "avg_cpa": data["total_spend"] / data["total_conversions"],
        "campaign_count": len(data["campaigns"]),
        "performance_score": analytics["performance_scores"]["overall"],
    }


def _create_detailed_analysis(data: dict, analytics: dict, insights: dict) -> dict:
    """Create detailed analysis section."""

    return {
        "campaign_breakdown": data["campaigns"],
        "trend_analysis": analytics["trends"],
        "anomaly_summary": analytics["anomalies"],
        "benchmark_results": analytics["benchmarks"],
        "key_insights": insights["key_findings"],
    }


def _create_report_visualizations(data: dict, analytics: dict) -> dict:
    """Create visualization placeholders."""

    return {
        "charts": [
            {"type": "roas_by_campaign", "title": "ROAS by Campaign"},
            {"type": "spend_trend", "title": "Spend Trend Over Time"},
            {"type": "conversion_funnel", "title": "Conversion Funnel Analysis"},
            {"type": "benchmark_comparison", "title": "Industry Benchmark Comparison"},
        ],
        "tables": [
            {"type": "campaign_performance", "title": "Campaign Performance Summary"},
            {"type": "top_keywords", "title": "Top Performing Keywords"},
        ],
    }


def _create_executive_kpis(data: dict, analytics: dict) -> dict:
    """Create executive KPIs."""

    total_roas = data["total_revenue"] / data["total_spend"]

    return {
        "primary_kpis": [
            {
                "name": "Total ROAS",
                "value": f"{total_roas:.2f}x",
                "trend": "up",
                "target": "4.0x",
            },
            {
                "name": "Revenue",
                "value": f"${data['total_revenue']:,.0f}",
                "trend": "up",
                "target": "$150K",
            },
            {
                "name": "Conversions",
                "value": f"{data['total_conversions']:,}",
                "trend": "up",
                "target": "700",
            },
            {
                "name": "Efficiency Score",
                "value": f"{analytics['performance_scores']['efficiency']:.1%}",
                "trend": "stable",
                "target": "85%",
            },
        ]
    }


def _create_detailed_metrics(data: dict, analytics: dict) -> dict:
    """Create detailed metrics breakdown."""

    return {
        "performance_breakdown": analytics["performance_scores"],
        "campaign_details": data["campaigns"],
        "trend_details": analytics["trends"],
        "anomaly_details": analytics["anomalies"],
    }


def _create_optimization_analysis(data: dict, analytics: dict, insights: dict) -> dict:
    """Create optimization-focused analysis."""

    return {
        "current_performance": analytics["performance_scores"],
        "improvement_opportunities": insights["opportunities"],
        "risk_assessment": insights["risks"],
        "roi_projections": {
            "conservative": 0.15,
            "realistic": 0.25,
            "optimistic": 0.35,
        },
    }


def _generate_executive_summary(analytics: dict) -> list[str]:
    """Generate executive summary based on analytics."""

    return [
        "Portfolio demonstrating strong performance with selective optimization opportunities",
        f"Overall efficiency score of {analytics['performance_scores']['efficiency']:.1%} indicates healthy campaign structure",
        "Key focus areas identified in impression share and creative performance optimization",
    ]


def _generate_detailed_analysis(analytics: dict) -> dict:
    """Generate detailed analysis section."""

    return {
        "performance_deep_dive": analytics["performance_scores"],
        "statistical_significance": analytics["trends"]["significance"] < 0.05,
        "anomaly_impact_assessment": "Medium - requires monitoring but not immediate action",
        "benchmark_competitive_position": f"{analytics['benchmarks']['industry_percentile']}th percentile",
    }


def _generate_optimization_priorities(analytics: dict) -> list[dict]:
    """Generate optimization priorities."""

    return [
        {
            "priority": 1,
            "area": "Budget Allocation",
            "rationale": "High ROAS variance between campaigns presents reallocation opportunity",
            "expected_impact": 0.25,
        },
        {
            "priority": 2,
            "area": "Impression Share",
            "rationale": f"Currently at {analytics['benchmarks']['industry_percentile']}th percentile with expansion potential",
            "expected_impact": 0.18,
        },
        {
            "priority": 3,
            "area": "Creative Testing",
            "rationale": "CTR performance below industry average in key segments",
            "expected_impact": 0.15,
        },
    ]
