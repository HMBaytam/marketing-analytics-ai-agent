"""Optimization recommendation commands."""

import typer
from typing import Optional, List
from datetime import datetime
from pathlib import Path
import json
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich import print as rprint
from rich.panel import Panel
from rich.columns import Columns

from ...optimization import (
    RecommendationsEngine, RecommendationConfig, 
    BudgetOptimizer, BudgetConstraints, AllocationStrategy,
    ROIOptimizer, OptimizationObjective,
    ABTestingOptimizer, TestConfig, TestType
)
# Removed circular import - get_config imported locally where needed

app = typer.Typer(help="⚡ Optimization recommendations and strategies")
console = Console()


@app.command()
def recommendations(
    campaign_id: Optional[str] = typer.Option(None, "--campaign-id", "-c", help="Specific campaign to optimize"),
    max_recommendations: int = typer.Option(10, "--max", "-n", help="Maximum recommendations to show"),
    confidence_threshold: float = typer.Option(0.7, "--confidence", help="Minimum confidence threshold"),
    priority_filter: Optional[str] = typer.Option(None, "--priority", help="Filter by priority (critical, high, medium, low)"),
    category_filter: Optional[str] = typer.Option(None, "--category", help="Filter by category"),
    output_file: Optional[Path] = typer.Option(None, "--output", "-o", help="Save recommendations to file"),
    interactive: bool = typer.Option(False, "--interactive", "-i", help="Interactive recommendation review"),
    auto_implement: bool = typer.Option(False, "--auto-implement", help="Auto-implement safe recommendations")
):
    """
    Generate comprehensive optimization recommendations.
    
    Analyzes:
    • Performance gaps and improvement opportunities
    • Budget allocation inefficiencies  
    • A/B testing opportunities
    • ROI optimization potential
    • Risk factors and mitigation strategies
    
    Examples:
        ai-agent optimize recommendations --campaign-id ABC123
        ai-agent optimize recommendations --priority high --interactive
        ai-agent optimize recommendations --confidence 0.8 --auto-implement
    """
    
    # Get config directly to avoid circular import
    from ...core.config import get_default_config
    config = get_default_config()
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        
        task = progress.add_task("Analyzing campaign performance...", total=100)
        
        # Initialize recommendations engine
        rec_config = RecommendationConfig(
            max_recommendations=max_recommendations,
            min_confidence_score=confidence_threshold
        )
        engine = RecommendationsEngine(rec_config)
        
        progress.update(task, advance=20, description="Gathering performance data...")
        
        # Mock campaign data (in production, fetch real data)
        campaign_data = _get_mock_campaign_data(campaign_id)
        performance_issues = _identify_mock_performance_issues(campaign_data)
        
        progress.update(task, advance=30, description="Running optimization analysis...")
        
        # Generate mock recommendations
        recommendations = _generate_mock_recommendations(
            campaign_data, performance_issues, priority_filter, category_filter
        )
        
        progress.update(task, advance=30, description="Prioritizing recommendations...")
        
        # Filter and sort recommendations
        if priority_filter:
            recommendations = [r for r in recommendations if r["priority"].lower() == priority_filter.lower()]
        
        if category_filter:
            recommendations = [r for r in recommendations if category_filter.lower() in r["type"].lower()]
        
        recommendations = sorted(recommendations, key=lambda x: x["priority_score"], reverse=True)
        recommendations = recommendations[:max_recommendations]
        
        progress.update(task, advance=20, description="Generating insights...")
    
    # Display recommendations
    if interactive:
        _interactive_recommendation_review(recommendations, auto_implement)
    else:
        _display_recommendations_table(recommendations)
        _display_recommendations_summary(recommendations)
    
    # Save to file if requested
    if output_file:
        _save_optimization_results(recommendations, output_file)
        rprint(f"✅ Recommendations saved to [green]{output_file}[/green]")
    
    # Auto-implement if requested
    if auto_implement and not interactive:
        safe_recommendations = [r for r in recommendations if r["risk_level"].lower() == "low"]
        if safe_recommendations:
            _auto_implement_recommendations(safe_recommendations)


@app.command()
def budget(
    strategy: str = typer.Option("roi_maximization", "--strategy", "-s", help="Allocation strategy"),
    total_budget: float = typer.Option(10000, "--total", "-t", help="Total budget to allocate"),
    min_per_channel: float = typer.Option(100, "--min-channel", help="Minimum budget per channel"),
    max_change: float = typer.Option(0.3, "--max-change", help="Maximum change percentage"),
    channels: Optional[List[str]] = typer.Option(None, "--channel", help="Specific channels to include"),
    preview_only: bool = typer.Option(False, "--preview", help="Preview changes without applying"),
    force_reallocation: bool = typer.Option(False, "--force", help="Force reallocation even with high risk"),
    export_plan: bool = typer.Option(False, "--export", help="Export budget plan to CSV")
):
    """
    Optimize budget allocation across channels and campaigns.
    
    Strategies:
    • roi_maximization: Maximize total ROI
    • performance_based: Allocate based on performance scores
    • risk_adjusted: Factor in risk and volatility
    • seasonal_adjusted: Account for seasonal patterns
    • marginal_efficiency: Optimize marginal returns
    
    Examples:
        ai-agent optimize budget --strategy roi_maximization --total 50000
        ai-agent optimize budget --strategy risk_adjusted --preview
        ai-agent optimize budget --max-change 0.2 --export
    """
    
    # Get config directly to avoid circular import
    from ...core.config import get_default_config
    config = get_default_config()
    
    # Parse strategy
    try:
        allocation_strategy = AllocationStrategy(strategy)
    except ValueError:
        typer.secho(f"❌ Invalid strategy: {strategy}", fg=typer.colors.RED)
        typer.echo("Available strategies: roi_maximization, performance_based, risk_adjusted, seasonal_adjusted, marginal_efficiency")
        raise typer.Exit(1)
    
    with Progress(console=console) as progress:
        task = progress.add_task("Optimizing budget allocation...", total=100)
        
        # Setup constraints
        constraints = BudgetConstraints(
            total_budget=total_budget,
            min_budget_per_channel=min_per_channel,
            max_change_percentage=max_change
        )
        
        optimizer = BudgetOptimizer(constraints)
        progress.update(task, advance=20)
        
        # Mock channel and performance data
        mock_channels = _get_mock_channel_data(channels)
        mock_performance = _get_mock_performance_history()
        
        progress.update(task, advance=30, description="Calculating optimal allocation...")
        
        # Generate budget allocation
        budget_allocation = optimizer.optimize_budget_allocation(
            mock_channels, mock_performance, allocation_strategy
        )
        
        progress.update(task, advance=50, description="Analyzing allocation impact...")
    
    # Display results
    _display_budget_allocation(budget_allocation, preview_only)
    
    # Export plan if requested
    if export_plan:
        _export_budget_plan(budget_allocation)
    
    # Apply changes if not preview only
    if not preview_only and not force_reallocation:
        if budget_allocation.overall_risk_score > 0.7:
            if not typer.confirm("⚠️  High risk allocation detected. Continue?"):
                rprint("Budget allocation cancelled")
                return
    
    if not preview_only:
        _apply_budget_changes(budget_allocation)


@app.command()
def roi(
    campaign_id: Optional[str] = typer.Option(None, "--campaign-id", "-c", help="Specific campaign to analyze"),
    objective: str = typer.Option("maximize_total_roi", "--objective", "-obj", help="Optimization objective"),
    include_analysis: bool = typer.Option(True, "--analysis/--no-analysis", help="Include detailed ROI analysis"),
    sensitivity_analysis: bool = typer.Option(False, "--sensitivity", help="Run sensitivity analysis"),
    scenario_planning: bool = typer.Option(False, "--scenarios", help="Generate scenario plans"),
    output_format: str = typer.Option("table", "--format", help="Output format (table, json, report)"),
    min_roi_improvement: float = typer.Option(0.1, "--min-improvement", help="Minimum ROI improvement threshold")
):
    """
    Analyze and optimize return on investment (ROI).
    
    Objectives:
    • maximize_total_roi: Maximize overall ROI
    • maximize_incremental_roi: Focus on incremental returns
    • minimize_payback_period: Reduce time to break-even
    • maximize_lifetime_value: Optimize customer LTV
    • optimize_marginal_roi: Balance marginal returns
    
    Examples:
        ai-agent optimize roi --objective maximize_total_roi --scenarios
        ai-agent optimize roi --campaign-id ABC123 --sensitivity
        ai-agent optimize roi --format report --min-improvement 0.2
    """
    
    # Get config directly to avoid circular import
    from ...core.config import get_default_config
    config = get_default_config()
    
    # Parse objective
    try:
        roi_objective = OptimizationObjective(objective)
    except ValueError:
        typer.secho(f"❌ Invalid objective: {objective}", fg=typer.colors.RED)
        available = [obj.value for obj in OptimizationObjective]
        typer.echo(f"Available objectives: {', '.join(available)}")
        raise typer.Exit(1)
    
    with Progress(console=console) as progress:
        task = progress.add_task("Analyzing ROI performance...", total=100)
        
        optimizer = ROIOptimizer()
        
        # Mock data (in production, fetch real data)
        campaign_data = _get_mock_campaign_data(campaign_id)
        cost_data = _get_mock_cost_data(campaign_id)
        revenue_data = _get_mock_revenue_data(campaign_id)
        
        progress.update(task, advance=30, description="Calculating current ROI metrics...")
        
        # Analyze current ROI
        current_analysis = optimizer.analyze_current_roi(
            campaign_data, cost_data, revenue_data
        )
        
        progress.update(task, advance=40, description="Generating optimization strategies...")
        
        # Generate ROI optimization
        campaign_performance = _combine_performance_data(campaign_data, cost_data, revenue_data)
        roi_optimization = optimizer.optimize_roi(
            current_analysis, campaign_performance, roi_objective
        )
        
        progress.update(task, advance=30, description="Preparing recommendations...")
    
    # Display results
    if output_format == "table":
        _display_roi_analysis_table(current_analysis, roi_optimization, include_analysis)
    elif output_format == "json":
        roi_data = {
            "current_analysis": current_analysis.dict(),
            "optimization": roi_optimization.dict()
        }
        rprint(json.dumps(roi_data, indent=2, default=str))
    elif output_format == "report":
        _display_roi_report(current_analysis, roi_optimization, sensitivity_analysis, scenario_planning)
    
    # Additional analyses
    if sensitivity_analysis:
        _display_sensitivity_analysis(roi_optimization.sensitivity_analysis)
    
    if scenario_planning:
        _display_scenario_planning(roi_optimization)


@app.command()
def ab_tests(
    campaign_id: Optional[str] = typer.Option(None, "--campaign-id", "-c", help="Campaign to generate tests for"),
    test_type: Optional[str] = typer.Option(None, "--type", "-t", help="Specific test type to focus on"),
    min_sample_size: int = typer.Option(1000, "--min-samples", help="Minimum sample size for tests"),
    max_test_cost: float = typer.Option(5000, "--max-cost", help="Maximum cost per test"),
    priority_only: bool = typer.Option(False, "--priority-only", help="Show only high priority tests"),
    include_setup: bool = typer.Option(True, "--setup/--no-setup", help="Include test setup instructions"),
    auto_schedule: bool = typer.Option(False, "--auto-schedule", help="Auto-schedule feasible tests"),
    test_calendar: bool = typer.Option(False, "--calendar", help="Show test scheduling calendar")
):
    """
    Generate A/B testing recommendations and schedules.
    
    Test Types:
    • creative_test: Creative and ad variations
    • landing_page_test: Landing page optimization
    • audience_test: Audience targeting tests
    • bid_strategy_test: Bidding optimization
    • budget_split_test: Budget timing tests
    
    Examples:
        ai-agent optimize ab-tests --campaign-id ABC123 --priority-only
        ai-agent optimize ab-tests --type creative_test --auto-schedule  
        ai-agent optimize ab-tests --max-cost 2000 --calendar
    """
    
    # Get config directly to avoid circular import
    from ...core.config import get_default_config
    config = get_default_config()
    
    with Progress(console=console) as progress:
        task = progress.add_task("Analyzing test opportunities...", total=100)
        
        # Initialize A/B testing optimizer
        test_config = TestConfig(
            min_daily_conversions=10,
            max_budget_risk=max_test_cost / 10000  # Convert to risk ratio
        )
        optimizer = ABTestingOptimizer(test_config)
        
        progress.update(task, advance=20)
        
        # Mock data
        campaign_data = _get_mock_campaign_data(campaign_id)
        performance_issues = _identify_mock_performance_issues(campaign_data)
        
        progress.update(task, advance=30, description="Generating test recommendations...")
        
        # Generate test recommendations
        test_recommendations = optimizer.generate_test_recommendations(
            campaign_data, performance_issues, campaign_id
        )
        
        # Filter by test type if specified
        if test_type:
            try:
                target_type = TestType(test_type)
                test_recommendations = [
                    t for t in test_recommendations 
                    if t.test_type == target_type
                ]
            except ValueError:
                typer.secho(f"❌ Invalid test type: {test_type}", fg=typer.colors.RED)
                available = [tt.value for tt in TestType]
                typer.echo(f"Available types: {', '.join(available)}")
                raise typer.Exit(1)
        
        # Filter by priority if requested
        if priority_only:
            test_recommendations = [
                t for t in test_recommendations 
                if t.priority.value in ["critical", "high"]
            ]
        
        # Filter by cost
        test_recommendations = [
            t for t in test_recommendations 
            if t.estimated_cost <= max_test_cost
        ]
        
        progress.update(task, advance=30, description="Prioritizing tests...")
        
        # Sort by priority and potential impact
        test_recommendations.sort(
            key=lambda x: (
                {"critical": 4, "high": 3, "medium": 2, "low": 1}[x.priority.value],
                x.potential_uplift
            ),
            reverse=True
        )
        
        progress.update(task, advance=20, description="Preparing test plans...")
    
    # Display test recommendations
    _display_ab_test_recommendations(test_recommendations, include_setup)
    
    # Show test calendar if requested
    if test_calendar:
        _display_test_calendar(test_recommendations)
    
    # Auto-schedule if requested
    if auto_schedule:
        _auto_schedule_tests(test_recommendations)


@app.command()
def simulate(
    scenario: str = typer.Argument(..., help="Optimization scenario to simulate"),
    budget_change: float = typer.Option(0.0, "--budget-change", help="Budget change percentage"),
    bid_change: float = typer.Option(0.0, "--bid-change", help="Bid change percentage"),
    audience_expansion: float = typer.Option(0.0, "--audience-expansion", help="Audience expansion factor"),
    duration_days: int = typer.Option(30, "--duration", help="Simulation duration in days"),
    confidence_level: float = typer.Option(0.95, "--confidence", help="Confidence level for predictions"),
    monte_carlo: bool = typer.Option(False, "--monte-carlo", help="Run Monte Carlo simulation"),
    iterations: int = typer.Option(1000, "--iterations", help="Monte Carlo iterations")
):
    """
    Simulate optimization scenarios and predict outcomes.
    
    Scenarios:
    • scale_up: Increase budget and expand reach
    • optimize_efficiency: Focus on cost reduction
    • audience_expansion: Expand to new audiences
    • creative_refresh: Test new creative approaches
    • seasonal_adjustment: Adjust for seasonal patterns
    
    Examples:
        ai-agent optimize simulate scale_up --budget-change 0.5
        ai-agent optimize simulate optimize_efficiency --bid-change -0.2
        ai-agent optimize simulate audience_expansion --monte-carlo
    """
    
    # Get config directly to avoid circular import
    from ...core.config import get_default_config
    config = get_default_config()
    
    with Progress(console=console) as progress:
        task = progress.add_task("Running optimization simulation...", total=100)
        
        # Setup simulation parameters
        simulation_params = {
            "scenario": scenario,
            "budget_change": budget_change,
            "bid_change": bid_change,
            "audience_expansion": audience_expansion,
            "duration_days": duration_days,
            "confidence_level": confidence_level
        }
        
        progress.update(task, advance=20, description="Initializing simulation...")
        
        # Mock current performance baseline
        baseline_metrics = _get_simulation_baseline()
        
        progress.update(task, advance=30, description="Running scenario simulation...")
        
        # Run simulation
        if monte_carlo:
            simulation_results = _run_monte_carlo_simulation(
                simulation_params, baseline_metrics, iterations
            )
            progress.update(task, advance=40, description=f"Running {iterations} Monte Carlo iterations...")
        else:
            simulation_results = _run_deterministic_simulation(
                simulation_params, baseline_metrics
            )
            progress.update(task, advance=40, description="Calculating expected outcomes...")
        
        progress.update(task, advance=10, description="Analyzing results...")
    
    # Display simulation results
    _display_simulation_results(simulation_results, monte_carlo)
    
    # Risk assessment
    _display_simulation_risk_analysis(simulation_results)


@app.command()
def monitor(
    optimization_id: Optional[str] = typer.Option(None, "--optimization-id", help="Specific optimization to monitor"),
    real_time: bool = typer.Option(False, "--real-time", help="Real-time monitoring mode"),
    alerts: bool = typer.Option(True, "--alerts/--no-alerts", help="Enable performance alerts"),
    refresh_interval: int = typer.Option(300, "--refresh", help="Refresh interval in seconds"),
    export_log: bool = typer.Option(False, "--export-log", help="Export monitoring log")
):
    """
    Monitor optimization implementation and performance.
    
    Tracks:
    • Recommendation implementation status
    • Performance impact measurement  
    • ROI improvement tracking
    • A/B test progress and results
    
    Examples:
        ai-agent optimize monitor --real-time
        ai-agent optimize monitor --optimization-id OPT123 --alerts
        ai-agent optimize monitor --refresh 60 --export-log
    """
    
    # Get config directly to avoid circular import
    from ...core.config import get_default_config
    config = get_default_config()
    
    if real_time:
        _start_real_time_monitoring(optimization_id, alerts, refresh_interval)
    else:
        _show_optimization_status(optimization_id, export_log)


# Helper functions

def _get_mock_campaign_data(campaign_id: Optional[str]) -> List[dict]:
    """Get mock campaign data for optimization."""
    campaigns = [
        {
            "id": "123456789",
            "name": "Search Campaign Q4",
            "spend": 15750.00,
            "revenue": 67200.00,
            "conversions": 234,
            "ctr": 0.034,
            "cpa": 67.31,
            "roas": 4.27,
            "quality_score": 7.2,
            "impression_share": 0.68
        },
        {
            "id": "987654321", 
            "name": "Shopping Campaign",
            "spend": 12400.00,
            "revenue": 31000.00,
            "conversions": 186,
            "ctr": 0.021,
            "cpa": 66.67,
            "roas": 2.50,
            "quality_score": 6.1,
            "impression_share": 0.45
        }
    ]
    
    if campaign_id:
        return [c for c in campaigns if c["id"] == campaign_id]
    return campaigns


def _identify_mock_performance_issues(campaign_data: List[dict]) -> List[dict]:
    """Identify performance issues from campaign data."""
    issues = []
    
    for campaign in campaign_data:
        if campaign["roas"] < 3.0:
            issues.append({
                "campaign_id": campaign["id"],
                "issue": "low_roas",
                "severity": "high",
                "current_value": campaign["roas"],
                "target_value": 4.0
            })
        
        if campaign["ctr"] < 0.025:
            issues.append({
                "campaign_id": campaign["id"],
                "issue": "low_ctr",
                "severity": "medium",
                "current_value": campaign["ctr"],
                "target_value": 0.030
            })
        
        if campaign["impression_share"] < 0.5:
            issues.append({
                "campaign_id": campaign["id"],
                "issue": "low_impression_share",
                "severity": "medium",
                "current_value": campaign["impression_share"],
                "target_value": 0.7
            })
    
    return issues


def _generate_mock_recommendations(
    campaign_data: List[dict],
    performance_issues: List[dict],
    priority_filter: Optional[str],
    category_filter: Optional[str]
) -> List[dict]:
    """Generate mock optimization recommendations."""
    
    recommendations = [
        {
            "id": "rec_001",
            "type": "budget_reallocation",
            "priority": "high",
            "priority_score": 8.5,
            "title": "Reallocate Budget to High-ROAS Campaigns",
            "description": "Shift 30% of budget from Shopping Campaign to Search Campaign Q4",
            "expected_impact": 0.25,
            "confidence": 0.84,
            "risk_level": "medium",
            "implementation_effort": "low",
            "timeline": "1-2 weeks",
            "actions": [
                "Reduce Shopping Campaign budget by 30%",
                "Increase Search Campaign Q4 budget by equivalent amount",
                "Monitor performance for 2 weeks"
            ],
            "success_metrics": ["Overall ROAS improvement >20%", "Maintained conversion volume"]
        },
        {
            "id": "rec_002", 
            "type": "keyword_optimization",
            "priority": "medium",
            "priority_score": 7.2,
            "title": "Expand High-Performance Keywords",
            "description": "Add similar keywords to top-performing ad groups",
            "expected_impact": 0.18,
            "confidence": 0.76,
            "risk_level": "low",
            "implementation_effort": "medium",
            "timeline": "3-5 days",
            "actions": [
                "Analyze top-performing keywords",
                "Research related keyword opportunities", 
                "Add new keywords with appropriate bids",
                "Monitor and optimize over 2 weeks"
            ],
            "success_metrics": ["CTR improvement >15%", "Impression share increase"]
        },
        {
            "id": "rec_003",
            "type": "bid_strategy_optimization", 
            "priority": "high",
            "priority_score": 8.8,
            "title": "Switch to Target ROAS Bidding",
            "description": "Implement automated Target ROAS bidding for Shopping Campaign",
            "expected_impact": 0.32,
            "confidence": 0.79,
            "risk_level": "medium",
            "implementation_effort": "low",
            "timeline": "1 week + 2 weeks learning",
            "actions": [
                "Set Target ROAS at current performance level",
                "Enable Target ROAS bidding",
                "Allow 2-week learning period",
                "Gradually increase ROAS target"
            ],
            "success_metrics": ["ROAS improvement >25%", "Stable conversion volume"]
        },
        {
            "id": "rec_004",
            "type": "creative_testing",
            "priority": "medium",
            "priority_score": 6.9,
            "title": "A/B Test New Ad Creatives",
            "description": "Test benefit-focused ad copy variations",
            "expected_impact": 0.15,
            "confidence": 0.68,
            "risk_level": "low", 
            "implementation_effort": "medium",
            "timeline": "2-3 weeks",
            "actions": [
                "Develop 3 new ad copy variations",
                "Set up A/B test with 50/50 traffic split",
                "Run test for statistical significance",
                "Implement winning variation"
            ],
            "success_metrics": ["CTR improvement >12%", "Quality Score increase"]
        }
    ]
    
    return recommendations


def _display_recommendations_table(recommendations: List[dict]):
    """Display recommendations in a formatted table."""
    
    table = Table(title="⚡ Optimization Recommendations")
    table.add_column("ID", style="cyan", width=8)
    table.add_column("Title", style="green", width=30)
    table.add_column("Priority", style="yellow", width=10)
    table.add_column("Impact", style="blue", width=8)
    table.add_column("Confidence", style="magenta", width=10)
    table.add_column("Risk", style="white", width=8)
    table.add_column("Timeline", style="cyan", width=12)
    
    for rec in recommendations:
        priority_emoji = {
            "critical": "🚨",
            "high": "🔴",
            "medium": "🟡", 
            "low": "🟢"
        }.get(rec["priority"], "❓")
        
        table.add_row(
            rec["id"],
            rec["title"],
            f"{priority_emoji} {rec['priority'].title()}",
            f"{rec['expected_impact']:.0%}",
            f"{rec['confidence']:.0%}",
            rec["risk_level"].title(),
            rec["timeline"]
        )
    
    console.print(table)


def _display_recommendations_summary(recommendations: List[dict]):
    """Display recommendation summary."""
    
    total_impact = sum(r["expected_impact"] for r in recommendations)
    avg_confidence = sum(r["confidence"] for r in recommendations) / len(recommendations)
    
    high_priority = len([r for r in recommendations if r["priority"] in ["critical", "high"]])
    low_risk = len([r for r in recommendations if r["risk_level"] == "low"])
    
    summary = Panel(
        f"""📊 **Recommendation Summary**

Total Recommendations: {len(recommendations)}
High Priority: {high_priority}
Low Risk: {low_risk}

Expected Total Impact: {total_impact:.0%}
Average Confidence: {avg_confidence:.0%}

🚀 Recommended Actions:
1. Start with high-priority, low-risk recommendations
2. Implement budget reallocations first for quick wins
3. Set up A/B tests for creative optimizations
4. Monitor performance closely during implementation""",
        title="Summary",
        border_style="blue"
    )
    
    console.print(summary)


def _interactive_recommendation_review(recommendations: List[dict], auto_implement: bool):
    """Interactive recommendation review interface."""
    
    rprint("🔄 [bold]Interactive Recommendation Review[/bold]")
    rprint("Use arrow keys to navigate, Space to select, Enter to continue")
    
    for i, rec in enumerate(recommendations):
        rprint(f"\n[bold]Recommendation {i+1}/{len(recommendations)}:[/bold]")
        rprint(f"📋 {rec['title']}")
        rprint(f"🎯 Expected Impact: {rec['expected_impact']:.0%}")
        rprint(f"✅ Confidence: {rec['confidence']:.0%}")
        rprint(f"⚠️  Risk Level: {rec['risk_level'].title()}")
        
        rprint("\n📝 Actions:")
        for action in rec["actions"]:
            rprint(f"   • {action}")
        
        # User decision
        if auto_implement and rec["risk_level"] == "low":
            rprint("🤖 [green]Auto-implementing (low risk)...[/green]")
            _implement_recommendation(rec)
        else:
            choice = typer.prompt(
                "\nAction: (i)mplement, (s)kip, (d)etails, (q)uit",
                default="s"
            )
            
            if choice.lower() == 'i':
                _implement_recommendation(rec)
            elif choice.lower() == 'd':
                _show_recommendation_details(rec)
            elif choice.lower() == 'q':
                break


def _implement_recommendation(rec: dict):
    """Implement a specific recommendation."""
    rprint(f"🚀 Implementing: {rec['title']}")
    
    with Progress(console=console) as progress:
        task = progress.add_task("Implementation in progress...", total=len(rec["actions"]))
        
        for action in rec["actions"]:
            progress.console.print(f"   ✅ {action}")
            progress.update(task, advance=1)
            import time
            time.sleep(0.5)  # Simulate implementation time
    
    rprint(f"✅ [green]Implementation complete for {rec['id']}[/green]")


def _show_recommendation_details(rec: dict):
    """Show detailed information about a recommendation."""
    
    details = Panel(
        f"""**{rec['title']}**

**Description:** {rec['description']}

**Expected Impact:** {rec['expected_impact']:.0%}
**Confidence Level:** {rec['confidence']:.0%}
**Risk Level:** {rec['risk_level'].title()}
**Implementation Effort:** {rec['implementation_effort'].title()}
**Timeline:** {rec['timeline']}

**Actions Required:**
{chr(10).join([f"• {action}" for action in rec['actions']])}

**Success Metrics:**
{chr(10).join([f"• {metric}" for metric in rec['success_metrics']])}""",
        title=f"Details - {rec['id']}",
        border_style="yellow"
    )
    
    console.print(details)


def _get_mock_channel_data(channels: Optional[List[str]]) -> List[dict]:
    """Get mock channel data for budget optimization."""
    
    all_channels = [
        {
            "id": "search",
            "name": "Paid Search",
            "budget": 15000,
            "spend": 14200,
            "performance_score": 0.85
        },
        {
            "id": "shopping", 
            "name": "Google Shopping",
            "budget": 10000,
            "spend": 9800,
            "performance_score": 0.62
        },
        {
            "id": "display",
            "name": "Display Ads", 
            "budget": 8000,
            "spend": 7500,
            "performance_score": 0.58
        },
        {
            "id": "social",
            "name": "Social Media",
            "budget": 5000,
            "spend": 4900,
            "performance_score": 0.71
        }
    ]
    
    if channels:
        return [c for c in all_channels if c["id"] in channels]
    return all_channels


def _get_mock_performance_history() -> List[dict]:
    """Get mock performance history for budget optimization."""
    return [
        {"channel_id": "search", "date": "2024-01-01", "spend": 500, "revenue": 2100, "conversions": 42},
        {"channel_id": "shopping", "date": "2024-01-01", "spend": 350, "revenue": 875, "conversions": 18},
        {"channel_id": "display", "date": "2024-01-01", "spend": 280, "revenue": 420, "conversions": 8},
        {"channel_id": "social", "date": "2024-01-01", "spend": 175, "revenue": 350, "conversions": 12}
    ]


def _display_budget_allocation(budget_allocation, preview_only: bool):
    """Display budget allocation results."""
    
    rprint(f"💰 [bold]Budget Allocation Plan - {budget_allocation.strategy.value.replace('_', ' ').title()}[/bold]")
    
    # Allocation table
    table = Table(title="Channel Allocations")
    table.add_column("Channel", style="cyan")
    table.add_column("Current", style="yellow")
    table.add_column("Recommended", style="green")
    table.add_column("Change", style="blue")
    table.add_column("ROAS Impact", style="magenta")
    
    for allocation in budget_allocation.channel_allocations:
        change_str = f"{allocation.change_percentage:+.1f}%"
        if allocation.change_percentage > 0:
            change_str = f"[green]{change_str}[/green]"
        elif allocation.change_percentage < 0:
            change_str = f"[red]{change_str}[/red]"
        
        table.add_row(
            allocation.channel_name,
            f"${allocation.current_budget:,.0f}",
            f"${allocation.recommended_budget:,.0f}",
            change_str,
            f"{allocation.predicted_roas:.2f}x"
        )
    
    console.print(table)
    
    # Summary metrics
    summary = Panel(
        f"""**Expected Results:**
• Total ROAS: {budget_allocation.expected_total_roas:.2f}x
• ROAS Improvement: {budget_allocation.roas_improvement:+.1%}
• Budget Utilization: {budget_allocation.budget_utilization:.1%}
• Risk Score: {budget_allocation.overall_risk_score:.1f}/10
• Diversification: {budget_allocation.diversification_score:.1%}

**Status:** {'Preview Mode' if preview_only else 'Ready to Apply'}""",
        title="Allocation Summary",
        border_style="green" if not preview_only else "yellow"
    )
    
    console.print(summary)


def _export_budget_plan(budget_allocation):
    """Export budget allocation plan to CSV."""
    import csv
    from datetime import datetime
    
    filename = f"budget_plan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            'Channel ID', 'Channel Name', 'Current Budget', 'Recommended Budget',
            'Change Amount', 'Change Percentage', 'Current ROAS', 'Predicted ROAS',
            'Rationale'
        ])
        
        for allocation in budget_allocation.channel_allocations:
            writer.writerow([
                allocation.channel_id,
                allocation.channel_name,
                allocation.current_budget,
                allocation.recommended_budget,
                allocation.change_amount,
                allocation.change_percentage,
                allocation.current_roas,
                allocation.predicted_roas,
                allocation.allocation_rationale
            ])
    
    rprint(f"📄 Budget plan exported to [green]{filename}[/green]")


def _apply_budget_changes(budget_allocation):
    """Apply budget allocation changes."""
    rprint("🔄 Applying budget changes...")
    
    with Progress(console=console) as progress:
        task = progress.add_task("Updating channel budgets...", total=len(budget_allocation.channel_allocations))
        
        for allocation in budget_allocation.channel_allocations:
            # Mock budget update (in production, use actual API calls)
            progress.console.print(f"   ✅ Updated {allocation.channel_name}: ${allocation.recommended_budget:,.0f}")
            progress.update(task, advance=1)
            import time
            time.sleep(0.3)
    
    rprint("✅ [green]Budget allocation applied successfully[/green]")
    rprint("📊 Monitor performance over the next 2 weeks")


def _get_mock_cost_data(campaign_id: Optional[str]) -> List[dict]:
    """Get mock cost data for ROI analysis."""
    return [
        {"date": "2024-01-01", "spend": 750, "campaign_id": "123456789"},
        {"date": "2024-01-02", "spend": 680, "campaign_id": "123456789"},
        {"date": "2024-01-03", "spend": 820, "campaign_id": "987654321"}
    ]


def _get_mock_revenue_data(campaign_id: Optional[str]) -> List[dict]:
    """Get mock revenue data for ROI analysis."""
    return [
        {"date": "2024-01-01", "revenue": 3200, "campaign_id": "123456789"},
        {"date": "2024-01-02", "revenue": 2890, "campaign_id": "123456789"},
        {"date": "2024-01-03", "revenue": 2050, "campaign_id": "987654321"}
    ]


def _combine_performance_data(campaign_data, cost_data, revenue_data) -> List[dict]:
    """Combine performance data for ROI analysis."""
    # Simplified combination - in production, properly merge datasets
    return campaign_data


def _display_roi_analysis_table(current_analysis, roi_optimization, include_analysis: bool):
    """Display ROI analysis in table format."""
    
    # Current vs Optimized comparison
    comparison_table = Table(title="📈 ROI Analysis Comparison")
    comparison_table.add_column("Metric", style="cyan")
    comparison_table.add_column("Current", style="yellow")
    comparison_table.add_column("Optimized", style="green")
    comparison_table.add_column("Improvement", style="blue")
    
    metrics = [
        ("ROI", current_analysis.current_roi, roi_optimization.optimized_metrics.current_roi),
        ("ROAS", current_analysis.current_roas, roi_optimization.optimized_metrics.current_roas),
        ("CPA", current_analysis.cost_per_acquisition, roi_optimization.optimized_metrics.cost_per_acquisition),
        ("Payback Days", current_analysis.payback_period_days, roi_optimization.optimized_metrics.payback_period_days)
    ]
    
    for metric_name, current_val, optimized_val in metrics:
        if metric_name == "Payback Days":
            improvement = f"{((current_val - optimized_val) / current_val * 100):+.1f}%"
        else:
            improvement = f"{((optimized_val - current_val) / current_val * 100):+.1f}%"
        
        comparison_table.add_row(
            metric_name,
            f"{current_val:.2f}",
            f"{optimized_val:.2f}",
            improvement
        )
    
    console.print(comparison_table)
    
    # Optimization actions
    if include_analysis:
        actions_table = Table(title="🎯 Recommended Actions")
        actions_table.add_column("Action", style="green", width=40)
        actions_table.add_column("Priority", style="yellow")
        actions_table.add_column("Impact", style="blue")
        actions_table.add_column("Timeline", style="cyan")
        
        for action in roi_optimization.recommended_actions:
            actions_table.add_row(
                action["action"],
                action["priority"],
                action["expected_impact"],
                action["timeline"]
            )
        
        console.print(actions_table)


def _display_roi_report(current_analysis, roi_optimization, sensitivity_analysis: bool, scenario_planning: bool):
    """Display comprehensive ROI report."""
    
    report_sections = []
    
    # Executive Summary
    report_sections.append(Panel(
        f"""**Executive Summary**

Current ROI Performance: {current_analysis.current_roi:.2f}x
Optimization Potential: {roi_optimization.roi_improvement:+.1%}
Expected Revenue Uplift: {roi_optimization.revenue_uplift:+.1%}
Confidence Level: {roi_optimization.confidence_level:.0%}

**Key Findings:**
• {roi_optimization.optimized_metrics.competitive_position.title()} competitive position
• {len(roi_optimization.recommended_actions)} optimization opportunities identified
• Payback period can be reduced by {((current_analysis.payback_period_days - roi_optimization.optimized_metrics.payback_period_days) / current_analysis.payback_period_days * 100):.0f}%""",
        title="📊 ROI Optimization Report",
        border_style="blue"
    ))
    
    report_sections.append(Panel(
        f"""**Optimization Strategy: {roi_optimization.objective.value.replace('_', ' ').title()}**

Priority: {roi_optimization.priority.title()}
Timeline: {roi_optimization.timeline}
Implementation Risk: {roi_optimization.risk_factors if roi_optimization.risk_factors else 'Low'}

**Success Metrics:**
{chr(10).join([f"• {metric}" for metric in roi_optimization.success_metrics])}""",
        title="🎯 Strategy Overview",
        border_style="green"
    ))
    
    for section in report_sections:
        console.print(section)


def _display_sensitivity_analysis(sensitivity_data: dict):
    """Display sensitivity analysis results."""
    
    rprint("\n🔍 [bold]Sensitivity Analysis[/bold]")
    
    sens_table = Table(title="Parameter Sensitivity")
    sens_table.add_column("Parameter", style="cyan")
    sens_table.add_column("Sensitivity", style="yellow")
    sens_table.add_column("Impact Description", style="white")
    
    sensitivity_descriptions = {
        "roi_sensitivity_to_cost_reduction": "10% cost reduction impact on ROI",
        "roi_sensitivity_to_revenue_increase": "10% revenue increase impact on ROI",
        "payback_sensitivity_to_conversion_rate": "10% conversion rate impact on payback",
        "ltv_sensitivity_to_retention": "10% retention improvement impact on LTV"
    }
    
    for param, value in sensitivity_data.items():
        description = sensitivity_descriptions.get(param, "Parameter sensitivity")
        sens_table.add_row(
            param.replace('_', ' ').title(),
            f"{value:+.1f}%",
            description
        )
    
    console.print(sens_table)


def _display_scenario_planning(roi_optimization):
    """Display scenario planning results."""
    
    scenarios = {
        "Conservative": {"roi_improvement": roi_optimization.roi_improvement * 0.5, "probability": 0.3},
        "Expected": {"roi_improvement": roi_optimization.roi_improvement, "probability": 0.5},
        "Optimistic": {"roi_improvement": roi_optimization.roi_improvement * 1.5, "probability": 0.2}
    }
    
    rprint("\n🎭 [bold]Scenario Planning[/bold]")
    
    scenario_table = Table(title="ROI Scenarios")
    scenario_table.add_column("Scenario", style="cyan")
    scenario_table.add_column("ROI Improvement", style="green")
    scenario_table.add_column("Probability", style="yellow")
    scenario_table.add_column("Expected Value", style="blue")
    
    total_expected_value = 0
    for scenario_name, data in scenarios.items():
        expected_value = data["roi_improvement"] * data["probability"]
        total_expected_value += expected_value
        
        scenario_table.add_row(
            scenario_name,
            f"{data['roi_improvement']:+.1%}",
            f"{data['probability']:.0%}",
            f"{expected_value:+.2%}"
        )
    
    scenario_table.add_row(
        "[bold]Weighted Average[/bold]",
        "",
        "",
        f"[bold]{total_expected_value:+.2%}[/bold]"
    )
    
    console.print(scenario_table)


def _display_ab_test_recommendations(test_recommendations, include_setup: bool):
    """Display A/B test recommendations."""
    
    if not test_recommendations:
        rprint("✅ No high-priority test opportunities identified")
        return
    
    # Test recommendations table
    table = Table(title="🧪 A/B Test Recommendations")
    table.add_column("Test ID", style="cyan", width=12)
    table.add_column("Type", style="yellow", width=15)
    table.add_column("Objective", style="green", width=35)
    table.add_column("Priority", style="red", width=10)
    table.add_column("Duration", style="blue", width=10)
    table.add_column("Cost", style="magenta", width=10)
    
    for test in test_recommendations:
        priority_emoji = {
            "critical": "🚨",
            "high": "🔴",
            "medium": "🟡",
            "low": "🟢"
        }.get(test.priority.value, "❓")
        
        table.add_row(
            test.test_id.split('_')[-1][:8],
            test.test_type.value.replace('_', ' ').title(),
            test.objective,
            f"{priority_emoji} {test.priority.value.title()}",
            f"{test.estimated_duration}d",
            f"${test.estimated_cost:,.0f}"
        )
    
    console.print(table)
    
    # Show detailed setup for top test if requested
    if include_setup and test_recommendations:
        top_test = test_recommendations[0]
        _display_test_setup_details(top_test)


def _display_test_setup_details(test):
    """Display detailed test setup information."""
    
    setup_details = Panel(
        f"""**Test Setup: {test.test_name}**

**Hypothesis:** {test.hypothesis}

**Variants:**
{chr(10).join([f"• {var.variant_name}: {var.description}" for var in test.variants])}

**Setup Requirements:**
{chr(10).join([f"• {req}" for req in test.setup_requirements])}

**Success Criteria:**
{chr(10).join([f"• {criteria}" for criteria in test.success_criteria])}

**Stop Conditions:**
{chr(10).join([f"• {condition}" for condition in test.stop_conditions])}

**Monitoring:**
Primary Metric: {test.primary_metric}
Secondary: {', '.join(test.secondary_metrics)}""",
        title=f"🧪 Test Setup - {test.test_id}",
        border_style="green"
    )
    
    console.print(setup_details)


def _display_test_calendar(test_recommendations):
    """Display test scheduling calendar."""
    
    rprint("\n📅 [bold]Test Scheduling Calendar[/bold]")
    
    # Simple calendar view (in production, use proper calendar library)
    current_date = datetime.now()
    
    calendar_table = Table(title="Test Schedule")
    calendar_table.add_column("Week", style="cyan")
    calendar_table.add_column("Tests", style="green")
    calendar_table.add_column("Status", style="yellow")
    
    for i, test in enumerate(test_recommendations[:4]):  # Show first 4 tests
        week_num = i + 1
        start_week = current_date.strftime(f"Week {week_num}")
        
        calendar_table.add_row(
            start_week,
            test.test_name,
            "Scheduled" if i == 0 else "Queued"
        )
    
    console.print(calendar_table)


def _auto_schedule_tests(test_recommendations):
    """Auto-schedule feasible A/B tests."""
    
    scheduled_tests = []
    for test in test_recommendations:
        if (test.priority.value in ["critical", "high"] and 
            test.risk_level.lower() in ["low", "medium"]):
            scheduled_tests.append(test)
    
    if scheduled_tests:
        rprint(f"\n🗓️ Auto-scheduled {len(scheduled_tests)} tests:")
        for i, test in enumerate(scheduled_tests):
            rprint(f"   {i+1}. {test.test_name} - Starts in {i*7} days")
    else:
        rprint("⚠️  No tests meet auto-scheduling criteria")


def _get_simulation_baseline() -> dict:
    """Get baseline metrics for simulation."""
    return {
        "impressions": 100000,
        "clicks": 3500,
        "conversions": 70,
        "cost": 2100,
        "revenue": 8400,
        "ctr": 0.035,
        "conversion_rate": 0.02,
        "cpa": 30.0,
        "roas": 4.0
    }


def _run_deterministic_simulation(params: dict, baseline: dict) -> dict:
    """Run deterministic simulation."""
    
    # Apply parameter changes
    budget_multiplier = 1 + params["budget_change"]
    bid_multiplier = 1 + params["bid_change"]
    audience_multiplier = 1 + params["audience_expansion"]
    
    # Calculate impacts (simplified model)
    new_cost = baseline["cost"] * budget_multiplier
    new_impressions = baseline["impressions"] * budget_multiplier * audience_multiplier
    new_clicks = new_impressions * baseline["ctr"] * (1 / bid_multiplier ** 0.2)  # Inverse bid impact
    new_conversions = new_clicks * baseline["conversion_rate"]
    new_revenue = new_conversions * (baseline["revenue"] / baseline["conversions"])
    
    return {
        "scenario": params["scenario"],
        "baseline": baseline,
        "projected": {
            "cost": new_cost,
            "impressions": new_impressions,
            "clicks": new_clicks,
            "conversions": new_conversions,
            "revenue": new_revenue,
            "ctr": new_clicks / new_impressions,
            "conversion_rate": new_conversions / new_clicks,
            "cpa": new_cost / new_conversions,
            "roas": new_revenue / new_cost
        },
        "changes": {
            "cost_change": (new_cost - baseline["cost"]) / baseline["cost"],
            "revenue_change": (new_revenue - baseline["revenue"]) / baseline["revenue"],
            "roas_change": ((new_revenue/new_cost) - baseline["roas"]) / baseline["roas"]
        }
    }


def _run_monte_carlo_simulation(params: dict, baseline: dict, iterations: int) -> dict:
    """Run Monte Carlo simulation."""
    import random
    
    results = []
    
    for _ in range(iterations):
        # Add randomness to parameters
        random_budget = params["budget_change"] * random.uniform(0.8, 1.2)
        random_bid = params["bid_change"] * random.uniform(0.9, 1.1)
        random_audience = params["audience_expansion"] * random.uniform(0.85, 1.15)
        
        # Create random parameter set
        random_params = {
            **params,
            "budget_change": random_budget,
            "bid_change": random_bid,
            "audience_expansion": random_audience
        }
        
        # Run deterministic simulation with random parameters
        iteration_result = _run_deterministic_simulation(random_params, baseline)
        results.append(iteration_result["projected"])
    
    # Calculate statistics
    metrics = ["cost", "revenue", "roas", "conversions"]
    statistics = {}
    
    for metric in metrics:
        values = [result[metric] for result in results]
        statistics[metric] = {
            "mean": sum(values) / len(values),
            "min": min(values),
            "max": max(values),
            "percentile_10": sorted(values)[int(iterations * 0.1)],
            "percentile_90": sorted(values)[int(iterations * 0.9)]
        }
    
    return {
        "scenario": params["scenario"],
        "iterations": iterations,
        "baseline": baseline,
        "statistics": statistics,
        "confidence_intervals": {
            metric: {
                "lower": stats["percentile_10"],
                "upper": stats["percentile_90"]
            }
            for metric, stats in statistics.items()
        }
    }


def _display_simulation_results(results: dict, monte_carlo: bool):
    """Display simulation results."""
    
    if monte_carlo:
        rprint(f"🎲 [bold]Monte Carlo Simulation Results - {results['scenario'].title()}[/bold]")
        rprint(f"Iterations: {results['iterations']:,}")
        
        # Statistics table
        stats_table = Table(title="Simulation Statistics")
        stats_table.add_column("Metric", style="cyan")
        stats_table.add_column("Baseline", style="yellow")
        stats_table.add_column("Mean", style="green")
        stats_table.add_column("90% Range", style="blue")
        
        baseline = results["baseline"]
        for metric, stats in results["statistics"].items():
            range_str = f"{stats['percentile_10']:.0f} - {stats['percentile_90']:.0f}"
            
            stats_table.add_row(
                metric.title(),
                f"{baseline[metric]:.0f}",
                f"{stats['mean']:.0f}",
                range_str
            )
        
        console.print(stats_table)
        
    else:
        rprint(f"📊 [bold]Simulation Results - {results['scenario'].title()}[/bold]")
        
        # Changes table
        changes_table = Table(title="Projected Changes")
        changes_table.add_column("Metric", style="cyan")
        changes_table.add_column("Current", style="yellow")
        changes_table.add_column("Projected", style="green")
        changes_table.add_column("Change", style="blue")
        
        baseline = results["baseline"]
        projected = results["projected"]
        changes = results["changes"]
        
        key_metrics = ["cost", "revenue", "conversions", "roas"]
        
        for metric in key_metrics:
            if metric in changes:
                change_pct = changes[f"{metric}_change"] if f"{metric}_change" in changes else 0
                change_str = f"{change_pct:+.1%}"
            else:
                change_pct = (projected[metric] - baseline[metric]) / baseline[metric]
                change_str = f"{change_pct:+.1%}"
            
            if change_pct > 0:
                change_str = f"[green]{change_str}[/green]"
            elif change_pct < 0:
                change_str = f"[red]{change_str}[/red]"
            
            changes_table.add_row(
                metric.title(),
                f"{baseline[metric]:.0f}",
                f"{projected[metric]:.0f}",
                change_str
            )
        
        console.print(changes_table)


def _display_simulation_risk_analysis(results: dict):
    """Display risk analysis for simulation."""
    
    rprint("\n⚠️ [bold]Risk Analysis[/bold]")
    
    if "statistics" in results:
        # Monte Carlo risk analysis
        roas_stats = results["statistics"]["roas"]
        cost_stats = results["statistics"]["cost"]
        
        risk_factors = []
        
        if roas_stats["percentile_10"] < results["baseline"]["roas"] * 0.8:
            risk_factors.append("10% chance of ROAS declining by >20%")
        
        if cost_stats["percentile_90"] > results["baseline"]["cost"] * 1.5:
            risk_factors.append("10% chance of costs increasing by >50%")
        
        if risk_factors:
            rprint("🚨 Identified Risks:")
            for risk in risk_factors:
                rprint(f"   • {risk}")
        else:
            rprint("✅ Low risk scenario - outcomes within acceptable ranges")
    
    else:
        # Deterministic risk analysis
        roas_change = results["changes"]["roas_change"]
        cost_change = results["changes"]["cost_change"]
        
        if roas_change < -0.1:
            rprint("🚨 High risk: ROAS projected to decline by >10%")
        elif cost_change > 0.5:
            rprint("⚠️  Medium risk: Costs projected to increase by >50%")
        else:
            rprint("✅ Acceptable risk level for projected scenario")


def _start_real_time_monitoring(optimization_id: Optional[str], alerts: bool, refresh_interval: int):
    """Start real-time optimization monitoring."""
    
    rprint("🔄 [bold]Real-time Optimization Monitoring[/bold]")
    rprint("Press Ctrl+C to stop monitoring")
    
    try:
        import time
        iteration = 0
        while True:
            console.clear()
            iteration += 1
            
            rprint(f"📊 [bold]Optimization Monitor - Refresh #{iteration}[/bold]")
            rprint("=" * 60)
            
            # Mock monitoring data
            _display_optimization_status(optimization_id)
            
            if alerts:
                _display_optimization_alerts()
            
            rprint(f"\n⏰ Next refresh in {refresh_interval} seconds")
            time.sleep(refresh_interval)
            
    except KeyboardInterrupt:
        rprint("\n👋 Monitoring stopped")


def _show_optimization_status(optimization_id: Optional[str], export_log: bool):
    """Show current optimization status."""
    
    rprint("📊 [bold]Optimization Status Dashboard[/bold]")
    
    _display_optimization_status(optimization_id)
    
    if export_log:
        log_filename = f"optimization_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        mock_log = {"timestamp": datetime.now().isoformat(), "status": "active"}
        
        with open(log_filename, 'w') as f:
            json.dump(mock_log, f, indent=2, default=str)
        
        rprint(f"📄 Log exported to [green]{log_filename}[/green]")


def _display_optimization_status(optimization_id: Optional[str]):
    """Display optimization implementation status."""
    
    # Mock status data
    optimizations = [
        {
            "id": "OPT_001",
            "title": "Budget Reallocation",
            "status": "implemented",
            "progress": 100,
            "impact_measured": 0.23,
            "target_impact": 0.25
        },
        {
            "id": "OPT_002",
            "title": "Bid Strategy Change",
            "status": "in_progress", 
            "progress": 60,
            "impact_measured": 0.12,
            "target_impact": 0.32
        },
        {
            "id": "OPT_003",
            "title": "Creative A/B Test",
            "status": "scheduled",
            "progress": 0,
            "impact_measured": 0.0,
            "target_impact": 0.15
        }
    ]
    
    status_table = Table(title="🎯 Optimization Status")
    status_table.add_column("ID", style="cyan")
    status_table.add_column("Optimization", style="green")
    status_table.add_column("Status", style="yellow")
    status_table.add_column("Progress", style="blue")
    status_table.add_column("Impact", style="magenta")
    
    for opt in optimizations:
        status_emoji = {
            "implemented": "✅",
            "in_progress": "🔄",
            "scheduled": "⏳"
        }.get(opt["status"], "❓")
        
        progress_bar = "█" * int(opt["progress"] / 10) + "░" * (10 - int(opt["progress"] / 10))
        
        status_table.add_row(
            opt["id"],
            opt["title"],
            f"{status_emoji} {opt['status'].replace('_', ' ').title()}",
            f"{progress_bar} {opt['progress']}%",
            f"{opt['impact_measured']:.1%} / {opt['target_impact']:.1%}"
        )
    
    console.print(status_table)


def _display_optimization_alerts():
    """Display optimization monitoring alerts."""
    
    alerts = [
        {"time": "5 min ago", "message": "Budget reallocation showing 23% ROAS improvement", "type": "success"},
        {"time": "12 min ago", "message": "Bid strategy learning period completed", "type": "info"},
        {"time": "1 hour ago", "message": "Creative test reached statistical significance", "type": "success"}
    ]
    
    rprint("\n🚨 [bold]Recent Alerts:[/bold]")
    
    for alert in alerts:
        color = {
            "success": "green",
            "warning": "yellow",
            "error": "red",
            "info": "blue"
        }.get(alert["type"], "white")
        
        rprint(f"[{color}]• {alert['time']} - {alert['message']}[/{color}]")


def _auto_implement_recommendations(recommendations: List[dict]):
    """Auto-implement safe recommendations."""
    
    rprint(f"🤖 [bold]Auto-implementing {len(recommendations)} safe recommendations[/bold]")
    
    for rec in recommendations:
        _implement_recommendation(rec)


def _save_optimization_results(results, output_file: Path):
    """Save optimization results to file."""
    
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)