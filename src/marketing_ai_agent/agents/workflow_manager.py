"""Workflow Manager for coordinating agent execution and state management."""

import logging
from datetime import datetime
from typing import Any

from langchain_anthropic import ChatAnthropic

from ..models.exporters import ExportConfig, ReportGenerator
from .orchestrator import AgentState, OrchestratorAgent

logger = logging.getLogger(__name__)


class WorkflowManager:
    """Manages the complete marketing analysis workflow execution."""

    def __init__(self, llm_config: dict[str, Any] | None = None):
        """
        Initialize workflow manager.

        Args:
            llm_config: Configuration for language models
        """
        self.llm_config = llm_config or {
            "model": "claude-3-sonnet-20240229",
            "temperature": 0.3,
            "max_tokens": 4000,
        }

        # Initialize orchestrator
        self.orchestrator = OrchestratorAgent(llm=ChatAnthropic(**self.llm_config))

        # Initialize report generator
        self.report_generator = ReportGenerator(
            config=ExportConfig(output_dir="./reports", include_timestamp=True)
        )

        self.execution_history: list[dict[str, Any]] = []

    async def execute_query(
        self, user_query: str, export_format: str = "markdown"
    ) -> dict[str, Any]:
        """
        Execute a complete marketing analysis query.

        Args:
            user_query: User's marketing question
            export_format: Output format (markdown, json, xlsx)

        Returns:
            Execution results with report path
        """
        execution_start = datetime.now()

        try:
            logger.info(f"Starting workflow execution for query: {user_query}")

            # Execute workflow
            final_state = await self.orchestrator.execute_workflow(user_query)

            # Generate report
            report_path = await self._generate_report(final_state, export_format)

            # Record execution
            execution_record = self._create_execution_record(
                user_query, final_state, execution_start, report_path
            )

            self.execution_history.append(execution_record)

            logger.info(f"Workflow completed successfully. Report: {report_path}")

            return {
                "success": len(final_state.error_messages) == 0,
                "report_path": report_path,
                "insights_count": len(final_state.analysis_insights),
                "recommendations_count": len(final_state.recommendations),
                "errors": final_state.error_messages,
                "execution_time": (datetime.now() - execution_start).total_seconds(),
                "state": final_state,
            }

        except Exception as e:
            logger.error(f"Workflow execution failed: {e}")
            return {
                "success": False,
                "report_path": None,
                "insights_count": 0,
                "recommendations_count": 0,
                "errors": [f"Workflow execution error: {str(e)}"],
                "execution_time": (datetime.now() - execution_start).total_seconds(),
                "state": None,
            }

    async def _generate_report(
        self, state: AgentState, export_format: str
    ) -> str | None:
        """Generate final report from workflow results."""
        try:
            if export_format.lower() == "markdown":
                return await self._generate_markdown_report(state)
            elif export_format.lower() == "json":
                return await self._generate_json_report(state)
            elif export_format.lower() == "xlsx":
                return await self._generate_excel_report(state)
            else:
                logger.warning(
                    f"Unknown export format: {export_format}, defaulting to markdown"
                )
                return await self._generate_markdown_report(state)

        except Exception as e:
            logger.error(f"Report generation failed: {e}")
            return None

    async def _generate_markdown_report(self, state: AgentState) -> str:
        """Generate Markdown report."""
        from pathlib import Path

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"marketing_analysis_{timestamp}.md"
        reports_dir = Path("./reports")
        reports_dir.mkdir(exist_ok=True)
        filepath = reports_dir / filename

        # Build markdown content
        content = self._build_markdown_content(state)

        # Write to file
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)

        return str(filepath)

    async def _generate_json_report(self, state: AgentState) -> str:
        """Generate JSON report."""
        import json
        from pathlib import Path

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"marketing_analysis_{timestamp}.json"
        reports_dir = Path("./reports")
        reports_dir.mkdir(exist_ok=True)
        filepath = reports_dir / filename

        # Convert state to JSON-serializable format
        report_data = {
            "query": state.user_query,
            "date_range": [
                state.date_range[0].isoformat() if state.date_range else None,
                state.date_range[1].isoformat() if state.date_range else None,
            ],
            "analysis_insights": state.analysis_insights,
            "recommendations": state.recommendations,
            "errors": state.error_messages,
            "completed_steps": state.completed_steps,
            "generated_at": datetime.now().isoformat(),
        }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=2, default=str)

        return str(filepath)

    async def _generate_excel_report(self, state: AgentState) -> str:
        """Generate Excel report with multiple sheets."""
        try:
            # Use the existing report generator for Excel output
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"marketing_analysis_{timestamp}"

            # Prepare data for Excel export
            campaigns = state.google_ads_data or []
            performance_data = state.performance_metrics or []
            analytics_data = state.ga4_data or []

            if campaigns or performance_data or analytics_data:
                report_path = (
                    self.report_generator.multi_exporter.export_marketing_report(
                        campaigns=campaigns,
                        performance_data=performance_data,
                        analytics_data=analytics_data,
                        filename=filename,
                    )
                )
                return report_path
            else:
                # Generate basic Excel with insights and recommendations
                return await self._generate_insights_excel(state, filename)

        except Exception as e:
            logger.error(f"Excel report generation failed: {e}")
            # Fallback to JSON
            return await self._generate_json_report(state)

    async def _generate_insights_excel(self, state: AgentState, filename: str) -> str:
        """Generate Excel report focused on insights and recommendations."""
        from pathlib import Path

        import pandas as pd

        reports_dir = Path("./reports")
        reports_dir.mkdir(exist_ok=True)
        filepath = reports_dir / f"{filename}.xlsx"

        with pd.ExcelWriter(filepath, engine="openpyxl") as writer:
            # Insights sheet
            if state.analysis_insights:
                insights_data = []
                for category, data in state.analysis_insights.items():
                    insights_data.append({"Category": category, "Data": str(data)})

                pd.DataFrame(insights_data).to_excel(
                    writer, sheet_name="Insights", index=False
                )

            # Recommendations sheet
            if state.recommendations:
                recommendations_data = []
                for i, rec in enumerate(state.recommendations, 1):
                    recommendations_data.append({"Priority": i, "Recommendation": rec})

                pd.DataFrame(recommendations_data).to_excel(
                    writer, sheet_name="Recommendations", index=False
                )

            # Summary sheet
            summary_data = [
                {
                    "Query": state.user_query,
                    "Generated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "Insights": len(state.analysis_insights),
                    "Recommendations": len(state.recommendations),
                    "Errors": len(state.error_messages),
                }
            ]

            pd.DataFrame(summary_data).to_excel(
                writer, sheet_name="Summary", index=False
            )

        return str(filepath)

    def _build_markdown_content(self, state: AgentState) -> str:
        """Build comprehensive markdown report content."""
        content_parts = []

        # Header
        content_parts.append("# Marketing Analysis Report\n")
        content_parts.append(
            f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        )
        content_parts.append(f"**Query:** {state.user_query}\n")

        if state.date_range:
            start_date = state.date_range[0].strftime("%Y-%m-%d")
            end_date = state.date_range[1].strftime("%Y-%m-%d")
            content_parts.append(f"**Date Range:** {start_date} to {end_date}\n")

        content_parts.append("\n---\n\n")

        # Executive Summary
        content_parts.append("## Executive Summary\n\n")
        if state.analysis_insights:
            summary_points = []

            # Add key metrics if available
            if "campaign_efficiency" in state.analysis_insights:
                eff_data = state.analysis_insights["campaign_efficiency"]
                if not eff_data.get("error"):
                    metrics = eff_data.get("metrics", {})
                    summary_points.append(
                        f"- **Average CPC:** ${metrics.get('average_cpc', 0):.2f}"
                    )
                    summary_points.append(
                        f"- **Average CTR:** {metrics.get('average_ctr', 0):.2f}%"
                    )
                    summary_points.append(
                        f"- **Active Campaigns:** {eff_data.get('active_campaigns', 0)}"
                    )

            if "spend_analysis" in state.analysis_insights:
                spend_data = state.analysis_insights["spend_analysis"]
                if not spend_data.get("error"):
                    summary_points.append(
                        f"- **Total Spend:** ${spend_data.get('total_spend', 0):,.2f}"
                    )
                    summary_points.append(
                        f"- **Total Conversions:** {spend_data.get('total_conversions', 0):,}"
                    )

            if summary_points:
                content_parts.extend(summary_points)
                content_parts.append("\n")

        # Analysis Insights
        if state.analysis_insights:
            content_parts.append("## Analysis Insights\n\n")

            for category, data in state.analysis_insights.items():
                if data.get("error"):
                    continue

                content_parts.append(f"### {category.replace('_', ' ').title()}\n\n")

                # Format specific insight types
                if category == "campaign_efficiency":
                    content_parts.append(self._format_efficiency_insights(data))
                elif category == "spend_analysis":
                    content_parts.append(self._format_spend_insights(data))
                elif category == "channel_performance":
                    content_parts.append(self._format_channel_insights(data))
                elif category == "ai_insights":
                    content_parts.append(self._format_ai_insights(data))
                else:
                    content_parts.append(f"```json\n{data}\n```\n\n")

        # Recommendations
        if state.recommendations:
            content_parts.append("## Recommendations\n\n")

            for i, recommendation in enumerate(state.recommendations, 1):
                content_parts.append(f"### {i}. {recommendation}\n")

        # Errors and Warnings
        if state.error_messages:
            content_parts.append("## Issues and Limitations\n\n")
            for error in state.error_messages:
                content_parts.append(f"⚠️ {error}\n")
            content_parts.append("\n")

        # Footer
        content_parts.append("---\n")
        content_parts.append(
            "*This report was generated by the Marketing Analytics AI Agent*\n"
        )

        return "".join(content_parts)

    def _format_efficiency_insights(self, data: dict[str, Any]) -> str:
        """Format campaign efficiency insights."""
        content = []

        metrics = data.get("metrics", {})
        if metrics:
            content.append("**Performance Metrics:**\n")
            content.append(f"- Average CPC: ${metrics.get('average_cpc', 0):.2f}\n")
            content.append(f"- Median CPC: ${metrics.get('median_cpc', 0):.2f}\n")
            content.append(f"- Average CTR: {metrics.get('average_ctr', 0):.2f}%\n")
            content.append(
                f"- Average Conversion Rate: {metrics.get('average_conversion_rate', 0):.2f}%\n\n"
            )

        top_performers = data.get("top_performers", [])
        if top_performers:
            content.append("**Top Performing Campaigns:**\n")
            for campaign in top_performers[:3]:
                content.append(
                    f"- {campaign['name']}: CPC ${campaign['cpc']:.2f}, "
                    f"CTR {campaign.get('ctr', 0):.2f}%, "
                    f"{campaign.get('conversions', 0)} conversions\n"
                )
            content.append("\n")

        return "".join(content)

    def _format_spend_insights(self, data: dict[str, Any]) -> str:
        """Format spend analysis insights."""
        content = []

        content.append(f"**Total Spend:** ${data.get('total_spend', 0):,.2f}\n")
        content.append(
            f"**Total Conversions:** {data.get('total_conversions', 0):,}\n\n"
        )

        campaign_distribution = data.get("campaign_distribution", [])
        if campaign_distribution:
            content.append("**Top Spending Campaigns:**\n")
            for campaign in campaign_distribution[:5]:
                content.append(
                    f"- {campaign['name']}: ${campaign['spend']:,.0f} "
                    f"({campaign['spend_share']:.1f}%)\n"
                )
            content.append("\n")

        return "".join(content)

    def _format_channel_insights(self, data: dict[str, Any]) -> str:
        """Format channel performance insights."""
        content = []

        content.append(f"**Total Sessions:** {data.get('total_sessions', 0):,}\n\n")

        channel_performance = data.get("channel_performance", {})
        if channel_performance:
            content.append("**Channel Performance:**\n")
            for channel, metrics in list(channel_performance.items())[:5]:
                content.append(
                    f"- **{channel}:** {metrics['sessions']:,} sessions "
                    f"({metrics['session_share']:.1f}%), "
                    f"{metrics.get('conversion_rate', 0):.2f}% conversion rate\n"
                )
            content.append("\n")

        return "".join(content)

    def _format_ai_insights(self, data: list[str]) -> str:
        """Format AI-generated insights."""
        content = []

        for i, insight in enumerate(data, 1):
            content.append(f"{i}. {insight}\n")

        content.append("\n")
        return "".join(content)

    def _create_execution_record(
        self,
        query: str,
        state: AgentState,
        start_time: datetime,
        report_path: str | None,
    ) -> dict[str, Any]:
        """Create execution record for history tracking."""
        return {
            "timestamp": start_time.isoformat(),
            "query": query,
            "duration_seconds": (datetime.now() - start_time).total_seconds(),
            "success": len(state.error_messages) == 0,
            "insights_generated": len(state.analysis_insights),
            "recommendations_generated": len(state.recommendations),
            "errors_count": len(state.error_messages),
            "completed_steps": state.completed_steps,
            "report_path": report_path,
        }

    def get_execution_history(self, limit: int = 10) -> list[dict[str, Any]]:
        """Get recent execution history."""
        return self.execution_history[-limit:] if limit > 0 else self.execution_history

    async def health_check(self) -> dict[str, Any]:
        """Perform system health check."""
        try:
            # Test orchestrator initialization
            AgentState(user_query="test query")

            health_status = {
                "orchestrator": "healthy",
                "report_generator": "healthy",
                "execution_history_count": len(self.execution_history),
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
            }

            return health_status

        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            }
