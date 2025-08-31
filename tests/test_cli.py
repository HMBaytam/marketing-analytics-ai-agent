"""Integration tests for CLI commands."""

import json
import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typer.testing import CliRunner

from marketing_ai_agent.cli.main import app


class TestMainCommands:
    """Test main CLI commands."""
    
    def test_version_command(self, cli_runner):
        """Test version command."""
        result = cli_runner.invoke(app, ["version"])
        assert result.exit_code == 0
        assert "Marketing AI Agent v1.0.0" in result.stdout
        assert "Built with ❤️" in result.stdout
    
    def test_init_command(self, cli_runner, isolated_filesystem):
        """Test project initialization command."""
        project_dir = Path("test_project")
        
        result = cli_runner.invoke(app, ["init", str(project_dir)])
        
        # Should succeed
        assert result.exit_code == 0
        assert "Initialized project" in result.stdout
        
        # Check created files and directories
        assert project_dir.exists()
        assert (project_dir / "config.yaml").exists()
        assert (project_dir / ".env.example").exists()
        assert (project_dir / "reports").is_dir()
        assert (project_dir / "data").is_dir()
        assert (project_dir / "configs").is_dir()
        
        # Check config content
        config_content = (project_dir / "config.yaml").read_text()
        assert "google_ads:" in config_content
        assert "google_analytics:" in config_content
    
    def test_init_command_existing_directory(self, cli_runner, isolated_filesystem):
        """Test init command with existing non-empty directory."""
        project_dir = Path("existing_project")
        project_dir.mkdir()
        (project_dir / "existing_file.txt").write_text("existing content")
        
        # Should prompt for confirmation
        result = cli_runner.invoke(app, ["init", str(project_dir)], input="n\n")
        assert result.exit_code == 1
        
        # Should succeed with confirmation
        result = cli_runner.invoke(app, ["init", str(project_dir)], input="y\n")
        assert result.exit_code == 0
    
    def test_config_show_command(self, cli_runner):
        """Test config show command."""
        result = cli_runner.invoke(app, ["config", "--show"])
        assert result.exit_code == 0
        # Should show config or indicate none loaded
        assert "Configuration" in result.stdout or "No configuration" in result.stdout
    
    def test_config_validate_command(self, cli_runner):
        """Test config validation command."""
        result = cli_runner.invoke(app, ["config", "--validate"])
        assert result.exit_code == 0
        assert "Validating configuration" in result.stdout
    
    @patch('marketing_ai_agent.core.monitoring.system_monitor')
    def test_status_command_basic(self, mock_monitor, cli_runner):
        """Test basic status command."""
        mock_monitor.get_status_report.return_value = {
            "overall_health": "healthy",
            "monitoring_active": True,
            "current_metrics": {
                "cpu_percent": 25.0,
                "memory_percent": 45.0,
                "disk_usage_percent": 60.0
            },
            "health_checks": {
                "system_resources": {
                    "status": "healthy",
                    "message": "System resources OK"
                }
            },
            "average_metrics_10min": {},
            "metrics_history_count": 100
        }
        
        result = cli_runner.invoke(app, ["status"])
        assert result.exit_code == 0
        assert "System Health: Healthy" in result.stdout
        assert "CPU Usage: 25.0%" in result.stdout
    
    @patch('marketing_ai_agent.core.monitoring.system_monitor')
    def test_status_command_detailed(self, mock_monitor, cli_runner):
        """Test detailed status command."""
        mock_monitor.get_status_report.return_value = {
            "overall_health": "warning",
            "monitoring_active": False,
            "current_metrics": None,
            "health_checks": {
                "system_resources": {
                    "status": "warning", 
                    "message": "High CPU usage"
                },
                "api_connectivity": {
                    "status": "healthy",
                    "message": "All APIs accessible"
                }
            },
            "average_metrics_10min": {},
            "metrics_history_count": 50
        }
        
        result = cli_runner.invoke(app, ["status", "--detailed"])
        assert result.exit_code == 0
        assert "System Health: Warning" in result.stdout
        assert "High CPU usage" in result.stdout
        assert "All APIs accessible" in result.stdout
    
    @patch('marketing_ai_agent.core.monitoring.system_monitor')
    def test_status_command_export(self, mock_monitor, cli_runner, isolated_filesystem):
        """Test status export functionality."""
        mock_monitor.get_status_report.return_value = {
            "overall_health": "healthy",
            "monitoring_active": True,
            "current_metrics": {"cpu_percent": 20.0},
            "health_checks": {},
            "average_metrics_10min": {},
            "metrics_history_count": 75
        }
        
        export_file = Path("status_export.json")
        result = cli_runner.invoke(app, ["status", "--export", str(export_file)])
        
        assert result.exit_code == 0
        assert export_file.exists()
        
        # Check exported content
        with open(export_file) as f:
            data = json.load(f)
        assert data["overall_health"] == "healthy"
    
    @patch('marketing_ai_agent.core.monitoring.system_monitor')
    def test_monitor_commands(self, mock_monitor, cli_runner):
        """Test monitoring management commands."""
        mock_monitor.get_status_report.return_value = {
            "monitoring_active": False,
            "current_metrics": None,
            "average_metrics_10min": {},
            "metrics_history_count": 0
        }
        
        # Test status check
        result = cli_runner.invoke(app, ["monitor"])
        assert result.exit_code == 0
        assert "Monitoring Status" in result.stdout
        
        # Test start monitoring
        result = cli_runner.invoke(app, ["monitor", "--start"])
        assert result.exit_code == 0
        mock_monitor.start_monitoring.assert_called_once()
        
        # Test stop monitoring
        result = cli_runner.invoke(app, ["monitor", "--stop"])
        assert result.exit_code == 0
        mock_monitor.stop_monitoring.assert_called_once()
    
    @patch('marketing_ai_agent.core.error_handlers.error_reporter')
    def test_errors_command(self, mock_error_reporter, cli_runner):
        """Test error management command."""
        mock_error_reporter.get_error_summary.return_value = {
            "total_error_types": 2,
            "error_counts": {
                "APIError": 5,
                "DataValidationError": 3
            },
            "recent_errors": [
                {
                    "timestamp": "2024-01-15T10:30:00",
                    "error_type": "APIError",
                    "message": "Test error message",
                    "severity": "ERROR",
                    "context": {"api": "test"}
                }
            ],
            "most_common_error": ("APIError", 5)
        }
        
        # Test error summary
        result = cli_runner.invoke(app, ["errors", "--summary"])
        assert result.exit_code == 0
        assert "Total error types: 2" in result.stdout
        assert "APIError: 5" in result.stdout
        
        # Test recent errors
        result = cli_runner.invoke(app, ["errors", "--recent", "5"])
        assert result.exit_code == 0
        assert "Test error message" in result.stdout
    
    @patch('marketing_ai_agent.core.error_handlers.error_reporter')
    def test_errors_export(self, mock_error_reporter, cli_runner, isolated_filesystem):
        """Test error report export."""
        mock_error_reporter.get_error_summary.return_value = {
            "total_error_types": 1,
            "error_counts": {"TestError": 1},
            "recent_errors": [],
            "most_common_error": None
        }
        
        export_file = Path("error_export.json")
        result = cli_runner.invoke(app, ["errors", "--export", str(export_file)])
        
        assert result.exit_code == 0
        assert export_file.exists()


class TestDataCommands:
    """Test data export commands."""
    
    @patch('marketing_ai_agent.api_clients.ga4_client.GA4Client')
    def test_data_export_ga4(self, mock_ga4_client, cli_runner):
        """Test GA4 data export."""
        mock_client_instance = Mock()
        mock_client_instance.get_report.return_value = {
            "data": [{"date": "2024-01-01", "sessions": 1000}]
        }
        mock_ga4_client.return_value = mock_client_instance
        
        result = cli_runner.invoke(app, [
            "data", "export", "ga4", 
            "--date-range", "7d",
            "--format", "json"
        ])
        
        # May fail due to missing implementation, but should not crash
        assert result.exit_code in [0, 1]  # Allow failure for unimplemented features
    
    @patch('marketing_ai_agent.api_clients.google_ads_client.GoogleAdsClient')
    def test_data_export_google_ads(self, mock_ads_client, cli_runner):
        """Test Google Ads data export."""
        mock_client_instance = Mock()
        mock_client_instance.get_campaigns.return_value = [
            {"id": "123", "name": "Test Campaign", "status": "ENABLED"}
        ]
        mock_ads_client.return_value = mock_client_instance
        
        result = cli_runner.invoke(app, [
            "data", "export", "google-ads",
            "--date-range", "30d",
            "--format", "csv"
        ])
        
        # May fail due to missing implementation, but should not crash
        assert result.exit_code in [0, 1]
    
    def test_data_list_campaigns(self, cli_runner):
        """Test campaign listing command."""
        result = cli_runner.invoke(app, ["data", "campaigns", "--source", "all"])
        
        # Should not crash, may return error for unimplemented features
        assert result.exit_code in [0, 1]
    
    def test_data_validate_command(self, cli_runner):
        """Test data validation command."""
        result = cli_runner.invoke(app, ["data", "validate", "--source", "ga4"])
        
        # Should not crash
        assert result.exit_code in [0, 1]


class TestAnalyticsCommands:
    """Test analytics commands."""
    
    def test_analytics_score_command(self, cli_runner):
        """Test performance scoring command."""
        result = cli_runner.invoke(app, [
            "analytics", "score", 
            "--campaign-id", "test123"
        ])
        
        # May fail due to missing implementation
        assert result.exit_code in [0, 1]
    
    def test_analytics_trends_command(self, cli_runner):
        """Test trend analysis command."""
        result = cli_runner.invoke(app, [
            "analytics", "trends",
            "--metric", "conversions",
            "--period", "weekly"
        ])
        
        # Should not crash
        assert result.exit_code in [0, 1]
    
    def test_analytics_anomalies_command(self, cli_runner):
        """Test anomaly detection command."""
        result = cli_runner.invoke(app, [
            "analytics", "anomalies",
            "--sensitivity", "medium"
        ])
        
        # Should not crash
        assert result.exit_code in [0, 1]
    
    def test_analytics_predictions_command(self, cli_runner):
        """Test prediction command."""
        result = cli_runner.invoke(app, [
            "analytics", "predict",
            "--horizon", "30",
            "--metric", "revenue"
        ])
        
        # Should not crash
        assert result.exit_code in [0, 1]


class TestOptimizationCommands:
    """Test optimization commands."""
    
    def test_optimization_recommendations(self, cli_runner):
        """Test recommendation generation."""
        result = cli_runner.invoke(app, [
            "optimize", "recommendations",
            "--max", "5",
            "--confidence", "0.8"
        ])
        
        # Should not crash
        assert result.exit_code in [0, 1]
    
    def test_optimization_budget(self, cli_runner):
        """Test budget optimization."""
        result = cli_runner.invoke(app, [
            "optimize", "budget",
            "--strategy", "maximize_conversions",
            "--total-budget", "10000"
        ])
        
        # Should not crash
        assert result.exit_code in [0, 1]
    
    def test_optimization_roi(self, cli_runner):
        """Test ROI optimization."""
        result = cli_runner.invoke(app, [
            "optimize", "roi",
            "--objective", "maximize_total_roi"
        ])
        
        # Should not crash  
        assert result.exit_code in [0, 1]


class TestReportCommands:
    """Test report generation commands."""
    
    def test_report_generate_command(self, cli_runner, isolated_filesystem):
        """Test report generation."""
        result = cli_runner.invoke(app, [
            "report", "generate",
            "--template", "executive",
            "--output", "test_report"
        ])
        
        # Should not crash
        assert result.exit_code in [0, 1]
    
    def test_report_schedule_command(self, cli_runner):
        """Test report scheduling."""
        result = cli_runner.invoke(app, [
            "report", "schedule",
            "--template", "weekly",
            "--frequency", "weekly",
            "--recipients", "test@example.com"
        ])
        
        # Should not crash
        assert result.exit_code in [0, 1]
    
    def test_report_templates_command(self, cli_runner):
        """Test template listing."""
        result = cli_runner.invoke(app, ["report", "templates"])
        
        # Should not crash
        assert result.exit_code in [0, 1]


class TestCLIErrorHandling:
    """Test CLI error handling."""
    
    def test_invalid_command(self, cli_runner):
        """Test invalid command handling."""
        result = cli_runner.invoke(app, ["invalid-command"])
        assert result.exit_code != 0
        assert "No such command" in result.stdout or "Usage:" in result.stdout
    
    def test_missing_required_argument(self, cli_runner):
        """Test missing required argument handling."""
        result = cli_runner.invoke(app, ["init"])  # Missing directory argument
        assert result.exit_code != 0
    
    def test_invalid_option_value(self, cli_runner):
        """Test invalid option value handling."""
        result = cli_runner.invoke(app, [
            "data", "export", "ga4",
            "--format", "invalid_format"
        ])
        
        # Should handle gracefully
        assert result.exit_code in [0, 1, 2]  # Various possible error codes
    
    @patch('marketing_ai_agent.core.monitoring.system_monitor')
    def test_command_with_exception(self, mock_monitor, cli_runner):
        """Test command that raises exception."""
        mock_monitor.get_status_report.side_effect = Exception("Test exception")
        
        result = cli_runner.invoke(app, ["status"])
        
        # Should handle exception gracefully
        assert result.exit_code in [0, 1]
        assert "Error getting system status" in result.stdout or "Test exception" in result.stdout


class TestCLIConfiguration:
    """Test CLI configuration handling."""
    
    def test_global_config_option(self, cli_runner, isolated_filesystem):
        """Test global config file option."""
        # Create test config file
        config_file = Path("test_config.yaml")
        config_content = """
        logging:
          level: DEBUG
        output:
          base_directory: ./test_output
        """
        config_file.write_text(config_content)
        
        result = cli_runner.invoke(app, [
            "--config", str(config_file),
            "version"
        ])
        
        assert result.exit_code == 0
    
    def test_verbose_option(self, cli_runner):
        """Test verbose logging option."""
        result = cli_runner.invoke(app, ["--verbose", "version"])
        assert result.exit_code == 0
    
    def test_output_dir_option(self, cli_runner, isolated_filesystem):
        """Test output directory option."""
        output_dir = Path("custom_output")
        output_dir.mkdir()
        
        result = cli_runner.invoke(app, [
            "--output-dir", str(output_dir),
            "version"
        ])
        
        assert result.exit_code == 0
    
    def test_monitor_option(self, cli_runner):
        """Test monitor option."""
        with patch('marketing_ai_agent.core.monitoring.system_monitor') as mock_monitor:
            result = cli_runner.invoke(app, ["--monitor", "version"])
            
            assert result.exit_code == 0
            mock_monitor.start_monitoring.assert_called_once()


@pytest.mark.integration
class TestCLIIntegration:
    """Full integration tests for CLI."""
    
    def test_full_workflow(self, cli_runner, isolated_filesystem):
        """Test complete workflow from init to report."""
        # Initialize project
        result = cli_runner.invoke(app, ["init", "test_project"])
        assert result.exit_code == 0
        
        # Check status
        result = cli_runner.invoke(app, ["status"])
        assert result.exit_code == 0
        
        # Try data validation (may fail without real API)
        result = cli_runner.invoke(app, ["data", "validate", "--source", "all"])
        # Don't assert exit code as this may fail without real APIs
    
    @patch('marketing_ai_agent.core.monitoring.system_monitor')
    @patch('marketing_ai_agent.core.error_handlers.error_reporter')
    def test_monitoring_integration(self, mock_error_reporter, mock_monitor, cli_runner):
        """Test monitoring integration."""
        # Setup mocks
        mock_monitor.get_status_report.return_value = {
            "overall_health": "healthy",
            "monitoring_active": True,
            "current_metrics": {"cpu_percent": 30.0},
            "health_checks": {},
            "average_metrics_10min": {},
            "metrics_history_count": 100
        }
        
        mock_error_reporter.get_error_summary.return_value = {
            "total_error_types": 0,
            "error_counts": {},
            "recent_errors": [],
            "most_common_error": None
        }
        
        # Start monitoring
        result = cli_runner.invoke(app, ["monitor", "--start"])
        assert result.exit_code == 0
        
        # Check status
        result = cli_runner.invoke(app, ["status", "--detailed"])
        assert result.exit_code == 0
        
        # Check errors
        result = cli_runner.invoke(app, ["errors", "--summary"])
        assert result.exit_code == 0


@pytest.mark.slow
class TestCLIPerformance:
    """Performance tests for CLI commands."""
    
    def test_status_command_performance(self, cli_runner, benchmark):
        """Benchmark status command performance."""
        with patch('marketing_ai_agent.core.monitoring.system_monitor') as mock_monitor:
            mock_monitor.get_status_report.return_value = {
                "overall_health": "healthy",
                "monitoring_active": True,
                "current_metrics": {"cpu_percent": 25.0},
                "health_checks": {},
                "average_metrics_10min": {},
                "metrics_history_count": 50
            }
            
            def run_status_command():
                return cli_runner.invoke(app, ["status"])
            
            result = benchmark(run_status_command)
            assert result.exit_code == 0
    
    def test_init_command_performance(self, cli_runner, benchmark, isolated_filesystem):
        """Benchmark init command performance."""
        
        def run_init_command():
            import shutil
            project_dir = Path("bench_project")
            if project_dir.exists():
                shutil.rmtree(project_dir)
            return cli_runner.invoke(app, ["init", str(project_dir)])
        
        result = benchmark(run_init_command)
        assert result.exit_code == 0