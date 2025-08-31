"""Demonstration script for error handling and logging features."""

import random
import time
from pathlib import Path

from ..core.error_handlers import (
    RetryConfig,
    circuit_breaker,
    error_reporter,
    handle_errors,
    retry_on_error,
)
from ..core.exceptions import (
    APIError,
    DataValidationError,
    RateLimitError,
    TemporaryServiceError,
)
from ..core.logging import AuditLogger, PerformanceLogger, get_logger, log_performance
from ..core.monitoring import system_monitor


class DemoService:
    """Demo service to showcase error handling patterns."""

    def __init__(self):
        self.logger = get_logger(__name__)
        self.audit_logger = AuditLogger("demo")
        self.failure_count = 0
        self.max_failures = 3

    @handle_errors(reraise=True, report=True, severity="ERROR")
    @log_performance("API call simulation")
    def simulate_api_call(self, success_rate: float = 0.7) -> dict:
        """Simulate an API call with configurable success rate."""

        with PerformanceLogger(self.logger, "API response processing"):
            # Simulate API delay
            time.sleep(random.uniform(0.1, 0.5))

            if random.random() > success_rate:
                if random.random() > 0.5:
                    raise APIError(
                        "Simulated API failure",
                        "Demo API",
                        status_code=500,
                        response_data={"error": "Internal server error"},
                    )
                else:
                    raise RateLimitError(
                        "Rate limit exceeded",
                        "Demo API",
                        retry_after=random.randint(1, 5),
                    )

            # Log successful API call
            self.audit_logger.log_api_call(
                api_name="Demo API",
                endpoint="/test",
                method="GET",
                status_code=200,
                duration=random.uniform(0.1, 0.5),
            )

            return {
                "status": "success",
                "data": {"value": random.randint(1, 100)},
                "timestamp": time.time(),
            }

    @retry_on_error(
        RetryConfig(
            max_attempts=5,
            initial_delay=1.0,
            backoff_factor=2.0,
            retryable_exceptions=(RateLimitError, TemporaryServiceError),
        )
    )
    def robust_api_call(self) -> dict:
        """API call with automatic retry logic."""
        return self.simulate_api_call(success_rate=0.3)  # Low success rate for demo

    @circuit_breaker(failure_threshold=3, recovery_timeout=10)
    @handle_errors(reraise=False, report=True)
    def protected_operation(self) -> dict:
        """Operation protected by circuit breaker."""
        self.failure_count += 1

        if self.failure_count <= self.max_failures:
            raise TemporaryServiceError(
                f"Simulated failure #{self.failure_count}",
                "Demo Service",
                retry_after=5,
            )

        # Reset after max failures
        self.failure_count = 0
        return {"status": "success", "message": "Circuit breaker test passed"}

    @handle_errors(reraise=False, report=True, severity="WARNING")
    def data_validation_demo(self, data: dict) -> bool:
        """Demonstrate data validation error handling."""

        required_fields = ["id", "name", "value"]

        for field in required_fields:
            if field not in data:
                raise DataValidationError(
                    f"Missing required field: {field}",
                    field_name=field,
                    field_value=None,
                )

            if not data[field]:
                raise DataValidationError(
                    f"Empty value for field: {field}",
                    field_name=field,
                    field_value=data[field],
                )

        if not isinstance(data["value"], int | float):
            raise DataValidationError(
                "Value must be numeric", field_name="value", field_value=data["value"]
            )

        self.logger.info(
            "Data validation successful",
            extra={"record_id": data["id"], "fields_validated": len(required_fields)},
        )

        return True


def run_error_demo():
    """Run comprehensive error handling demonstration."""

    logger = get_logger("demo_runner")
    logger.info("Starting error handling demonstration")

    # Start monitoring for the demo
    system_monitor.start_monitoring(metrics_interval=30)

    demo_service = DemoService()

    print("🚀 Marketing AI Agent - Error Handling Demo")
    print("=" * 50)

    # Demo 1: Basic error handling and reporting
    print("\n1️⃣ Basic Error Handling Demo")
    print("-" * 30)

    for i in range(5):
        try:
            result = demo_service.simulate_api_call(success_rate=0.6)
            print(f"✅ API call {i+1} succeeded: {result['data']}")
        except Exception as e:
            print(f"❌ API call {i+1} failed: {str(e)}")

    # Demo 2: Retry mechanism
    print("\n2️⃣ Automatic Retry Demo")
    print("-" * 25)

    try:
        result = demo_service.robust_api_call()
        print(f"✅ Robust API call succeeded after retries: {result}")
    except Exception as e:
        print(f"❌ Robust API call failed after all retries: {str(e)}")

    # Demo 3: Circuit breaker
    print("\n3️⃣ Circuit Breaker Demo")
    print("-" * 23)

    for i in range(8):
        try:
            result = demo_service.protected_operation()
            print(f"✅ Protected operation {i+1} succeeded: {result}")
        except Exception as e:
            print(f"❌ Protected operation {i+1} failed: {str(e)}")

        time.sleep(0.5)  # Brief delay between attempts

    # Demo 4: Data validation
    print("\n4️⃣ Data Validation Demo")
    print("-" * 25)

    test_data = [
        {"id": 1, "name": "valid_record", "value": 42.5},
        {"name": "missing_id", "value": 100},
        {"id": 2, "name": "", "value": 50},
        {"id": 3, "name": "invalid_value", "value": "not_a_number"},
    ]

    for data in test_data:
        try:
            demo_service.data_validation_demo(data)
            print(f"✅ Validation passed for: {data}")
        except Exception as e:
            print(f"❌ Validation failed for {data}: {str(e)}")

    # Demo 5: Error reporting
    print("\n5️⃣ Error Summary Report")
    print("-" * 25)

    error_summary = error_reporter.get_error_summary()
    print(f"📊 Total error types: {error_summary['total_error_types']}")
    print("📈 Error counts by type:")

    for error_type, count in error_summary["error_counts"].items():
        print(f"   • {error_type}: {count}")

    if error_summary["most_common_error"]:
        most_common = error_summary["most_common_error"]
        print(f"🔝 Most common error: {most_common[0]} ({most_common[1]} times)")

    print(f"\n🕐 Recent errors: {len(error_summary['recent_errors'])}")

    # Demo 6: System monitoring
    print("\n6️⃣ System Monitoring Report")
    print("-" * 27)

    status_report = system_monitor.get_status_report()
    print(f"🏥 Overall health: {status_report['overall_health']}")
    print(f"📊 Monitoring active: {status_report['monitoring_active']}")

    if status_report["current_metrics"]:
        metrics = status_report["current_metrics"]
        print("💻 Current system metrics:")
        print(f"   CPU: {metrics['cpu_percent']:.1f}%")
        print(f"   Memory: {metrics['memory_percent']:.1f}%")
        print(f"   Disk: {metrics['disk_usage_percent']:.1f}%")

    # Export demo results
    print("\n7️⃣ Exporting Demo Results")
    print("-" * 27)

    export_dir = Path("demo_results")
    export_dir.mkdir(exist_ok=True)

    # Export error report
    error_export_path = export_dir / "error_report.json"
    import json

    with open(error_export_path, "w") as f:
        json.dump(error_summary, f, indent=2, default=str)
    print(f"📄 Error report: {error_export_path}")

    # Export metrics
    try:
        metrics_export_path = export_dir / "metrics.json"
        system_monitor.export_metrics(str(metrics_export_path))
        print(f"📊 Metrics report: {metrics_export_path}")
    except Exception as e:
        print(f"⚠️  Could not export metrics: {str(e)}")

    # Stop monitoring
    system_monitor.stop_monitoring()

    print("\n🎯 Demo completed! Check the logs/ directory for detailed logs.")
    print("📁 Demo results exported to demo_results/ directory.")

    logger.info(
        "Error handling demonstration completed",
        extra={
            "error_types_generated": len(error_summary["error_counts"]),
            "total_errors": sum(error_summary["error_counts"].values()),
            "demo_duration": "estimated_5_minutes",
        },
    )


if __name__ == "__main__":
    run_error_demo()
