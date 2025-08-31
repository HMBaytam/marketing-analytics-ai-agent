"""System monitoring and health check utilities."""

import psutil
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from collections import deque
import json

from .logging import get_logger, AuditLogger
from .exceptions import MarketingAIAgentError


@dataclass
class HealthCheck:
    """Health check result."""
    name: str
    status: str  # "healthy", "warning", "critical"
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    duration: Optional[float] = None


@dataclass
class PerformanceMetrics:
    """Performance metrics snapshot."""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    disk_usage_percent: float
    network_io: Dict[str, int]
    process_count: int
    response_time: Optional[float] = None


class MetricsCollector:
    """Collects and stores performance metrics."""
    
    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self.metrics_history = deque(maxlen=max_history)
        self.logger = get_logger(__name__)
        self._lock = threading.Lock()
        self._monitoring = False
        self._monitor_thread = None
        self._monitor_interval = 60  # seconds
    
    def collect_metrics(self) -> PerformanceMetrics:
        """Collect current system metrics."""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_used_mb = memory.used / (1024 * 1024)
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_usage_percent = (disk.used / disk.total) * 100
            
            # Network I/O
            network = psutil.net_io_counters()
            network_io = {
                "bytes_sent": network.bytes_sent,
                "bytes_recv": network.bytes_recv,
                "packets_sent": network.packets_sent,
                "packets_recv": network.packets_recv
            }
            
            # Process count
            process_count = len(psutil.pids())
            
            metrics = PerformanceMetrics(
                timestamp=datetime.now(),
                cpu_percent=cpu_percent,
                memory_percent=memory_percent,
                memory_used_mb=memory_used_mb,
                disk_usage_percent=disk_usage_percent,
                network_io=network_io,
                process_count=process_count
            )
            
            with self._lock:
                self.metrics_history.append(metrics)
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Failed to collect metrics: {str(e)}")
            raise MarketingAIAgentError(f"Metrics collection failed: {str(e)}")
    
    def start_monitoring(self, interval: int = 60):
        """Start continuous metrics monitoring."""
        if self._monitoring:
            return
        
        self._monitor_interval = interval
        self._monitoring = True
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        self.logger.info(f"Started metrics monitoring with {interval}s interval")
    
    def stop_monitoring(self):
        """Stop metrics monitoring."""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)
        self.logger.info("Stopped metrics monitoring")
    
    def _monitor_loop(self):
        """Monitoring loop running in background thread."""
        while self._monitoring:
            try:
                self.collect_metrics()
                time.sleep(self._monitor_interval)
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {str(e)}")
                time.sleep(self._monitor_interval)
    
    def get_recent_metrics(self, minutes: int = 10) -> List[PerformanceMetrics]:
        """Get metrics from the last N minutes."""
        cutoff_time = datetime.now() - timedelta(minutes=minutes)
        
        with self._lock:
            return [
                metric for metric in self.metrics_history
                if metric.timestamp >= cutoff_time
            ]
    
    def get_average_metrics(self, minutes: int = 10) -> Dict[str, float]:
        """Get average metrics over the last N minutes."""
        recent_metrics = self.get_recent_metrics(minutes)
        
        if not recent_metrics:
            return {}
        
        return {
            "avg_cpu_percent": sum(m.cpu_percent for m in recent_metrics) / len(recent_metrics),
            "avg_memory_percent": sum(m.memory_percent for m in recent_metrics) / len(recent_metrics),
            "avg_disk_usage_percent": sum(m.disk_usage_percent for m in recent_metrics) / len(recent_metrics),
            "avg_memory_used_mb": sum(m.memory_used_mb for m in recent_metrics) / len(recent_metrics)
        }


class HealthChecker:
    """Performs various health checks."""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.audit_logger = AuditLogger("health")
        self.checks = {}
        self.thresholds = {
            "cpu_warning": 70,
            "cpu_critical": 90,
            "memory_warning": 80,
            "memory_critical": 95,
            "disk_warning": 80,
            "disk_critical": 95,
            "response_time_warning": 5.0,
            "response_time_critical": 10.0
        }
    
    def register_check(self, name: str, check_func: Callable[[], HealthCheck]):
        """Register a custom health check."""
        self.checks[name] = check_func
        self.logger.info(f"Registered health check: {name}")
    
    def check_system_resources(self) -> HealthCheck:
        """Check system resource usage."""
        try:
            start_time = time.time()
            
            # Get current metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            duration = time.time() - start_time
            
            # Determine status based on thresholds
            status = "healthy"
            issues = []
            
            if cpu_percent >= self.thresholds["cpu_critical"]:
                status = "critical"
                issues.append(f"CPU usage critical: {cpu_percent:.1f}%")
            elif cpu_percent >= self.thresholds["cpu_warning"]:
                status = "warning"
                issues.append(f"CPU usage high: {cpu_percent:.1f}%")
            
            if memory.percent >= self.thresholds["memory_critical"]:
                status = "critical"
                issues.append(f"Memory usage critical: {memory.percent:.1f}%")
            elif memory.percent >= self.thresholds["memory_warning"]:
                if status != "critical":
                    status = "warning"
                issues.append(f"Memory usage high: {memory.percent:.1f}%")
            
            disk_percent = (disk.used / disk.total) * 100
            if disk_percent >= self.thresholds["disk_critical"]:
                status = "critical"
                issues.append(f"Disk usage critical: {disk_percent:.1f}%")
            elif disk_percent >= self.thresholds["disk_warning"]:
                if status != "critical":
                    status = "warning"
                issues.append(f"Disk usage high: {disk_percent:.1f}%")
            
            message = "System resources OK" if status == "healthy" else "; ".join(issues)
            
            return HealthCheck(
                name="system_resources",
                status=status,
                message=message,
                details={
                    "cpu_percent": cpu_percent,
                    "memory_percent": memory.percent,
                    "memory_available_gb": memory.available / (1024**3),
                    "disk_percent": disk_percent,
                    "disk_free_gb": disk.free / (1024**3)
                },
                duration=duration
            )
            
        except Exception as e:
            return HealthCheck(
                name="system_resources",
                status="critical",
                message=f"Failed to check system resources: {str(e)}",
                details={"error": str(e)}
            )
    
    def check_api_connectivity(self) -> HealthCheck:
        """Check API connectivity (placeholder)."""
        try:
            start_time = time.time()
            
            # Placeholder for actual API checks
            # This would test GA4, Google Ads APIs, etc.
            api_checks = {
                "ga4_api": {"status": "healthy", "response_time": 0.5},
                "google_ads_api": {"status": "healthy", "response_time": 0.8}
            }
            
            duration = time.time() - start_time
            
            # Check if any APIs are down
            failed_apis = [name for name, check in api_checks.items() if check["status"] != "healthy"]
            
            if failed_apis:
                status = "critical"
                message = f"API connectivity issues: {', '.join(failed_apis)}"
            else:
                status = "healthy"
                message = "All APIs accessible"
            
            return HealthCheck(
                name="api_connectivity",
                status=status,
                message=message,
                details=api_checks,
                duration=duration
            )
            
        except Exception as e:
            return HealthCheck(
                name="api_connectivity",
                status="critical",
                message=f"API connectivity check failed: {str(e)}",
                details={"error": str(e)}
            )
    
    def check_database_connection(self) -> HealthCheck:
        """Check database connectivity (placeholder)."""
        try:
            start_time = time.time()
            
            # Placeholder for database connectivity check
            # This would test connection to any databases used
            
            duration = time.time() - start_time
            
            return HealthCheck(
                name="database_connection",
                status="healthy",
                message="Database connection OK",
                details={"response_time": duration},
                duration=duration
            )
            
        except Exception as e:
            return HealthCheck(
                name="database_connection",
                status="critical",
                message=f"Database connection failed: {str(e)}",
                details={"error": str(e)}
            )
    
    def run_all_checks(self) -> Dict[str, HealthCheck]:
        """Run all registered health checks."""
        results = {}
        
        # Built-in checks
        built_in_checks = {
            "system_resources": self.check_system_resources,
            "api_connectivity": self.check_api_connectivity,
            "database_connection": self.check_database_connection
        }
        
        # Run built-in checks
        for name, check_func in built_in_checks.items():
            try:
                results[name] = check_func()
            except Exception as e:
                results[name] = HealthCheck(
                    name=name,
                    status="critical",
                    message=f"Health check failed: {str(e)}",
                    details={"error": str(e)}
                )
        
        # Run custom checks
        for name, check_func in self.checks.items():
            try:
                results[name] = check_func()
            except Exception as e:
                results[name] = HealthCheck(
                    name=name,
                    status="critical",
                    message=f"Custom health check failed: {str(e)}",
                    details={"error": str(e)}
                )
        
        # Log health check results
        self.audit_logger.logger.info("Health check completed", extra={
            "event_type": "health_check",
            "checks_run": len(results),
            "healthy_checks": len([r for r in results.values() if r.status == "healthy"]),
            "warning_checks": len([r for r in results.values() if r.status == "warning"]),
            "critical_checks": len([r for r in results.values() if r.status == "critical"])
        })
        
        return results
    
    def get_overall_health(self) -> str:
        """Get overall system health status."""
        results = self.run_all_checks()
        
        if any(check.status == "critical" for check in results.values()):
            return "critical"
        elif any(check.status == "warning" for check in results.values()):
            return "warning"
        else:
            return "healthy"


class SystemMonitor:
    """Main system monitoring coordinator."""
    
    def __init__(self):
        self.metrics_collector = MetricsCollector()
        self.health_checker = HealthChecker()
        self.logger = get_logger(__name__)
        self.audit_logger = AuditLogger("monitoring")
        self._monitoring_active = False
    
    def start_monitoring(self, metrics_interval: int = 60, health_check_interval: int = 300):
        """Start system monitoring."""
        if self._monitoring_active:
            return
        
        self._monitoring_active = True
        
        # Start metrics collection
        self.metrics_collector.start_monitoring(metrics_interval)
        
        # Start periodic health checks
        self._start_health_monitoring(health_check_interval)
        
        self.logger.info("System monitoring started")
        self.audit_logger.logger.info("System monitoring started", extra={
            "event_type": "monitoring_start",
            "metrics_interval": metrics_interval,
            "health_check_interval": health_check_interval
        })
    
    def stop_monitoring(self):
        """Stop system monitoring."""
        self._monitoring_active = False
        self.metrics_collector.stop_monitoring()
        self.logger.info("System monitoring stopped")
    
    def _start_health_monitoring(self, interval: int):
        """Start periodic health checks."""
        def health_check_loop():
            while self._monitoring_active:
                try:
                    results = self.health_checker.run_all_checks()
                    overall_health = self.health_checker.get_overall_health()
                    
                    if overall_health == "critical":
                        self.logger.error("System health check: CRITICAL")
                    elif overall_health == "warning":
                        self.logger.warning("System health check: WARNING")
                    else:
                        self.logger.info("System health check: HEALTHY")
                    
                    time.sleep(interval)
                except Exception as e:
                    self.logger.error(f"Health check loop error: {str(e)}")
                    time.sleep(interval)
        
        health_thread = threading.Thread(target=health_check_loop, daemon=True)
        health_thread.start()
    
    def get_status_report(self) -> Dict[str, Any]:
        """Get comprehensive system status report."""
        # Get health check results
        health_results = self.health_checker.run_all_checks()
        overall_health = self.health_checker.get_overall_health()
        
        # Get recent metrics
        recent_metrics = self.metrics_collector.get_recent_metrics(10)
        avg_metrics = self.metrics_collector.get_average_metrics(10)
        
        # Get current metrics
        try:
            current_metrics = self.metrics_collector.collect_metrics()
        except Exception as e:
            current_metrics = None
            self.logger.error(f"Failed to get current metrics: {str(e)}")
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "overall_health": overall_health,
            "health_checks": {
                name: {
                    "status": check.status,
                    "message": check.message,
                    "duration": check.duration,
                    "details": check.details
                }
                for name, check in health_results.items()
            },
            "current_metrics": {
                "cpu_percent": current_metrics.cpu_percent if current_metrics else None,
                "memory_percent": current_metrics.memory_percent if current_metrics else None,
                "disk_usage_percent": current_metrics.disk_usage_percent if current_metrics else None,
                "memory_used_mb": current_metrics.memory_used_mb if current_metrics else None
            } if current_metrics else None,
            "average_metrics_10min": avg_metrics,
            "metrics_history_count": len(recent_metrics),
            "monitoring_active": self._monitoring_active
        }
        
        return report
    
    def export_metrics(self, filepath: str, format: str = "json"):
        """Export metrics history to file."""
        try:
            recent_metrics = self.metrics_collector.get_recent_metrics(60)  # Last hour
            
            if format.lower() == "json":
                data = [
                    {
                        "timestamp": m.timestamp.isoformat(),
                        "cpu_percent": m.cpu_percent,
                        "memory_percent": m.memory_percent,
                        "memory_used_mb": m.memory_used_mb,
                        "disk_usage_percent": m.disk_usage_percent,
                        "process_count": m.process_count
                    }
                    for m in recent_metrics
                ]
                
                with open(filepath, 'w') as f:
                    json.dump(data, f, indent=2)
            
            self.logger.info(f"Exported {len(recent_metrics)} metrics to {filepath}")
            
        except Exception as e:
            self.logger.error(f"Failed to export metrics: {str(e)}")
            raise MarketingAIAgentError(f"Metrics export failed: {str(e)}")


# Global system monitor instance
system_monitor = SystemMonitor()