"""Comprehensive logging system for the Marketing AI Agent."""

import json
import logging
import logging.handlers
import sys
import traceback
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Any

from .exceptions import MarketingAIAgentError


class JSONFormatter(logging.Formatter):
    """Custom JSON formatter for structured logging."""

    def format(self, record):
        log_data = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = {
                "type": record.exc_info[0].__name__,
                "message": str(record.exc_info[1]),
                "traceback": traceback.format_exception(*record.exc_info),
            }

        # Add extra fields
        for key, value in record.__dict__.items():
            if key not in [
                "name",
                "msg",
                "args",
                "levelname",
                "levelno",
                "pathname",
                "filename",
                "module",
                "lineno",
                "funcName",
                "created",
                "msecs",
                "relativeCreated",
                "thread",
                "threadName",
                "processName",
                "process",
                "getMessage",
                "exc_info",
                "exc_text",
                "stack_info",
            ]:
                log_data[key] = value

        return json.dumps(log_data)


class ColoredConsoleFormatter(logging.Formatter):
    """Console formatter with color support."""

    COLORS = {
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[32m",  # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[35m",  # Magenta
        "RESET": "\033[0m",  # Reset
    }

    def format(self, record):
        color = self.COLORS.get(record.levelname, self.COLORS["RESET"])
        record.levelname = f"{color}{record.levelname}{self.COLORS['RESET']}"
        return super().format(record)


class PerformanceLogger:
    """Context manager for performance logging."""

    def __init__(
        self, logger: logging.Logger, operation: str, level: int = logging.INFO
    ):
        self.logger = logger
        self.operation = operation
        self.level = level
        self.start_time = None

    def __enter__(self):
        self.start_time = datetime.now()
        self.logger.log(
            self.level,
            f"Starting {self.operation}",
            extra={"operation": self.operation, "event": "start"},
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        end_time = datetime.now()
        duration = (end_time - self.start_time).total_seconds()

        if exc_type:
            self.logger.error(
                f"Failed {self.operation} after {duration:.2f}s",
                extra={
                    "operation": self.operation,
                    "event": "failed",
                    "duration": duration,
                    "error_type": exc_type.__name__,
                    "error_message": str(exc_val),
                },
            )
        else:
            self.logger.log(
                self.level,
                f"Completed {self.operation} in {duration:.2f}s",
                extra={
                    "operation": self.operation,
                    "event": "completed",
                    "duration": duration,
                },
            )


class LoggerManager:
    """Centralized logger management."""

    _instance = None
    _loggers = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not hasattr(self, "_initialized"):
            self._initialized = True
            self.log_dir = Path("logs")
            self.log_dir.mkdir(exist_ok=True)

    def get_logger(self, name: str, level: int = logging.INFO) -> logging.Logger:
        """Get or create a logger with the specified configuration."""
        if name in self._loggers:
            return self._loggers[name]

        logger = logging.getLogger(name)
        logger.setLevel(level)

        # Clear existing handlers to avoid duplicates
        logger.handlers.clear()

        # Console handler with colors
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_formatter = ColoredConsoleFormatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

        # File handler for general logs
        file_handler = logging.handlers.RotatingFileHandler(
            self.log_dir / f"{name}.log",
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5,
        )
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(module)s:%(funcName)s:%(lineno)d - %(message)s"
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

        # JSON file handler for structured logs
        json_handler = logging.handlers.RotatingFileHandler(
            self.log_dir / f"{name}.json",
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5,
        )
        json_handler.setLevel(logging.DEBUG)
        json_handler.setFormatter(JSONFormatter())
        logger.addHandler(json_handler)

        # Error file handler for errors only
        error_handler = logging.handlers.RotatingFileHandler(
            self.log_dir / f"{name}_errors.log",
            maxBytes=5 * 1024 * 1024,  # 5MB
            backupCount=3,
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(file_formatter)
        logger.addHandler(error_handler)

        self._loggers[name] = logger
        return logger

    def configure_root_logger(self, level: int = logging.INFO):
        """Configure the root logger."""
        root_logger = logging.getLogger()
        root_logger.setLevel(level)

        # Clear existing handlers
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)

        # Add console handler only for root logger
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_formatter = ColoredConsoleFormatter(
            "%(asctime)s - %(levelname)s - %(message)s"
        )
        console_handler.setFormatter(console_formatter)
        root_logger.addHandler(console_handler)


def get_logger(name: str = None, level: int = logging.INFO) -> logging.Logger:
    """Get a configured logger instance."""
    if name is None:
        # Get caller's module name
        import inspect

        frame = inspect.currentframe().f_back
        name = frame.f_globals.get("__name__", "unknown")

    manager = LoggerManager()
    return manager.get_logger(name, level)


def log_performance(operation: str, level: int = logging.INFO):
    """Decorator for automatic performance logging."""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            logger = get_logger(func.__module__)
            with PerformanceLogger(logger, f"{func.__name__}({operation})", level):
                return func(*args, **kwargs)

        return wrapper

    return decorator


def log_errors(logger: logging.Logger | None = None, reraise: bool = True):
    """Decorator for automatic error logging."""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            nonlocal logger
            if logger is None:
                logger = get_logger(func.__module__)

            try:
                return func(*args, **kwargs)
            except MarketingAIAgentError as e:
                logger.error(
                    f"Marketing AI Agent error in {func.__name__}: {e.message}",
                    extra={
                        "function": func.__name__,
                        "error_code": e.error_code,
                        "context": e.context,
                    },
                )
                if reraise:
                    raise
            except Exception as e:
                logger.error(
                    f"Unexpected error in {func.__name__}: {str(e)}",
                    extra={"function": func.__name__, "error_type": type(e).__name__},
                    exc_info=True,
                )
                if reraise:
                    raise

        return wrapper

    return decorator


def log_function_calls(
    logger: logging.Logger | None = None, level: int = logging.DEBUG
):
    """Decorator for logging function calls."""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            nonlocal logger
            if logger is None:
                logger = get_logger(func.__module__)

            # Log function entry
            logger.log(
                level,
                f"Calling {func.__name__}",
                extra={
                    "function": func.__name__,
                    "args_count": len(args),
                    "kwargs_keys": list(kwargs.keys()),
                },
            )

            try:
                result = func(*args, **kwargs)
                logger.log(
                    level,
                    f"Completed {func.__name__}",
                    extra={"function": func.__name__, "success": True},
                )
                return result
            except Exception as e:
                logger.log(
                    level,
                    f"Failed {func.__name__}: {str(e)}",
                    extra={
                        "function": func.__name__,
                        "success": False,
                        "error_type": type(e).__name__,
                    },
                )
                raise

        return wrapper

    return decorator


class AuditLogger:
    """Specialized logger for audit trails."""

    def __init__(self, name: str = "audit"):
        self.logger = get_logger(f"audit.{name}")

    def log_api_call(
        self,
        api_name: str,
        endpoint: str,
        method: str,
        status_code: int,
        duration: float,
    ):
        """Log API call details."""
        self.logger.info(
            "API call completed",
            extra={
                "event_type": "api_call",
                "api_name": api_name,
                "endpoint": endpoint,
                "method": method,
                "status_code": status_code,
                "duration": duration,
            },
        )

    def log_data_export(
        self, source: str, records_count: int, file_path: str, user: str = None
    ):
        """Log data export operations."""
        self.logger.info(
            "Data export completed",
            extra={
                "event_type": "data_export",
                "source": source,
                "records_count": records_count,
                "file_path": file_path,
                "user": user,
            },
        )

    def log_model_training(
        self,
        model_name: str,
        model_type: str,
        training_duration: float,
        accuracy: float = None,
    ):
        """Log model training operations."""
        self.logger.info(
            "Model training completed",
            extra={
                "event_type": "model_training",
                "model_name": model_name,
                "model_type": model_type,
                "training_duration": training_duration,
                "accuracy": accuracy,
            },
        )

    def log_optimization_run(
        self,
        optimization_type: str,
        campaigns_affected: int,
        estimated_impact: dict[str, Any],
    ):
        """Log optimization operations."""
        self.logger.info(
            "Optimization run completed",
            extra={
                "event_type": "optimization_run",
                "optimization_type": optimization_type,
                "campaigns_affected": campaigns_affected,
                "estimated_impact": estimated_impact,
            },
        )

    def log_report_generation(
        self,
        template_name: str,
        output_format: str,
        file_path: str,
        generation_time: float,
    ):
        """Log report generation."""
        self.logger.info(
            "Report generated",
            extra={
                "event_type": "report_generation",
                "template_name": template_name,
                "output_format": output_format,
                "file_path": file_path,
                "generation_time": generation_time,
            },
        )


# Initialize logging system
def initialize_logging(level: int = logging.INFO):
    """Initialize the logging system."""
    manager = LoggerManager()
    manager.configure_root_logger(level)

    # Create main application logger
    logger = get_logger("marketing_ai_agent", level)
    logger.info(
        "Logging system initialized",
        extra={
            "log_level": logging.getLevelName(level),
            "log_directory": str(manager.log_dir),
        },
    )

    return logger
