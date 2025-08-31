"""Custom exceptions for the Marketing AI Agent."""

from typing import Any, Dict, Optional


class MarketingAIAgentError(Exception):
    """Base exception for all Marketing AI Agent errors."""
    
    def __init__(self, message: str, error_code: Optional[str] = None, context: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.context = context or {}


class ConfigurationError(MarketingAIAgentError):
    """Raised when there's a configuration issue."""
    
    def __init__(self, message: str, config_key: Optional[str] = None):
        super().__init__(message, "CONFIG_ERROR", {"config_key": config_key})
        self.config_key = config_key


class APIError(MarketingAIAgentError):
    """Base class for API-related errors."""
    
    def __init__(self, message: str, api_name: str, status_code: Optional[int] = None, response_data: Optional[Dict] = None):
        super().__init__(message, "API_ERROR", {
            "api_name": api_name,
            "status_code": status_code,
            "response_data": response_data
        })
        self.api_name = api_name
        self.status_code = status_code
        self.response_data = response_data


class GA4APIError(APIError):
    """Raised when GA4 API calls fail."""
    
    def __init__(self, message: str, status_code: Optional[int] = None, response_data: Optional[Dict] = None):
        super().__init__(message, "GA4", status_code, response_data)


class GoogleAdsAPIError(APIError):
    """Raised when Google Ads API calls fail."""
    
    def __init__(self, message: str, status_code: Optional[int] = None, response_data: Optional[Dict] = None):
        super().__init__(message, "Google Ads", status_code, response_data)


class DataValidationError(MarketingAIAgentError):
    """Raised when data validation fails."""
    
    def __init__(self, message: str, field_name: Optional[str] = None, field_value: Any = None):
        super().__init__(message, "DATA_VALIDATION_ERROR", {
            "field_name": field_name,
            "field_value": field_value
        })
        self.field_name = field_name
        self.field_value = field_value


class ModelError(MarketingAIAgentError):
    """Raised when ML model operations fail."""
    
    def __init__(self, message: str, model_name: Optional[str] = None, model_type: Optional[str] = None):
        super().__init__(message, "MODEL_ERROR", {
            "model_name": model_name,
            "model_type": model_type
        })
        self.model_name = model_name
        self.model_type = model_type


class AnalyticsError(MarketingAIAgentError):
    """Raised when analytics calculations fail."""
    
    def __init__(self, message: str, analysis_type: Optional[str] = None):
        super().__init__(message, "ANALYTICS_ERROR", {"analysis_type": analysis_type})
        self.analysis_type = analysis_type


class OptimizationError(MarketingAIAgentError):
    """Raised when optimization operations fail."""
    
    def __init__(self, message: str, optimization_type: Optional[str] = None):
        super().__init__(message, "OPTIMIZATION_ERROR", {"optimization_type": optimization_type})
        self.optimization_type = optimization_type


class ReportGenerationError(MarketingAIAgentError):
    """Raised when report generation fails."""
    
    def __init__(self, message: str, template_name: Optional[str] = None, output_format: Optional[str] = None):
        super().__init__(message, "REPORT_ERROR", {
            "template_name": template_name,
            "output_format": output_format
        })
        self.template_name = template_name
        self.output_format = output_format


class FileOperationError(MarketingAIAgentError):
    """Raised when file operations fail."""
    
    def __init__(self, message: str, file_path: Optional[str] = None, operation: Optional[str] = None):
        super().__init__(message, "FILE_ERROR", {
            "file_path": file_path,
            "operation": operation
        })
        self.file_path = file_path
        self.operation = operation


class RetryableError(MarketingAIAgentError):
    """Base class for errors that can be retried."""
    
    def __init__(self, message: str, retry_after: Optional[int] = None):
        super().__init__(message, "RETRYABLE_ERROR", {"retry_after": retry_after})
        self.retry_after = retry_after


class RateLimitError(RetryableError):
    """Raised when API rate limits are exceeded."""
    
    def __init__(self, message: str, api_name: str, retry_after: Optional[int] = None):
        super().__init__(message, retry_after)
        self.error_code = "RATE_LIMIT_ERROR"
        self.context.update({"api_name": api_name})
        self.api_name = api_name


class TemporaryServiceError(RetryableError):
    """Raised when a service is temporarily unavailable."""
    
    def __init__(self, message: str, service_name: str, retry_after: Optional[int] = None):
        super().__init__(message, retry_after)
        self.error_code = "TEMPORARY_SERVICE_ERROR"
        self.context.update({"service_name": service_name})
        self.service_name = service_name