"""Configuration management for Marketing AI Agent."""

import os
from pathlib import Path
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field
import yaml
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class APIConfig(BaseModel):
    """API configuration settings."""
    
    google_ads: Dict[str, str] = Field(default_factory=dict, description="Google Ads API settings")
    google_analytics: Dict[str, str] = Field(default_factory=dict, description="Google Analytics API settings")
    openai: Dict[str, str] = Field(default_factory=dict, description="OpenAI API settings")
    anthropic: Dict[str, str] = Field(default_factory=dict, description="Anthropic API settings")


class OutputConfig(BaseModel):
    """Output configuration settings."""
    
    base_directory: str = Field(default="./reports", description="Base output directory")
    format: str = Field(default="markdown", description="Default output format")
    include_charts: bool = Field(default=True, description="Include charts in reports")
    template_directory: Optional[str] = Field(default=None, description="Custom template directory")


class LoggingConfig(BaseModel):
    """Logging configuration settings."""
    
    level: str = Field(default="INFO", description="Log level")
    file: Optional[str] = Field(default=None, description="Log file path")
    format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="Log format"
    )


class AnalyticsConfig(BaseModel):
    """Analytics configuration settings."""
    
    default_date_range: str = Field(default="30d", description="Default date range for analysis")
    significance_threshold: float = Field(default=0.05, description="Statistical significance threshold")
    confidence_level: float = Field(default=0.95, description="Confidence level for intervals")
    min_sample_size: int = Field(default=100, description="Minimum sample size for analysis")


class OptimizationConfig(BaseModel):
    """Optimization configuration settings."""
    
    confidence_threshold: float = Field(default=0.7, description="Minimum confidence for recommendations")
    max_recommendations: int = Field(default=10, description="Maximum recommendations to generate")
    include_experimental: bool = Field(default=True, description="Include experimental recommendations")
    budget_risk_threshold: float = Field(default=0.1, description="Maximum budget risk for optimizations")


class Config(BaseModel):
    """Main configuration class."""
    
    api: APIConfig = Field(default_factory=APIConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    analytics: AnalyticsConfig = Field(default_factory=AnalyticsConfig)
    optimization: OptimizationConfig = Field(default_factory=OptimizationConfig)
    
    @classmethod
    def from_env(cls) -> "Config":
        """Create configuration from environment variables."""
        
        # Google Ads API from environment
        google_ads_config = {}
        if os.getenv("GOOGLE_ADS_CLIENT_ID"):
            google_ads_config = {
                "client_id": os.getenv("GOOGLE_ADS_CLIENT_ID"),
                "client_secret": os.getenv("GOOGLE_ADS_CLIENT_SECRET"),
                "refresh_token": os.getenv("GOOGLE_ADS_REFRESH_TOKEN"),
                "developer_token": os.getenv("GOOGLE_ADS_DEVELOPER_TOKEN")
            }
        
        # Google Analytics from environment
        google_analytics_config = {}
        if os.getenv("GA_PROPERTY_ID"):
            google_analytics_config = {
                "property_id": os.getenv("GA_PROPERTY_ID"),
                "credentials_file": os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
            }
        
        # AI API configurations
        openai_config = {}
        if os.getenv("OPENAI_API_KEY"):
            openai_config = {"api_key": os.getenv("OPENAI_API_KEY")}
        
        anthropic_config = {}
        if os.getenv("ANTHROPIC_API_KEY"):
            anthropic_config = {"api_key": os.getenv("ANTHROPIC_API_KEY")}
        
        return cls(
            api=APIConfig(
                google_ads=google_ads_config,
                google_analytics=google_analytics_config,
                openai=openai_config,
                anthropic=anthropic_config
            ),
            output=OutputConfig(
                base_directory=os.getenv("OUTPUT_DIRECTORY", "./reports")
            ),
            logging=LoggingConfig(
                level=os.getenv("LOG_LEVEL", "INFO"),
                file=os.getenv("LOG_FILE")
            )
        )


def load_config(config_path: Path) -> Config:
    """Load configuration from YAML file."""
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config_data = yaml.safe_load(f)
    
    # Merge with environment variables
    env_config = Config.from_env()
    
    # Override with file configuration
    if config_data:
        # API configuration
        if 'api' in config_data:
            if 'google_ads' in config_data['api'] and env_config.api.google_ads:
                config_data['api']['google_ads'].update(env_config.api.google_ads)
            if 'google_analytics' in config_data['api'] and env_config.api.google_analytics:
                config_data['api']['google_analytics'].update(env_config.api.google_analytics)
        
        # Create config from merged data
        return Config(**config_data)
    
    return env_config


def save_config(config: Config, config_path: Path) -> None:
    """Save configuration to YAML file.
    
    Args:
        config: Configuration object to save
        config_path: Path where to save the configuration file
        
    Raises:
        FileNotFoundError: If parent directory doesn't exist
        PermissionError: If unable to write to the specified path
        yaml.YAMLError: If unable to serialize configuration to YAML
    """
    # Ensure parent directory exists
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert config to dictionary for YAML serialization
    config_dict = config.model_dump()
    
    # Write configuration to file
    with open(config_path, 'w') as f:
        yaml.safe_dump(config_dict, f, default_flow_style=False, indent=2)


def get_default_config() -> Config:
    """Get default configuration with environment variables."""
    return Config.from_env()