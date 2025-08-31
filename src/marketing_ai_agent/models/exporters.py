"""Data export functions for CSV, JSON, and Excel formats."""

import csv
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd
from pydantic import BaseModel

from .campaign import Campaign, AdGroup, Keyword, CampaignPerformance
from .metrics import DailyMetrics, AggregatedMetrics
from .analytics import TrafficData, ConversionEvent, ChannelPerformance

logger = logging.getLogger(__name__)


class ExportConfig(BaseModel):
    """Export configuration model."""
    
    output_dir: str = "./exports"
    include_timestamp: bool = True
    decimal_places: int = 4
    date_format: str = "%Y-%m-%d"
    datetime_format: str = "%Y-%m-%d %H:%M:%S"
    
    class Config:
        arbitrary_types_allowed = True


class DataExporter:
    """Data exporter for marketing data models."""
    
    def __init__(self, config: Optional[ExportConfig] = None):
        """
        Initialize data exporter.
        
        Args:
            config: Export configuration
        """
        self.config = config or ExportConfig()
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def _generate_filename(self, base_name: str, format_type: str) -> str:
        """Generate filename with optional timestamp."""
        if self.config.include_timestamp:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            return f"{base_name}_{timestamp}.{format_type}"
        return f"{base_name}.{format_type}"
    
    def _prepare_data_for_export(self, data: List[Any]) -> List[Dict[str, Any]]:
        """
        Prepare data for export by converting to dictionaries.
        
        Args:
            data: List of Pydantic models
            
        Returns:
            List of dictionaries
        """
        prepared_data = []
        
        for item in data:
            if hasattr(item, 'dict'):
                # Pydantic model
                item_dict = item.dict()
                
                # Convert Decimal to float for JSON compatibility
                for key, value in item_dict.items():
                    if hasattr(value, '__float__'):
                        try:
                            item_dict[key] = round(float(value), self.config.decimal_places)
                        except (ValueError, TypeError):
                            pass
                    elif isinstance(value, datetime):
                        item_dict[key] = value.strftime(self.config.datetime_format)
                    elif hasattr(value, 'date') and hasattr(value.date, '__call__'):
                        item_dict[key] = value.strftime(self.config.date_format)
                
                prepared_data.append(item_dict)
            else:
                # Assume it's already a dictionary
                prepared_data.append(item)
        
        return prepared_data
    
    def export_to_csv(self, data: List[Any], filename: str) -> str:
        """
        Export data to CSV format.
        
        Args:
            data: List of data objects
            filename: Base filename (without extension)
            
        Returns:
            Path to exported file
        """
        if not data:
            logger.warning("No data to export")
            return ""
        
        try:
            prepared_data = self._prepare_data_for_export(data)
            full_filename = self._generate_filename(filename, "csv")
            file_path = self.output_dir / full_filename
            
            # Get all unique field names
            fieldnames = set()
            for item in prepared_data:
                fieldnames.update(item.keys())
            
            fieldnames = sorted(list(fieldnames))
            
            with open(file_path, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(prepared_data)
            
            logger.info(f"Exported {len(prepared_data)} records to {file_path}")
            return str(file_path)
            
        except Exception as e:
            logger.error(f"Failed to export CSV: {e}")
            return ""
    
    def export_to_json(self, data: List[Any], filename: str, pretty: bool = True) -> str:
        """
        Export data to JSON format.
        
        Args:
            data: List of data objects
            filename: Base filename (without extension)
            pretty: Whether to format JSON with indentation
            
        Returns:
            Path to exported file
        """
        if not data:
            logger.warning("No data to export")
            return ""
        
        try:
            prepared_data = self._prepare_data_for_export(data)
            full_filename = self._generate_filename(filename, "json")
            file_path = self.output_dir / full_filename
            
            with open(file_path, 'w', encoding='utf-8') as jsonfile:
                json.dump(
                    prepared_data, 
                    jsonfile, 
                    indent=2 if pretty else None,
                    ensure_ascii=False,
                    default=str
                )
            
            logger.info(f"Exported {len(prepared_data)} records to {file_path}")
            return str(file_path)
            
        except Exception as e:
            logger.error(f"Failed to export JSON: {e}")
            return ""
    
    def export_to_excel(self, data: List[Any], filename: str, sheet_name: str = "Data") -> str:
        """
        Export data to Excel format.
        
        Args:
            data: List of data objects
            filename: Base filename (without extension)
            sheet_name: Name of the Excel sheet
            
        Returns:
            Path to exported file
        """
        if not data:
            logger.warning("No data to export")
            return ""
        
        try:
            prepared_data = self._prepare_data_for_export(data)
            full_filename = self._generate_filename(filename, "xlsx")
            file_path = self.output_dir / full_filename
            
            df = pd.DataFrame(prepared_data)
            
            with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
                df.to_excel(writer, sheet_name=sheet_name, index=False)
            
            logger.info(f"Exported {len(prepared_data)} records to {file_path}")
            return str(file_path)
            
        except Exception as e:
            logger.error(f"Failed to export Excel: {e}")
            return ""
    
    def export_campaigns(
        self, 
        campaigns: List[Campaign], 
        format_type: str = "csv",
        filename: str = "campaigns"
    ) -> str:
        """
        Export campaign data.
        
        Args:
            campaigns: List of campaign models
            format_type: Export format (csv, json, xlsx)
            filename: Base filename
            
        Returns:
            Path to exported file
        """
        if format_type.lower() == "csv":
            return self.export_to_csv(campaigns, filename)
        elif format_type.lower() == "json":
            return self.export_to_json(campaigns, filename)
        elif format_type.lower() == "xlsx":
            return self.export_to_excel(campaigns, filename, "Campaigns")
        else:
            raise ValueError(f"Unsupported format: {format_type}")
    
    def export_performance_data(
        self, 
        performance_data: List[Union[DailyMetrics, CampaignPerformance]], 
        format_type: str = "csv",
        filename: str = "performance"
    ) -> str:
        """
        Export performance data.
        
        Args:
            performance_data: List of performance models
            format_type: Export format (csv, json, xlsx)
            filename: Base filename
            
        Returns:
            Path to exported file
        """
        if format_type.lower() == "csv":
            return self.export_to_csv(performance_data, filename)
        elif format_type.lower() == "json":
            return self.export_to_json(performance_data, filename)
        elif format_type.lower() == "xlsx":
            return self.export_to_excel(performance_data, filename, "Performance")
        else:
            raise ValueError(f"Unsupported format: {format_type}")
    
    def export_analytics_data(
        self, 
        analytics_data: List[Union[TrafficData, ConversionEvent]], 
        format_type: str = "csv",
        filename: str = "analytics"
    ) -> str:
        """
        Export analytics data.
        
        Args:
            analytics_data: List of analytics models
            format_type: Export format (csv, json, xlsx)
            filename: Base filename
            
        Returns:
            Path to exported file
        """
        if format_type.lower() == "csv":
            return self.export_to_csv(analytics_data, filename)
        elif format_type.lower() == "json":
            return self.export_to_json(analytics_data, filename)
        elif format_type.lower() == "xlsx":
            return self.export_to_excel(analytics_data, filename, "Analytics")
        else:
            raise ValueError(f"Unsupported format: {format_type}")


class MultiSheetExporter:
    """Exporter for creating multi-sheet Excel reports."""
    
    def __init__(self, config: Optional[ExportConfig] = None):
        """
        Initialize multi-sheet exporter.
        
        Args:
            config: Export configuration
        """
        self.config = config or ExportConfig()
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.exporter = DataExporter(config)
    
    def export_marketing_report(
        self,
        campaigns: Optional[List[Campaign]] = None,
        ad_groups: Optional[List[AdGroup]] = None,
        keywords: Optional[List[Keyword]] = None,
        performance_data: Optional[List[DailyMetrics]] = None,
        analytics_data: Optional[List[TrafficData]] = None,
        filename: str = "marketing_report"
    ) -> str:
        """
        Export comprehensive marketing report to multi-sheet Excel.
        
        Args:
            campaigns: Campaign data
            ad_groups: Ad group data
            keywords: Keyword data
            performance_data: Performance metrics
            analytics_data: Analytics data
            filename: Base filename
            
        Returns:
            Path to exported file
        """
        try:
            full_filename = self.exporter._generate_filename(filename, "xlsx")
            file_path = self.output_dir / full_filename
            
            with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
                # Export each data type to separate sheets
                if campaigns:
                    campaigns_data = self.exporter._prepare_data_for_export(campaigns)
                    pd.DataFrame(campaigns_data).to_excel(writer, sheet_name='Campaigns', index=False)
                
                if ad_groups:
                    ad_groups_data = self.exporter._prepare_data_for_export(ad_groups)
                    pd.DataFrame(ad_groups_data).to_excel(writer, sheet_name='Ad Groups', index=False)
                
                if keywords:
                    keywords_data = self.exporter._prepare_data_for_export(keywords)
                    pd.DataFrame(keywords_data).to_excel(writer, sheet_name='Keywords', index=False)
                
                if performance_data:
                    performance_data_prepared = self.exporter._prepare_data_for_export(performance_data)
                    pd.DataFrame(performance_data_prepared).to_excel(writer, sheet_name='Performance', index=False)
                
                if analytics_data:
                    analytics_data_prepared = self.exporter._prepare_data_for_export(analytics_data)
                    pd.DataFrame(analytics_data_prepared).to_excel(writer, sheet_name='Analytics', index=False)
                
                # Create summary sheet
                self._create_summary_sheet(writer, campaigns, performance_data, analytics_data)
            
            logger.info(f"Exported marketing report to {file_path}")
            return str(file_path)
            
        except Exception as e:
            logger.error(f"Failed to export marketing report: {e}")
            return ""
    
    def _create_summary_sheet(
        self,
        writer: pd.ExcelWriter,
        campaigns: Optional[List[Campaign]] = None,
        performance_data: Optional[List[DailyMetrics]] = None,
        analytics_data: Optional[List[TrafficData]] = None
    ) -> None:
        """Create summary sheet with key metrics."""
        try:
            summary_data = []
            
            # Campaign summary
            if campaigns:
                total_campaigns = len(campaigns)
                active_campaigns = len([c for c in campaigns if c.status.value == "ENABLED"])
                total_impressions = sum(c.impressions or 0 for c in campaigns)
                total_clicks = sum(c.clicks or 0 for c in campaigns)
                total_cost = sum(float(c.cost or 0) for c in campaigns)
                
                summary_data.extend([
                    {"Metric": "Total Campaigns", "Value": total_campaigns},
                    {"Metric": "Active Campaigns", "Value": active_campaigns},
                    {"Metric": "Total Impressions", "Value": total_impressions},
                    {"Metric": "Total Clicks", "Value": total_clicks},
                    {"Metric": "Total Cost", "Value": f"${total_cost:,.2f}"},
                ])
            
            # Performance summary
            if performance_data:
                date_range_start = min(p.date for p in performance_data)
                date_range_end = max(p.date for p in performance_data)
                
                summary_data.extend([
                    {"Metric": "Date Range Start", "Value": date_range_start.strftime(self.config.date_format)},
                    {"Metric": "Date Range End", "Value": date_range_end.strftime(self.config.date_format)},
                    {"Metric": "Performance Records", "Value": len(performance_data)},
                ])
            
            # Analytics summary
            if analytics_data:
                total_sessions = sum(a.sessions for a in analytics_data)
                total_users = sum(a.users for a in analytics_data)
                
                summary_data.extend([
                    {"Metric": "Total Sessions", "Value": total_sessions},
                    {"Metric": "Total Users", "Value": total_users},
                    {"Metric": "Analytics Records", "Value": len(analytics_data)},
                ])
            
            # Export timestamp
            summary_data.append({
                "Metric": "Export Timestamp", 
                "Value": datetime.now().strftime(self.config.datetime_format)
            })
            
            if summary_data:
                pd.DataFrame(summary_data).to_excel(writer, sheet_name='Summary', index=False)
            
        except Exception as e:
            logger.error(f"Failed to create summary sheet: {e}")


class ReportGenerator:
    """High-level report generator with predefined formats."""
    
    def __init__(self, config: Optional[ExportConfig] = None):
        """
        Initialize report generator.
        
        Args:
            config: Export configuration
        """
        self.config = config or ExportConfig()
        self.exporter = DataExporter(config)
        self.multi_exporter = MultiSheetExporter(config)
    
    def generate_campaign_performance_report(
        self,
        campaigns: List[Campaign],
        performance_data: List[DailyMetrics],
        format_type: str = "xlsx",
        filename: str = "campaign_performance_report"
    ) -> str:
        """
        Generate campaign performance report.
        
        Args:
            campaigns: Campaign data
            performance_data: Performance metrics
            format_type: Export format
            filename: Base filename
            
        Returns:
            Path to generated report
        """
        if format_type.lower() == "xlsx":
            return self.multi_exporter.export_marketing_report(
                campaigns=campaigns,
                performance_data=performance_data,
                filename=filename
            )
        else:
            # For other formats, combine data
            combined_data = []
            
            # Create lookup for campaign data
            campaign_lookup = {c.id: c for c in campaigns}
            
            for perf in performance_data:
                campaign = campaign_lookup.get(perf.campaign_id)
                if campaign:
                    combined_record = {
                        # Campaign data
                        "campaign_name": campaign.name,
                        "campaign_status": campaign.status.value,
                        "channel_type": campaign.advertising_channel_type.value,
                        # Performance data
                        "date": perf.date,
                        "impressions": perf.impressions,
                        "clicks": perf.clicks,
                        "cost": float(perf.cost),
                        "conversions": perf.conversions,
                        "ctr": perf.ctr,
                        "conversion_rate": perf.conversion_rate,
                    }
                    combined_data.append(combined_record)
            
            return self.exporter.export_to_csv(combined_data, filename) if format_type.lower() == "csv" else \
                   self.exporter.export_to_json(combined_data, filename)
    
    def generate_keyword_analysis_report(
        self,
        keywords: List[Keyword],
        format_type: str = "xlsx",
        filename: str = "keyword_analysis_report"
    ) -> str:
        """
        Generate keyword analysis report.
        
        Args:
            keywords: Keyword data
            format_type: Export format
            filename: Base filename
            
        Returns:
            Path to generated report
        """
        if format_type.lower() == "xlsx":
            return self.multi_exporter.export_marketing_report(
                keywords=keywords,
                filename=filename
            )
        else:
            return self.exporter.export_keywords(keywords, format_type, filename)
    
    def generate_analytics_overview_report(
        self,
        traffic_data: List[TrafficData],
        conversion_events: List[ConversionEvent],
        format_type: str = "xlsx",
        filename: str = "analytics_overview_report"
    ) -> str:
        """
        Generate analytics overview report.
        
        Args:
            traffic_data: Traffic data
            conversion_events: Conversion events
            format_type: Export format
            filename: Base filename
            
        Returns:
            Path to generated report
        """
        if format_type.lower() == "xlsx":
            return self.multi_exporter.export_marketing_report(
                analytics_data=traffic_data,
                filename=filename
            )
        else:
            # Combine traffic and conversion data
            combined_data = traffic_data + conversion_events
            return self.exporter.export_analytics_data(combined_data, format_type, filename)