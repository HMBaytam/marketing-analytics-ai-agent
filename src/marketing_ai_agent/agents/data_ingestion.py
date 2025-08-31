"""Data Ingestion Agent for fetching marketing data from APIs."""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any

from langchain_anthropic import ChatAnthropic
from pydantic import BaseModel

from ..api_clients.google_ads_client import GoogleAdsAPIClient
from ..api_clients.ga4_client import GA4APIClient
from ..models.transformers import GoogleAdsTransformer, GA4Transformer
from ..models.cache import CacheManager, CacheConfig
from .orchestrator import AgentState

logger = logging.getLogger(__name__)


class DataIngestionAgent:
    """Agent responsible for fetching and preparing marketing data."""
    
    def __init__(self, llm: Optional[ChatAnthropic] = None):
        """
        Initialize data ingestion agent.
        
        Args:
            llm: Language model for query interpretation
        """
        self.llm = llm or ChatAnthropic(
            model="claude-3-haiku-20240307",
            temperature=0.1,
            max_tokens=2000
        )
        
        # Initialize API clients
        self.google_ads_client = None
        self.ga4_client = None
        
        # Initialize transformers and cache
        self.google_ads_transformer = GoogleAdsTransformer()
        self.ga4_transformer = GA4Transformer()
        self.cache_manager = CacheManager(CacheConfig(
            cache_dir="./.cache/marketing_data",
            default_ttl=1800,  # 30 minutes
            max_cache_size=50
        ))
    
    async def ingest_data(self, state: AgentState) -> AgentState:
        """
        Ingest marketing data based on user query and task plan.
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated state with ingested data
        """
        try:
            logger.info("Starting data ingestion process")
            
            # Determine required data sources
            data_requirements = await self._analyze_data_requirements(state)
            
            # Initialize clients if needed
            await self._initialize_clients()
            
            # Fetch data concurrently
            tasks = []
            
            if data_requirements.get("google_ads", False):
                tasks.append(self._fetch_google_ads_data(state, data_requirements))
            
            if data_requirements.get("ga4", False):
                tasks.append(self._fetch_ga4_data(state, data_requirements))
            
            # Execute data fetching
            if tasks:
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Process results
                for i, result in enumerate(results):
                    if isinstance(result, Exception):
                        logger.error(f"Data ingestion task {i} failed: {result}")
                        state.error_messages.append(f"Data fetch error: {str(result)}")
            
            # Log ingestion summary
            data_summary = self._create_ingestion_summary(state)
            logger.info(f"Data ingestion completed: {data_summary}")
            
            return state
            
        except Exception as e:
            logger.error(f"Data ingestion failed: {e}")
            state.error_messages.append(f"Data ingestion error: {str(e)}")
            return state
    
    async def _analyze_data_requirements(self, state: AgentState) -> Dict[str, Any]:
        """
        Analyze user query to determine required data sources.
        
        Args:
            state: Current workflow state
            
        Returns:
            Dictionary specifying data requirements
        """
        query = state.user_query.lower()
        
        # Default requirements
        requirements = {
            "google_ads": False,
            "ga4": False,
            "metrics": [],
            "dimensions": [],
            "entities": []
        }
        
        # Google Ads indicators
        google_ads_keywords = [
            "campaign", "ad group", "keyword", "ads", "cpc", "impression",
            "click", "spend", "cost", "paid search", "google ads", "adwords"
        ]
        
        if any(keyword in query for keyword in google_ads_keywords):
            requirements["google_ads"] = True
            requirements["metrics"].extend(["impressions", "clicks", "cost", "conversions"])
            requirements["dimensions"].extend(["campaign", "ad_group", "keyword"])
        
        # GA4 indicators
        ga4_keywords = [
            "traffic", "session", "user", "pageview", "event", "conversion",
            "organic", "referral", "direct", "analytics", "ga4"
        ]
        
        if any(keyword in query for keyword in ga4_keywords):
            requirements["ga4"] = True
            requirements["metrics"].extend(["sessions", "users", "pageviews", "events"])
            requirements["dimensions"].extend(["source", "medium", "channel"])
        
        # Performance analysis typically needs both
        performance_keywords = [
            "performance", "efficiency", "roi", "roas", "attribution", "contribution"
        ]
        
        if any(keyword in query for keyword in performance_keywords):
            requirements["google_ads"] = True
            requirements["ga4"] = True
        
        # Entity-specific requirements
        if "channel" in query:
            requirements["entities"].append("channel_performance")
        if "conversion" in query:
            requirements["entities"].append("conversion_events")
        
        logger.info(f"Data requirements: {requirements}")
        return requirements
    
    async def _initialize_clients(self):
        """Initialize API clients if not already done."""
        try:
            if self.google_ads_client is None:
                self.google_ads_client = GoogleAdsAPIClient()
            
            if self.ga4_client is None:
                self.ga4_client = GA4APIClient()
                
        except Exception as e:
            logger.error(f"Client initialization failed: {e}")
            raise
    
    async def _fetch_google_ads_data(self, state: AgentState, requirements: Dict[str, Any]) -> None:
        """
        Fetch Google Ads data.
        
        Args:
            state: Current workflow state
            requirements: Data requirements specification
        """
        try:
            if not self.google_ads_client:
                raise ValueError("Google Ads client not initialized")
            
            logger.info("Fetching Google Ads data")
            
            # Default customer ID (should be configurable)
            customer_id = "1234567890"  # Replace with actual customer ID
            
            date_range = state.date_range or self._get_default_date_range()
            start_date = date_range[0].strftime("%Y-%m-%d")
            end_date = date_range[1].strftime("%Y-%m-%d")
            
            # Check cache first
            cache_key_params = {
                "customer_id": customer_id,
                "start_date": start_date,
                "end_date": end_date,
                "metrics": sorted(requirements.get("metrics", []))
            }
            
            cached_data = await self.cache_manager.get_google_ads_campaigns(
                customer_id, cache_key_params
            )
            
            if cached_data:
                logger.info("Using cached Google Ads data")
                state.google_ads_data = cached_data
                return
            
            # Fetch fresh data
            campaigns = await self.google_ads_client.get_campaigns(
                customer_id=customer_id,
                start_date=start_date,
                end_date=end_date
            )
            
            if campaigns:
                # Transform to Pydantic models
                transformed_campaigns = []
                for campaign_row in campaigns:
                    campaign = self.google_ads_transformer.transform_campaign(campaign_row, customer_id)
                    if campaign:
                        transformed_campaigns.append(campaign)
                
                state.google_ads_data = transformed_campaigns
                
                # Cache the results
                await self.cache_manager.set_google_ads_campaigns(
                    customer_id, cache_key_params, transformed_campaigns
                )
                
                logger.info(f"Fetched {len(transformed_campaigns)} campaigns from Google Ads")
            else:
                logger.warning("No Google Ads campaigns found")
                state.error_messages.append("No Google Ads data available for the specified period")
                
        except Exception as e:
            logger.error(f"Google Ads data fetch failed: {e}")
            state.error_messages.append(f"Google Ads fetch error: {str(e)}")
    
    async def _fetch_ga4_data(self, state: AgentState, requirements: Dict[str, Any]) -> None:
        """
        Fetch GA4 data.
        
        Args:
            state: Current workflow state
            requirements: Data requirements specification
        """
        try:
            if not self.ga4_client:
                raise ValueError("GA4 client not initialized")
            
            logger.info("Fetching GA4 data")
            
            # Default property ID (should be configurable)
            property_id = "properties/123456789"  # Replace with actual property ID
            
            date_range = state.date_range or self._get_default_date_range()
            start_date = date_range[0].strftime("%Y-%m-%d")
            end_date = date_range[1].strftime("%Y-%m-%d")
            
            # Check cache first
            cache_key_params = {
                "property_id": property_id,
                "start_date": start_date,
                "end_date": end_date,
                "metrics": sorted(requirements.get("metrics", []))
            }
            
            cached_data = await self.cache_manager.get_ga4_traffic(
                property_id, cache_key_params
            )
            
            if cached_data:
                logger.info("Using cached GA4 data")
                state.ga4_data = cached_data
                return
            
            # Fetch fresh data
            traffic_data = await self.ga4_client.get_traffic_data(
                property_id=property_id,
                start_date=start_date,
                end_date=end_date,
                dimensions=["date", "sessionDefaultChannelGroup"],
                metrics=["sessions", "users", "screenPageViews", "conversions"]
            )
            
            if traffic_data:
                # Transform to Pydantic models
                transformed_data = []
                for row in traffic_data:
                    traffic = self.ga4_transformer.transform_traffic_data(row)
                    if traffic:
                        transformed_data.append(traffic)
                
                state.ga4_data = transformed_data
                
                # Cache the results
                await self.cache_manager.set_ga4_traffic(
                    property_id, cache_key_params, transformed_data
                )
                
                logger.info(f"Fetched {len(transformed_data)} traffic records from GA4")
            else:
                logger.warning("No GA4 data found")
                state.error_messages.append("No GA4 data available for the specified period")
                
        except Exception as e:
            logger.error(f"GA4 data fetch failed: {e}")
            state.error_messages.append(f"GA4 fetch error: {str(e)}")
    
    def _get_default_date_range(self) -> tuple[datetime, datetime]:
        """Get default date range (last 30 days)."""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        return (start_date, end_date)
    
    def _create_ingestion_summary(self, state: AgentState) -> str:
        """Create summary of ingested data."""
        summary_parts = []
        
        if state.google_ads_data:
            summary_parts.append(f"{len(state.google_ads_data)} Google Ads campaigns")
        
        if state.ga4_data:
            summary_parts.append(f"{len(state.ga4_data)} GA4 traffic records")
        
        if state.performance_metrics:
            summary_parts.append(f"{len(state.performance_metrics)} performance metrics")
        
        if state.conversion_events:
            summary_parts.append(f"{len(state.conversion_events)} conversion events")
        
        return ", ".join(summary_parts) if summary_parts else "No data ingested"
    
    async def validate_data_quality(self, state: AgentState) -> Dict[str, Any]:
        """
        Validate quality of ingested data.
        
        Args:
            state: Current workflow state
            
        Returns:
            Data quality report
        """
        quality_report = {
            "overall_quality": "good",
            "issues": [],
            "recommendations": []
        }
        
        try:
            # Check Google Ads data quality
            if state.google_ads_data:
                for campaign in state.google_ads_data:
                    if not campaign.impressions and not campaign.clicks:
                        quality_report["issues"].append(
                            f"Campaign '{campaign.name}' has no impression or click data"
                        )
            
            # Check GA4 data quality
            if state.ga4_data:
                total_sessions = sum(traffic.sessions for traffic in state.ga4_data)
                if total_sessions == 0:
                    quality_report["issues"].append("No GA4 session data found")
            
            # Overall quality assessment
            if len(quality_report["issues"]) > 3:
                quality_report["overall_quality"] = "poor"
                quality_report["recommendations"].append("Consider expanding date range or checking data sources")
            elif len(quality_report["issues"]) > 1:
                quality_report["overall_quality"] = "fair"
                quality_report["recommendations"].append("Some data gaps detected, results may be incomplete")
            
            return quality_report
            
        except Exception as e:
            logger.error(f"Data quality validation failed: {e}")
            quality_report["overall_quality"] = "unknown"
            quality_report["issues"].append(f"Quality validation error: {str(e)}")
            return quality_report