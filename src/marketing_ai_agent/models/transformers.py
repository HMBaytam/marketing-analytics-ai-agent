"""Data transformation functions to convert API responses to structured models."""

import logging
from datetime import date, datetime
from decimal import Decimal
from typing import Any

from ..models.analytics import (
    ChannelGrouping,
    ConversionEvent,
    DeviceCategory,
    TrafficData,
)
from ..models.campaign import (
    AdGroup,
    AdGroupStatus,
    AdGroupType,
    AdvertisingChannelType,
    Campaign,
    CampaignPerformance,
    CampaignStatus,
    Keyword,
    KeywordMatchType,
    KeywordStatus,
)
from ..models.metrics import DailyMetrics

logger = logging.getLogger(__name__)


class GoogleAdsTransformer:
    """Transformer for Google Ads API responses."""

    @staticmethod
    def transform_campaign(row: Any, customer_id: str) -> Campaign | None:
        """
        Transform Google Ads API campaign row to Campaign model.

        Args:
            row: Google Ads API response row
            customer_id: Customer account ID

        Returns:
            Campaign model instance or None if transformation fails
        """
        try:
            campaign = row.campaign
            metrics = getattr(row, "metrics", None)

            # Parse dates
            start_date = None
            end_date = None

            if hasattr(campaign, "start_date") and campaign.start_date:
                start_date = datetime.strptime(campaign.start_date, "%Y-%m-%d").date()

            if hasattr(campaign, "end_date") and campaign.end_date:
                end_date = datetime.strptime(campaign.end_date, "%Y-%m-%d").date()

            # Parse status
            status = CampaignStatus.UNKNOWN
            if hasattr(campaign, "status"):
                try:
                    status = CampaignStatus(campaign.status.name)
                except ValueError:
                    logger.warning(f"Unknown campaign status: {campaign.status.name}")

            # Parse channel type
            channel_type = AdvertisingChannelType.UNKNOWN
            if hasattr(campaign, "advertising_channel_type"):
                try:
                    channel_type = AdvertisingChannelType(
                        campaign.advertising_channel_type.name
                    )
                except ValueError:
                    logger.warning(
                        f"Unknown channel type: {campaign.advertising_channel_type.name}"
                    )

            # Create campaign model
            campaign_model = Campaign(
                id=str(campaign.id),
                name=campaign.name,
                status=status,
                advertising_channel_type=channel_type,
                start_date=start_date,
                end_date=end_date,
                customer_id=customer_id,
            )

            # Add metrics if available
            if metrics:
                campaign_model.impressions = getattr(metrics, "impressions", 0)
                campaign_model.clicks = getattr(metrics, "clicks", 0)
                campaign_model.conversions = getattr(metrics, "conversions", 0.0)
                campaign_model.cost_micros = getattr(metrics, "cost_micros", 0)

                # Calculate derived metrics
                campaign_model.calculate_metrics()

            return campaign_model

        except Exception as e:
            logger.error(f"Failed to transform campaign: {e}")
            return None

    @staticmethod
    def transform_ad_group(row: Any, customer_id: str) -> AdGroup | None:
        """
        Transform Google Ads API ad group row to AdGroup model.

        Args:
            row: Google Ads API response row
            customer_id: Customer account ID

        Returns:
            AdGroup model instance or None if transformation fails
        """
        try:
            ad_group = row.ad_group
            campaign = row.campaign
            metrics = getattr(row, "metrics", None)

            # Parse status
            status = AdGroupStatus.UNKNOWN
            if hasattr(ad_group, "status"):
                try:
                    status = AdGroupStatus(ad_group.status.name)
                except ValueError:
                    logger.warning(f"Unknown ad group status: {ad_group.status.name}")

            # Parse type
            ad_group_type = AdGroupType.UNKNOWN
            if hasattr(ad_group, "type_"):
                try:
                    ad_group_type = AdGroupType(ad_group.type_.name)
                except ValueError:
                    logger.warning(f"Unknown ad group type: {ad_group.type_.name}")

            # Create ad group model
            ad_group_model = AdGroup(
                id=str(ad_group.id),
                name=ad_group.name,
                status=status,
                type=ad_group_type,
                campaign_id=str(campaign.id),
                campaign_name=campaign.name,
                customer_id=customer_id,
            )

            # Add bidding information
            if hasattr(ad_group, "cpc_bid_micros"):
                ad_group_model.cpc_bid_micros = ad_group.cpc_bid_micros

            # Add metrics if available
            if metrics:
                ad_group_model.impressions = getattr(metrics, "impressions", 0)
                ad_group_model.clicks = getattr(metrics, "clicks", 0)
                ad_group_model.conversions = getattr(metrics, "conversions", 0.0)
                ad_group_model.cost_micros = getattr(metrics, "cost_micros", 0)

                # Calculate derived metrics
                ad_group_model.calculate_metrics()

            return ad_group_model

        except Exception as e:
            logger.error(f"Failed to transform ad group: {e}")
            return None

    @staticmethod
    def transform_keyword(row: Any, customer_id: str) -> Keyword | None:
        """
        Transform Google Ads API keyword row to Keyword model.

        Args:
            row: Google Ads API response row
            customer_id: Customer account ID

        Returns:
            Keyword model instance or None if transformation fails
        """
        try:
            keyword = row.ad_group_criterion
            ad_group = row.ad_group
            campaign = row.campaign
            metrics = getattr(row, "metrics", None)

            # Parse match type
            match_type = KeywordMatchType.UNKNOWN
            if hasattr(keyword.keyword, "match_type"):
                try:
                    match_type = KeywordMatchType(keyword.keyword.match_type.name)
                except ValueError:
                    logger.warning(
                        f"Unknown match type: {keyword.keyword.match_type.name}"
                    )

            # Parse status
            status = KeywordStatus.UNKNOWN
            if hasattr(keyword, "status"):
                try:
                    status = KeywordStatus(keyword.status.name)
                except ValueError:
                    logger.warning(f"Unknown keyword status: {keyword.status.name}")

            # Create keyword model
            keyword_model = Keyword(
                text=keyword.keyword.text,
                match_type=match_type,
                status=status,
                ad_group_id=str(ad_group.id),
                ad_group_name=ad_group.name,
                campaign_id=str(campaign.id),
                campaign_name=campaign.name,
                customer_id=customer_id,
            )

            # Add quality score
            if hasattr(keyword, "quality_info") and keyword.quality_info:
                keyword_model.quality_score = getattr(
                    keyword.quality_info, "quality_score", None
                )

            # Add metrics if available
            if metrics:
                keyword_model.impressions = getattr(metrics, "impressions", 0)
                keyword_model.clicks = getattr(metrics, "clicks", 0)
                keyword_model.conversions = getattr(metrics, "conversions", 0.0)
                keyword_model.cost_micros = getattr(metrics, "cost_micros", 0)

                # Calculate derived metrics
                keyword_model.calculate_metrics()

            return keyword_model

        except Exception as e:
            logger.error(f"Failed to transform keyword: {e}")
            return None

    @staticmethod
    def transform_performance_data(
        row: Any, customer_id: str
    ) -> CampaignPerformance | None:
        """
        Transform performance data with date dimension.

        Args:
            row: Google Ads API response row
            customer_id: Customer account ID

        Returns:
            CampaignPerformance model instance or None if transformation fails
        """
        try:
            campaign = row.campaign
            segments = row.segments
            metrics = row.metrics

            # Parse date
            performance_date = datetime.strptime(segments.date, "%Y-%m-%d").date()

            # Create performance model
            performance_model = CampaignPerformance(
                date=performance_date,
                campaign_id=str(campaign.id),
                campaign_name=campaign.name,
                customer_id=customer_id,
                impressions=getattr(metrics, "impressions", 0),
                clicks=getattr(metrics, "clicks", 0),
                conversions=getattr(metrics, "conversions", 0.0),
                cost_micros=getattr(metrics, "cost_micros", 0),
            )

            # Calculate derived metrics
            performance_model.calculate_metrics()

            return performance_model

        except Exception as e:
            logger.error(f"Failed to transform performance data: {e}")
            return None


class GA4Transformer:
    """Transformer for Google Analytics 4 API responses."""

    @staticmethod
    def transform_traffic_data(row: Any, property_id: str) -> TrafficData | None:
        """
        Transform GA4 API traffic row to TrafficData model.

        Args:
            row: GA4 API response row
            property_id: GA4 property ID

        Returns:
            TrafficData model instance or None if transformation fails
        """
        try:
            # Extract dimensions
            dimensions = {}
            for i, dim_value in enumerate(row.dimension_values):
                dimensions[f"dim_{i}"] = dim_value.value

            # Extract metrics
            metrics = {}
            for i, metric_value in enumerate(row.metric_values):
                metrics[f"metric_{i}"] = metric_value.value

            # Parse date (assuming first dimension is date)
            traffic_date = datetime.strptime(
                dimensions.get("dim_0", ""), "%Y%m%d"
            ).date()

            # Map dimensions and metrics based on expected order
            # This mapping should match the order in the original query
            traffic_model = TrafficData(
                date=traffic_date,
                property_id=property_id,
                country=dimensions.get("dim_1"),
                device_category=GA4Transformer._parse_device_category(
                    dimensions.get("dim_2")
                ),
                channel_grouping=GA4Transformer._parse_channel_grouping(
                    dimensions.get("dim_3")
                ),
                sessions=int(metrics.get("metric_0", 0)),
                users=int(metrics.get("metric_1", 0)),
                new_users=int(metrics.get("metric_2", 0)),
                pageviews=int(metrics.get("metric_3", 0)),
                bounce_rate=float(metrics.get("metric_4", 0))
                if metrics.get("metric_4")
                else None,
                average_session_duration=float(metrics.get("metric_5", 0))
                if metrics.get("metric_5")
                else None,
            )

            return traffic_model

        except Exception as e:
            logger.error(f"Failed to transform traffic data: {e}")
            return None

    @staticmethod
    def transform_conversion_event(
        row: Any, property_id: str
    ) -> ConversionEvent | None:
        """
        Transform GA4 API conversion row to ConversionEvent model.

        Args:
            row: GA4 API response row
            property_id: GA4 property ID

        Returns:
            ConversionEvent model instance or None if transformation fails
        """
        try:
            # Extract dimensions
            dimensions = {}
            for i, dim_value in enumerate(row.dimension_values):
                dimensions[f"dim_{i}"] = dim_value.value

            # Extract metrics
            metrics = {}
            for i, metric_value in enumerate(row.metric_values):
                metrics[f"metric_{i}"] = metric_value.value

            # Parse date
            event_date = datetime.strptime(dimensions.get("dim_0", ""), "%Y%m%d").date()

            # Create conversion event model
            conversion_model = ConversionEvent(
                date=event_date,
                property_id=property_id,
                event_name=dimensions.get("dim_1", ""),
                channel_grouping=GA4Transformer._parse_channel_grouping(
                    dimensions.get("dim_2")
                ),
                conversions=float(metrics.get("metric_0", 0)),
                total_revenue=Decimal(metrics.get("metric_1", 0))
                if metrics.get("metric_1")
                else None,
                purchase_revenue=Decimal(metrics.get("metric_2", 0))
                if metrics.get("metric_2")
                else None,
                event_count=int(metrics.get("metric_3", 0)),
            )

            return conversion_model

        except Exception as e:
            logger.error(f"Failed to transform conversion event: {e}")
            return None

    @staticmethod
    def _parse_device_category(value: str | None) -> DeviceCategory | None:
        """Parse device category from GA4 response."""
        if not value:
            return None

        try:
            return DeviceCategory(value.lower())
        except ValueError:
            logger.warning(f"Unknown device category: {value}")
            return DeviceCategory.UNKNOWN

    @staticmethod
    def _parse_channel_grouping(value: str | None) -> ChannelGrouping | None:
        """Parse channel grouping from GA4 response."""
        if not value:
            return None

        # Map common GA4 channel groupings to our enum
        channel_mapping = {
            "Direct": ChannelGrouping.DIRECT,
            "Organic Search": ChannelGrouping.ORGANIC_SEARCH,
            "Paid Search": ChannelGrouping.PAID_SEARCH,
            "Paid Social": ChannelGrouping.PAID_SOCIAL,
            "Organic Social": ChannelGrouping.ORGANIC_SOCIAL,
            "Email": ChannelGrouping.EMAIL,
            "Affiliates": ChannelGrouping.AFFILIATES,
            "Referral": ChannelGrouping.REFERRAL,
            "Display": ChannelGrouping.DISPLAY,
            "(not set)": ChannelGrouping.UNASSIGNED,
        }

        return channel_mapping.get(value, ChannelGrouping.UNASSIGNED)


class DataCleaner:
    """Utility class for cleaning and validating transformed data."""

    @staticmethod
    def clean_campaign_list(campaigns: list[Campaign]) -> list[Campaign]:
        """
        Clean and validate a list of campaigns.

        Args:
            campaigns: List of campaign models

        Returns:
            Cleaned list of campaigns
        """
        cleaned = []

        for campaign in campaigns:
            try:
                # Validate required fields
                if not campaign.id or not campaign.name:
                    logger.warning(
                        f"Skipping campaign with missing ID or name: {campaign}"
                    )
                    continue

                # Clean metrics
                campaign.impressions = max(0, campaign.impressions or 0)
                campaign.clicks = max(0, campaign.clicks or 0)
                campaign.conversions = max(0.0, campaign.conversions or 0.0)
                campaign.cost_micros = max(0, campaign.cost_micros or 0)

                # Recalculate metrics after cleaning
                campaign.calculate_metrics()

                cleaned.append(campaign)

            except Exception as e:
                logger.error(f"Error cleaning campaign {campaign.id}: {e}")
                continue

        return cleaned

    @staticmethod
    def clean_metrics_list(metrics: list[DailyMetrics]) -> list[DailyMetrics]:
        """
        Clean and validate a list of daily metrics.

        Args:
            metrics: List of daily metrics models

        Returns:
            Cleaned list of metrics
        """
        cleaned = []

        for metric in metrics:
            try:
                # Validate required fields
                if not metric.entity_id or not metric.date:
                    logger.warning(f"Skipping metric with missing ID or date: {metric}")
                    continue

                # Clean metric values
                metric.impressions = max(0, metric.impressions)
                metric.clicks = max(0, metric.clicks)
                metric.conversions = max(0.0, metric.conversions)
                metric.cost_micros = max(0, metric.cost_micros)

                cleaned.append(metric)

            except Exception as e:
                logger.error(f"Error cleaning metric for {metric.entity_id}: {e}")
                continue

        return cleaned

    @staticmethod
    def remove_duplicates(data: list[Any], key_func) -> list[Any]:
        """
        Remove duplicates from a list based on a key function.

        Args:
            data: List of data objects
            key_func: Function to extract unique key from each object

        Returns:
            List with duplicates removed
        """
        seen = set()
        unique_data = []

        for item in data:
            key = key_func(item)
            if key not in seen:
                seen.add(key)
                unique_data.append(item)
            else:
                logger.debug(f"Removing duplicate item with key: {key}")

        return unique_data

    @staticmethod
    def fill_missing_dates(
        data: list[Any],
        start_date: date,
        end_date: date,
        date_attr: str = "date",
        fill_func=None,
    ) -> list[Any]:
        """
        Fill missing dates in time series data.

        Args:
            data: List of data objects with date attribute
            start_date: Start date for the range
            end_date: End date for the range
            date_attr: Name of the date attribute
            fill_func: Optional function to create fill objects for missing dates

        Returns:
            List with missing dates filled
        """
        if not data or not fill_func:
            return data

        # Create a dictionary of existing data by date
        data_by_date = {getattr(item, date_attr): item for item in data}

        # Generate all dates in range
        current_date = start_date
        filled_data = []

        while current_date <= end_date:
            if current_date in data_by_date:
                filled_data.append(data_by_date[current_date])
            else:
                # Create fill object for missing date
                fill_object = fill_func(current_date)
                if fill_object:
                    filled_data.append(fill_object)
                    logger.debug(f"Filled missing date: {current_date}")

            # Move to next date
            current_date = date.fromordinal(current_date.toordinal() + 1)

        return filled_data
