"""Factory classes for generating test data using Factory Boy."""

import random
from datetime import timedelta
from typing import Any

import factory
from faker import Faker

fake = Faker()


class CampaignDataFactory(factory.DictFactory):
    """Factory for generating campaign data."""

    id = factory.LazyFunction(lambda: f"campaign_{random.randint(10000, 99999)}")
    name = factory.LazyFunction(lambda: f"{fake.company()} - {fake.catch_phrase()}")
    status = factory.LazyFunction(
        lambda: random.choice(["ENABLED", "PAUSED", "REMOVED"])
    )
    budget = factory.LazyFunction(lambda: round(random.uniform(100.0, 10000.0), 2))
    currency = "USD"

    # Performance metrics
    impressions = factory.LazyFunction(lambda: random.randint(1000, 100000))
    clicks = factory.LazyAttribute(
        lambda obj: int(obj.impressions * random.uniform(0.01, 0.10))
    )
    cost = factory.LazyAttribute(
        lambda obj: round(obj.clicks * random.uniform(0.50, 3.00), 2)
    )
    conversions = factory.LazyAttribute(
        lambda obj: int(obj.clicks * random.uniform(0.01, 0.15))
    )

    # Calculated metrics
    ctr = factory.LazyAttribute(
        lambda obj: round((obj.clicks / obj.impressions) * 100, 2)
    )
    cpc = factory.LazyAttribute(
        lambda obj: round(obj.cost / obj.clicks, 2) if obj.clicks > 0 else 0
    )
    cpa = factory.LazyAttribute(
        lambda obj: round(obj.cost / obj.conversions, 2) if obj.conversions > 0 else 0
    )
    conversion_rate = factory.LazyAttribute(
        lambda obj: round((obj.conversions / obj.clicks) * 100, 2)
        if obj.clicks > 0
        else 0
    )

    # Dates
    start_date = factory.LazyFunction(
        lambda: fake.date_between(start_date="-1y", end_date="-1m")
    )
    end_date = factory.LazyAttribute(
        lambda obj: obj.start_date + timedelta(days=random.randint(30, 365))
    )
    created_at = factory.LazyFunction(
        lambda: fake.date_time_between(start_date="-2y", end_date="now")
    )
    updated_at = factory.LazyAttribute(
        lambda obj: fake.date_time_between(start_date=obj.created_at, end_date="now")
    )

    # Campaign settings
    bid_strategy = factory.LazyFunction(
        lambda: random.choice(
            [
                "TARGET_CPA",
                "TARGET_ROAS",
                "MAXIMIZE_CLICKS",
                "MAXIMIZE_CONVERSIONS",
                "MANUAL_CPC",
            ]
        )
    )
    target_cpa = factory.LazyFunction(lambda: round(random.uniform(10.0, 100.0), 2))
    target_roas = factory.LazyFunction(lambda: round(random.uniform(200.0, 800.0), 0))

    # Geographic and demographic data
    locations = factory.LazyFunction(
        lambda: random.sample(
            [
                "United States",
                "Canada",
                "United Kingdom",
                "Germany",
                "France",
                "Australia",
            ],
            random.randint(1, 3),
        )
    )

    languages = factory.LazyFunction(
        lambda: random.sample(
            ["English", "Spanish", "French", "German", "Italian"], random.randint(1, 2)
        )
    )


class KeywordDataFactory(factory.DictFactory):
    """Factory for generating keyword data."""

    campaign_id = factory.LazyFunction(
        lambda: f"campaign_{random.randint(10000, 99999)}"
    )
    ad_group_id = factory.LazyFunction(
        lambda: f"adgroup_{random.randint(10000, 99999)}"
    )
    keyword = factory.LazyFunction(
        lambda: " ".join(fake.words(nb=random.randint(1, 4)))
    )
    match_type = factory.LazyFunction(
        lambda: random.choice(["EXACT", "PHRASE", "BROAD"])
    )

    # Performance metrics
    impressions = factory.LazyFunction(lambda: random.randint(100, 10000))
    clicks = factory.LazyAttribute(
        lambda obj: int(obj.impressions * random.uniform(0.01, 0.20))
    )
    cost = factory.LazyAttribute(
        lambda obj: round(obj.clicks * random.uniform(0.25, 5.00), 2)
    )
    conversions = factory.LazyAttribute(
        lambda obj: int(obj.clicks * random.uniform(0.0, 0.25))
    )

    # Quality metrics
    quality_score = factory.LazyFunction(lambda: random.randint(1, 10))
    first_page_cpc = factory.LazyFunction(lambda: round(random.uniform(0.50, 10.00), 2))
    top_of_page_cpc = factory.LazyFunction(
        lambda: round(random.uniform(1.00, 15.00), 2)
    )

    # Status
    status = factory.LazyFunction(
        lambda: random.choice(["ENABLED", "PAUSED", "REMOVED"])
    )
    approval_status = factory.LazyFunction(
        lambda: random.choice(["APPROVED", "UNDER_REVIEW", "DISAPPROVED"])
    )


class AnalyticsDataFactory(factory.DictFactory):
    """Factory for generating GA4 analytics data."""

    date = factory.LazyFunction(
        lambda: fake.date_between(start_date="-1y", end_date="now")
    )
    source = factory.LazyFunction(
        lambda: random.choice(
            ["google", "facebook", "twitter", "linkedin", "direct", "organic", "email"]
        )
    )
    medium = factory.LazyFunction(
        lambda: random.choice(
            ["cpc", "social", "organic", "email", "referral", "direct"]
        )
    )
    campaign = factory.LazyFunction(
        lambda: f"campaign_{fake.word()}_{random.randint(1, 100)}"
    )

    # User metrics
    users = factory.LazyFunction(lambda: random.randint(100, 5000))
    new_users = factory.LazyAttribute(
        lambda obj: int(obj.users * random.uniform(0.3, 0.8))
    )
    sessions = factory.LazyAttribute(
        lambda obj: int(obj.users * random.uniform(1.0, 3.0))
    )
    bounce_rate = factory.LazyFunction(lambda: round(random.uniform(0.2, 0.8), 3))

    # Engagement metrics
    session_duration = factory.LazyFunction(lambda: random.randint(30, 600))  # seconds
    page_views = factory.LazyAttribute(
        lambda obj: int(obj.sessions * random.uniform(1.5, 8.0))
    )
    pages_per_session = factory.LazyAttribute(
        lambda obj: round(obj.page_views / obj.sessions, 2)
    )

    # Conversion metrics
    conversions = factory.LazyAttribute(
        lambda obj: int(obj.sessions * random.uniform(0.01, 0.20))
    )
    conversion_rate = factory.LazyAttribute(
        lambda obj: round((obj.conversions / obj.sessions) * 100, 2)
    )
    goal_completions = factory.LazyAttribute(
        lambda obj: obj.conversions + random.randint(0, 10)
    )

    # Revenue metrics
    revenue = factory.LazyAttribute(
        lambda obj: round(obj.conversions * random.uniform(10.0, 500.0), 2)
    )
    average_order_value = factory.LazyAttribute(
        lambda obj: round(obj.revenue / obj.conversions, 2)
        if obj.conversions > 0
        else 0
    )

    # Device and location data
    device_category = factory.LazyFunction(
        lambda: random.choice(["desktop", "mobile", "tablet"])
    )
    country = factory.LazyFunction(lambda: fake.country())
    city = factory.LazyFunction(lambda: fake.city())

    # Custom dimensions
    user_type = factory.LazyFunction(lambda: random.choice(["new", "returning"]))
    landing_page = factory.LazyFunction(lambda: f"/{fake.uri_path()}")


class OptimizationResultFactory(factory.DictFactory):
    """Factory for generating optimization results."""

    id = factory.LazyFunction(lambda: f"opt_{random.randint(100000, 999999)}")
    campaign_id = factory.LazyFunction(
        lambda: f"campaign_{random.randint(10000, 99999)}"
    )
    optimization_type = factory.LazyFunction(
        lambda: random.choice(
            [
                "BUDGET_OPTIMIZATION",
                "BID_OPTIMIZATION",
                "KEYWORD_OPTIMIZATION",
                "AUDIENCE_OPTIMIZATION",
                "AD_CREATIVE_OPTIMIZATION",
            ]
        )
    )

    # Recommendation details
    title = factory.LazyFunction(
        lambda: fake.sentence(nb_words=6)[:-1]
    )  # Remove period
    description = factory.LazyFunction(lambda: fake.paragraph(nb_sentences=3))
    confidence_score = factory.LazyFunction(lambda: round(random.uniform(0.5, 1.0), 3))
    priority = factory.LazyFunction(lambda: random.choice(["HIGH", "MEDIUM", "LOW"]))

    # Current state
    current_value = factory.LazyFunction(
        lambda: round(random.uniform(100.0, 10000.0), 2)
    )
    recommended_value = factory.LazyFunction(
        lambda: round(random.uniform(100.0, 10000.0), 2)
    )

    # Impact projections
    estimated_impact = factory.LazyFunction(
        lambda: round(random.uniform(-0.5, 2.0), 3)
    )  # -50% to +200%
    estimated_cost_change = factory.LazyFunction(
        lambda: round(random.uniform(-1000.0, 1000.0), 2)
    )
    estimated_conversion_change = factory.LazyFunction(lambda: random.randint(-50, 200))
    estimated_revenue_change = factory.LazyFunction(
        lambda: round(random.uniform(-5000.0, 15000.0), 2)
    )

    # Implementation
    implementation_difficulty = factory.LazyFunction(
        lambda: random.choice(["EASY", "MEDIUM", "HARD"])
    )
    estimated_time_to_implement = factory.LazyFunction(
        lambda: random.randint(1, 48)
    )  # hours

    # Status and dates
    status = factory.LazyFunction(
        lambda: random.choice(
            ["PENDING", "APPROVED", "IMPLEMENTED", "REJECTED", "EXPIRED"]
        )
    )
    created_at = factory.LazyFunction(
        lambda: fake.date_time_between(start_date="-30d", end_date="now")
    )
    expires_at = factory.LazyAttribute(
        lambda obj: obj.created_at + timedelta(days=random.randint(7, 30))
    )

    # Supporting data
    supporting_metrics = factory.LazyFunction(
        lambda: {
            "historical_performance": {
                "avg_ctr": round(random.uniform(1.0, 8.0), 2),
                "avg_conversion_rate": round(random.uniform(0.5, 15.0), 2),
                "avg_cpa": round(random.uniform(5.0, 100.0), 2),
            },
            "competitor_analysis": {
                "market_share": round(random.uniform(0.05, 0.30), 3),
                "competitor_cpc": round(random.uniform(0.50, 5.00), 2),
            },
        }
    )


class ReportDataFactory(factory.DictFactory):
    """Factory for generating report data."""

    id = factory.LazyFunction(lambda: f"report_{random.randint(100000, 999999)}")
    title = factory.LazyFunction(
        lambda: f"{fake.company()} Marketing Performance Report"
    )
    template = factory.LazyFunction(
        lambda: random.choice(["executive", "detailed", "technical"])
    )
    date_range = factory.LazyFunction(lambda: random.choice(["7d", "30d", "90d", "1y"]))

    # Report metadata
    generated_at = factory.LazyFunction(
        lambda: fake.date_time_between(start_date="-7d", end_date="now")
    )
    generated_by = factory.LazyFunction(lambda: fake.user_name())
    format = factory.LazyFunction(
        lambda: random.choice(["markdown", "html", "pdf", "json"])
    )

    # Summary metrics
    summary = factory.LazyFunction(
        lambda: {
            "total_campaigns": random.randint(5, 50),
            "total_spend": round(random.uniform(5000.0, 100000.0), 2),
            "total_impressions": random.randint(100000, 5000000),
            "total_clicks": random.randint(5000, 250000),
            "total_conversions": random.randint(100, 5000),
            "average_ctr": round(random.uniform(1.0, 8.0), 2),
            "average_cpc": round(random.uniform(0.50, 5.00), 2),
            "average_conversion_rate": round(random.uniform(1.0, 15.0), 2),
            "roas": round(random.uniform(150.0, 800.0), 0),
        }
    )

    # Performance trends
    trends = factory.LazyFunction(
        lambda: {
            "spend_trend": random.choice(["increasing", "decreasing", "stable"]),
            "conversion_trend": random.choice(["increasing", "decreasing", "stable"]),
            "efficiency_trend": random.choice(["improving", "declining", "stable"]),
        }
    )

    # Top performers
    top_campaigns = factory.LazyFunction(
        lambda: [CampaignDataFactory.build() for _ in range(random.randint(3, 10))]
    )

    # Recommendations
    recommendations = factory.LazyFunction(
        lambda: [OptimizationResultFactory.build() for _ in range(random.randint(2, 8))]
    )

    # Charts and visualizations
    charts = factory.LazyFunction(
        lambda: [
            {
                "type": chart_type,
                "title": f"{chart_type.replace('_', ' ').title()} Chart",
                "data_points": random.randint(7, 30),
            }
            for chart_type in random.sample(
                ["line_chart", "bar_chart", "pie_chart", "scatter_plot", "heatmap"],
                random.randint(2, 4),
            )
        ]
    )

    # Report sections
    sections = factory.LazyFunction(
        lambda: [
            "executive_summary",
            "performance_overview",
            "campaign_analysis",
            "optimization_opportunities",
            "recommendations",
            "appendix",
        ]
    )


class PerformanceMetricsFactory(factory.DictFactory):
    """Factory for generating system performance metrics."""

    timestamp = factory.LazyFunction(
        lambda: fake.date_time_between(start_date="-1h", end_date="now")
    )

    # System metrics
    cpu_percent = factory.LazyFunction(lambda: round(random.uniform(10.0, 90.0), 1))
    memory_percent = factory.LazyFunction(lambda: round(random.uniform(30.0, 85.0), 1))
    memory_used_mb = factory.LazyFunction(
        lambda: round(random.uniform(1000.0, 8000.0), 1)
    )
    disk_usage_percent = factory.LazyFunction(
        lambda: round(random.uniform(40.0, 95.0), 1)
    )

    # Network I/O
    network_io = factory.LazyFunction(
        lambda: {
            "bytes_sent": random.randint(1000000, 100000000),
            "bytes_recv": random.randint(5000000, 500000000),
            "packets_sent": random.randint(10000, 1000000),
            "packets_recv": random.randint(20000, 2000000),
        }
    )

    # Process information
    process_count = factory.LazyFunction(lambda: random.randint(100, 300))
    response_time = factory.LazyFunction(lambda: round(random.uniform(0.1, 2.0), 3))


class ErrorDataFactory(factory.DictFactory):
    """Factory for generating error data."""

    timestamp = factory.LazyFunction(
        lambda: fake.date_time_between(start_date="-7d", end_date="now")
    )
    error_type = factory.LazyFunction(
        lambda: random.choice(
            [
                "APIError",
                "DataValidationError",
                "ConfigurationError",
                "ModelError",
                "OptimizationError",
                "ReportGenerationError",
                "RateLimitError",
            ]
        )
    )

    message = factory.LazyFunction(
        lambda: fake.sentence(nb_words=random.randint(5, 12))
    )
    severity = factory.LazyFunction(
        lambda: random.choice(["ERROR", "WARNING", "CRITICAL"])
    )

    context = factory.LazyFunction(
        lambda: {
            "function": fake.pystr(min_chars=5, max_chars=15),
            "module": f"{fake.word()}.{fake.word()}",
            "user_id": fake.uuid4() if random.choice([True, False]) else None,
            "request_id": fake.uuid4(),
            "additional_data": {
                "param_count": random.randint(1, 10),
                "execution_time": round(random.uniform(0.1, 5.0), 3),
            },
        }
    )


# Utility functions for creating related test data
def create_campaign_with_keywords(keyword_count: int = 5) -> dict[str, Any]:
    """Create a campaign with associated keywords."""
    campaign = CampaignDataFactory.build()
    keywords = [
        KeywordDataFactory.build(campaign_id=campaign["id"])
        for _ in range(keyword_count)
    ]

    return {
        "campaign": campaign,
        "keywords": keywords,
        "total_keyword_impressions": sum(k["impressions"] for k in keywords),
        "total_keyword_clicks": sum(k["clicks"] for k in keywords),
        "total_keyword_cost": sum(k["cost"] for k in keywords),
    }


def create_analytics_funnel() -> list[dict[str, Any]]:
    """Create a complete analytics funnel with decreasing conversion rates."""
    base_sessions = random.randint(10000, 50000)

    funnel_steps = [
        {"step": "sessions", "count": base_sessions, "rate": 100.0},
        {
            "step": "page_views",
            "count": int(base_sessions * random.uniform(1.2, 3.0)),
            "rate": 0,
        },
        {
            "step": "engaged_sessions",
            "count": int(base_sessions * random.uniform(0.4, 0.8)),
            "rate": 0,
        },
        {
            "step": "add_to_cart",
            "count": int(base_sessions * random.uniform(0.1, 0.3)),
            "rate": 0,
        },
        {
            "step": "purchase_intent",
            "count": int(base_sessions * random.uniform(0.05, 0.15)),
            "rate": 0,
        },
        {
            "step": "conversions",
            "count": int(base_sessions * random.uniform(0.01, 0.08)),
            "rate": 0,
        },
    ]

    # Calculate conversion rates
    for i, step in enumerate(funnel_steps[1:], 1):
        step["rate"] = round((step["count"] / funnel_steps[i - 1]["count"]) * 100, 2)

    return funnel_steps


def create_optimization_scenario() -> dict[str, Any]:
    """Create a complete optimization scenario with before/after projections."""
    campaign = CampaignDataFactory.build()
    recommendations = [
        OptimizationResultFactory.build(campaign_id=campaign["id"]) for _ in range(3)
    ]

    # Calculate projected improvements
    total_cost_change = sum(rec["estimated_cost_change"] for rec in recommendations)
    total_conversion_change = sum(
        rec["estimated_conversion_change"] for rec in recommendations
    )
    total_revenue_change = sum(
        rec["estimated_revenue_change"] for rec in recommendations
    )

    return {
        "campaign": campaign,
        "recommendations": recommendations,
        "current_performance": {
            "cost": campaign["cost"],
            "conversions": campaign["conversions"],
            "revenue": campaign["conversions"] * random.uniform(20.0, 200.0),
        },
        "projected_performance": {
            "cost": campaign["cost"] + total_cost_change,
            "conversions": campaign["conversions"] + total_conversion_change,
            "revenue": (campaign["conversions"] * random.uniform(20.0, 200.0))
            + total_revenue_change,
        },
        "improvement_summary": {
            "cost_change_percent": round(
                (total_cost_change / campaign["cost"]) * 100, 2
            ),
            "conversion_change_percent": round(
                (total_conversion_change / campaign["conversions"]) * 100, 2
            ),
            "roi_improvement": round(
                total_revenue_change / max(abs(total_cost_change), 1), 2
            ),
        },
    }
