"""A/B testing recommendations and optimization system."""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
import math

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class TestType(str, Enum):
    """Types of A/B tests."""
    CREATIVE_TEST = "creative_test"
    LANDING_PAGE_TEST = "landing_page_test"
    AUDIENCE_TEST = "audience_test"
    BID_STRATEGY_TEST = "bid_strategy_test"
    AD_COPY_TEST = "ad_copy_test"
    BUDGET_SPLIT_TEST = "budget_split_test"
    TIMING_TEST = "timing_test"
    CHANNEL_TEST = "channel_test"


class TestPriority(str, Enum):
    """Test priority levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class TestConfig(BaseModel):
    """Configuration for A/B test setup."""
    
    # Test parameters
    test_duration_days: int = Field(default=14, description="Test duration in days")
    traffic_split: float = Field(default=0.5, description="Traffic split between variants")
    significance_level: float = Field(default=0.05, description="Statistical significance level")
    minimum_detectable_effect: float = Field(default=0.1, description="Minimum effect size to detect")
    
    # Sample size parameters
    power: float = Field(default=0.8, description="Statistical power")
    baseline_conversion_rate: float = Field(default=0.02, description="Baseline conversion rate")
    
    # Test constraints
    max_budget_risk: float = Field(default=0.1, description="Maximum budget at risk")
    min_daily_conversions: int = Field(default=10, description="Minimum daily conversions needed")
    allow_overlapping_tests: bool = Field(default=False, description="Allow overlapping tests")


class TestVariant(BaseModel):
    """Individual test variant configuration."""
    
    variant_id: str = Field(description="Variant identifier")
    variant_name: str = Field(description="Variant display name")
    description: str = Field(description="Variant description")
    
    # Configuration
    configuration: Dict[str, Any] = Field(description="Variant-specific configuration")
    traffic_allocation: float = Field(description="Traffic allocation percentage")
    
    # Expected metrics
    expected_ctr: Optional[float] = Field(default=None, description="Expected CTR")
    expected_conversion_rate: Optional[float] = Field(default=None, description="Expected conversion rate")
    expected_cpa: Optional[float] = Field(default=None, description="Expected CPA")
    
    # Rationale
    hypothesis: str = Field(description="Hypothesis for this variant")
    expected_outcome: str = Field(description="Expected outcome")


class TestRecommendation(BaseModel):
    """A/B test recommendation."""
    
    test_id: str = Field(description="Unique test identifier")
    test_type: TestType = Field(description="Type of A/B test")
    priority: TestPriority = Field(description="Test priority")
    
    # Test details
    test_name: str = Field(description="Test name")
    objective: str = Field(description="Test objective")
    hypothesis: str = Field(description="Primary hypothesis")
    
    # Test configuration
    variants: List[TestVariant] = Field(description="Test variants")
    primary_metric: str = Field(description="Primary success metric")
    secondary_metrics: List[str] = Field(description="Secondary metrics to track")
    
    # Test parameters
    estimated_duration: int = Field(description="Estimated test duration in days")
    required_sample_size: int = Field(description="Required sample size per variant")
    estimated_cost: float = Field(description="Estimated test cost")
    potential_uplift: float = Field(description="Potential performance uplift")
    
    # Risk assessment
    risk_level: str = Field(description="Test risk level")
    risk_factors: List[str] = Field(description="Identified risk factors")
    mitigation_strategies: List[str] = Field(description="Risk mitigation strategies")
    
    # Implementation
    setup_requirements: List[str] = Field(description="Setup requirements")
    success_criteria: List[str] = Field(description="Success criteria")
    stop_conditions: List[str] = Field(description="Early stop conditions")
    
    # Context
    campaign_id: Optional[str] = Field(default=None, description="Associated campaign")
    channel: Optional[str] = Field(default=None, description="Associated channel")
    created_at: datetime = Field(default_factory=datetime.now)


class ABTestingOptimizer:
    """A/B testing optimization and recommendation engine."""
    
    def __init__(self, config: TestConfig):
        self.config = config
        self.active_tests = []
        self.test_history = []
    
    def generate_test_recommendations(
        self,
        campaign_data: List[Dict[str, Any]],
        performance_issues: List[Dict[str, Any]],
        campaign_id: Optional[str] = None
    ) -> List[TestRecommendation]:
        """Generate A/B test recommendations based on campaign performance."""
        
        logger.info(f"Generating A/B test recommendations for campaign: {campaign_id}")
        
        try:
            # Analyze current performance
            performance_analysis = self._analyze_performance(campaign_data)
            
            # Identify test opportunities
            test_opportunities = self._identify_test_opportunities(
                performance_analysis, performance_issues
            )
            
            # Generate specific test recommendations
            recommendations = []
            for opportunity in test_opportunities:
                test_rec = self._create_test_recommendation(
                    opportunity, performance_analysis, campaign_id
                )
                if test_rec:
                    recommendations.append(test_rec)
            
            # Prioritize recommendations
            prioritized_recommendations = self._prioritize_test_recommendations(recommendations)
            
            # Filter by constraints
            feasible_recommendations = self._filter_feasible_tests(
                prioritized_recommendations, campaign_data
            )
            
            logger.info(f"Generated {len(feasible_recommendations)} feasible test recommendations")
            return feasible_recommendations
            
        except Exception as e:
            logger.error(f"Test recommendation generation failed: {str(e)}")
            raise
    
    def _analyze_performance(self, campaign_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze current campaign performance for test opportunities."""
        
        if not campaign_data:
            return {}
        
        # Calculate performance metrics
        total_impressions = sum(record.get('impressions', 0) for record in campaign_data)
        total_clicks = sum(record.get('clicks', 0) for record in campaign_data)
        total_conversions = sum(record.get('conversions', 0) for record in campaign_data)
        total_spend = sum(record.get('spend', 0) for record in campaign_data)
        total_revenue = sum(record.get('revenue', 0) for record in campaign_data)
        
        # Calculate derived metrics
        ctr = total_clicks / total_impressions if total_impressions > 0 else 0
        conversion_rate = total_conversions / total_clicks if total_clicks > 0 else 0
        cpa = total_spend / total_conversions if total_conversions > 0 else 0
        roas = total_revenue / total_spend if total_spend > 0 else 0
        
        # Performance trends
        if len(campaign_data) > 7:
            recent_data = campaign_data[-7:]  # Last 7 days
            earlier_data = campaign_data[-14:-7]  # Previous 7 days
            
            recent_ctr = sum(r.get('clicks', 0) for r in recent_data) / max(1, sum(r.get('impressions', 0) for r in recent_data))
            earlier_ctr = sum(r.get('clicks', 0) for r in earlier_data) / max(1, sum(r.get('impressions', 0) for r in earlier_data))
            
            ctr_trend = (recent_ctr - earlier_ctr) / earlier_ctr if earlier_ctr > 0 else 0
        else:
            ctr_trend = 0
        
        # Identify problem areas
        problem_areas = []
        if ctr < 0.015:  # Low CTR
            problem_areas.append('low_ctr')
        if conversion_rate < 0.01:  # Low conversion rate
            problem_areas.append('low_conversion_rate')
        if roas < 2.0:  # Low ROAS
            problem_areas.append('low_roas')
        if ctr_trend < -0.1:  # Declining CTR
            problem_areas.append('declining_ctr')
        
        return {
            'ctr': ctr,
            'conversion_rate': conversion_rate,
            'cpa': cpa,
            'roas': roas,
            'ctr_trend': ctr_trend,
            'total_conversions': total_conversions,
            'daily_conversions': total_conversions / max(1, len(campaign_data)),
            'problem_areas': problem_areas,
            'data_quality': 'sufficient' if len(campaign_data) > 30 else 'limited'
        }
    
    def _identify_test_opportunities(
        self,
        performance_analysis: Dict[str, Any],
        performance_issues: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Identify specific A/B test opportunities."""
        
        opportunities = []
        problem_areas = performance_analysis.get('problem_areas', [])
        
        # Creative testing opportunities
        if 'low_ctr' in problem_areas or 'declining_ctr' in problem_areas:
            opportunities.append({
                'type': TestType.CREATIVE_TEST,
                'priority': TestPriority.HIGH,
                'reason': 'Low or declining CTR indicates creative fatigue',
                'metrics_to_improve': ['ctr', 'engagement'],
                'potential_impact': 0.25
            })
            
            opportunities.append({
                'type': TestType.AD_COPY_TEST,
                'priority': TestPriority.HIGH,
                'reason': 'Ad copy optimization can improve CTR',
                'metrics_to_improve': ['ctr', 'quality_score'],
                'potential_impact': 0.20
            })
        
        # Landing page testing opportunities
        if 'low_conversion_rate' in problem_areas:
            opportunities.append({
                'type': TestType.LANDING_PAGE_TEST,
                'priority': TestPriority.CRITICAL,
                'reason': 'Low conversion rate suggests landing page issues',
                'metrics_to_improve': ['conversion_rate', 'cpa'],
                'potential_impact': 0.30
            })
        
        # Audience testing opportunities
        if 'low_roas' in problem_areas:
            opportunities.append({
                'type': TestType.AUDIENCE_TEST,
                'priority': TestPriority.HIGH,
                'reason': 'Poor ROAS may indicate audience mismatch',
                'metrics_to_improve': ['roas', 'conversion_rate'],
                'potential_impact': 0.35
            })
        
        # Bid strategy testing
        if performance_analysis.get('cpa', 0) > 50:  # High CPA threshold
            opportunities.append({
                'type': TestType.BID_STRATEGY_TEST,
                'priority': TestPriority.MEDIUM,
                'reason': 'High CPA suggests bidding optimization needed',
                'metrics_to_improve': ['cpa', 'roas'],
                'potential_impact': 0.15
            })
        
        # Budget split testing
        opportunities.append({
            'type': TestType.BUDGET_SPLIT_TEST,
            'priority': TestPriority.MEDIUM,
            'reason': 'Test optimal budget allocation timing',
            'metrics_to_improve': ['efficiency', 'roas'],
            'potential_impact': 0.10
        })
        
        # Timing optimization
        if len(performance_issues) == 0:  # No urgent issues, can test optimization
            opportunities.append({
                'type': TestType.TIMING_TEST,
                'priority': TestPriority.LOW,
                'reason': 'Optimize ad scheduling for better performance',
                'metrics_to_improve': ['efficiency', 'ctr'],
                'potential_impact': 0.08
            })
        
        return opportunities
    
    def _create_test_recommendation(
        self,
        opportunity: Dict[str, Any],
        performance_analysis: Dict[str, Any],
        campaign_id: Optional[str]
    ) -> Optional[TestRecommendation]:
        """Create detailed test recommendation from opportunity."""
        
        test_type = opportunity['type']
        
        try:
            # Generate test-specific details
            if test_type == TestType.CREATIVE_TEST:
                return self._create_creative_test(opportunity, performance_analysis, campaign_id)
            elif test_type == TestType.LANDING_PAGE_TEST:
                return self._create_landing_page_test(opportunity, performance_analysis, campaign_id)
            elif test_type == TestType.AUDIENCE_TEST:
                return self._create_audience_test(opportunity, performance_analysis, campaign_id)
            elif test_type == TestType.BID_STRATEGY_TEST:
                return self._create_bid_strategy_test(opportunity, performance_analysis, campaign_id)
            elif test_type == TestType.AD_COPY_TEST:
                return self._create_ad_copy_test(opportunity, performance_analysis, campaign_id)
            elif test_type == TestType.BUDGET_SPLIT_TEST:
                return self._create_budget_split_test(opportunity, performance_analysis, campaign_id)
            elif test_type == TestType.TIMING_TEST:
                return self._create_timing_test(opportunity, performance_analysis, campaign_id)
            
        except Exception as e:
            logger.error(f"Failed to create {test_type} recommendation: {str(e)}")
            return None
        
        return None
    
    def _create_creative_test(
        self,
        opportunity: Dict[str, Any],
        performance_analysis: Dict[str, Any],
        campaign_id: Optional[str]
    ) -> TestRecommendation:
        """Create creative A/B test recommendation."""
        
        # Calculate sample size
        baseline_ctr = performance_analysis.get('ctr', 0.02)
        sample_size = self._calculate_sample_size(
            baseline_ctr, 
            baseline_ctr * (1 + opportunity['potential_impact'])
        )
        
        variants = [
            TestVariant(
                variant_id="control",
                variant_name="Current Creative",
                description="Existing creative assets",
                configuration={"creative_type": "current"},
                traffic_allocation=0.5,
                expected_ctr=baseline_ctr,
                hypothesis="Current creative serves as baseline",
                expected_outcome="Baseline performance"
            ),
            TestVariant(
                variant_id="variant_a",
                variant_name="New Creative Concept",
                description="Fresh creative with different visual approach",
                configuration={"creative_type": "new_concept", "visual_style": "modern"},
                traffic_allocation=0.5,
                expected_ctr=baseline_ctr * (1 + opportunity['potential_impact']),
                hypothesis="New visual approach will improve engagement",
                expected_outcome=f"CTR improvement of {opportunity['potential_impact']*100:.1f}%"
            )
        ]
        
        return TestRecommendation(
            test_id=f"creative_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            test_type=TestType.CREATIVE_TEST,
            priority=opportunity['priority'],
            test_name="Creative Performance Test",
            objective="Improve click-through rates with new creative approach",
            hypothesis="New creative concept will outperform current creative by improving visual appeal and message clarity",
            variants=variants,
            primary_metric="ctr",
            secondary_metrics=["engagement_rate", "quality_score", "cpc"],
            estimated_duration=self.config.test_duration_days,
            required_sample_size=sample_size,
            estimated_cost=self._estimate_test_cost(sample_size, performance_analysis),
            potential_uplift=opportunity['potential_impact'],
            risk_level="Medium",
            risk_factors=["Creative production costs", "Potential CTR decline"],
            mitigation_strategies=["A/B test with traffic split", "Monitor daily performance"],
            setup_requirements=[
                "Develop new creative assets",
                "Setup campaign variants",
                "Configure tracking"
            ],
            success_criteria=[
                f"CTR improvement >10%",
                "Statistical significance achieved",
                "No significant CPA increase"
            ],
            stop_conditions=[
                "CTR drops >20% below baseline",
                "CPA increases >30%",
                "Budget exhausted"
            ],
            campaign_id=campaign_id
        )
    
    def _create_landing_page_test(
        self,
        opportunity: Dict[str, Any],
        performance_analysis: Dict[str, Any],
        campaign_id: Optional[str]
    ) -> TestRecommendation:
        """Create landing page A/B test recommendation."""
        
        baseline_conversion_rate = performance_analysis.get('conversion_rate', 0.02)
        sample_size = self._calculate_sample_size(
            baseline_conversion_rate,
            baseline_conversion_rate * (1 + opportunity['potential_impact'])
        )
        
        variants = [
            TestVariant(
                variant_id="control",
                variant_name="Current Landing Page",
                description="Existing landing page",
                configuration={"page_version": "current"},
                traffic_allocation=0.5,
                expected_conversion_rate=baseline_conversion_rate,
                hypothesis="Current page serves as baseline",
                expected_outcome="Baseline conversion rate"
            ),
            TestVariant(
                variant_id="variant_a",
                variant_name="Optimized Landing Page",
                description="Landing page with improved UX and clearer CTA",
                configuration={"page_version": "optimized", "improvements": ["clearer_cta", "reduced_form_fields", "social_proof"]},
                traffic_allocation=0.5,
                expected_conversion_rate=baseline_conversion_rate * (1 + opportunity['potential_impact']),
                hypothesis="UX improvements will increase conversion rate",
                expected_outcome=f"Conversion rate improvement of {opportunity['potential_impact']*100:.1f}%"
            )
        ]
        
        return TestRecommendation(
            test_id=f"landing_page_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            test_type=TestType.LANDING_PAGE_TEST,
            priority=opportunity['priority'],
            test_name="Landing Page Optimization Test",
            objective="Improve conversion rates through landing page optimization",
            hypothesis="Simplified form and clearer value proposition will increase conversions",
            variants=variants,
            primary_metric="conversion_rate",
            secondary_metrics=["bounce_rate", "time_on_page", "cpa"],
            estimated_duration=self.config.test_duration_days,
            required_sample_size=sample_size,
            estimated_cost=self._estimate_test_cost(sample_size, performance_analysis),
            potential_uplift=opportunity['potential_impact'],
            risk_level="Low",
            risk_factors=["Development time", "Potential conversion decline"],
            mitigation_strategies=["Test with limited traffic first", "Monitor user feedback"],
            setup_requirements=[
                "Develop new landing page version",
                "Setup URL routing and tracking",
                "Configure conversion tracking"
            ],
            success_criteria=[
                f"Conversion rate improvement >15%",
                "Statistical significance achieved",
                "Improved user experience metrics"
            ],
            stop_conditions=[
                "Conversion rate drops >15% below baseline",
                "Bounce rate increases significantly",
                "Technical issues detected"
            ],
            campaign_id=campaign_id
        )
    
    def _create_audience_test(
        self,
        opportunity: Dict[str, Any],
        performance_analysis: Dict[str, Any],
        campaign_id: Optional[str]
    ) -> TestRecommendation:
        """Create audience targeting A/B test recommendation."""
        
        baseline_roas = performance_analysis.get('roas', 2.0)
        sample_size = self._calculate_sample_size(0.02, 0.025)  # Simplified
        
        variants = [
            TestVariant(
                variant_id="control",
                variant_name="Current Audience",
                description="Current audience targeting settings",
                configuration={"audience_type": "current"},
                traffic_allocation=0.5,
                hypothesis="Current audience serves as baseline",
                expected_outcome="Baseline ROAS"
            ),
            TestVariant(
                variant_id="variant_a",
                variant_name="Lookalike Audience",
                description="Lookalike audience based on high-value customers",
                configuration={"audience_type": "lookalike", "source": "high_value_customers", "similarity": 0.8},
                traffic_allocation=0.5,
                hypothesis="Lookalike audience will have higher intent and conversion rates",
                expected_outcome=f"ROAS improvement of {opportunity['potential_impact']*100:.1f}%"
            )
        ]
        
        return TestRecommendation(
            test_id=f"audience_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            test_type=TestType.AUDIENCE_TEST,
            priority=opportunity['priority'],
            test_name="Audience Targeting Optimization Test",
            objective="Improve ROAS through better audience targeting",
            hypothesis="Lookalike audience based on high-value customers will drive better performance",
            variants=variants,
            primary_metric="roas",
            secondary_metrics=["conversion_rate", "cpa", "ltv"],
            estimated_duration=21,  # Longer for audience tests
            required_sample_size=sample_size,
            estimated_cost=self._estimate_test_cost(sample_size, performance_analysis),
            potential_uplift=opportunity['potential_impact'],
            risk_level="Medium",
            risk_factors=["Audience size limitations", "Learning period required"],
            mitigation_strategies=["Gradual budget allocation", "Extended learning period"],
            setup_requirements=[
                "Create lookalike audience",
                "Setup audience exclusions",
                "Configure tracking and attribution"
            ],
            success_criteria=[
                f"ROAS improvement >20%",
                "Lower CPA than baseline",
                "Maintained or increased conversion volume"
            ],
            stop_conditions=[
                "ROAS drops >10% below baseline",
                "CPA increases >25%",
                "Insufficient audience reach"
            ],
            campaign_id=campaign_id
        )
    
    def _create_bid_strategy_test(
        self,
        opportunity: Dict[str, Any],
        performance_analysis: Dict[str, Any],
        campaign_id: Optional[str]
    ) -> TestRecommendation:
        """Create bid strategy A/B test recommendation."""
        
        current_cpa = performance_analysis.get('cpa', 50)
        sample_size = 1000  # Simplified for bid strategy tests
        
        variants = [
            TestVariant(
                variant_id="control",
                variant_name="Current Bid Strategy",
                description="Current bidding approach",
                configuration={"bid_strategy": "current"},
                traffic_allocation=0.5,
                expected_cpa=current_cpa,
                hypothesis="Current bid strategy serves as baseline",
                expected_outcome="Baseline CPA performance"
            ),
            TestVariant(
                variant_id="variant_a",
                variant_name="Target CPA Bidding",
                description="Automated target CPA bidding strategy",
                configuration={"bid_strategy": "target_cpa", "target_cpa": current_cpa * 0.85},
                traffic_allocation=0.5,
                expected_cpa=current_cpa * 0.85,
                hypothesis="Automated bidding will optimize for lower CPA",
                expected_outcome=f"CPA reduction of {opportunity['potential_impact']*100:.1f}%"
            )
        ]
        
        return TestRecommendation(
            test_id=f"bid_strategy_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            test_type=TestType.BID_STRATEGY_TEST,
            priority=opportunity['priority'],
            test_name="Bid Strategy Optimization Test",
            objective="Reduce CPA through optimized bidding strategy",
            hypothesis="Target CPA bidding will achieve lower cost per acquisition",
            variants=variants,
            primary_metric="cpa",
            secondary_metrics=["roas", "conversion_volume", "impression_share"],
            estimated_duration=self.config.test_duration_days + 7,  # Longer for bid strategy
            required_sample_size=sample_size,
            estimated_cost=self._estimate_test_cost(sample_size, performance_analysis),
            potential_uplift=opportunity['potential_impact'],
            risk_level="Medium",
            risk_factors=["Learning period volatility", "Volume fluctuations"],
            mitigation_strategies=["Extended learning period", "Gradual implementation"],
            setup_requirements=[
                "Configure target CPA bidding",
                "Set appropriate learning budget",
                "Setup enhanced tracking"
            ],
            success_criteria=[
                f"CPA reduction >10%",
                "Maintained conversion volume",
                "Stable performance after learning period"
            ],
            stop_conditions=[
                "CPA increases >20% above baseline",
                "Conversion volume drops >30%",
                "Bidding instability persists"
            ],
            campaign_id=campaign_id
        )
    
    def _create_ad_copy_test(
        self,
        opportunity: Dict[str, Any],
        performance_analysis: Dict[str, Any],
        campaign_id: Optional[str]
    ) -> TestRecommendation:
        """Create ad copy A/B test recommendation."""
        
        baseline_ctr = performance_analysis.get('ctr', 0.02)
        sample_size = self._calculate_sample_size(baseline_ctr, baseline_ctr * 1.2)
        
        variants = [
            TestVariant(
                variant_id="control",
                variant_name="Current Ad Copy",
                description="Existing ad copy and messaging",
                configuration={"copy_version": "current"},
                traffic_allocation=0.5,
                expected_ctr=baseline_ctr,
                hypothesis="Current ad copy serves as baseline",
                expected_outcome="Baseline CTR"
            ),
            TestVariant(
                variant_id="variant_a",
                variant_name="Benefit-Focused Copy",
                description="Ad copy emphasizing key benefits and value proposition",
                configuration={"copy_version": "benefit_focused", "style": "value_proposition"},
                traffic_allocation=0.5,
                expected_ctr=baseline_ctr * 1.2,
                hypothesis="Benefit-focused messaging will improve CTR",
                expected_outcome="CTR improvement of 20%"
            )
        ]
        
        return TestRecommendation(
            test_id=f"ad_copy_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            test_type=TestType.AD_COPY_TEST,
            priority=opportunity['priority'],
            test_name="Ad Copy Optimization Test",
            objective="Improve CTR through optimized ad messaging",
            hypothesis="Benefit-focused ad copy will resonate better with target audience",
            variants=variants,
            primary_metric="ctr",
            secondary_metrics=["quality_score", "cpc", "conversion_rate"],
            estimated_duration=self.config.test_duration_days,
            required_sample_size=sample_size,
            estimated_cost=self._estimate_test_cost(sample_size, performance_analysis),
            potential_uplift=opportunity['potential_impact'],
            risk_level="Low",
            risk_factors=["Minimal setup risk"],
            mitigation_strategies=["Quick implementation", "Easy rollback"],
            setup_requirements=[
                "Write new ad copy variations",
                "Setup ad group variants",
                "Configure tracking"
            ],
            success_criteria=[
                "CTR improvement >15%",
                "Improved Quality Score",
                "Statistical significance"
            ],
            stop_conditions=[
                "CTR drops >10% below baseline",
                "Quality Score decreases significantly"
            ],
            campaign_id=campaign_id
        )
    
    def _create_budget_split_test(
        self,
        opportunity: Dict[str, Any],
        performance_analysis: Dict[str, Any],
        campaign_id: Optional[str]
    ) -> TestRecommendation:
        """Create budget allocation timing test."""
        
        sample_size = 500  # Simplified for budget tests
        
        variants = [
            TestVariant(
                variant_id="control",
                variant_name="Even Budget Distribution",
                description="Distribute budget evenly throughout the day",
                configuration={"budget_strategy": "even_distribution"},
                traffic_allocation=0.5,
                hypothesis="Even distribution serves as baseline",
                expected_outcome="Baseline efficiency"
            ),
            TestVariant(
                variant_id="variant_a",
                variant_name="Peak Hours Focus",
                description="Concentrate budget during identified peak performance hours",
                configuration={"budget_strategy": "peak_hours", "peak_hours": [9, 12, 18, 20]},
                traffic_allocation=0.5,
                hypothesis="Focusing budget during peak hours will improve efficiency",
                expected_outcome="Improved cost efficiency and ROAS"
            )
        ]
        
        return TestRecommendation(
            test_id=f"budget_split_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            test_type=TestType.BUDGET_SPLIT_TEST,
            priority=opportunity['priority'],
            test_name="Budget Timing Optimization Test",
            objective="Optimize budget allocation timing for better efficiency",
            hypothesis="Concentrating spend during peak hours will improve overall efficiency",
            variants=variants,
            primary_metric="roas",
            secondary_metrics=["cpa", "impression_share", "conversion_rate"],
            estimated_duration=self.config.test_duration_days,
            required_sample_size=sample_size,
            estimated_cost=self._estimate_test_cost(sample_size, performance_analysis),
            potential_uplift=opportunity['potential_impact'],
            risk_level="Low",
            risk_factors=["Potential missed opportunities outside peak hours"],
            mitigation_strategies=["Monitor performance across all hours", "Gradual shift implementation"],
            setup_requirements=[
                "Identify peak performance hours",
                "Setup ad scheduling",
                "Configure budget pacing"
            ],
            success_criteria=[
                "ROAS improvement >8%",
                "Better budget utilization",
                "Maintained conversion volume"
            ],
            stop_conditions=[
                "ROAS drops below baseline",
                "Significant volume loss"
            ],
            campaign_id=campaign_id
        )
    
    def _create_timing_test(
        self,
        opportunity: Dict[str, Any],
        performance_analysis: Dict[str, Any],
        campaign_id: Optional[str]
    ) -> TestRecommendation:
        """Create ad scheduling timing test."""
        
        sample_size = 800
        
        variants = [
            TestVariant(
                variant_id="control",
                variant_name="Current Schedule",
                description="Current ad scheduling settings",
                configuration={"schedule": "current"},
                traffic_allocation=0.5,
                hypothesis="Current schedule serves as baseline",
                expected_outcome="Baseline performance"
            ),
            TestVariant(
                variant_id="variant_a",
                variant_name="Optimized Schedule",
                description="Data-driven optimized ad scheduling",
                configuration={"schedule": "optimized", "high_performance_hours": [8, 12, 17, 19, 21]},
                traffic_allocation=0.5,
                hypothesis="Optimized scheduling will improve efficiency",
                expected_outcome="Better cost efficiency and performance"
            )
        ]
        
        return TestRecommendation(
            test_id=f"timing_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            test_type=TestType.TIMING_TEST,
            priority=opportunity['priority'],
            test_name="Ad Scheduling Optimization Test",
            objective="Optimize ad scheduling for better performance",
            hypothesis="Data-driven ad scheduling will improve cost efficiency",
            variants=variants,
            primary_metric="efficiency_score",
            secondary_metrics=["ctr", "conversion_rate", "cpa"],
            estimated_duration=self.config.test_duration_days,
            required_sample_size=sample_size,
            estimated_cost=self._estimate_test_cost(sample_size, performance_analysis),
            potential_uplift=opportunity['potential_impact'],
            risk_level="Low",
            risk_factors=["Seasonal variations"],
            mitigation_strategies=["Account for seasonal patterns", "Monitor daily performance"],
            setup_requirements=[
                "Analyze historical performance by hour",
                "Setup optimized ad schedule",
                "Configure performance monitoring"
            ],
            success_criteria=[
                "Efficiency improvement >5%",
                "Better resource utilization",
                "Consistent performance patterns"
            ],
            stop_conditions=[
                "Performance drops below baseline",
                "Irregular performance patterns"
            ],
            campaign_id=campaign_id
        )
    
    def _calculate_sample_size(
        self,
        baseline_rate: float,
        target_rate: float,
        power: float = 0.8,
        alpha: float = 0.05
    ) -> int:
        """Calculate required sample size for A/B test."""
        
        # Simplified sample size calculation for proportion test
        # Using normal approximation
        
        if baseline_rate <= 0 or target_rate <= 0:
            return 10000  # Default fallback
        
        effect_size = abs(target_rate - baseline_rate) / baseline_rate
        if effect_size < 0.01:
            return 50000  # Very large sample needed for small effects
        
        # Simplified calculation (in practice, use proper statistical formulas)
        z_alpha = 1.96  # for alpha = 0.05
        z_beta = 0.84   # for power = 0.8
        
        pooled_rate = (baseline_rate + target_rate) / 2
        
        # Sample size per group
        n = (2 * pooled_rate * (1 - pooled_rate) * (z_alpha + z_beta) ** 2) / (target_rate - baseline_rate) ** 2
        
        return max(100, int(n))
    
    def _estimate_test_cost(
        self,
        sample_size: int,
        performance_analysis: Dict[str, Any]
    ) -> float:
        """Estimate cost of running the A/B test."""
        
        daily_conversions = performance_analysis.get('daily_conversions', 10)
        if daily_conversions <= 0:
            return sample_size * 2.0  # Fallback estimation
        
        conversion_rate = performance_analysis.get('conversion_rate', 0.02)
        cpa = performance_analysis.get('cpa', 50)
        
        # Estimate required conversions per variant
        conversions_needed = sample_size * conversion_rate
        
        # Estimate cost per variant
        cost_per_variant = conversions_needed * cpa
        
        return cost_per_variant * 2  # Two variants
    
    def _prioritize_test_recommendations(
        self,
        recommendations: List[TestRecommendation]
    ) -> List[TestRecommendation]:
        """Prioritize test recommendations based on impact and feasibility."""
        
        def priority_score(rec):
            # Convert priority to numeric
            priority_scores = {
                TestPriority.CRITICAL: 4,
                TestPriority.HIGH: 3,
                TestPriority.MEDIUM: 2,
                TestPriority.LOW: 1
            }
            
            priority_val = priority_scores.get(rec.priority, 2)
            impact_score = rec.potential_uplift * 10
            feasibility_score = 1 / (rec.estimated_cost / 1000 + 1)  # Inverse of cost
            
            return priority_val * 0.4 + impact_score * 0.4 + feasibility_score * 0.2
        
        return sorted(recommendations, key=priority_score, reverse=True)
    
    def _filter_feasible_tests(
        self,
        recommendations: List[TestRecommendation],
        campaign_data: List[Dict[str, Any]]
    ) -> List[TestRecommendation]:
        """Filter recommendations based on feasibility constraints."""
        
        feasible_tests = []
        
        for rec in recommendations:
            # Check minimum daily conversions
            daily_conversions = sum(r.get('conversions', 0) for r in campaign_data[-7:]) / 7
            if daily_conversions < self.config.min_daily_conversions:
                logger.warning(f"Test {rec.test_id} filtered: insufficient daily conversions")
                continue
            
            # Check budget constraints
            if rec.estimated_cost > self.config.max_budget_risk * 10000:  # Simplified
                logger.warning(f"Test {rec.test_id} filtered: cost exceeds budget risk threshold")
                continue
            
            # Check for overlapping tests (if not allowed)
            if not self.config.allow_overlapping_tests and self.active_tests:
                logger.warning(f"Test {rec.test_id} filtered: overlapping tests not allowed")
                continue
            
            feasible_tests.append(rec)
        
        return feasible_tests