"""Rule-based optimization system for marketing campaigns."""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from enum import Enum

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ConditionOperator(str, Enum):
    """Operators for rule conditions."""
    GREATER_THAN = ">"
    LESS_THAN = "<"
    GREATER_EQUAL = ">="
    LESS_EQUAL = "<="
    EQUAL = "=="
    NOT_EQUAL = "!="
    IN = "in"
    NOT_IN = "not_in"
    CONTAINS = "contains"


class RuleCondition(BaseModel):
    """Individual condition within an optimization rule."""
    
    metric: str = Field(description="Metric to evaluate")
    operator: ConditionOperator = Field(description="Comparison operator")
    value: Union[float, int, str, List[str]] = Field(description="Comparison value")
    weight: float = Field(default=1.0, description="Condition weight in rule evaluation")


class RuleAction(BaseModel):
    """Action to take when rule conditions are met."""
    
    action_type: str = Field(description="Type of action to take")
    parameters: Dict[str, Any] = Field(description="Action parameters")
    priority: float = Field(default=1.0, description="Action priority")
    description: str = Field(description="Human-readable action description")


class OptimizationRule(BaseModel):
    """Complete optimization rule with conditions and actions."""
    
    id: str = Field(description="Unique rule identifier")
    name: str = Field(description="Rule name")
    description: str = Field(description="Rule description")
    
    # Rule logic
    conditions: List[RuleCondition] = Field(description="Rule conditions")
    condition_logic: str = Field(default="AND", description="Logic between conditions (AND/OR)")
    actions: List[RuleAction] = Field(description="Actions to take")
    
    # Rule metadata
    category: str = Field(description="Rule category")
    confidence: float = Field(default=0.8, description="Rule confidence score")
    enabled: bool = Field(default=True, description="Whether rule is active")
    
    # Execution tracking
    last_triggered: Optional[datetime] = Field(default=None, description="Last trigger time")
    trigger_count: int = Field(default=0, description="Total trigger count")
    success_rate: float = Field(default=0.0, description="Historical success rate")


class RuleBasedOptimizer:
    """Rule-based optimization engine."""
    
    def __init__(self, rules: Optional[List[OptimizationRule]] = None):
        self.rules = rules or self._load_default_rules()
        self.execution_history = []
    
    def evaluate_rules(
        self,
        campaign_data: Dict[str, Any],
        campaign_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Evaluate all rules against campaign data."""
        
        logger.info(f"Evaluating {len(self.rules)} rules for campaign: {campaign_id}")
        
        triggered_actions = []
        
        for rule in self.rules:
            if not rule.enabled:
                continue
            
            try:
                if self._evaluate_rule(rule, campaign_data):
                    logger.info(f"Rule triggered: {rule.name}")
                    
                    # Record trigger
                    rule.last_triggered = datetime.now()
                    rule.trigger_count += 1
                    
                    # Add actions
                    for action in rule.actions:
                        triggered_actions.append({
                            'rule_id': rule.id,
                            'rule_name': rule.name,
                            'rule_confidence': rule.confidence,
                            'action': action,
                            'campaign_id': campaign_id,
                            'triggered_at': datetime.now()
                        })
                
            except Exception as e:
                logger.error(f"Error evaluating rule {rule.id}: {str(e)}")
                continue
        
        # Sort actions by priority
        triggered_actions.sort(key=lambda x: x['action'].priority, reverse=True)
        
        logger.info(f"Triggered {len(triggered_actions)} actions from rule evaluation")
        return triggered_actions
    
    def _evaluate_rule(self, rule: OptimizationRule, data: Dict[str, Any]) -> bool:
        """Evaluate a single rule against campaign data."""
        
        condition_results = []
        
        for condition in rule.conditions:
            result = self._evaluate_condition(condition, data)
            condition_results.append(result)
        
        # Apply condition logic
        if rule.condition_logic.upper() == "AND":
            return all(condition_results)
        elif rule.condition_logic.upper() == "OR":
            return any(condition_results)
        else:
            # Default to AND
            return all(condition_results)
    
    def _evaluate_condition(self, condition: RuleCondition, data: Dict[str, Any]) -> bool:
        """Evaluate a single condition."""
        
        metric_value = self._get_metric_value(condition.metric, data)
        
        if metric_value is None:
            return False
        
        try:
            if condition.operator == ConditionOperator.GREATER_THAN:
                return float(metric_value) > float(condition.value)
            elif condition.operator == ConditionOperator.LESS_THAN:
                return float(metric_value) < float(condition.value)
            elif condition.operator == ConditionOperator.GREATER_EQUAL:
                return float(metric_value) >= float(condition.value)
            elif condition.operator == ConditionOperator.LESS_EQUAL:
                return float(metric_value) <= float(condition.value)
            elif condition.operator == ConditionOperator.EQUAL:
                return metric_value == condition.value
            elif condition.operator == ConditionOperator.NOT_EQUAL:
                return metric_value != condition.value
            elif condition.operator == ConditionOperator.IN:
                return metric_value in condition.value
            elif condition.operator == ConditionOperator.NOT_IN:
                return metric_value not in condition.value
            elif condition.operator == ConditionOperator.CONTAINS:
                return str(condition.value) in str(metric_value)
            
        except (ValueError, TypeError) as e:
            logger.warning(f"Condition evaluation error: {str(e)}")
            return False
        
        return False
    
    def _get_metric_value(self, metric_path: str, data: Dict[str, Any]) -> Any:
        """Get metric value from nested data structure."""
        
        keys = metric_path.split('.')
        value = data
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return None
        
        return value
    
    def add_rule(self, rule: OptimizationRule):
        """Add a new optimization rule."""
        self.rules.append(rule)
        logger.info(f"Added rule: {rule.name}")
    
    def remove_rule(self, rule_id: str):
        """Remove an optimization rule."""
        self.rules = [r for r in self.rules if r.id != rule_id]
        logger.info(f"Removed rule: {rule_id}")
    
    def get_rule_performance(self) -> List[Dict[str, Any]]:
        """Get performance statistics for all rules."""
        
        performance = []
        for rule in self.rules:
            performance.append({
                'id': rule.id,
                'name': rule.name,
                'trigger_count': rule.trigger_count,
                'success_rate': rule.success_rate,
                'last_triggered': rule.last_triggered,
                'enabled': rule.enabled
            })
        
        return performance
    
    def _load_default_rules(self) -> List[OptimizationRule]:
        """Load default optimization rules."""
        
        return [
            # High CPA Rule
            OptimizationRule(
                id="high_cpa_rule",
                name="High CPA Alert",
                description="Trigger when CPA exceeds threshold",
                conditions=[
                    RuleCondition(
                        metric="performance.cpa",
                        operator=ConditionOperator.GREATER_THAN,
                        value=50.0
                    )
                ],
                actions=[
                    RuleAction(
                        action_type="budget_reduction",
                        parameters={"reduction_percentage": 20},
                        priority=0.8,
                        description="Reduce budget by 20% due to high CPA"
                    ),
                    RuleAction(
                        action_type="keyword_optimization",
                        parameters={"focus": "negative_keywords"},
                        priority=0.7,
                        description="Add negative keywords to reduce CPA"
                    )
                ],
                category="cost_optimization",
                confidence=0.85
            ),
            
            # Low CTR Rule
            OptimizationRule(
                id="low_ctr_rule",
                name="Low CTR Optimization",
                description="Optimize when CTR falls below threshold",
                conditions=[
                    RuleCondition(
                        metric="performance.ctr",
                        operator=ConditionOperator.LESS_THAN,
                        value=0.02
                    )
                ],
                actions=[
                    RuleAction(
                        action_type="creative_testing",
                        parameters={"test_type": "ad_copy", "variations": 3},
                        priority=0.9,
                        description="Test new ad copy variations"
                    ),
                    RuleAction(
                        action_type="audience_expansion",
                        parameters={"expansion_percentage": 15},
                        priority=0.6,
                        description="Expand audience targeting"
                    )
                ],
                category="engagement_optimization",
                confidence=0.8
            ),
            
            # High ROAS Rule
            OptimizationRule(
                id="high_roas_rule",
                name="Scale High ROAS Campaigns",
                description="Increase budget for high-performing campaigns",
                conditions=[
                    RuleCondition(
                        metric="performance.roas",
                        operator=ConditionOperator.GREATER_THAN,
                        value=4.0
                    ),
                    RuleCondition(
                        metric="performance.spend",
                        operator=ConditionOperator.GREATER_THAN,
                        value=1000
                    )
                ],
                condition_logic="AND",
                actions=[
                    RuleAction(
                        action_type="budget_increase",
                        parameters={"increase_percentage": 25, "max_budget": 10000},
                        priority=0.9,
                        description="Increase budget by 25% for high ROAS campaign"
                    ),
                    RuleAction(
                        action_type="bid_optimization",
                        parameters={"strategy": "maximize_conversions"},
                        priority=0.7,
                        description="Switch to maximize conversions bidding"
                    )
                ],
                category="scaling_optimization",
                confidence=0.9
            ),
            
            # Seasonal Performance Rule
            OptimizationRule(
                id="seasonal_performance_rule",
                name="Seasonal Performance Adjustment",
                description="Adjust for seasonal performance patterns",
                conditions=[
                    RuleCondition(
                        metric="temporal.day_of_week",
                        operator=ConditionOperator.IN,
                        value=["Saturday", "Sunday"]
                    ),
                    RuleCondition(
                        metric="performance.conversion_rate",
                        operator=ConditionOperator.LESS_THAN,
                        value=0.01
                    )
                ],
                condition_logic="AND",
                actions=[
                    RuleAction(
                        action_type="schedule_adjustment",
                        parameters={"reduce_weekend_spend": True, "reduction": 30},
                        priority=0.7,
                        description="Reduce weekend spend due to low conversion rate"
                    )
                ],
                category="temporal_optimization",
                confidence=0.75
            ),
            
            # Anomaly Response Rule
            OptimizationRule(
                id="anomaly_response_rule",
                name="Performance Anomaly Response",
                description="Respond to detected performance anomalies",
                conditions=[
                    RuleCondition(
                        metric="analytics.anomaly_severity",
                        operator=ConditionOperator.IN,
                        value=["critical", "high"]
                    )
                ],
                actions=[
                    RuleAction(
                        action_type="emergency_pause",
                        parameters={"pause_duration_hours": 2},
                        priority=1.0,
                        description="Temporarily pause campaign due to anomaly"
                    ),
                    RuleAction(
                        action_type="alert_notification",
                        parameters={"urgency": "high", "channels": ["email", "slack"]},
                        priority=0.9,
                        description="Send high-priority anomaly alert"
                    )
                ],
                category="risk_management",
                confidence=0.95
            ),
            
            # Competitive Response Rule
            OptimizationRule(
                id="competitive_response_rule",
                name="Competitive Position Response",
                description="Respond to competitive positioning changes",
                conditions=[
                    RuleCondition(
                        metric="benchmarks.industry_percentile",
                        operator=ConditionOperator.LESS_THAN,
                        value=25
                    ),
                    RuleCondition(
                        metric="trends.trend_direction",
                        operator=ConditionOperator.EQUAL,
                        value="declining"
                    )
                ],
                condition_logic="AND",
                actions=[
                    RuleAction(
                        action_type="competitive_analysis",
                        parameters={"focus": "top_performers", "analysis_depth": "deep"},
                        priority=0.8,
                        description="Analyze top competitor strategies"
                    ),
                    RuleAction(
                        action_type="strategy_pivot",
                        parameters={"test_new_approaches": True},
                        priority=0.7,
                        description="Test new strategic approaches"
                    )
                ],
                category="competitive_optimization",
                confidence=0.7
            ),
            
            # Budget Utilization Rule
            OptimizationRule(
                id="budget_utilization_rule",
                name="Budget Utilization Optimization",
                description="Optimize budget utilization rates",
                conditions=[
                    RuleCondition(
                        metric="budget.utilization_rate",
                        operator=ConditionOperator.LESS_THAN,
                        value=0.8
                    ),
                    RuleCondition(
                        metric="performance.efficiency_score",
                        operator=ConditionOperator.GREATER_THAN,
                        value=0.7
                    )
                ],
                condition_logic="AND",
                actions=[
                    RuleAction(
                        action_type="bid_increase",
                        parameters={"increase_percentage": 15, "gradual": True},
                        priority=0.8,
                        description="Gradually increase bids to improve budget utilization"
                    ),
                    RuleAction(
                        action_type="audience_expansion",
                        parameters={"expansion_type": "similar", "similarity_threshold": 0.8},
                        priority=0.6,
                        description="Expand to similar audiences"
                    )
                ],
                category="budget_optimization",
                confidence=0.8
            )
        ]