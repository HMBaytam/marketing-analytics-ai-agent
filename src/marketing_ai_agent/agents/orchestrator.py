"""Orchestrator Agent for coordinating marketing analysis tasks."""

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

from langchain.schema import BaseMessage, HumanMessage, SystemMessage
from langchain_anthropic import ChatAnthropic
from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field

from ..models.campaign import Campaign
from ..models.metrics import DailyMetrics
from ..models.analytics import TrafficData, ConversionEvent

logger = logging.getLogger(__name__)


class AgentState(BaseModel):
    """State shared across all agents in the workflow."""
    
    user_query: str = Field(..., description="Original user question")
    task_plan: List[str] = Field(default_factory=list, description="Step-by-step execution plan")
    current_step: int = Field(default=0, description="Current execution step")
    
    # Data storage
    google_ads_data: Optional[List[Campaign]] = Field(default=None, description="Google Ads campaign data")
    ga4_data: Optional[List[TrafficData]] = Field(default=None, description="GA4 traffic data")
    performance_metrics: Optional[List[DailyMetrics]] = Field(default=None, description="Performance metrics")
    conversion_events: Optional[List[ConversionEvent]] = Field(default=None, description="Conversion data")
    
    # Analysis results
    analysis_insights: Dict[str, Any] = Field(default_factory=dict, description="Analysis findings")
    recommendations: List[str] = Field(default_factory=list, description="Generated recommendations")
    
    # Metadata
    date_range: Optional[Tuple[datetime, datetime]] = Field(default=None, description="Analysis date range")
    error_messages: List[str] = Field(default_factory=list, description="Error messages")
    completed_steps: List[str] = Field(default_factory=list, description="Successfully completed steps")
    
    class Config:
        arbitrary_types_allowed = True


class OrchestratorAgent:
    """Main orchestrator agent that coordinates the marketing analysis workflow."""
    
    def __init__(self, llm: Optional[ChatAnthropic] = None):
        """
        Initialize orchestrator agent.
        
        Args:
            llm: Language model for task planning
        """
        self.llm = llm or ChatAnthropic(
            model="claude-3-sonnet-20240229",
            temperature=0.3,
            max_tokens=4000
        )
        self.graph = self._build_workflow_graph()
    
    def _build_workflow_graph(self) -> StateGraph:
        """Build the LangGraph workflow."""
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("plan_tasks", self.plan_tasks)
        workflow.add_node("data_ingestion", self.route_to_data_ingestion)
        workflow.add_node("analysis", self.route_to_analysis)
        workflow.add_node("generate_recommendations", self.route_to_recommendations)
        workflow.add_node("finalize", self.finalize_results)
        
        # Define flow
        workflow.set_entry_point("plan_tasks")
        
        workflow.add_edge("plan_tasks", "data_ingestion")
        workflow.add_edge("data_ingestion", "analysis")
        workflow.add_edge("analysis", "generate_recommendations")
        workflow.add_edge("generate_recommendations", "finalize")
        workflow.add_edge("finalize", END)
        
        return workflow.compile()
    
    async def plan_tasks(self, state: AgentState) -> AgentState:
        """
        Plan the execution steps based on user query.
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated state with task plan
        """
        try:
            logger.info(f"Planning tasks for query: {state.user_query}")
            
            planning_prompt = self._create_planning_prompt(state.user_query)
            response = await self.llm.ainvoke([
                SystemMessage(content=planning_prompt),
                HumanMessage(content=state.user_query)
            ])
            
            # Extract task plan from response
            task_plan = self._parse_task_plan(response.content)
            date_range = self._extract_date_range(state.user_query)
            
            state.task_plan = task_plan
            state.date_range = date_range
            state.completed_steps.append("task_planning")
            
            logger.info(f"Generated task plan with {len(task_plan)} steps")
            return state
            
        except Exception as e:
            logger.error(f"Task planning failed: {e}")
            state.error_messages.append(f"Task planning error: {str(e)}")
            return state
    
    async def route_to_data_ingestion(self, state: AgentState) -> AgentState:
        """
        Route to data ingestion agent.
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated state with ingestion results
        """
        from .data_ingestion import DataIngestionAgent
        
        try:
            logger.info("Starting data ingestion")
            
            ingestion_agent = DataIngestionAgent()
            state = await ingestion_agent.ingest_data(state)
            
            state.completed_steps.append("data_ingestion")
            state.current_step += 1
            
            return state
            
        except Exception as e:
            logger.error(f"Data ingestion routing failed: {e}")
            state.error_messages.append(f"Data ingestion error: {str(e)}")
            return state
    
    async def route_to_analysis(self, state: AgentState) -> AgentState:
        """
        Route to campaign analysis agent.
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated state with analysis results
        """
        from .campaign_analyzer import CampaignAnalyzerAgent
        
        try:
            logger.info("Starting campaign analysis")
            
            analyzer_agent = CampaignAnalyzerAgent()
            state = await analyzer_agent.analyze_campaigns(state)
            
            state.completed_steps.append("campaign_analysis")
            state.current_step += 1
            
            return state
            
        except Exception as e:
            logger.error(f"Campaign analysis routing failed: {e}")
            state.error_messages.append(f"Campaign analysis error: {str(e)}")
            return state
    
    async def route_to_recommendations(self, state: AgentState) -> AgentState:
        """
        Route to recommendation generation agent.
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated state with recommendations
        """
        from .recommendation_generator import RecommendationAgent
        
        try:
            logger.info("Starting recommendation generation")
            
            recommendation_agent = RecommendationAgent()
            state = await recommendation_agent.generate_recommendations(state)
            
            state.completed_steps.append("recommendation_generation")
            state.current_step += 1
            
            return state
            
        except Exception as e:
            logger.error(f"Recommendation generation routing failed: {e}")
            state.error_messages.append(f"Recommendation generation error: {str(e)}")
            return state
    
    async def finalize_results(self, state: AgentState) -> AgentState:
        """
        Finalize and validate results.
        
        Args:
            state: Current workflow state
            
        Returns:
            Final state with validated results
        """
        try:
            logger.info("Finalizing workflow results")
            
            # Validate results
            if not state.analysis_insights:
                state.error_messages.append("No analysis insights generated")
            
            if not state.recommendations:
                state.error_messages.append("No recommendations generated")
            
            # Log completion
            completed_count = len(state.completed_steps)
            total_steps = len(state.task_plan)
            
            logger.info(f"Workflow completed: {completed_count}/{total_steps} steps successful")
            
            if state.error_messages:
                logger.warning(f"Workflow completed with {len(state.error_messages)} errors")
            
            state.completed_steps.append("finalization")
            return state
            
        except Exception as e:
            logger.error(f"Result finalization failed: {e}")
            state.error_messages.append(f"Finalization error: {str(e)}")
            return state
    
    def _create_planning_prompt(self, user_query: str) -> str:
        """Create planning prompt for task decomposition."""
        return f"""You are a marketing analytics expert. Given a user query about marketing performance, 
        create a step-by-step execution plan.
        
        Available capabilities:
        1. Google Ads data ingestion (campaigns, ad groups, keywords, performance)
        2. GA4 data ingestion (traffic, conversions, user behavior)
        3. Campaign performance analysis
        4. Conversion attribution analysis  
        5. ROI and efficiency calculations
        6. Trend analysis and forecasting
        7. Recommendation generation
        
        For each step, specify:
        - Data sources required
        - Analysis type needed
        - Expected outputs
        
        Return your plan as a numbered list of concrete steps.
        
        User Query: {user_query}
        """
    
    def _parse_task_plan(self, plan_text: str) -> List[str]:
        """Parse task plan from LLM response."""
        lines = plan_text.strip().split('\n')
        tasks = []
        
        for line in lines:
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith('-')):
                # Remove numbering and clean up
                task = line.split('.', 1)[-1].split('-', 1)[-1].strip()
                if task:
                    tasks.append(task)
        
        return tasks or ["Analyze marketing performance", "Generate insights", "Create recommendations"]
    
    def _extract_date_range(self, query: str) -> Optional[Tuple[datetime, datetime]]:
        """Extract date range from user query."""
        query_lower = query.lower()
        now = datetime.now()
        
        # Quarter references
        if 'q1' in query_lower or 'first quarter' in query_lower:
            year = now.year if now.month >= 1 else now.year - 1
            return (datetime(year, 1, 1), datetime(year, 3, 31))
        elif 'q2' in query_lower or 'second quarter' in query_lower:
            year = now.year if now.month >= 4 else now.year - 1
            return (datetime(year, 4, 1), datetime(year, 6, 30))
        elif 'q3' in query_lower or 'third quarter' in query_lower:
            year = now.year if now.month >= 7 else now.year - 1
            return (datetime(year, 7, 1), datetime(year, 9, 30))
        elif 'q4' in query_lower or 'fourth quarter' in query_lower:
            year = now.year if now.month >= 10 else now.year - 1
            return (datetime(year, 10, 1), datetime(year, 12, 31))
        
        # Month references
        elif 'last month' in query_lower:
            if now.month == 1:
                start = datetime(now.year - 1, 12, 1)
                end = datetime(now.year, 1, 1) - timedelta(days=1)
            else:
                start = datetime(now.year, now.month - 1, 1)
                next_month = now.month if now.month < 12 else 1
                next_year = now.year if now.month < 12 else now.year + 1
                end = datetime(next_year, next_month, 1) - timedelta(days=1)
            return (start, end)
        
        # Default to last 30 days
        end_date = now.date()
        start_date = end_date - timedelta(days=30)
        return (datetime.combine(start_date, datetime.min.time()),
                datetime.combine(end_date, datetime.max.time()))
    
    async def execute_workflow(self, user_query: str) -> AgentState:
        """
        Execute the complete marketing analysis workflow.
        
        Args:
            user_query: User's marketing question
            
        Returns:
            Final state with results
        """
        initial_state = AgentState(user_query=user_query)
        
        try:
            logger.info(f"Starting workflow execution for: {user_query}")
            final_state = await self.graph.ainvoke(initial_state)
            
            logger.info("Workflow execution completed")
            return final_state
            
        except Exception as e:
            logger.error(f"Workflow execution failed: {e}")
            initial_state.error_messages.append(f"Workflow execution error: {str(e)}")
            return initial_state