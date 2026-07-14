# src/core/graph.py
"""
LangGraph orchestration for the AI Data Analyst system.
"""
from typing import Dict, Any, Optional
import logging
from langgraph.graph import StateGraph, END
from langgraph.checkpoint import MemorySaver
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from ..agents.sql_agent import SQLAgent
from ..agents.viz_agent import VisualizationAgent
from ..database.db_utils import DatabaseManager
from ..prompts.templates import PromptTemplates
from .state import AgentState, QueryType

logger = logging.getLogger(__name__)

class DataAnalystGraph:
    """
    Main LangGraph orchestration for the AI Data Analyst.
    """
    
    def __init__(
        self,
        groq_api_key: str,
        db_path: str,
        model_name: str = "llama-3.1-70b-versatile",
        temperature: float = 0.1
    ):
        """
        Initialize the data analyst graph.
        
        Args:
            groq_api_key: Groq API key
            db_path: Path to database file
            model_name: Groq model to use
            temperature: LLM temperature
        """
        # Initialize LLM
        self.llm = ChatGroq(
            groq_api_key=groq_api_key,
            model_name=model_name,
            temperature=temperature
        )
        
        # Initialize database
        self.db_manager = DatabaseManager(db_path)
        
        # Initialize agents
        self.sql_agent = SQLAgent(self.llm, self.db_manager)
        self.viz_agent = VisualizationAgent(self.llm)
        
        # Build graph
        self.graph = self._build_graph()
        self.memory = MemorySaver()
        self.app = self.graph.compile(checkpointer=self.memory)
    
    def _build_graph(self) -> StateGraph:
        """
        Build the LangGraph workflow.
        
        Returns:
            Compiled StateGraph
        """
        graph = StateGraph(AgentState)
        
        # Add nodes
        graph.add_node("router", self._router_node)
        graph.add_node("sql_gen", self._sql_gen_node)
        graph.add_node("sql_exec", self._sql_exec_node)
        graph.add_node("python_analysis", self._python_analysis_node)
        graph.add_node("viz_gen", self._viz_gen_node)
        graph.add_node("synthesize", self._synthesize_node)
        
        # Add edges
        graph.set_entry_point("router")
        
        # Conditional routing from router
        graph.add_conditional_edges(
            "router",
            self._route_query,
            {
                "sql": "sql_gen",
                "python": "python_analysis",
                "hybrid": "sql_gen",  # Start with SQL then move to Python
                "unknown": "synthesize"
            }
        )
        
        # SQL path
        graph.add_edge("sql_gen", "sql_exec")
        graph.add_conditional_edges(
            "sql_exec",
            self._handle_sql_result,
            {
                "success": "viz_gen",
                "retry": "sql_gen",
                "error": "synthesize"
            }
        )
        
        # Visualization path
        graph.add_edge("viz_gen", "synthesize")
        graph.add_edge("python_analysis", "viz_gen")
        graph.add_edge("synthesize", END)
        
        return graph
    
    def _router_node(self, state: AgentState) -> AgentState:
        """
        Route the query to appropriate processing path.
        """
        question = state['question']
        
        router_prompt = ChatPromptTemplate.from_template(
            PromptTemplates.ROUTER
        )
        chain = router_prompt | self.llm | StrOutputParser()
        
        try:
            classification = chain.invoke({"question": question})
            classification = classification.strip().lower()
            
            if "sql" in classification:
                state['query_type'] = QueryType.SQL
            elif "python" in classification:
                state['query_type'] = QueryType.PYTHON
            elif "hybrid" in classification:
                state['query_type'] = QueryType.HYBRID
            else:
                state['query_type'] = QueryType.UNKNOWN
            
            logger.info(f"Query classified as: {state['query_type']}")
            
        except Exception as e:
            logger.error(f"Routing failed: {str(e)}")
            state['query_type'] = QueryType.SQL  # Default to SQL
        
        # Add schema to state
        state['schema_text'] = self.db_manager.get_schema_text()
        
        return state
    
    def _sql_gen_node(self, state: AgentState) -> AgentState:
        """
        Generate SQL query.
        """
        return self.sql_agent.process(state)
    
    def _sql_exec_node(self, state: AgentState) -> AgentState:
        """
        Execute SQL query.
        """
        sql_query = state.get('sql_query')
        if not sql_query:
            state['error'] = "No SQL query to execute"
            return state
        
        df, error = self.sql_agent.execute_sql(sql_query)
        
        if error:
            state['error'] = error
            state['sql_attempts'] = state.get('sql_attempts', 0) + 1
        else:
            state['dataframe'] = df
            state['data_description'] = self._describe_dataframe(df)
        
        return state
    
    def _python_analysis_node(self, state: AgentState) -> AgentState:
        """
        Perform Python-based analysis.
        """
        # Placeholder for Python analysis logic
        # This would be implemented similarly to SQL agent
        state['analysis_results'] = "Python analysis placeholder"
        return state
    
    def _viz_gen_node(self, state: AgentState) -> AgentState:
        """
        Generate visualization.
        """
        df = state.get('dataframe')
        if df is None or df.empty:
            state['error'] = "No data available for visualization"
            return state
        
        # Generate chart
        state = self.viz_agent.process(state)
        
        return state
    
    def _synthesize_node(self, state: AgentState) -> AgentState:
        """
        Synthesize final answer with insights.
        """
        question = state['question']
        data_description = state.get('data_description', 'No data available')
        chart_path = state.get('chart_path', '')
        error = state.get('error', '')
        
        synthesis_prompt = ChatPromptTemplate.from_template(
            PromptTemplates.SYNTHESIS
        )
        
        chain = synthesis_prompt | self.llm | StrOutputParser()
        
        try:
            # Prepare synthesis context
            context = {
                "analysis_results": data_description,
                "chart_path": chart_path or "No visualization generated",
                "question": question
            }
            
            if error:
                context["analysis_results"] = f"Error occurred: {error}\n\n{data_description}"
            
            answer = chain.invoke(context)
            state['final_answer'] = answer
            
        except Exception as e:
            logger.error(f"Synthesis failed: {str(e)}")
            state['final_answer'] = f"I encountered an error while analyzing your question: {str(e)}"
        
        return state
    
    def _route_query(self, state: AgentState) -> str:
        """
        Route decision based on query type.
        """
        query_type = state.get('query_type', QueryType.UNKNOWN)
        
        if query_type == QueryType.SQL:
            return "sql"
        elif query_type == QueryType.PYTHON:
            return "python"
        elif query_type == QueryType.HYBRID:
            return "hybrid"
        else:
            return "unknown"
    
    def _handle_sql_result(self, state: AgentState) -> str:
        """
        Handle SQL execution result.
        """
        if state.get('error'):
            attempts = state.get('sql_attempts', 0)
            if attempts < 3:
                return "retry"
            return "error"
        
        df = state.get('dataframe')
        if df is not None and not df.empty:
            return "success"
        
        state['error'] = "Query returned no results"
        return "error"
    
    def _describe_dataframe(self, df):
        """Create descriptive summary of dataframe."""
        desc = f"DataFrame with {len(df)} rows and {len(df.columns)} columns\n\n"
        
        for col in df.columns:
            desc += f"- {col}: {df[col].dtype}\n"
        
        desc += f"\nSample Data (first 5 rows):\n{df.head().to_string()}"
        
        return desc
    
    def run(self, question: str, thread_id: str = "default") -> Dict[str, Any]:
        """
        Run the data analyst workflow.
        
        Args:
            question: User question
            thread_id: Thread ID for conversation memory
            
        Returns:
            Final state with results
        """
        initial_state = {
            "question": question,
            "messages": [{"role": "user", "content": question}],
            "query_type": QueryType.UNKNOWN,
            "sql_attempts": 0,
            "retry_count": 0
        }
        
        try:
            config = {"configurable": {"thread_id": thread_id}}
            result = self.app.invoke(initial_state, config)
            
            return {
                "answer": result.get('final_answer', 'No answer generated'),
                "chart_path": result.get('chart_path'),
                "chart_html": result.get('chart_html'),
                "dataframe": result.get('dataframe'),
                "sql_query": result.get('sql_query'),
                "error": result.get('error')
            }
            
        except Exception as e:
            logger.error(f"Workflow execution failed: {str(e)}")
            return {
                "answer": f"An error occurred: {str(e)}",
                "error": str(e)
            }