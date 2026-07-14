"""
LangGraph orchestration for the AI Data Analyst system.
"""
from typing import Dict, Any, Optional
import logging
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.io import to_html
from pathlib import Path
import uuid
import traceback

from ..agents.sql_agent import SQLAgent
from ..agents.viz_agent import AdvancedVisualizationAgent
from ..database.db_utils import DatabaseManager
from ..prompts.templates import PromptTemplates
from .state import AgentState, QueryType

logger = logging.getLogger(__name__)

class DataAnalystGraph:
    def __init__(self, groq_api_key: str, db_path: str, model_name: str = "llama-3.3-70b-versatile", temperature: float = 0.1):
        self.llm = ChatGroq(groq_api_key=groq_api_key, model_name=model_name, temperature=temperature)
        self.db_manager = DatabaseManager(db_path)
        self.sql_agent = SQLAgent(self.llm, self.db_manager)
        self.viz_agent = AdvancedVisualizationAgent()
        self.graph = self._build_graph()
        self.memory = MemorySaver()
        self.app = self.graph.compile(checkpointer=self.memory)
    
    def _build_graph(self) -> StateGraph:
        graph = StateGraph(AgentState)
        
        graph.add_node("router", self._router_node)
        graph.add_node("sql_gen", self._sql_gen_node)
        graph.add_node("sql_exec", self._sql_exec_node)
        graph.add_node("viz_gen", self._viz_gen_node)
        graph.add_node("synthesize", self._synthesize_node)
        
        graph.set_entry_point("router")
        
        # FORCED flow - always go through visualization
        graph.add_edge("router", "sql_gen")
        graph.add_edge("sql_gen", "sql_exec")
        graph.add_edge("sql_exec", "viz_gen")
        graph.add_edge("viz_gen", "synthesize")
        graph.add_edge("synthesize", END)
        
        return graph
    
    def _router_node(self, state: AgentState) -> AgentState:
        state['schema_text'] = self.db_manager.get_schema_text()
        state['query_type'] = QueryType.SQL
        return state
    
    def _sql_gen_node(self, state: AgentState) -> AgentState:
        return self.sql_agent.process(state)
    
    def _sql_exec_node(self, state: AgentState) -> AgentState:
        sql_query = state.get('sql_query')
        if not sql_query:
            state['error'] = "No SQL query to execute"
            return state
        
        df, error = self.sql_agent.execute_sql(sql_query)
        
        if error:
            state['error'] = error
        else:
            state['dataframe_data'] = df.to_dict('records') if df is not None else []
            state['dataframe_columns'] = df.columns.tolist() if df is not None else []
            state['data_description'] = self._describe_dataframe(df)
            state['error'] = None
        
        return state
    
    def _viz_gen_node(self, state: AgentState) -> AgentState:
        """Generate advanced dashboard with multiple visualizations."""
        logger.info("🎨 Creating advanced dashboard...")
        
        try:
            # Get data
            dataframe_data = state.get('dataframe_data', [])
            dataframe_columns = state.get('dataframe_columns', [])
            
            if not dataframe_data:
                state['error'] = "No data available for visualization"
                return state
            
            # Use the advanced agent
            state = self.viz_agent.process(state)
            
            if state.get('chart_html'):
                logger.info("✅ Advanced dashboard created successfully")
            else:
                logger.warning("⚠️ Dashboard creation had issues")
            
        except Exception as e:
            logger.error(f"Dashboard creation failed: {str(e)}")
            state['error'] = f"Dashboard failed: {str(e)}"
        
        return state
    
    def _synthesize_node(self, state: AgentState) -> AgentState:
        question = state['question']
        data_description = state.get('data_description', 'No data available')
        has_chart = bool(state.get('chart_html'))
        error = state.get('error', '')
        
        synthesis_prompt = ChatPromptTemplate.from_template(PromptTemplates.SYNTHESIS)
        chain = synthesis_prompt | self.llm | StrOutputParser()
        
        try:
            context = {
                "analysis_results": data_description,
                "chart_path": "✅ Dashboard generated" if has_chart else "⚠️ Chart not available",
                "question": question
            }
            
            if error:
                context["analysis_results"] = f"Note: {error}\n\n{data_description}"
            
            answer = chain.invoke(context)
            state['final_answer'] = answer
            
        except Exception as e:
            logger.error(f"Synthesis failed: {str(e)}")
            state['final_answer'] = f"Analysis complete. Data: {data_description[:500]}"
        
        return state
    
    def _describe_dataframe(self, df):
        if df is None or df.empty:
            return "No data available"
        desc = f"DataFrame with {len(df)} rows and {len(df.columns)} columns\n\n"
        for col in df.columns:
            desc += f"- {col}: {df[col].dtype}\n"
        desc += f"\nSample:\n{df.head().to_string()}"
        return desc
    
    def run(self, question: str, thread_id: str = "default") -> Dict[str, Any]:
        initial_state = {
            "question": question,
            "messages": [{"role": "user", "content": question}],
            "query_type": QueryType.SQL,
            "sql_attempts": 0,
            "retry_count": 0,
            "dataframe_data": [],
            "dataframe_columns": []
        }
        
        try:
            config = {"configurable": {"thread_id": thread_id}}
            result = self.app.invoke(initial_state, config)
            
            dataframe_data = result.get('dataframe_data', [])
            dataframe_columns = result.get('dataframe_columns', [])
            df = pd.DataFrame(dataframe_data, columns=dataframe_columns) if dataframe_data else None
            
            return {
                "answer": result.get('final_answer', 'No answer generated'),
                "chart_path": result.get('chart_path'),
                "chart_html": result.get('chart_html'),
                "dataframe": df,
                "sql_query": result.get('sql_query'),
                "error": result.get('error')
            }
            
        except Exception as e:
            logger.error(f"Workflow failed: {str(e)}")
            logger.error(traceback.format_exc())
            return {
                "answer": f"Error: {str(e)}",
                "error": str(e)
            }