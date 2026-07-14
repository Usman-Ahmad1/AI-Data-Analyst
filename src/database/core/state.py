# src/core/state.py
"""
State definitions for LangGraph workflow.
"""
from typing import TypedDict, Optional, List, Any, Dict
from pydantic import BaseModel, Field
from enum import Enum

class QueryType(str, Enum):
    """Types of queries the system can handle."""
    SQL = "sql"
    PYTHON = "python"
    HYBRID = "hybrid"
    UNKNOWN = "unknown"

class AgentState(TypedDict):
    """State passed between agents in LangGraph."""
    # Input
    question: str
    messages: List[Dict[str, str]]
    
    # Context
    schema_text: str
    query_type: QueryType
    
    # SQL processing
    sql_query: Optional[str]
    sql_error: Optional[str]
    sql_attempts: int
    
    # Data
    dataframe: Optional[Any]  # pandas DataFrame
    data_description: Optional[str]
    
    # Analysis
    python_code: Optional[str]
    analysis_results: Optional[str]
    
    # Visualization
    chart_code: Optional[str]
    chart_path: Optional[str]
    chart_html: Optional[str]
    
    # Final output
    final_answer: Optional[str]
    insights: Optional[List[str]]
    
    # Error handling
    error: Optional[str]
    retry_count: int

class SQLGenerationRequest(BaseModel):
    """Request for SQL generation."""
    question: str = Field(..., description="Natural language question")
    schema_text: str = Field(..., description="Database schema")
    max_attempts: int = Field(3, description="Maximum generation attempts")

class VisualizationRequest(BaseModel):
    """Request for visualization generation."""
    data_description: str = Field(..., description="Description of data")
    question: str = Field(..., description="Original question")
    chart_type: Optional[str] = Field(None, description="Preferred chart type")
    