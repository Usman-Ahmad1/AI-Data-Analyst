from typing import TypedDict, Optional, List, Any, Dict
from enum import Enum
import pandas as pd

class QueryType(str, Enum):
    SQL = "sql"
    PYTHON = "python"
    HYBRID = "hybrid"
    UNKNOWN = "unknown"

class AgentState(TypedDict):
    """State passed between agents in LangGraph."""
    question: str
    messages: List[Dict[str, str]]
    schema_text: str
    query_type: QueryType
    sql_query: Optional[str]
    sql_error: Optional[str]
    sql_attempts: int
    dataframe_data: Optional[List[Dict]]  # Store as list of dicts
    dataframe_columns: Optional[List[str]]
    data_description: Optional[str]
    python_code: Optional[str]
    analysis_results: Optional[str]
    chart_code: Optional[str]
    chart_path: Optional[str]
    chart_html: Optional[str]
    final_answer: Optional[str]
    insights: Optional[List[str]]
    error: Optional[str]
    retry_count: int

def dataframe_to_dict(df: pd.DataFrame) -> dict:
    """Convert DataFrame to serializable format."""
    if df is None:
        return {"data": [], "columns": []}
    return {
        "data": df.to_dict('records'),
        "columns": df.columns.tolist()
    }

def dict_to_dataframe(data: dict) -> pd.DataFrame:
    """Convert dict back to DataFrame."""
    if not data or not data.get("data"):
        return pd.DataFrame()
    return pd.DataFrame(data["data"], columns=data.get("columns"))
