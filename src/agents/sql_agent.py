# src/agents/sql_agent.py
"""
SQL generation and execution agent.
"""
from typing import Dict, Any, Optional, Tuple
import logging
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import pandas as pd

from ..core.state import AgentState
from ..database.db_utils import DatabaseManager
from ..prompts.templates import PromptTemplates

logger = logging.getLogger(__name__)

class SQLAgent:
    """
    Agent responsible for SQL generation and execution.
    """
    
    def __init__(
        self,
        llm: ChatGroq,
        db_manager: DatabaseManager,
        max_attempts: int = 3
    ):
        """
        Initialize SQL agent.
        """
        self.llm = llm
        self.db_manager = db_manager
        self.max_attempts = max_attempts
        
        # Setup prompts
        self.sql_prompt = ChatPromptTemplate.from_template(
            PromptTemplates.SQL_GENERATION
        )
        
        self.output_parser = StrOutputParser()
        self.chain = self.sql_prompt | self.llm | self.output_parser
    
    def generate_sql(self, question: str, schema_text: str) -> Tuple[str, Optional[str]]:
        """
        Generate SQL from natural language question.
        """
        try:
            # Check if schema is empty
            if not schema_text or len(schema_text.strip()) == 0:
                logger.error("Schema text is empty")
                return "", "Database schema is empty. Please check database connection."
            
            examples = self._format_examples()
            
            logger.info(f"Generating SQL for: {question[:50]}...")
            
            # Invoke the chain
            response = self.chain.invoke({
                "schema_text": schema_text,
                "examples": examples,
                "question": question
            })
            
            # Clean response
            sql_query = self._clean_sql(response)
            
            # Validate SQL
            if not sql_query or sql_query.strip() in ["", ";"]:
                logger.warning("Generated SQL is empty")
                return "", "Failed to generate SQL query"
            
            # Basic SQL validation - check for SELECT
            if "SELECT" not in sql_query.upper():
                logger.warning(f"Generated query doesn't contain SELECT: {sql_query[:100]}")
                return "", "Generated query is not a SELECT statement"
            
            logger.info(f"Generated SQL: {sql_query[:100]}...")
            return sql_query, None
            
        except Exception as e:
            logger.error(f"SQL generation failed: {str(e)}", exc_info=True)
            return "", str(e)
    
    def _format_examples(self) -> str:
        """Format few-shot examples."""
        examples = PromptTemplates.get_sql_examples()
        if not examples:
            return "No examples available"
        
        formatted = []
        for ex in examples:
            formatted.append(f"Question: {ex['question']}\nSQL: {ex['sql']}")
        return "\n\n".join(formatted)
    
    def _clean_sql(self, response: str) -> str:
        """Clean SQL response from markdown and extra text."""
        # Remove markdown code blocks
        response = response.replace("```sql", "").replace("```", "")
        
        # Remove leading/trailing whitespace
        response = response.strip()
        
        # If response contains multiple lines, try to find SQL
        lines = response.split('\n')
        sql_lines = []
        in_sql = False
        
        for line in lines:
            # Skip empty lines at start
            if not sql_lines and not line.strip():
                continue
            
            # Look for SQL keywords to identify start
            if not in_sql and any(kw in line.upper() for kw in ['SELECT', 'WITH', 'INSERT', 'UPDATE']):
                in_sql = True
            
            if in_sql:
                sql_lines.append(line)
        
        if sql_lines:
            response = '\n'.join(sql_lines)
        
        # Ensure ends with semicolon
        response = response.strip()
        if response and not response.endswith(";"):
            response += ";"
        
        return response
    
    def execute_sql(self, sql_query: str) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
        """
        Execute SQL query safely.
        """
        if not sql_query or sql_query.strip() == ";":
            return None, "No valid SQL query to execute"
        
        try:
            logger.info(f"Executing SQL: {sql_query[:100]}...")
            df = self.db_manager.execute_query(sql_query)
            
            if df is None or df.empty:
                logger.warning("Query returned empty result")
                return pd.DataFrame(), "Query returned no results"
            
            logger.info(f"Query returned {len(df)} rows")
            return df, None
            
        except Exception as e:
            logger.error(f"SQL execution failed: {str(e)}")
            return None, str(e)
    
    def process(self, state: AgentState) -> AgentState:
        """
        Process SQL generation and execution.
        """
        question = state.get('question', '')
        schema_text = state.get('schema_text', '')
        
        if not question:
            state['error'] = "No question provided"
            return state
        
        if not schema_text:
            state['error'] = "No schema available"
            return state
        
        # Generate SQL with retry
        sql_query = None
        error = None
        attempts = 0
        
        while attempts < self.max_attempts:
            attempts += 1
            logger.info(f"SQL generation attempt {attempts}/{self.max_attempts}")
            
            sql_query, error = self.generate_sql(question, schema_text)
            
            if sql_query and not error:
                break
                
            if attempts < self.max_attempts:
                logger.info(f"Retrying SQL generation...")
        
        if error or not sql_query:
            state['sql_error'] = error or "Failed to generate SQL"
            state['sql_attempts'] = attempts
            state['error'] = f"SQL generation failed after {attempts} attempts: {error}"
            return state
        
        state['sql_query'] = sql_query
        state['sql_attempts'] = attempts
        
        # Execute SQL
        df, error = self.execute_sql(sql_query)
        
        if error:
            state['sql_error'] = error
            state['error'] = f"SQL execution failed: {error}"
        else:
            # Store DataFrame as serializable format
            state['dataframe_data'] = df.to_dict('records') if df is not None else []
            state['dataframe_columns'] = df.columns.tolist() if df is not None else []
            state['data_description'] = self._describe_dataframe(df)
            state['error'] = None
        
        return state
    
    def _describe_dataframe(self, df: pd.DataFrame) -> str:
        """Create a descriptive summary of dataframe."""
        if df is None or df.empty:
            return "No data available"
        
        desc = f"DataFrame with {len(df)} rows and {len(df.columns)} columns\n\n"
        
        # Add column info
        for col in df.columns:
            desc += f"- {col}: {df[col].dtype}\n"
            if df[col].dtype in ['int64', 'float64']:
                desc += f"  Range: {df[col].min():.2f} to {df[col].max():.2f}\n"
                desc += f"  Mean: {df[col].mean():.2f}\n"
        
        # Add sample data
        desc += f"\nSample Data (first 5 rows):\n{df.head().to_string()}"
        
        return desc
    
    def validate_and_retry(self, state: AgentState) -> bool:
        """
        Determine if retry is needed.
        """
        if state.get('error'):
            return state.get('sql_attempts', 0) < self.max_attempts
        return False