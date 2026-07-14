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

from ..core.state import AgentState, SQLGenerationRequest
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
        
        Args:
            llm: Groq LLM instance
            db_manager: Database manager instance
            max_attempts: Maximum SQL generation attempts
        """
        self.llm = llm
        self.db_manager = db_manager
        self.max_attempts = max_attempts
        
        # Setup prompts
        self.sql_prompt = ChatPromptTemplate.from_template(
            PromptTemplates.SQL_GENERATION
        )
        
        self.output_parser = StrOutputParser()
        
        # Build chain
        self.chain = self.sql_prompt | self.llm | self.output_parser
    
    def generate_sql(self, question: str, schema_text: str) -> Tuple[str, Optional[str]]:
        """
        Generate SQL from natural language question.
        
        Args:
            question: Natural language question
            schema_text: Database schema description
            
        Returns:
            (sql_query, error_message)
        """
        try:
            examples = self._format_examples()
            
            response = self.chain.invoke({
                "schema_text": schema_text,
                "examples": examples,
                "question": question
            })
            
            # Clean response - remove markdown code blocks if present
            sql_query = self._clean_sql(response)
            
            return sql_query, None
            
        except Exception as e:
            logger.error(f"SQL generation failed: {str(e)}")
            return "", str(e)
    
    def _format_examples(self) -> str:
        """Format few-shot examples."""
        examples = PromptTemplates.get_sql_examples()
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
        # Ensure ends with semicolon
        if not response.endswith(";"):
            response += ";"
        return response
    
    def execute_sql(self, sql_query: str) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
        """
        Execute SQL query safely.
        
        Args:
            sql_query: SQL query to execute
            
        Returns:
            (dataframe, error_message)
        """
        try:
            df = self.db_manager.execute_query(sql_query)
            return df, None
        except Exception as e:
            logger.error(f"SQL execution failed: {str(e)}")
            return None, str(e)
    
    def process(self, state: AgentState) -> AgentState:
        """
        Process SQL generation and execution.
        
        Args:
            state: Current agent state
            
        Returns:
            Updated state
        """
        question = state['question']
        schema_text = state['schema_text']
        
        # Generate SQL
        sql_query, error = self.generate_sql(question, schema_text)
        
        if error:
            state['sql_error'] = error
            state['sql_attempts'] = state.get('sql_attempts', 0) + 1
            return state
        
        state['sql_query'] = sql_query
        
        # Execute SQL
        df, error = self.execute_sql(sql_query)
        
        if error:
            state['sql_error'] = error
            state['sql_attempts'] = state.get('sql_attempts', 0) + 1
        else:
            state['dataframe'] = df
            state['data_description'] = self._describe_dataframe(df)
            state['sql_attempts'] = 0
        
        return state
    
    def _describe_dataframe(self, df: pd.DataFrame) -> str:
        """Create a descriptive summary of dataframe."""
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
        
        Args:
            state: Current state
            
        Returns:
            True if retry should be attempted
        """
        if state.get('error'):
            return state.get('sql_attempts', 0) < self.max_attempts
        return False