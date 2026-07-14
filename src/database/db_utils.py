# src/database/db_utils.py
"""
Database utilities for safe query execution and schema management.
"""
import sqlite3
from typing import List, Dict, Any, Optional, Tuple
from contextlib import contextmanager
import pandas as pd
from sqlalchemy import create_engine, text, inspect
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError
import logging
from pathlib import Path
import urllib.request
import os

logger = logging.getLogger(__name__)

class DatabaseManager:
    """
    Manages database connections and operations with security features.
    """
    
    def __init__(self, db_path: str):
        """
        Initialize database manager.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        
        # Auto-download database if not exists
        if not self.db_path.exists():
            logger.info(f"Database not found at {db_path}. Downloading...")
            self._download_database()
        
        self._engine: Optional[Engine] = None
        self._schema_cache: Optional[Dict] = None
        self._max_rows = 1000  # Safety limit
    
    def _download_database(self):
        """Download Chinook database if not exists."""
        try:
            # Create data directory
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Download from GitHub
            url = "https://github.com/lerocha/chinook-database/raw/master/ChinookDatabase/DataSources/Chinook_Sqlite.sqlite"
            urllib.request.urlretrieve(url, self.db_path)
            
            if self.db_path.exists() and self.db_path.stat().st_size > 0:
                logger.info(f"✅ Database downloaded successfully: {self.db_path}")
            else:
                raise Exception("Download failed - file is empty")
                
        except Exception as e:
            logger.error(f"Failed to download database: {str(e)}")
            raise FileNotFoundError(f"Could not download database to {self.db_path}. Error: {str(e)}")
    
    @property
    def engine(self) -> Engine:
        """Lazy-load SQLAlchemy engine."""
        if self._engine is None:
            self._engine = create_engine(
                f"sqlite:///{self.db_path}",
                echo=False,
                pool_size=5,
                max_overflow=10
            )
        return self._engine
    
    @contextmanager
    def get_connection(self):
        """
        Context manager for database connections with automatic cleanup.
        """
        connection = self.engine.connect()
        try:
            yield connection
        finally:
            connection.close()
    
    def get_schema(self, refresh: bool = False) -> Dict[str, Any]:
        """
        Get database schema with table structures and relationships.
        """
        if self._schema_cache is not None and not refresh:
            return self._schema_cache
        
        schema_info = {}
        inspector = inspect(self.engine)
        
        for table_name in inspector.get_table_names():
            if table_name.startswith('sqlite_'):
                continue
                
            columns = inspector.get_columns(table_name)
            foreign_keys = inspector.get_foreign_keys(table_name)
            
            schema_info[table_name] = {
                'columns': [
                    {
                        'name': col['name'],
                        'type': str(col['type']),
                        'nullable': col['nullable'],
                        'primary_key': col.get('primary_key', False)
                    }
                    for col in columns
                ],
                'foreign_keys': [
                    {
                        'constrained_columns': fk['constrained_columns'],
                        'referred_table': fk['referred_table'],
                        'referred_columns': fk['referred_columns']
                    }
                    for fk in foreign_keys
                ]
            }
        
        self._schema_cache = schema_info
        return schema_info
    
    def get_schema_text(self) -> str:
        """
        Convert schema to text format for LLM prompts.
        """
        schema = self.get_schema()
        schema_text = "DATABASE SCHEMA:\n\n"
        
        for table_name, table_info in schema.items():
            schema_text += f"Table: {table_name}\n"
            schema_text += "Columns:\n"
            for col in table_info['columns']:
                pk_marker = " (PRIMARY KEY)" if col['primary_key'] else ""
                null_marker = " NOT NULL" if not col['nullable'] else ""
                schema_text += f"  - {col['name']}: {col['type']}{pk_marker}{null_marker}\n"
            
            if table_info['foreign_keys']:
                schema_text += "Foreign Keys:\n"
                for fk in table_info['foreign_keys']:
                    schema_text += f"  - {fk['constrained_columns']} → {fk['referred_table']}({fk['referred_columns']})\n"
            
            schema_text += "\n"
        
        return schema_text
    
    def validate_query(self, query: str) -> Tuple[bool, str]:
        """
        Validate SQL query for security.
        """
        query_upper = query.upper().strip()
        
        dangerous_ops = ['DROP', 'DELETE', 'UPDATE', 'INSERT', 'ALTER', 'CREATE', 'TRUNCATE']
        for op in dangerous_ops:
            if op in query_upper:
                return False, f"Query contains dangerous operation: {op}. Only SELECT queries are allowed."
        
        if ';' in query and not query.endswith(';'):
            statements = [s.strip() for s in query.split(';') if s.strip()]
            if len(statements) > 1:
                return False, "Multiple SQL statements are not allowed."
        
        if '/*' in query or '*/' in query:
            return False, "Block comments are not allowed."
        
        return True, ""
    
    def execute_query(self, query: str, limit: Optional[int] = None) -> pd.DataFrame:
        """
        Execute SQL query safely.
        """
        is_valid, error_msg = self.validate_query(query)
        if not is_valid:
            raise ValueError(error_msg)
        
        limit_clause = f" LIMIT {limit or self._max_rows}"
        if 'LIMIT' not in query.upper():
            query = query.rstrip(';') + limit_clause
        
        try:
            logger.info(f"Executing query: {query[:100]}...")
            with self.get_connection() as conn:
                df = pd.read_sql(query, conn)
                logger.info(f"Query returned {len(df)} rows")
                return df
        except SQLAlchemyError as e:
            logger.error(f"Database error: {str(e)}")
            raise
    
    def get_table_sample(self, table_name: str, limit: int = 5) -> pd.DataFrame:
        """
        Get sample data from a table.
        """
        return self.execute_query(f"SELECT * FROM {table_name} LIMIT {limit}")