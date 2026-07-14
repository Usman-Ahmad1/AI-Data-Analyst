# src/utils/security.py
"""
Security utilities for input validation and sandboxing.
"""
import re
from typing import Any, Dict, List, Optional
import ast
import logging

logger = logging.getLogger(__name__)

class SecurityValidator:
    """
    Validates inputs and prevents injection attacks.
    """
    
    @staticmethod
    def validate_table_name(table_name: str) -> bool:
        """
        Validate table name against SQL injection.
        
        Args:
            table_name: Table name to validate
            
        Returns:
            True if valid
        """
        # Only allow alphanumeric and underscores
        return bool(re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', table_name))
    
    @staticmethod
    def validate_column_name(column_name: str) -> bool:
        """
        Validate column name against SQL injection.
        
        Args:
            column_name: Column name to validate
            
        Returns:
            True if valid
        """
        return bool(re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', column_name))
    
    @staticmethod
    def validate_python_code(code: str) -> tuple[bool, Optional[str]]:
        """
        Validate Python code for dangerous operations.
        
        Args:
            code: Python code to validate
            
        Returns:
            (is_valid, error_message)
        """
        try:
            tree = ast.parse(code)
            
            # Check for dangerous imports and functions
            dangerous_names = {
                'exec', 'eval', 'compile', '__import__', 'open', 
                'file', 'input', 'raw_input', 'eval', 'execfile',
                'os', 'sys', 'subprocess', 'socket', 'shutil'
            }
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name.split('.')[0] in dangerous_names:
                            return False, f"Dangerous import: {alias.name}"
                
                elif isinstance(node, ast.ImportFrom):
                    if node.module and node.module.split('.')[0] in dangerous_names:
                        return False, f"Dangerous import: {node.module}"
                
                elif isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name):
                        if node.func.id in dangerous_names:
                            return False, f"Dangerous function call: {node.func.id}"
            
            return True, None
            
        except SyntaxError as e:
            return False, f"Syntax error in Python code: {str(e)}"
