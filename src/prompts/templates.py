# src/prompts/templates.py
"""
Prompt templates for different agents.
"""
from typing import List, Dict, Any

class PromptTemplates:
    """Centralized prompt template management."""
    
    SYSTEM_PROMPT = """You are an AI Data Analyst with expertise in SQL, Python, and data visualization.
    Your goal is to help users understand their data through queries, analysis, and visualizations.
    
    Follow these principles:
    1. Always validate your understanding before proceeding
    2. Choose the right tool (SQL or Python) for each task
    3. Generate accurate and efficient code
    4. Explain your reasoning clearly
    5. Handle errors gracefully and suggest alternatives
    """
    
    SQL_GENERATION = """You are an expert SQL query generator.

    DATABASE SCHEMA:
    {schema_text}
    
    RULES:
    1. Generate ONLY valid SQL for SQLite
    2. Always include appropriate LIMIT clauses
    3. Use proper JOINs instead of subqueries when possible
    4. Avoid SELECT * - specify only needed columns
    5. Use aliases for readability
    6. Consider performance - add WHERE clauses to filter data
    
    FEW-SHOT EXAMPLES:
    {examples}
    
    USER QUESTION: {question}
    
    Generate a SQL query that answers this question. Return ONLY the SQL query, no explanations.
    
    SQL:"""
    
    PYTHON_ANALYSIS = """You are an expert Python data analyst using pandas.
    
    Data Description:
    {data_description}
    
    User Question: {question}
    
    RULES:
    1. Write clean, efficient pandas code
    2. Use proper error handling
    3. Include comments for complex operations
    4. Store results in a 'result' variable
    5. Generate meaningful output
    
    Available libraries: pandas as pd, numpy as np
    
    Python Code:"""
    
    VIZ_GENERATION = """You are an expert data visualization expert using Plotly.
    
    Data Description:
    {data_description}
    
    User Question: {question}
    
    RULES:
    1. Choose the most appropriate chart type
    2. Use Plotly Express for simplicity
    3. Include proper titles, labels, and formatting
    4. Make visualizations interactive and informative
    5. Save the figure to a variable named 'fig'
    
    Generate Python code that creates a Plotly visualization.
    
    Python Code:"""
    
    SYNTHESIS = """You are a business insights expert.
    
    Data Analysis Results:
    {analysis_results}
    
    Visualization: {chart_path}
    
    Original Question: {question}
    
    Provide a comprehensive answer that:
    1. Summarizes the key findings
    2. Provides actionable business insights
    3. Recommends next steps
    4. Highlights any limitations or caveats
    
    Answer:"""
    
    ROUTER = """You are a query routing expert.
    
    Question: {question}
    
    Determine the best approach:
    - SQL: For structured data queries, aggregations, joins
    - Python: For complex calculations, statistical analysis, custom logic
    - Hybrid: For queries requiring both SQL and Python
    
    Classify the query type as one of: sql, python, hybrid
    
    Classification:"""

    @classmethod
    def get_sql_examples(cls) -> List[Dict[str, str]]:
        """Get few-shot SQL examples for Chinook database."""
        return [
            {
                "question": "Which products sold the most?",
                "sql": """SELECT 
                    Track.Name as Product,
                    SUM(InvoiceLine.Quantity) as TotalSold
                FROM Track
                JOIN InvoiceLine ON Track.TrackId = InvoiceLine.TrackId
                GROUP BY Track.TrackId
                ORDER BY TotalSold DESC
                LIMIT 10;"""
            },
            {
                "question": "What's the revenue by country?",
                "sql": """SELECT 
                    BillingCountry as Country,
                    ROUND(SUM(Total), 2) as TotalRevenue
                FROM Invoice
                GROUP BY BillingCountry
                ORDER BY TotalRevenue DESC;"""
            },
            {
                "question": "Who are the top 5 customers by spending?",
                "sql": """SELECT 
                    Customer.FirstName || ' ' || Customer.LastName as Customer,
                    ROUND(SUM(Invoice.Total), 2) as TotalSpent
                FROM Customer
                JOIN Invoice ON Customer.CustomerId = Invoice.CustomerId
                GROUP BY Customer.CustomerId
                ORDER BY TotalSpent DESC
                LIMIT 5;"""
            }
        ]
