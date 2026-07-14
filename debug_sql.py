# debug_sql.py
"""
Comprehensive debug script for SQL generation.
"""
import sys
import os
from pathlib import Path
import logging
import traceback

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv()

from src.database.db_utils import DatabaseManager
from src.agents.sql_agent import SQLAgent
from src.prompts.templates import PromptTemplates
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

# Setup logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def debug_sql_generation():
    """Comprehensive debug of SQL generation."""
    print("=" * 70)
    print("🔍 SQL Generation Debug")
    print("=" * 70)
    
    # 1. Check environment
    print("\n📋 1. Environment Check:")
    print("-" * 50)
    
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        print("❌ GROQ_API_KEY not found in .env")
        print("   Please add: GROQ_API_KEY=gsk_... to .env")
        return
    print(f"✅ GROQ_API_KEY found: {groq_api_key[:10]}...")
    
    db_path = os.getenv("DATABASE_PATH", "./data/chinook.db")
    if not os.path.exists(db_path):
        print(f"❌ Database not found: {db_path}")
        return
    print(f"✅ Database found: {db_path}")
    
    # 2. Test database connection
    print("\n📋 2. Database Connection:")
    print("-" * 50)
    try:
        db_manager = DatabaseManager(db_path)
        schema_text = db_manager.get_schema_text()
        print(f"✅ Schema loaded: {len(schema_text)} characters")
        print(f"   First 500 chars:\n{schema_text[:500]}")
    except Exception as e:
        print(f"❌ Database error: {str(e)}")
        traceback.print_exc()
        return
    
    # 3. Test LLM connection
    print("\n📋 3. LLM Connection:")
    print("-" * 50)
    try:
        llm = ChatGroq(
            groq_api_key=groq_api_key,
            model_name="llama-3.1-70b-versatile",
            temperature=0.1
        )
        print("✅ LLM initialized")
        
        # Test LLM with simple prompt
        test_response = llm.invoke("Say 'Hello' if you can hear me")
        print(f"✅ LLM test response: {test_response.content[:50]}...")
    except Exception as e:
        print(f"❌ LLM error: {str(e)}")
        traceback.print_exc()
        return
    
    # 4. Test SQL Agent
    print("\n📋 4. SQL Agent Test:")
    print("-" * 50)
    try:
        sql_agent = SQLAgent(llm, db_manager)
        print("✅ SQL Agent initialized")
    except Exception as e:
        print(f"❌ SQL Agent error: {str(e)}")
        traceback.print_exc()
        return
    
    # 5. Test SQL Generation
    print("\n📋 5. SQL Generation Test:")
    print("-" * 50)
    
    test_questions = [
        "What are the top 5 best-selling products?",
        "Show me sales trends over time",
        "Who are the top customers by spending?"
    ]
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n[{i}] Question: {question}")
        print("-" * 30)
        
        try:
            # Generate SQL
            sql, error = sql_agent.generate_sql(question, schema_text)
            
            if error:
                print(f"❌ Generation Error: {error}")
                print(f"   SQL returned: '{sql}'")
            else:
                print(f"✅ Generated SQL:")
                print(f"   {sql}")
                
                # Try to execute
                df, exec_error = sql_agent.execute_sql(sql)
                if exec_error:
                    print(f"❌ Execution Error: {exec_error}")
                else:
                    print(f"✅ Execution successful: {len(df)} rows")
                    if len(df) > 0:
                        print(f"   Preview:\n{df.head()}")
        
        except Exception as e:
            print(f"❌ Unexpected error: {str(e)}")
            traceback.print_exc()
    
    # 6. Test the full chain
    print("\n📋 6. Full Chain Test:")
    print("-" * 50)
    
    try:
        # Direct chain test
        prompt = ChatPromptTemplate.from_template(
            PromptTemplates.SQL_GENERATION
        )
        chain = prompt | llm
        
        print("Testing chain with simplified prompt...")
        response = chain.invoke({
            "schema_text": schema_text[:1000],  # Limit for testing
            "examples": PromptTemplates.get_sql_examples(),
            "question": "What are the top 5 products?"
        })
        print(f"✅ Chain response: {response.content[:200]}...")
    except Exception as e:
        print(f"❌ Chain error: {str(e)}")
        traceback.print_exc()
    
    print("\n" + "=" * 70)
    print("✅ Debug complete!")
    print("=" * 70)

if __name__ == "__main__":
    debug_sql_generation()