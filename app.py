# -*- coding: utf-8 -*-
import sys
import io
import os
import urllib.request
from pathlib import Path
import sqlite3
import zipfile
import logging
import traceback
import re
from datetime import datetime

# ============================================
# STEP 1: SETUP ENCODING
# ============================================
if not hasattr(sys.stdout, '_wrapped'):
    try:
        if hasattr(sys.stdout, 'buffer') and sys.stdout.buffer is not None:
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
            sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
            sys.stdout._wrapped = True
            sys.stderr._wrapped = True
    except (ValueError, AttributeError, OSError):
        pass

# ============================================
# STEP 2: SETUP PATH
# ============================================
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# ============================================
# STEP 3: IMPORT STREAMLIT FIRST
# ============================================
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================
# STEP 4: PAGE CONFIGURATION
# ============================================
st.set_page_config(
    page_title="AI Data Analyst",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# STEP 5: IMPORT CORE MODULES
# ============================================
try:
    from src.core.graph import DataAnalystGraph
except ImportError as e:
    st.error(f"Failed to import core modules: {str(e)}")
    st.info("Please ensure all source files are in the correct location.")
    sys.exit(1)

# ============================================
# STEP 6: DATABASE SETUP FUNCTION (ONLY ONCE!)
# ============================================
def setup_database():
    """Download and setup the Chinook database if not exists."""
    
    # Get database path
    db_path = os.getenv("DATABASE_PATH", "./data/chinook.db")
    db_file = Path(db_path)
    data_dir = db_file.parent
    
    # Create directory
    data_dir.mkdir(exist_ok=True)
    
    # Check if database exists
    if db_file.exists() and db_file.stat().st_size > 0:
        logger.info(f"✅ Database found: {db_path}")
        return str(db_file)
    
    # Download database
    logger.info(f"📥 Downloading database to: {db_path}")
    
    try:
        # Try primary source
        url = "https://github.com/lerocha/chinook-database/raw/master/ChinookDatabase/DataSources/Chinook_Sqlite.sqlite"
        urllib.request.urlretrieve(url, db_path)
        
        # Verify download
        if db_file.exists() and db_file.stat().st_size > 0:
            logger.info(f"✅ Database downloaded! Size: {db_file.stat().st_size} bytes")
            return str(db_file)
        else:
            raise Exception("Downloaded file is empty")
            
    except Exception as e:
        logger.error(f"❌ Download failed: {str(e)}")
        st.error(f"❌ Failed to download database: {str(e)}")
        return None

# ============================================
# STEP 7: INITIALIZE ANALYST
# ============================================
def initialize_analyst():
    """Initialize the data analyst with proper error handling."""
    try:
        # Get API key
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            st.error("❌ GROQ_API_KEY not found!")
            st.info("Please add your Groq API key to .env file or Streamlit secrets")
            return None
        
        # Setup database
        db_path = setup_database()
        if db_path is None:
            st.error("❌ Database setup failed!")
            return None
        
        # Get model name
        model_name = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
        
        # Initialize the graph
        logger.info(f"Initializing DataAnalystGraph with model: {model_name}")
        return DataAnalystGraph(
            groq_api_key=groq_api_key,
            db_path=db_path,
            model_name=model_name,
            temperature=0.1
        )
        
    except Exception as e:
        logger.error(f"Failed to initialize analyst: {str(e)}")
        st.error(f"❌ Failed to initialize: {str(e)}")
        return None

# ============================================
# STEP 8: CSS STYLING
# ============================================
def load_css():
    """Load custom CSS."""
    st.markdown("""
    <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        .dashboard-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 1.5rem 2rem;
            border-radius: 16px;
            margin-bottom: 1.5rem;
            color: white;
            box-shadow: 0 4px 20px rgba(102, 126, 234, 0.3);
        }
        .dashboard-header h1 { font-size: 2rem; font-weight: 700; margin: 0; }
        .dashboard-header p { font-size: 1rem; opacity: 0.9; margin: 0.3rem 0 0 0; }
        .dashboard-header .badge {
            background: rgba(255,255,255,0.2);
            padding: 4px 16px;
            border-radius: 20px;
            font-size: 0.8rem;
            display: inline-block;
            margin-top: 0.3rem;
        }
        .metric-card {
            background: white;
            border-radius: 12px;
            padding: 1.2rem;
            box-shadow: 0 2px 8px rgba(0,0,0,0.06);
            border: 1px solid #f0f0f0;
            transition: transform 0.2s, box-shadow 0.2s;
            height: 100%;
        }
        .metric-card:hover {
            transform: translateY(-4px);
            box-shadow: 0 8px 24px rgba(0,0,0,0.10);
        }
        .metric-card .metric-icon { font-size: 2rem; margin-bottom: 0.3rem; }
        .metric-card .metric-value { font-size: 1.8rem; font-weight: 700; color: #1a1a2e; }
        .metric-card .metric-label { font-size: 0.85rem; color: #6b7280; font-weight: 500; }
        .insight-card {
            background: white;
            border-radius: 12px;
            padding: 1.2rem 1.5rem;
            margin-bottom: 0.8rem;
            border-left: 4px solid #667eea;
            box-shadow: 0 2px 8px rgba(0,0,0,0.04);
            transition: all 0.2s;
        }
        .insight-card:hover { transform: translateX(4px); }
        .insight-card .insight-title { font-weight: 600; color: #1a1a2e; margin: 0; }
        .insight-card .insight-text { color: #4b5563; margin: 0.3rem 0 0 0; font-size: 0.95rem; line-height: 1.5; }
        .section-title { font-size: 1.2rem; font-weight: 600; color: #1a1a2e; margin: 1.5rem 0 1rem 0; display: flex; align-items: center; gap: 0.5rem; }
    </style>
    """, unsafe_allow_html=True)

# ============================================
# STEP 9: HELPER FUNCTIONS
# ============================================
def parse_insights(answer_text):
    """Parse the answer text into structured insights."""
    insights = []
    sections = re.split(r'\n(?=\d+\.\s)', answer_text)
    for section in sections:
        if not section.strip():
            continue
        lines = section.strip().split('\n')
        if lines:
            title = lines[0].strip()
            content = ' '.join([l.strip() for l in lines[1:] if l.strip()])
            if content:
                insights.append({'title': title, 'content': content})
    return insights if insights else [{'title': 'Analysis', 'content': answer_text}]

def extract_metrics(df):
    """Extract key metrics from DataFrame."""
    metrics = {}
    if df is None or df.empty:
        return metrics
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    for col in numeric_cols[:3]:
        try:
            metrics[f"Total {col}"] = f"{df[col].sum():,.2f}"
        except:
            pass
        try:
            if col not in metrics:
                metrics[f"Avg {col}"] = f"{df[col].mean():,.2f}"
        except:
            pass
    if len(metrics) == 0:
        metrics['Rows'] = str(len(df))
        metrics['Columns'] = str(len(df.columns))
    return metrics

def render_header(question):
    """Render the dashboard header."""
    st.markdown(f"""
    <div class="dashboard-header">
        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
            <div>
                <h1>📊 AI Data Analyst</h1>
                <p>{question if question else 'Ask me anything about your data!'}</p>
                <span class="badge">✨ AI Powered Analysis</span>
            </div>
            <div style="text-align: right; font-size: 0.9rem; opacity: 0.8;">
                <div>📅 {datetime.now().strftime('%B %d, %Y')}</div>
                <div>⏰ {datetime.now().strftime('%I:%M %p')}</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def render_sql(sql_query):
    if sql_query:
        with st.expander("📝 View SQL Query", expanded=False):
            st.code(sql_query, language='sql')

def render_charts(chart_html, chart_path):
    if not chart_html and not chart_path:
        return
    st.markdown('<div class="section-title"><span class="section-icon">📈</span> Interactive Dashboard</div>', unsafe_allow_html=True)
    html_content = ""
    if chart_html:
        html_content = chart_html
    elif chart_path and Path(chart_path).exists():
        try:
            with open(chart_path, 'r', encoding='utf-8') as f:
                html_content = f.read()
        except:
            pass
    if html_content:
        st.components.v1.html(html_content, height=650, scrolling=True)
        st.caption("🖱️ Hover, zoom, and interact with the charts")

def render_data_preview(df):
    if df is not None and not df.empty:
        st.markdown('<div class="section-title"><span class="section-icon">📋</span> Data Preview</div>', unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Rows", len(df))
        with col2:
            st.metric("Total Columns", len(df.columns))
        with col3:
            st.metric("Memory Usage", f"{df.memory_usage(deep=True).sum() / 1024:.1f} KB")
        st.dataframe(df, use_container_width=True)

def render_welcome():
    st.markdown("""
    <div style="text-align: center; padding: 3rem 1rem;">
        <div style="font-size: 4rem; margin-bottom: 1rem;">🚀</div>
        <h2 style="color: #1a1a2e; margin: 0;">Welcome to AI Data Analyst</h2>
        <p style="color: #6b7280; font-size: 1.1rem; max-width: 600px; margin: 0.5rem auto;">
            Ask questions about your data in natural language and get instant insights with beautiful visualizations.
        </p>
        <div style="display: flex; gap: 1rem; justify-content: center; flex-wrap: wrap; margin-top: 1.5rem;">
            <span style="background: #f0f0f0; padding: 0.5rem 1rem; border-radius: 8px;">📊 Top Products</span>
            <span style="background: #f0f0f0; padding: 0.5rem 1rem; border-radius: 8px;">💰 Revenue Analysis</span>
            <span style="background: #f0f0f0; padding: 0.5rem 1rem; border-radius: 8px;">👤 Customer Insights</span>
            <span style="background: #f0f0f0; padding: 0.5rem 1rem; border-radius: 8px;">📈 Sales Trends</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ============================================
# STEP 10: MAIN APPLICATION
# ============================================
def main():
    # Load CSS
    load_css()
    
    # Initialize session state
    if 'analyst' not in st.session_state:
        with st.spinner("🔄 Initializing AI Analyst..."):
            st.session_state.analyst = initialize_analyst()
    
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    if 'thread_id' not in st.session_state:
        import uuid
        st.session_state.thread_id = str(uuid.uuid4())
    
    # Sidebar
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; padding: 0.5rem 0;">
            <div style="font-size: 3rem;">📊</div>
            <h3 style="margin: 0; color: #1a1a2e;">AI Analyst</h3>
            <p style="color: #6b7280; font-size: 0.8rem; margin: 0;">Intelligent Data Insights</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("#### 💡 Quick Examples")
        examples = [
            "🏆 Top 5 best-selling products",
            "👤 Most valuable customers",
            "📈 Sales trends over time",
            "💰 Revenue by country"
        ]
        for example in examples:
            if st.button(example, key=example[:20], use_container_width=True):
                st.session_state.current_query = example
        
        st.markdown("---")
        st.markdown("#### 📊 Session Info")
        st.write(f"🆔 Thread: `{st.session_state.thread_id[:8]}...`")
        st.write(f"💬 Messages: {len(st.session_state.messages)}")
        
        if st.button("🔄 New Session", use_container_width=True):
            import uuid
            st.session_state.thread_id = str(uuid.uuid4())
            st.session_state.messages = []
            st.rerun()
        
        st.markdown("---")
        st.caption("✨ Powered by Groq Llama 3.3")
    
    # Main content
    query_input = st.chat_input("💬 Ask me anything about your data...")
    
    if 'current_query' in st.session_state:
        query = st.session_state.current_query
        del st.session_state.current_query
    else:
        query = query_input
    
    if not query:
        render_header("")
        render_welcome()
    else:
        render_header(query)
        
        with st.chat_message("user"):
            st.write(query)
        
        with st.chat_message("assistant"):
            with st.spinner("🤔 Analyzing your data..."):
                try:
                    if st.session_state.analyst is None:
                        st.error("❌ Analyst not initialized. Please check your API key and database.")
                        return
                    
                    result = st.session_state.analyst.run(
                        query,
                        thread_id=st.session_state.thread_id
                    )
                    
                    if result.get('error'):
                        st.error(f"❌ {result['error']}")
                    else:
                        df = result.get('dataframe')
                        sql_query = result.get('sql_query')
                        answer = result.get('answer', '')
                        
                        render_sql(sql_query)
                        
                        if df is not None and not df.empty:
                            metrics = extract_metrics(df)
                            if metrics:
                                st.markdown('<div class="section-title"><span class="section-icon">📊</span> Key Metrics</div>', unsafe_allow_html=True)
                                cols = st.columns(min(len(metrics), 4))
                                icons = ['📊', '💰', '📈', '👥']
                                for i, (key, value) in enumerate(metrics.items()):
                                    col = cols[i % len(cols)]
                                    icon = icons[i % len(icons)]
                                    with col:
                                        st.markdown(f"""
                                        <div class="metric-card">
                                            <div class="metric-icon">{icon}</div>
                                            <div class="metric-value">{value}</div>
                                            <div class="metric-label">{key}</div>
                                        </div>
                                        """, unsafe_allow_html=True)
                        
                        insights = parse_insights(answer)
                        if insights:
                            st.markdown('<div class="section-title"><span class="section-icon">💡</span> Analysis Insights</div>', unsafe_allow_html=True)
                            for insight in insights:
                                st.markdown(f"""
                                <div class="insight-card">
                                    <p class="insight-title">💡 {insight['title']}</p>
                                    <p class="insight-text">{insight['content'][:500]}</p>
                                </div>
                                """, unsafe_allow_html=True)
                        
                        chart_html = result.get('chart_html')
                        chart_path = result.get('chart_path')
                        if chart_html or chart_path:
                            render_charts(chart_html, chart_path)
                        
                        if df is not None and not df.empty:
                            render_data_preview(df)
                    
                except Exception as e:
                    st.error(f"❌ An error occurred: {str(e)}")
                    logger.error(f"Processing error: {str(e)}")
                    logger.error(traceback.format_exc())

if __name__ == "__main__":
    main()