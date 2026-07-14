# src/agents/viz_agent.py - Fixed version
from typing import Dict, Any, Optional, Tuple
import logging
import plotly.graph_objects as go
import plotly.express as px
from plotly.io import to_html
import pandas as pd
from pathlib import Path
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from ..core.state import AgentState
from ..prompts.templates import PromptTemplates
from ..utils.security import SecurityValidator

logger = logging.getLogger(__name__)

class VisualizationAgent:
    def __init__(self, llm: ChatGroq, output_dir: str = "charts"):
        self.llm = llm
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.viz_prompt = ChatPromptTemplate.from_template(
            PromptTemplates.VIZ_GENERATION
        )
        self.output_parser = StrOutputParser()
        self.chain = self.viz_prompt | self.llm | self.output_parser
        self.validator = SecurityValidator()
    
    def process(self, state: AgentState) -> AgentState:
        """Process visualization generation - simplified."""
        # This is now a fallback - the main chart creation happens in graph.py
        # Just pass through and let the graph handle it
        return state
    
    def save_chart(self, fig: go.Figure, filename: str = None) -> Tuple[str, str]:
        """Save chart to file."""
        if filename is None:
            import uuid
            filename = f"chart_{uuid.uuid4().hex[:8]}.html"
        
        file_path = self.output_dir / filename
        html_string = to_html(fig, include_plotlyjs='cdn', full_html=True)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(html_string)
        
        return str(file_path), html_string