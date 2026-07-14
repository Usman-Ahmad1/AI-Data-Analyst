"""
Advanced Visualization Agent with Multiple Chart Types and Dashboard Layout
"""
from typing import Dict, Any, Optional, Tuple, List
import logging
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from plotly.io import to_html
import pandas as pd
import numpy as np
from pathlib import Path
import uuid
from datetime import datetime
import json
import re

from ..core.state import AgentState

logger = logging.getLogger(__name__)

class AdvancedVisualizationAgent:
    """
    Creates advanced dashboards with multiple chart types.
    """
    
    def __init__(self, output_dir: str = "charts"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
    
    def create_dashboard(self, df: pd.DataFrame, question: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Create an advanced dashboard with multiple visualizations.
        """
        try:
            if df is None or df.empty:
                return None, None
            
            # Get column information
            numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            
            # Create individual chart figures
            charts = []
            
            # 1. Main Bar Chart
            if len(categorical_cols) >= 1 and len(numeric_cols) >= 1:
                bar_chart = self._create_bar_chart(df, categorical_cols[0], numeric_cols[0])
                if bar_chart:
                    charts.append(bar_chart)
            
            # 2. Pie Chart (top categories)
            if len(categorical_cols) >= 1 and len(numeric_cols) >= 1:
                pie_chart = self._create_pie_chart(df, categorical_cols[0], numeric_cols[0])
                if pie_chart:
                    charts.append(pie_chart)
            
            # 3. Distribution Chart
            if len(numeric_cols) >= 1:
                dist_chart = self._create_distribution_chart(df, numeric_cols[0])
                if dist_chart:
                    charts.append(dist_chart)
            
            # 4. Trend Line Chart
            if len(numeric_cols) >= 2:
                line_chart = self._create_line_chart(df, numeric_cols[:2])
                if line_chart:
                    charts.append(line_chart)
            
            # 5. Heatmap (correlation)
            if len(numeric_cols) >= 3:
                heatmap = self._create_heatmap(df, numeric_cols)
                if heatmap:
                    charts.append(heatmap)
            
            # Create the dashboard HTML with embedded charts
            dashboard_html = self._create_dashboard_html(charts, question, df)
            
            # Save the dashboard
            filename = f"dashboard_{uuid.uuid4().hex[:8]}.html"
            file_path = self.output_dir / filename
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(dashboard_html)
            
            logger.info(f"✅ Dashboard created with {len(charts)} charts")
            return str(file_path), dashboard_html
            
        except Exception as e:
            logger.error(f"Dashboard creation failed: {str(e)}")
            return None, None
    
    def _create_bar_chart(self, df: pd.DataFrame, x_col: str, y_col: str) -> Optional[go.Figure]:
        """Create an enhanced bar chart."""
        try:
            data = df.sort_values(y_col, ascending=False).head(20)
            
            fig = px.bar(
                data,
                x=x_col,
                y=y_col,
                title=f"{y_col} by {x_col}",
                color=y_col,
                color_continuous_scale='Viridis',
                text=y_col,
                height=400
            )
            
            fig.update_traces(
                texttemplate='%{text:.2f}',
                textposition='outside'
            )
            fig.update_layout(
                xaxis_tickangle=-45,
                showlegend=False,
                margin=dict(l=50, r=50, t=60, b=80),
                template='plotly_white'
            )
            return fig
        except Exception as e:
            logger.warning(f"Bar chart creation failed: {str(e)}")
            return None
    
    def _create_pie_chart(self, df: pd.DataFrame, category_col: str, value_col: str) -> Optional[go.Figure]:
        """Create a pie chart for distribution."""
        try:
            grouped = df.groupby(category_col)[value_col].sum().reset_index()
            grouped = grouped.sort_values(value_col, ascending=False).head(10)
            
            fig = px.pie(
                grouped,
                values=value_col,
                names=category_col,
                title=f"Distribution of {value_col}",
                height=400,
                hole=0.3
            )
            
            fig.update_traces(
                textposition='inside',
                textinfo='percent+label',
                hovertemplate='<b>%{label}</b><br>Value: %{value:,.0f}<br>Percentage: %{percent}<extra></extra>'
            )
            fig.update_layout(
                template='plotly_white',
                margin=dict(l=20, r=20, t=60, b=20)
            )
            return fig
        except Exception as e:
            logger.warning(f"Pie chart creation failed: {str(e)}")
            return None
    
    def _create_distribution_chart(self, df: pd.DataFrame, col: str) -> Optional[go.Figure]:
        """Create a distribution histogram."""
        try:
            fig = px.histogram(
                df,
                x=col,
                title=f"Distribution of {col}",
                marginal='box',
                nbins=30,
                height=400,
                color_discrete_sequence=['#636EFA']
            )
            
            fig.update_layout(
                bargap=0.1,
                showlegend=False,
                template='plotly_white',
                margin=dict(l=50, r=50, t=60, b=50)
            )
            return fig
        except Exception as e:
            logger.warning(f"Distribution chart creation failed: {str(e)}")
            return None
    
    def _create_line_chart(self, df: pd.DataFrame, y_cols: List[str]) -> Optional[go.Figure]:
        """Create a line chart for trends."""
        try:
            fig = px.line(
                df,
                x=df.index if len(df) > 0 else range(len(df)),
                y=y_cols,
                title="Trend Analysis",
                markers=True,
                height=400
            )
            
            fig.update_layout(
                xaxis_title="Index",
                yaxis_title="Value",
                hovermode='x unified',
                template='plotly_white',
                margin=dict(l=50, r=50, t=60, b=50),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            return fig
        except Exception as e:
            logger.warning(f"Line chart creation failed: {str(e)}")
            return None
    
    def _create_heatmap(self, df: pd.DataFrame, numeric_cols: List[str]) -> Optional[go.Figure]:
        """Create a correlation heatmap."""
        try:
            corr = df[numeric_cols].corr()
            
            fig = px.imshow(
                corr,
                text_auto=True,
                aspect='auto',
                color_continuous_scale='RdBu_r',
                title="Correlation Heatmap",
                height=400,
                zmin=-1,
                zmax=1
            )
            
            fig.update_layout(
                xaxis_tickangle=-45,
                template='plotly_white',
                margin=dict(l=50, r=50, t=60, b=80)
            )
            return fig
        except Exception as e:
            logger.warning(f"Heatmap creation failed: {str(e)}")
            return None
    
    def _create_dashboard_html(self, charts: List[go.Figure], question: str, df: pd.DataFrame) -> str:
        """Create the complete dashboard HTML with properly embedded charts."""
        
        # Generate HTML for each chart individually
        chart_htmls = []
        for i, fig in enumerate(charts):
            if fig:
                try:
                    # Generate complete standalone HTML for each chart
                    chart_div = to_html(fig, include_plotlyjs='cdn', full_html=False)
                    chart_htmls.append(chart_div)
                except Exception as e:
                    logger.warning(f"Failed to convert chart {i}: {str(e)}")
        
        # Build the dashboard HTML
        html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Data Dashboard</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js" charset="utf-8"></script>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #f5f7fa;
            padding: 20px;
            color: #1a1a2e;
        }}
        .dashboard-container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 16px;
            box-shadow: 0 4px 24px rgba(0,0,0,0.08);
            padding: 30px;
        }}
        .dashboard-header {{
            border-bottom: 2px solid #e8ecf1;
            padding-bottom: 20px;
            margin-bottom: 30px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
        }}
        .dashboard-title {{
            font-size: 28px;
            font-weight: 700;
            color: #1a1a2e;
        }}
        .dashboard-subtitle {{
            color: #6b7280;
            font-size: 14px;
            margin-top: 5px;
        }}
        .dashboard-meta {{
            background: #f8fafc;
            padding: 10px 20px;
            border-radius: 8px;
            font-size: 14px;
            color: #4b5563;
            display: flex;
            gap: 20px;
            flex-wrap: wrap;
        }}
        .chart-grid {{
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 25px;
            margin-top: 20px;
        }}
        .chart-card {{
            background: white;
            border: 1px solid #e8ecf1;
            border-radius: 12px;
            padding: 15px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.04);
            transition: transform 0.2s;
        }}
        .chart-card:hover {{
            transform: translateY(-2px);
            box-shadow: 0 4px 16px rgba(0,0,0,0.08);
        }}
        .chart-full {{
            grid-column: 1 / -1;
        }}
        .chart-container {{
            width: 100%;
            min-height: 420px;
        }}
        .chart-container .plotly-graph-div {{
            width: 100% !important;
            height: 420px !important;
        }}
        .badge {{
            background: #e8ecf1;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 12px;
            color: #4b5563;
        }}
        @media (max-width: 768px) {{
            .chart-grid {{
                grid-template-columns: 1fr;
            }}
            .dashboard-header {{
                flex-direction: column;
                align-items: flex-start;
                gap: 10px;
            }}
        }}
    </style>
</head>
<body>
    <div class="dashboard-container">
        <div class="dashboard-header">
            <div>
                <div class="dashboard-title">📊 AI Data Dashboard</div>
                <div class="dashboard-subtitle">{question[:100]}</div>
            </div>
            <div class="dashboard-meta">
                <span>📅 {datetime.now().strftime('%B %d, %Y')}</span>
                <span>📊 {len(df)} rows</span>
                <span>📋 {len(df.columns)} columns</span>
                <span class="badge">AI Generated</span>
            </div>
        </div>
        
        <div class="chart-grid">
"""
        
        # Add charts to grid
        for i, chart_div in enumerate(chart_htmls):
            is_full = len(chart_htmls) == 1
            grid_class = "chart-full" if is_full else ""
            html += f"""
            <div class="chart-card {grid_class}">
                <div class="chart-container">
                    {chart_div}
                </div>
            </div>
            """
        
        # Add summary stats
        html += f"""
        </div>
        
        <div style="margin-top: 30px; padding: 20px; background: #f8fafc; border-radius: 12px; border: 1px solid #e8ecf1;">
            <h3>📋 Summary Statistics</h3>
            <table style="width:100%; border-collapse: collapse; margin-top: 10px;">
                <tr style="background-color: #f0f0f0;">
                    <th style="padding: 8px; border: 1px solid #ddd; text-align: left;">Metric</th>
                    <th style="padding: 8px; border: 1px solid #ddd; text-align: right;">Value</th>
                </tr>
                <tr>
                    <td style="padding: 8px; border: 1px solid #ddd;">Total Rows</td>
                    <td style="padding: 8px; border: 1px solid #ddd; text-align: right;">{len(df)}</td>
                </tr>
                <tr>
                    <td style="padding: 8px; border: 1px solid #ddd;">Total Columns</td>
                    <td style="padding: 8px; border: 1px solid #ddd; text-align: right;">{len(df.columns)}</td>
                </tr>
                <tr>
                    <td style="padding: 8px; border: 1px solid #ddd;">Charts Generated</td>
                    <td style="padding: 8px; border: 1px solid #ddd; text-align: right;">{len(charts)}</td>
                </tr>
            </table>
        </div>
    </div>
</body>
</html>
"""
        
        return html
    
    def process(self, state: AgentState) -> AgentState:
        """Process and create advanced dashboard."""
        try:
            dataframe_data = state.get('dataframe_data', [])
            dataframe_columns = state.get('dataframe_columns', [])
            
            if not dataframe_data:
                state['error'] = "No data available for visualization"
                return state
            
            df = pd.DataFrame(dataframe_data, columns=dataframe_columns)
            
            if df is None or df.empty:
                state['error'] = "Empty data for visualization"
                return state
            
            question = state.get('question', 'Data Analysis')
            
            # Create advanced dashboard
            file_path, html_string = self.create_dashboard(df, question)
            
            if file_path and html_string:
                state['chart_path'] = file_path
                state['chart_html'] = html_string
                state['error'] = None
                logger.info(f"✅ Advanced dashboard created: {file_path}")
            else:
                state['error'] = "Failed to create dashboard"
            
        except Exception as e:
            logger.error(f"Dashboard creation failed: {str(e)}")
            state['error'] = f"Dashboard failed: {str(e)}"
        
        return state