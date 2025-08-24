"""
Visualization Agent - Advanced Code and Data Visualization
Uses Generative models and specialized rendering engines for visual representations
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import networkx as nx
import graphviz
from PIL import Image, ImageDraw, ImageFont
import base64
from io import BytesIO
import json
import ast
import re
from collections import defaultdict, Counter
import colorsys
import pandas as pd

@dataclass
class VisualizationConfig:
    """Configuration for visualizations"""
    theme: str = "dark"  # dark, light, colorful
    output_format: str = "html"  # html, png, svg, json
    interactive: bool = True
    width: int = 1200
    height: int = 800
    dpi: int = 300
    font_size: int = 12
    color_palette: List[str] = None

@dataclass
class VisualizationResult:
    """Results from visualization generation"""
    visualization_type: str
    output_data: Union[str, bytes]
    metadata: Dict[str, Any]
    interactive_elements: List[Dict[str, Any]]
    export_formats: List[str]

class GenerativeVisualizationNetwork(nn.Module):
    """Neural network for generating visualization layouts"""
    
    def __init__(self, input_dim=512, hidden_dim=256, output_dim=128):
        super().__init__()
        
        # Encoder for code/data features
        self.feature_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Layout generator
        self.layout_generator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Tanh()  # Normalize coordinates
        )
        
        # Style generator
        self.style_generator = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),  # Colors, sizes, shapes
            nn.Sigmoid()
        )
        
        # Attention mechanism for element importance
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=8)
        
    def forward(self, features):
        # Encode features
        encoded = self.feature_encoder(features)
        
        # Generate layout coordinates
        layout = self.layout_generator(encoded)
        
        # Generate style parameters
        style = self.style_generator(encoded)
        
        # Apply attention for importance weighting
        attended_features, attention_weights = self.attention(
            encoded.unsqueeze(0), encoded.unsqueeze(0), encoded.unsqueeze(0)
        )
        
        return {
            'layout': layout,
            'style': style,
            'attention': attention_weights.squeeze(0),
            'features': attended_features.squeeze(0)
        }

class CodeStructureVisualizer:
    """Specialized visualizer for code structure"""
    
    def __init__(self):
        self.node_types = {
            'class': {'shape': 'box', 'color': '#3498db', 'style': 'rounded'},
            'function': {'shape': 'ellipse', 'color': '#2ecc71', 'style': 'filled'},
            'variable': {'shape': 'diamond', 'color': '#f39c12', 'style': 'filled'},
            'import': {'shape': 'pentagon', 'color': '#9b59b6', 'style': 'filled'},
            'module': {'shape': 'folder', 'color': '#34495e', 'style': 'filled'}
        }
        
    def visualize_ast_structure(self, ast_analysis: Dict[str, Any], config: VisualizationConfig) -> str:
        """Create AST structure visualization"""
        
        # Create directed graph
        dot = graphviz.Digraph(comment='Code Structure')
        dot.attr(rankdir='TB', size='12,8', dpi=str(config.dpi))
        
        # Set theme
        if config.theme == 'dark':
            dot.attr(bgcolor='#2c3e50', fontcolor='white')
        
        # Add nodes for classes
        for cls in ast_analysis.get('classes', []):
            dot.node(
                cls['name'],
                f"{cls['name']}\\nLine: {cls['line_number']}\\nMethods: {len(cls.get('methods', []))}",
                **self.node_types['class']
            )
            
            # Add method nodes
            for method in cls.get('methods', []):
                method_id = f"{cls['name']}.{method}"
                dot.node(method_id, method, **self.node_types['function'])
                dot.edge(cls['name'], method_id)
        
        # Add standalone functions
        for func in ast_analysis.get('functions', []):
            if not any(func['name'] in cls.get('methods', []) for cls in ast_analysis.get('classes', [])):
                dot.node(
                    func['name'],
                    f"{func['name']}\\nLine: {func['line_number']}\\nParams: {func.get('parameters', 0)}",
                    **self.node_types['function']
                )
        
        # Add import dependencies
        for imp in ast_analysis.get('imports', []):
            module_name = imp['module'].split('.')[0]
            dot.node(module_name, module_name, **self.node_types['import'])
        
        return dot.source
    
    def visualize_complexity_heatmap(self, ast_analysis: Dict[str, Any], config: VisualizationConfig) -> go.Figure:
        """Create complexity heatmap"""
        
        functions = ast_analysis.get('functions', [])
        if not functions:
            return go.Figure()
        
        # Prepare data
        func_names = [f['name'] for f in functions]
        complexities = [f.get('complexity', 1) for f in functions]
        parameters = [f.get('parameters', 0) for f in functions]
        line_numbers = [f.get('line_number', 0) for f in functions]
        
        # Create heatmap data
        z_data = []
        for i, func in enumerate(functions):
            z_data.append([complexities[i], parameters[i], line_numbers[i] / 100])
        
        fig = go.Figure(data=go.Heatmap(
            z=z_data,
            x=['Complexity', 'Parameters', 'Line Number (scaled)'],
            y=func_names,
            colorscale='Viridis' if config.theme == 'dark' else 'Blues',
            hoverongaps=False
        ))
        
        fig.update_layout(
            title='Function Complexity Heatmap',
            xaxis_title='Metrics',
            yaxis_title='Functions',
            width=config.width,
            height=config.height,
            template='plotly_dark' if config.theme == 'dark' else 'plotly_white'
        )
        
        return fig
    
    def visualize_call_graph(self, ast_analysis: Dict[str, Any], config: VisualizationConfig) -> go.Figure:
        """Create interactive call graph"""
        
        call_graph = ast_analysis.get('call_graph', nx.DiGraph())
        
        if call_graph.number_of_nodes() == 0:
            return go.Figure()
        
        # Generate layout
        try:
            pos = nx.spring_layout(call_graph, k=1, iterations=50)
        except:
            pos = {node: (i, 0) for i, node in enumerate(call_graph.nodes())}
        
        # Prepare node traces
        node_x = [pos[node][0] for node in call_graph.nodes()]
        node_y = [pos[node][1] for node in call_graph.nodes()]
        node_text = list(call_graph.nodes())
        node_colors = [call_graph.degree(node) for node in call_graph.nodes()]
        
        # Prepare edge traces
        edge_x, edge_y = [], []
        for edge in call_graph.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
        
        # Create figure
        fig = go.Figure()
        
        # Add edges
        fig.add_trace(go.Scatter(
            x=edge_x, y=edge_y,
            mode='lines',
            line=dict(width=1, color='#888'),
            hoverinfo='none',
            showlegend=False
        ))
        
        # Add nodes
        fig.add_trace(go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            marker=dict(
                size=[10 + degree * 2 for degree in node_colors],
                color=node_colors,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Connections")
            ),
            text=node_text,
            textposition="middle center",
            hovertemplate='<b>%{text}</b><br>Connections: %{marker.color}<extra></extra>',
            showlegend=False
        ))
        
        fig.update_layout(
            title='Code Call Graph',
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20, l=5, r=5, t=40),
            annotations=[
                dict(
                    text="Node size indicates number of connections",
                    showarrow=False,
                    xref="paper", yref="paper",
                    x=0.005, y=-0.002,
                    xanchor='left', yanchor='bottom',
                    font=dict(size=12)
                )
            ],
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            width=config.width,
            height=config.height,
            template='plotly_dark' if config.theme == 'dark' else 'plotly_white'
        )
        
        return fig

class DataFlowVisualizer:
    """Specialized visualizer for data flow and processing"""
    
    def __init__(self):
        self.flow_colors = {
            'source': '#27ae60',
            'transform': '#3498db',
            'sink': '#e74c3c',
            'decision': '#f39c12',
            'process': '#9b59b6'
        }
    
    def visualize_data_pipeline(self, pipeline_data: Dict[str, Any], config: VisualizationConfig) -> go.Figure:
        """Create data pipeline visualization"""
        
        # Create subplots for different views
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Pipeline Flow', 'Data Volume', 'Processing Time', 'Quality Metrics'),
            specs=[[{"type": "scatter"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "scatter"}]]
        )
        
        # Pipeline flow (top-left)
        stages = pipeline_data.get('stages', [])
        if stages:
            x_pos = list(range(len(stages)))
            y_pos = [0] * len(stages)
            
            fig.add_trace(
                go.Scatter(
                    x=x_pos, y=y_pos,
                    mode='markers+lines+text',
                    marker=dict(size=20, color=[self.flow_colors.get(stage.get('type', 'process'), '#333') for stage in stages]),
                    text=[stage.get('name', f'Stage {i}') for i, stage in enumerate(stages)],
                    textposition="top center",
                    line=dict(width=3),
                    showlegend=False
                ),
                row=1, col=1
            )
        
        # Data volume (top-right)
        volumes = pipeline_data.get('data_volumes', [])
        if volumes:
            fig.add_trace(
                go.Bar(
                    x=[f"Stage {i}" for i in range(len(volumes))],
                    y=volumes,
                    marker_color='#3498db',
                    showlegend=False
                ),
                row=1, col=2
            )
        
        # Processing time (bottom-left)
        times = pipeline_data.get('processing_times', [])
        if times:
            fig.add_trace(
                go.Bar(
                    x=[f"Stage {i}" for i in range(len(times))],
                    y=times,
                    marker_color='#e74c3c',
                    showlegend=False
                ),
                row=2, col=1
            )
        
        # Quality metrics (bottom-right)
        quality = pipeline_data.get('quality_metrics', {})
        if quality:
            fig.add_trace(
                go.Scatter(
                    x=list(quality.keys()),
                    y=list(quality.values()),
                    mode='markers+lines',
                    marker=dict(size=12, color='#f39c12'),
                    line=dict(width=2),
                    showlegend=False
                ),
                row=2, col=2
            )
        
        fig.update_layout(
            title_text="Data Pipeline Analysis Dashboard",
            width=config.width,
            height=config.height,
            template='plotly_dark' if config.theme == 'dark' else 'plotly_white'
        )
        
        return fig
    
    def create_data_schema_diagram(self, schema_data: Dict[str, Any], config: VisualizationConfig) -> str:
        """Create data schema visualization using Graphviz"""
        
        dot = graphviz.Digraph(comment='Data Schema')
        dot.attr(rankdir='TB', size='12,8')
        
        if config.theme == 'dark':
            dot.attr(bgcolor='#2c3e50', fontcolor='white')
        
        # Add tables/collections
        for table_name, columns in schema_data.items():
            if isinstance(columns, dict):
                # SQL-style table
                label = f"{table_name}\\n"
                for col_name, col_type in columns.items():
                    label += f"• {col_name}: {col_type}\\n"
                
                dot.node(
                    table_name,
                    label,
                    shape='record',
                    style='filled',
                    fillcolor='#ecf0f1',
                    fontcolor='black'
                )
            else:
                # Simple node
                dot.node(
                    table_name,
                    table_name,
                    shape='box',
                    style='filled',
                    fillcolor='#3498db',
                    fontcolor='white'
                )
        
        return dot.source

class MetricsVisualizer:
    """Specialized visualizer for code metrics and quality"""
    
    def create_quality_dashboard(self, metrics_data: Dict[str, Any], config: VisualizationConfig) -> go.Figure:
        """Create comprehensive quality metrics dashboard"""
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                'Overall Quality Score', 'Complexity Distribution',
                'Security Issues', 'Code Smells',
                'Technical Debt', 'Maintainability Trend'
            ),
            specs=[[{"type": "indicator"}, {"type": "histogram"}],
                   [{"type": "pie"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "scatter"}]]
        )
        
        # Overall quality score (gauge)
        quality_score = metrics_data.get('quality_score', 0.5)
        fig.add_trace(
            go.Indicator(
                mode="gauge+number+delta",
                value=quality_score * 100,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Quality Score"},
                delta={'reference': 70},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': self._get_quality_color(quality_score)},
                    'steps': [
                        {'range': [0, 50], 'color': "lightgray"},
                        {'range': [50, 85], 'color': "gray"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90
                    }
                }
            ),
            row=1, col=1
        )
        
        # Complexity distribution
        complexities = metrics_data.get('function_complexities', [1, 2, 3, 5, 8, 13])
        fig.add_trace(
            go.Histogram(
                x=complexities,
                nbinsx=10,
                marker_color='#3498db',
                showlegend=False
            ),
            row=1, col=2
        )
        
        # Security issues pie chart
        security_issues = metrics_data.get('security_issues', {})
        if security_issues:
            fig.add_trace(
                go.Pie(
                    labels=list(security_issues.keys()),
                    values=list(security_issues.values()),
                    showlegend=False
                ),
                row=2, col=1
            )
        
        # Code smells bar chart
        code_smells = metrics_data.get('code_smells', {})
        if code_smells:
            fig.add_trace(
                go.Bar(
                    x=list(code_smells.keys()),
                    y=list(code_smells.values()),
                    marker_color='#e74c3c',
                    showlegend=False
                ),
                row=2, col=2
            )
        
        # Technical debt
        debt_data = metrics_data.get('technical_debt', {})
        if debt_data:
            fig.add_trace(
                go.Bar(
                    x=list(debt_data.keys()),
                    y=list(debt_data.values()),
                    marker_color='#f39c12',
                    showlegend=False
                ),
                row=3, col=1
            )
        
        # Maintainability trend
        trend_data = metrics_data.get('maintainability_trend', [])
        if trend_data:
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(trend_data))),
                    y=trend_data,
                    mode='lines+markers',
                    marker_color='#27ae60',
                    showlegend=False
                ),
                row=3, col=2
            )
        
        fig.update_layout(
            title_text="Code Quality Dashboard",
            width=config.width,
            height=config.height,
            template='plotly_dark' if config.theme == 'dark' else 'plotly_white'
        )
        
        return fig
    
    def _get_quality_color(self, score: float) -> str:
        """Get color based on quality score"""
        if score >= 0.8:
            return "#27ae60"  # Green
        elif score >= 0.6:
            return "#f39c12"  # Orange
        else:
            return "#e74c3c"   # Red

class VisualizationAgent:
    """
    Advanced Visualization Agent for comprehensive visual representations
    Combines multiple visualization techniques with generative models
    """
    
    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        
        # Initialize neural network
        self.generative_net = GenerativeVisualizationNetwork().to(device)
        
        # Initialize specialized visualizers
        self.code_visualizer = CodeStructureVisualizer()
        self.data_visualizer = DataFlowVisualizer()
        self.metrics_visualizer = MetricsVisualizer()
        
        # Default configuration
        self.default_config = VisualizationConfig()
        
        # Visualization cache
        self.visualization_cache = {}
        
        # Supported formats
        self.supported_formats = ['html', 'png', 'svg', 'json', 'pdf']
        
        # Color palettes
        self.color_palettes = {
            'default': ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c'],
            'dark': ['#34495e', '#2c3e50', '#7f8c8d', '#95a5a6', '#bdc3c7', '#ecf0f1'],
            'vibrant': ['#ff6b6b', '#4ecdc4', '#45b7d1', '#f9ca24', '#f0932b', '#eb4d4b'],
            'pastel': ['#ffb3ba', '#ffdfba', '#ffffba', '#baffc9', '#bae1ff', '#d4baff']
        }
    
    def visualize_code_structure(self, 
                               ast_analysis: Dict[str, Any], 
                               config: Optional[VisualizationConfig] = None) -> VisualizationResult:
        """Create comprehensive code structure visualization"""
        
        config = config or self.default_config
        
        # Generate multiple visualizations
        visualizations = {}
        
        # AST structure diagram
        ast_diagram = self.code_visualizer.visualize_ast_structure(ast_analysis, config)
        visualizations['ast_structure'] = ast_diagram
        
        # Complexity heatmap
        complexity_heatmap = self.code_visualizer.visualize_complexity_heatmap(ast_analysis, config)
        visualizations['complexity_heatmap'] = complexity_heatmap.to_html(include_plotlyjs='cdn')
        
        # Call graph
        call_graph = self.code_visualizer.visualize_call_graph(ast_analysis, config)
        visualizations['call_graph'] = call_graph.to_html(include_plotlyjs='cdn')
        
        # Create combined HTML output
        combined_html = self._create_combined_visualization(visualizations, "Code Structure Analysis", config)
        
        return VisualizationResult(
            visualization_type="code_structure",
            output_data=combined_html,
            metadata={
                'total_classes': len(ast_analysis.get('classes', [])),
                'total_functions': len(ast_analysis.get('functions', [])),
                'max_complexity': max([f.get('complexity', 1) for f in ast_analysis.get('functions', [])], default=1)
            },
            interactive_elements=[
                {'type': 'hover', 'description': 'Hover over nodes for details'},
                {'type': 'zoom', 'description': 'Zoom and pan in call graph'},
                {'type': 'filter', 'description': 'Click legend items to filter'}
            ],
            export_formats=['html', 'png', 'svg']
        )
    
    def visualize_data_flow(self, 
                          pipeline_data: Dict[str, Any], 
                          schema_data: Optional[Dict[str, Any]] = None,
                          config: Optional[VisualizationConfig] = None) -> VisualizationResult:
        """Create data flow and pipeline visualization"""
        
        config = config or self.default_config
        
        visualizations = {}
        
        # Data pipeline visualization
        pipeline_fig = self.data_visualizer.visualize_data_pipeline(pipeline_data, config)
        visualizations['pipeline'] = pipeline_fig.to_html(include_plotlyjs='cdn')
        
        # Schema diagram (if provided)
        if schema_data:
            schema_diagram = self.data_visualizer.create_data_schema_diagram(schema_data, config)
            visualizations['schema'] = schema_diagram
        
        # Create combined visualization
        combined_html = self._create_combined_visualization(visualizations, "Data Flow Analysis", config)
        
        return VisualizationResult(
            visualization_type="data_flow",
            output_data=combined_html,
            metadata={
                'pipeline_stages': len(pipeline_data.get('stages', [])),
                'data_sources': len(pipeline_data.get('sources', [])),
                'total_processing_time': sum(pipeline_data.get('processing_times', []))
            },
            interactive_elements=[
                {'type': 'dashboard', 'description': 'Interactive dashboard with multiple views'},
                {'type': 'drill_down', 'description': 'Click on stages for detailed metrics'}
            ],
            export_formats=['html', 'png', 'pdf']
        )
    
    def visualize_code_quality(self, 
                             metrics_data: Dict[str, Any], 
                             config: Optional[VisualizationConfig] = None) -> VisualizationResult:
        """Create comprehensive code quality dashboard"""
        
        config = config or self.default_config
        
        # Generate quality dashboard
        quality_dashboard = self.metrics_visualizer.create_quality_dashboard(metrics_data, config)
        dashboard_html = quality_dashboard.to_html(include_plotlyjs='cdn')
        
        # Create recommendations panel
        recommendations_html = self._create_recommendations_panel(
            metrics_data.get('recommendations', []), config
        )
        
        # Combine visualizations
        combined_html = f"""
        <div style="display: grid; grid-template-rows: 1fr auto; height: 100vh;">
            <div>{dashboard_html}</div>
            <div style="padding: 20px; background-color: {'#34495e' if config.theme == 'dark' else '#f8f9fa'};">
                {recommendations_html}
            </div>
        </div>
        """
        
        return VisualizationResult(
            visualization_type="code_quality",
            output_data=combined_html,
            metadata={
                'overall_quality': metrics_data.get('quality_score', 0),
                'security_issues_count': len(metrics_data.get('security_issues', [])),
                'code_smells_count': len(metrics_data.get('code_smells', []))
            },
            interactive_elements=[
                {'type': 'gauge', 'description': 'Interactive quality score gauge'},
                {'type': 'charts', 'description': 'Multiple interactive charts'},
                {'type': 'recommendations', 'description': 'Actionable recommendations panel'}
            ],
            export_formats=['html', 'png', 'pdf']
        )
    
    def create_architecture_diagram(self, 
                                  components: List[Dict[str, Any]], 
                                  relationships: List[Dict[str, Any]],
                                  config: Optional[VisualizationConfig] = None) -> VisualizationResult:
        """Create system architecture diagram"""
        
        config = config or self.default_config
        
        # Create architecture graph using Graphviz
        dot = graphviz.Digraph(comment='System Architecture')
        dot.attr(rankdir='TB', size='14,10', dpi=str(config.dpi))
        
        if config.theme == 'dark':
            dot.attr(bgcolor='#2c3e50', fontcolor='white')
        
        # Define component styles
        component_styles = {
            'service': {'shape': 'box', 'style': 'rounded,filled', 'fillcolor': '#3498db'},
            'database': {'shape': 'cylinder', 'style': 'filled', 'fillcolor': '#e74c3c'},
            'api': {'shape': 'hexagon', 'style': 'filled', 'fillcolor': '#2ecc71'},
            'ui': {'shape': 'note', 'style': 'filled', 'fillcolor': '#f39c12'},
            'external': {'shape': 'doubleoctagon', 'style': 'filled', 'fillcolor': '#9b59b6'}
        }
        
        # Add components
        for component in components:
            comp_type = component.get('type', 'service')
            style = component_styles.get(comp_type, component_styles['service'])
            
            dot.node(
                component['id'],
                f"{component['name']}\\n({comp_type})",
                **style
            )
        
        # Add relationships
        for rel in relationships:
            dot.edge(
                rel['source'],
                rel['target'],
                label=rel.get('type', ''),
                style=rel.get('style', 'solid')
            )
        
        # Generate architecture diagram
        arch_source = dot.source
        
        # Create metadata
        metadata = {
            'total_components': len(components),
            'total_relationships': len(relationships),
            'component_types': Counter([comp.get('type', 'service') for comp in components])
        }
        
        return VisualizationResult(
            visualization_type="architecture",
            output_data=arch_source,
            metadata=metadata,
            interactive_elements=[],
            export_formats=['svg', 'png', 'pdf']
        )
    
    def generate_interactive_report(self, 
                                  analysis_results: Dict[str, Any],
                                  config: Optional[VisualizationConfig] = None) -> VisualizationResult:
        """Generate comprehensive interactive analysis report"""
        
        config = config or self.default_config
        
        # Extract data from analysis results
        code_analysis = analysis_results.get('code_analysis', {})
        data_analysis = analysis_results.get('data_analysis', {})
        quality_metrics = analysis_results.get('quality_metrics', {})
        
        # Generate individual visualizations
        visualizations = []
        
        # Code structure section
        if code_analysis:
            code_viz = self.visualize_code_structure(code_analysis, config)
            visualizations.append({
                'title': 'Code Structure Analysis',
                'content': code_viz.output_data,
                'type': 'code'
            })
        
        # Data flow section
        if data_analysis:
            data_viz = self.visualize_data_flow(data_analysis, config=config)
            visualizations.append({
                'title': 'Data Flow Analysis',
                'content': data_viz.output_data,
                'type': 'data'
            })
        
        # Quality metrics section
        if quality_metrics:
            quality_viz = self.visualize_code_quality(quality_metrics, config)
            visualizations.append({
                'title': 'Quality Assessment',
                'content': quality_viz.output_data,
                'type': 'quality'
            })
        
        # Create comprehensive report
        report_html = self._create_interactive_report_html(visualizations, analysis_results, config)
        
        return VisualizationResult(
            visualization_type="comprehensive_report",
            output_data=report_html,
            metadata={
                'sections': len(visualizations),
                'analysis_timestamp': analysis_results.get('timestamp', 'unknown'),
                'total_insights': sum([len(v.get('insights', [])) for v in visualizations])
            },
            interactive_elements=[
                {'type': 'navigation', 'description': 'Section navigation menu'},
                {'type': 'filters', 'description': 'Cross-section filtering'},
                {'type': 'export', 'description': 'Export individual sections'}
            ],
            export_formats=['html', 'pdf']
        )
    
    def _create_combined_visualization(self, 
                                     visualizations: Dict[str, str], 
                                     title: str, 
                                     config: VisualizationConfig) -> str:
        """Create combined HTML visualization"""
        
        theme_styles = self._get_theme_styles(config.theme)
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{title}</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                {theme_styles}
                .visualization-container {{
                    margin: 20px;
                    border-radius: 8px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }}
                .viz-section {{
                    margin-bottom: 30px;
                    padding: 20px;
                }}
                .viz-title {{
                    font-size: 1.5em;
                    margin-bottom: 15px;
                    font-weight: bold;
                }}
                .tab-container {{
                    display: flex;
                    background-color: var(--bg-secondary);
                    border-radius: 8px 8px 0 0;
                }}
                .tab {{
                    padding: 10px 20px;
                    cursor: pointer;
                    background-color: var(--bg-secondary);
                    border: none;
                    color: var(--text-primary);
                }}
                .tab.active {{
                    background-color: var(--bg-primary);
                    border-bottom: 2px solid var(--accent-color);
                }}
                .tab-content {{
                    padding: 20px;
                    background-color: var(--bg-primary);
                    border-radius: 0 0 8px 8px;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>{title}</h1>
        """
        
        # Add tabbed interface for multiple visualizations
        if len(visualizations) > 1:
            html_content += '<div class="tab-container">'
            for i, (viz_name, _) in enumerate(visualizations.items()):
                active_class = "active" if i == 0 else ""
                html_content += f'<button class="tab {active_class}" onclick="showTab(\'{viz_name}\')">{viz_name.replace("_", " ").title()}</button>'
            html_content += '</div>'
            
            for i, (viz_name, viz_content) in enumerate(visualizations.items()):
                display_style = "block" if i == 0 else "none"
                html_content += f'''
                <div id="{viz_name}" class="tab-content" style="display: {display_style};">
                    {viz_content}
                </div>
                '''
        else:
            # Single visualization
            viz_content = next(iter(visualizations.values()))
            html_content += f'<div class="viz-section">{viz_content}</div>'
        
        # Add JavaScript for tab functionality
        html_content += """
            </div>
            <script>
                function showTab(tabName) {
                    // Hide all tab contents
                    var tabContents = document.getElementsByClassName('tab-content');
                    for (var i = 0; i < tabContents.length; i++) {
                        tabContents[i].style.display = 'none';
                    }
                    
                    // Remove active class from all tabs
                    var tabs = document.getElementsByClassName('tab');
                    for (var i = 0; i < tabs.length; i++) {
                        tabs[i].classList.remove('active');
                    }
                    
                    // Show selected tab content
                    document.getElementById(tabName).style.display = 'block';
                    
                    // Add active class to clicked tab
                    event.target.classList.add('active');
                }
            </script>
        </body>
        </html>
        """
        
        return html_content
    
    def _create_recommendations_panel(self, recommendations: List[str], config: VisualizationConfig) -> str:
        """Create recommendations panel HTML"""
        
        if not recommendations:
            return "<p>No specific recommendations available.</p>"
        
        panel_html = """
        <div style="background-color: var(--bg-secondary); padding: 15px; border-radius: 8px;">
            <h3 style="color: var(--accent-color); margin-bottom: 10px;">🔍 Recommendations</h3>
            <ul style="margin: 0; padding-left: 20px;">
        """
        
        for rec in recommendations:
            panel_html += f"<li style='margin-bottom: 8px; color: var(--text-primary);'>{rec}</li>"
        
        panel_html += "</ul></div>"
        return panel_html
    
    def _create_interactive_report_html(self, 
                                      visualizations: List[Dict], 
                                      analysis_results: Dict, 
                                      config: VisualizationConfig) -> str:
        """Create comprehensive interactive report HTML"""
        
        theme_styles = self._get_theme_styles(config.theme)
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>AI Multi-Agent Analysis Report</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                {theme_styles}
                .report-header {{
                    text-align: center;
                    padding: 20px;
                    background: linear-gradient(135deg, var(--accent-color), var(--accent-secondary));
                    color: white;
                    margin-bottom: 30px;
                }}
                .section-nav {{
                    position: fixed;
                    left: 0;
                    top: 200px;
                    width: 200px;
                    padding: 20px;
                    background-color: var(--bg-secondary);
                    border-radius: 0 8px 8px 0;
                    z-index: 1000;
                }}
                .nav-item {{
                    display: block;
                    padding: 10px;
                    margin: 5px 0;
                    color: var(--text-primary);
                    text-decoration: none;
                    border-radius: 4px;
                    transition: background-color 0.3s;
                }}
                .nav-item:hover {{
                    background-color: var(--bg-primary);
                }}
                .content {{
                    margin-left: 240px;
                    padding: 20px;
                }}
                .section {{
                    margin-bottom: 50px;
                    padding: 30px;
                    background-color: var(--bg-primary);
                    border-radius: 12px;
                    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                }}
                .summary-stats {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                    gap: 20px;
                    margin: 20px 0;
                }}
                .stat-card {{
                    background-color: var(--bg-secondary);
                    padding: 20px;
                    border-radius: 8px;
                    text-align: center;
                }}
                .stat-value {{
                    font-size: 2em;
                    font-weight: bold;
                    color: var(--accent-color);
                }}
                .stat-label {{
                    color: var(--text-secondary);
                    margin-top: 5px;
                }}
            </style>
        </head>
        <body>
            <div class="report-header">
                <h1>🤖 AI Multi-Agent Analysis Report</h1>
                <p>Comprehensive code and system analysis by specialized AI agents</p>
                <div class="summary-stats">
                    <div class="stat-card">
                        <div class="stat-value">{len(visualizations)}</div>
                        <div class="stat-label">Analysis Sections</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value">{analysis_results.get('total_files', 'N/A')}</div>
                        <div class="stat-label">Files Analyzed</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value">{analysis_results.get('analysis_duration', 'N/A')}</div>
                        <div class="stat-label">Analysis Time</div>
                    </div>
                </div>
            </div>
            
            <nav class="section-nav">
                <h3 style="color: var(--accent-color); margin-bottom: 15px;">📋 Sections</h3>
        """
        
        # Add navigation links
        for viz in visualizations:
            section_id = viz['title'].lower().replace(' ', '_')
            html += f'<a href="#{section_id}" class="nav-item">{viz["title"]}</a>'
        
        html += """
            </nav>
            
            <div class="content">
        """
        
        # Add visualization sections
        for viz in visualizations:
            section_id = viz['title'].lower().replace(' ', '_')
            html += f"""
            <section id="{section_id}" class="section">
                <h2>{viz['title']}</h2>
                {viz['content']}
            </section>
            """
        
        html += """
            </div>
            
            <script>
                // Smooth scrolling for navigation
                document.querySelectorAll('.nav-item').forEach(item => {
                    item.addEventListener('click', function(e) {
                        e.preventDefault();
                        const target = document.querySelector(this.getAttribute('href'));
                        if (target) {
                            target.scrollIntoView({ behavior: 'smooth' });
                        }
                    });
                });
            </script>
        </body>
        </html>
        """
        
        return html
    
    def _get_theme_styles(self, theme: str) -> str:
        """Get CSS styles for specified theme"""
        
        if theme == 'dark':
            return """
            :root {
                --bg-primary: #2c3e50;
                --bg-secondary: #34495e;
                --text-primary: #ecf0f1;
                --text-secondary: #bdc3c7;
                --accent-color: #3498db;
                --accent-secondary: #2ecc71;
            }
            body {
                background-color: var(--bg-primary);
                color: var(--text-primary);
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                margin: 0;
                padding: 0;
            }
            .container {
                max-width: 1400px;
                margin: 0 auto;
                padding: 20px;
            }
            """
        else:
            return """
            :root {
                --bg-primary: #ffffff;
                --bg-secondary: #f8f9fa;
                --text-primary: #2c3e50;
                --text-secondary: #7f8c8d;
                --accent-color: #3498db;
                --accent-secondary: #2ecc71;
            }
            body {
                background-color: var(--bg-primary);
                color: var(--text-primary);
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                margin: 0;
                padding: 0;
            }
            .container {
                max-width: 1400px;
                margin: 0 auto;
                padding: 20px;
            }
            """
    
    def export_visualization(self, 
                           visualization: VisualizationResult, 
                           format: str, 
                           output_path: str) -> bool:
        """Export visualization to specified format"""
        
        if format not in self.supported_formats:
            return False
        
        try:
            if format == 'html':
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(visualization.output_data)
            elif format == 'json':
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump({
                        'type': visualization.visualization_type,
                        'metadata': visualization.metadata,
                        'interactive_elements': visualization.interactive_elements
                    }, f, indent=2)
            
            return True
        except Exception as e:
            print(f"Export failed: {e}")
            return False
    
    def get_visualization_summary(self) -> Dict[str, Any]:
        """Get summary of visualization capabilities and cache status"""
        
        return {
            'supported_formats': self.supported_formats,
            'color_palettes': list(self.color_palettes.keys()),
            'cache_size': len(self.visualization_cache),
            'visualizers': {
                'code_structure': 'AST analysis, complexity heatmaps, call graphs',
                'data_flow': 'Pipeline visualization, schema diagrams',
                'metrics': 'Quality dashboards, trend analysis',
                'architecture': 'System architecture diagrams'
            }
        }

# Usage example
if __name__ == "__main__":
    agent = VisualizationAgent()
    
    # Sample AST analysis data
    sample_ast = {
        'classes': [
            {'name': 'Calculator', 'line_number': 5, 'methods': ['add', 'subtract', 'multiply']},
            {'name': 'DataProcessor', 'line_number': 20, 'methods': ['process', 'validate']}
        ],
        'functions': [
            {'name': 'add', 'line_number': 7, 'parameters': 2, 'complexity': 1},
            {'name': 'process', 'line_number': 22, 'parameters': 3, 'complexity': 8},
            {'name': 'validate', 'line_number': 35, 'parameters': 1, 'complexity': 5}
        ],
        'imports': [
            {'module': 'os', 'line_number': 1},
            {'module': 'numpy', 'line_number': 2}
        ]
    }
    
    # Create code structure visualization
    config = VisualizationConfig(theme='dark', interactive=True)
    result = agent.visualize_code_structure(sample_ast, config)
    
    print(f"Visualization type: {result.visualization_type}")
    print(f"Metadata: {result.metadata}")
    print(f"Interactive elements: {len(result.interactive_elements)}")
    
    # Export to file
    agent.export_visualization(result, 'html', 'code_structure.html')
    print("Visualization exported to code_structure.html")