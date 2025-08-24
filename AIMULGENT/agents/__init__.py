"""
AIMULGENT Agents Package
Specialized AI agents for comprehensive code analysis and processing
"""

from .perception_agent import PerceptionAgent, PerceptionResult
from .memory_agent import MemoryAgent, MemoryEntry, MemorySearchResult
from .data_agent import DataAgent, DataProcessingResult, DataFlowNode
from .analysis_agent import AnalysisAgent, AnalysisResult, CodeMetrics, SecurityIssue, CodeSmell
from .visualization_agent import VisualizationAgent, VisualizationResult, VisualizationConfig

__all__ = [
    # Agents
    'PerceptionAgent',
    'MemoryAgent',
    'DataAgent', 
    'AnalysisAgent',
    'VisualizationAgent',
    
    # Result classes
    'PerceptionResult',
    'MemorySearchResult',
    'DataProcessingResult',
    'AnalysisResult',
    'VisualizationResult',
    
    # Data classes
    'MemoryEntry',
    'DataFlowNode',
    'CodeMetrics',
    'SecurityIssue',
    'CodeSmell',
    'VisualizationConfig'
]

__version__ = "1.0.0"
__author__ = "AI Research Team"
__email__ = "research@aimulgent.ai"
__description__ = "Specialized AI agents for multi-agent coding systems"