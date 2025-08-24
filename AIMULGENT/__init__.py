"""
AIMULGENT - AI Multiple Agents for Coding
State-of-the-art multi-agent system for comprehensive code analysis and development assistance

This package provides specialized AI agents that work together to analyze code,
understand software architecture, process data, and generate insightful visualizations.

Key Components:
- Perception Agent: Visual and structural code analysis using Vision Transformers and CNNs
- Memory Agent: Episodic and semantic memory with vector databases for context management  
- Data Agent: Advanced data processing and pipeline execution with Graph Neural Networks
- Analysis Agent: Deep code analysis using specialized transformers (CodeT5/CodeBERT)
- Visualization Agent: Interactive visualizations using generative models and rendering engines
- Observer/Coordinator: Event-driven multi-agent orchestration with observer pattern

Features:
- Multi-modal code understanding (text, visual, structural)
- Real-time agent coordination and task distribution
- Advanced neural architectures specialized for code
- Interactive visualizations and comprehensive reporting
- Scalable architecture supporting enterprise-grade deployments
- Integration with 2025 state-of-the-art frameworks (CrewAI, AutoGen, LangGraph)

Usage:
    from aimulgent import AIMultiAgentSystem
    
    # Initialize system
    system = AIMultiAgentSystem()
    
    # Start agents
    await system.start()
    
    # Analyze code
    results = await system.analyze_code(code_content, file_path)
    
    # Stop system
    await system.stop()
"""

# Import main system class
from .main import AIMultiAgentSystem

# Import core components
from .core import (
    ObserverCoordinator,
    Observer, 
    Agent,
    CoordinationEvent,
    Task,
    AgentState,
    EventType,
    Priority
)

# Import specialized agents
from .agents import (
    PerceptionAgent,
    MemoryAgent,
    DataAgent,
    AnalysisAgent,
    VisualizationAgent,
    PerceptionResult,
    MemoryEntry,
    MemorySearchResult,
    DataProcessingResult,
    AnalysisResult,
    CodeMetrics,
    SecurityIssue,
    CodeSmell,
    VisualizationResult,
    VisualizationConfig
)

# Package metadata
__version__ = "1.0.0"
__author__ = "AI Research Team"
__email__ = "research@aimulgent.ai"
__description__ = "AI Multiple Agents for Coding - Advanced multi-agent system for comprehensive code analysis"
__url__ = "https://github.com/aimulgent/aimulgent"
__license__ = "MIT"

# Export all public components
__all__ = [
    # Main system
    'AIMultiAgentSystem',
    
    # Core coordination
    'ObserverCoordinator',
    'Observer',
    'Agent',
    'CoordinationEvent',
    'Task',
    'AgentState',
    'EventType',
    'Priority',
    
    # Specialized agents
    'PerceptionAgent',
    'MemoryAgent', 
    'DataAgent',
    'AnalysisAgent',
    'VisualizationAgent',
    
    # Result classes
    'PerceptionResult',
    'MemoryEntry',
    'MemorySearchResult',
    'DataProcessingResult',
    'AnalysisResult',
    'CodeMetrics',
    'SecurityIssue',
    'CodeSmell',
    'VisualizationResult',
    'VisualizationConfig',
    
    # Package info
    '__version__',
    '__author__',
    '__email__',
    '__description__',
    '__url__',
    '__license__'
]

# System information
SYSTEM_INFO = {
    'name': 'AIMULGENT',
    'version': __version__,
    'description': __description__,
    'agents': {
        'perception': 'Visual and structural code analysis',
        'memory': 'Episodic and semantic memory management',
        'data': 'Advanced data processing and pipeline execution',
        'analysis': 'Deep code analysis and quality assessment',
        'visualization': 'Interactive visualizations and reporting'
    },
    'features': [
        'Multi-modal code understanding',
        'Real-time agent coordination', 
        'Specialized neural networks for code',
        'Interactive visualization generation',
        'Enterprise-scale architecture',
        '2025 state-of-the-art integration'
    ],
    'supported_languages': [
        'Python',
        'JavaScript', 
        'TypeScript',
        'Java',
        'C++',
        'C#',
        'Go',
        'Rust',
        'Ruby',
        'PHP'
    ],
    'neural_architectures': [
        'Vision Transformer (ViT)',
        'Graph Neural Networks (GNN)',
        'Code-specialized Transformers (CodeT5, CodeBERT)',
        'Generative Adversarial Networks (GAN)',
        'Episodic Memory Networks',
        'Active Predictive Coding'
    ],
    'integration_frameworks': [
        'CrewAI (role-based teams)',
        'AutoGen (conversational orchestration)', 
        'LangGraph (graph-based coordination)',
        'Neuro-symbolic AI integration',
        'Agent Communication Protocols (MCP, ACP, A2A, ANP)'
    ]
}

def get_system_info():
    """Get comprehensive system information"""
    return SYSTEM_INFO.copy()

def print_banner():
    """Print AIMULGENT banner with system information"""
    banner = f"""
    ╔══════════════════════════════════════════════════════════════╗
    ║                          AIMULGENT                           ║
    ║                AI Multiple Agents for Coding                 ║
    ║                                                              ║
    ║  🤖 Version: {__version__:<48} ║
    ║  🧠 State-of-the-art Multi-Agent System                     ║
    ║  🔍 Comprehensive Code Analysis & Understanding              ║
    ║  📊 Advanced Visualization & Reporting                       ║
    ║  🚀 Enterprise-Grade Performance & Scalability              ║
    ║                                                              ║
    ║  Specialized Agents:                                         ║
    ║  • Perception   → Visual & structural analysis              ║
    ║  • Memory       → Context & knowledge management            ║
    ║  • Data         → Processing & pipeline execution           ║
    ║  • Analysis     → Deep code understanding & quality         ║
    ║  • Visualization → Interactive charts & reports             ║
    ║                                                              ║
    ║  Built with 2025 cutting-edge AI frameworks                 ║
    ║  CrewAI • AutoGen • LangGraph • Neuro-symbolic AI           ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)

# Initialize logging configuration
import logging
import sys

def setup_logging(level=logging.INFO, format_string=None):
    """Setup package-wide logging configuration"""
    
    if format_string is None:
        format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    logging.basicConfig(
        level=level,
        format=format_string,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('aimulgent.log', mode='a', encoding='utf-8')
        ]
    )
    
    # Set specific loggers
    logging.getLogger('aimulgent').setLevel(level)
    logging.getLogger('transformers').setLevel(logging.WARNING)
    logging.getLogger('torch').setLevel(logging.WARNING)

# Package initialization
def initialize(config_path=None, log_level='INFO'):
    """Initialize AIMULGENT package with configuration"""
    
    # Setup logging
    log_level_obj = getattr(logging, log_level.upper())
    setup_logging(level=log_level_obj)
    
    logger = logging.getLogger(__name__)
    logger.info(f"AIMULGENT v{__version__} initialized")
    
    return True

# Convenience function for quick setup
async def quick_analyze(code: str, file_path: str = None, config: dict = None):
    """Quick code analysis using default configuration"""
    
    system = AIMultiAgentSystem(config)
    
    try:
        await system.start()
        result = await system.analyze_code(code, file_path)
        return result
    finally:
        await system.stop()

# Version check function
def check_dependencies():
    """Check if all required dependencies are available"""
    
    required_packages = [
        'torch', 'transformers', 'numpy', 'pandas', 'matplotlib',
        'plotly', 'networkx', 'faiss', 'sqlite3', 'asyncio'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"Warning: Missing required packages: {', '.join(missing_packages)}")
        print("Install with: pip install -r requirements.txt")
        return False
    
    return True

# Auto-check on import (can be disabled with environment variable)
import os
if not os.environ.get('AIMULGENT_SKIP_DEPENDENCY_CHECK'):
    check_dependencies()