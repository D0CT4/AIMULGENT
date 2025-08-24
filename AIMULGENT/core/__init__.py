"""
AIMULGENT Core Package
Core coordination and orchestration components for multi-agent system
"""

from .observer_coordinator import (
    ObserverCoordinator,
    Observer,
    Agent,
    LoggingObserver,
    MetricsObserver,
    CoordinationEvent,
    Task,
    AgentInfo,
    AgentState,
    EventType,
    Priority
)

__all__ = [
    # Main coordinator
    'ObserverCoordinator',
    
    # Base classes
    'Observer',
    'Agent',
    
    # Observer implementations
    'LoggingObserver',
    'MetricsObserver',
    
    # Data classes
    'CoordinationEvent',
    'Task',
    'AgentInfo',
    
    # Enums
    'AgentState',
    'EventType',
    'Priority'
]

__version__ = "1.0.0"
__author__ = "AI Research Team"
__email__ = "research@aimulgent.ai"
__description__ = "Core coordination system for AI multi-agent orchestration"