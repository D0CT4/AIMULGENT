"""
AIMULGENT Base Agent
Abstract base class for all agents following single responsibility principle.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """
    Abstract base class for all AIMULGENT agents.
    
    Each agent should:
    - Have a single responsibility
    - Be under 500 lines of code  
    - Follow KISS principles
    - Implement proper error handling
    """
    
    def __init__(self, agent_id: str, capabilities: List[str]):
        self.agent_id = agent_id
        self.capabilities = capabilities
        self.tasks_processed = 0
        self.logger = logging.getLogger(f"aimulgent.agents.{agent_id}")
    
    @abstractmethod
    async def process_task(
        self, 
        task_type: str, 
        input_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Process a task of the specified type.
        
        Args:
            task_type: Type of task to process
            input_data: Input data for the task
            
        Returns:
            Task results
            
        Raises:
            ValueError: If task type is not supported
            Exception: If task processing fails
        """
        pass
    
    def can_handle(self, task_type: str) -> bool:
        """Check if agent can handle the given task type."""
        return task_type in self.capabilities
    
    def get_info(self) -> Dict[str, Any]:
        """Get agent information."""
        return {
            "agent_id": self.agent_id,
            "capabilities": self.capabilities,
            "tasks_processed": self.tasks_processed,
            "status": "active"
        }