"""
AIMULGENT Main System
Core system orchestrating all agents following KISS principles.
"""

import asyncio
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from aimulgent.core.config import Settings, get_settings
from aimulgent.core.coordinator import Coordinator
from aimulgent.agents.analysis import AnalysisAgent
from aimulgent.agents.base import BaseAgent

logger = logging.getLogger(__name__)


class AIMULGENTSystem:
    """
    Main AIMULGENT system managing all agents and coordination.
    
    Follows KISS principle with simplified architecture focused on core functionality.
    """
    
    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()
        self.coordinator = Coordinator(
            max_concurrent_tasks=self.settings.coordinator.max_concurrent_tasks
        )
        self.agents: Dict[str, BaseAgent] = {}
        self.running = False
        
        # Setup logging
        logging.basicConfig(
            level=getattr(logging, self.settings.log_level),
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        
        # Initialize data directory
        self.settings.data_dir.mkdir(exist_ok=True)
    
    async def start(self) -> None:
        """Start the AIMULGENT system."""
        if self.running:
            logger.warning("System already running")
            return
        
        logger.info("Starting AIMULGENT system")
        
        # Start coordinator
        await self.coordinator.start()
        
        # Initialize agents
        await self._initialize_agents()
        
        self.running = True
        logger.info("AIMULGENT system started successfully")
    
    async def stop(self) -> None:
        """Stop the AIMULGENT system."""
        if not self.running:
            return
        
        logger.info("Stopping AIMULGENT system")
        
        # Stop coordinator
        await self.coordinator.stop()
        
        self.running = False
        logger.info("AIMULGENT system stopped")
    
    async def analyze_code(
        self, 
        code: str, 
        file_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Analyze code using the analysis agent.
        
        Args:
            code: Source code to analyze
            file_path: Optional path to the source file
            
        Returns:
            Analysis results including quality metrics and recommendations
        """
        if not self.running:
            raise RuntimeError("System not running. Call start() first.")
        
        # Submit analysis task
        task_id = await self.coordinator.submit_task(
            task_type="code_analysis",
            input_data={
                "code": code,
                "file_path": file_path
            }
        )
        
        # Get result
        result = await self.coordinator.get_task_result(task_id)
        
        return {
            "file_path": file_path,
            "analysis": result,
            "timestamp": "now",  # Would use actual timestamp
            "system_info": {
                "version": self.settings.version,
                "agents_active": len(self.agents)
            }
        }
    
    async def process_data(
        self, 
        data: Dict[str, Any], 
        processing_type: str = "general"
    ) -> Dict[str, Any]:
        """
        Process data using the data agent.
        
        Args:
            data: Data to process
            processing_type: Type of processing to perform
            
        Returns:
            Processing results
        """
        if not self.running:
            raise RuntimeError("System not running. Call start() first.")
        
        task_id = await self.coordinator.submit_task(
            task_type="data_processing",
            input_data={
                "data": data,
                "processing_type": processing_type
            }
        )
        
        return await self.coordinator.get_task_result(task_id)
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        coordinator_status = self.coordinator.get_status()
        
        return {
            "system": {
                "name": self.settings.app_name,
                "version": self.settings.version,
                "running": self.running,
                "debug": self.settings.debug
            },
            "coordinator": coordinator_status,
            "agents": {
                agent_id: {
                    "type": type(agent).__name__,
                    "capabilities": getattr(agent, "capabilities", [])
                }
                for agent_id, agent in self.agents.items()
            }
        }
    
    async def _initialize_agents(self) -> None:
        """Initialize and register all agents."""
        
        # Analysis agent (main agent for now)
        if self.settings.agents["analysis"].enabled:
            analysis_agent = AnalysisAgent()
            self.agents["analysis"] = analysis_agent
            await analysis_agent.start()
            
            await self.coordinator.register_agent(
                agent_id="analysis",
                capabilities=self.settings.agents["analysis"].capabilities,
                agent_instance=analysis_agent
            )
            logger.info("Analysis agent initialized")
        
        # Additional agents would be initialized here following the same pattern
        # Each agent should be under 500 lines and have a single responsibility
        
        logger.info(f"Initialized {len(self.agents)} agents")


async def create_system(settings: Optional[Settings] = None) -> AIMULGENTSystem:
    """
    Factory function to create and initialize AIMULGENT system.
    
    Args:
        settings: Optional custom settings
        
    Returns:
        Initialized AIMULGENT system
    """
    system = AIMULGENTSystem(settings)
    await system.start()
    return system