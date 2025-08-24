"""
AIMULGENT Configuration Management
Manages system settings with Pydantic validation.
"""

from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional

from pydantic import Field
from pydantic_settings import BaseSettings


class AgentConfig(BaseSettings):
    """Configuration for individual agents."""
    
    enabled: bool = True
    capabilities: List[str] = Field(default_factory=list)
    max_concurrent_tasks: int = 3
    timeout_seconds: int = 300


class CoordinatorConfig(BaseSettings):
    """Configuration for the coordination system."""
    
    max_concurrent_tasks: int = 10
    heartbeat_interval: float = 30.0
    heartbeat_timeout: float = 90.0
    event_queue_size: int = 1000


class Settings(BaseSettings):
    """Main application settings."""
    
    # System settings
    app_name: str = "AIMULGENT"
    version: str = "1.0.0"
    debug: bool = False
    
    # Data directory
    data_dir: Path = Field(default=Path("data"))
    
    # Coordinator settings
    coordinator: CoordinatorConfig = Field(default_factory=CoordinatorConfig)
    
    # Agent configurations
    agents: Dict[str, AgentConfig] = Field(default_factory=lambda: {
        "perception": AgentConfig(
            capabilities=["visual_analysis", "code_structure", "pattern_recognition"]
        ),
        "memory": AgentConfig(
            capabilities=["episodic_memory", "semantic_memory", "retrieval"]
        ),
        "data": AgentConfig(
            capabilities=["data_processing", "schema_analysis", "pipeline_execution"]
        ),
        "analysis": AgentConfig(
            capabilities=["code_analysis", "security_analysis", "quality_assessment"]
        ),
        "visualization": AgentConfig(
            capabilities=["code_visualization", "data_visualization", "report_generation"]
        )
    })
    
    # Database settings
    database_url: str = "sqlite:///./aimulgent.db"
    
    # Logging
    log_level: str = "INFO"
    log_file: Optional[Path] = None
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()