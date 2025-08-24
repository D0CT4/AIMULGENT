"""
AIMULGENT - AI Multiple Agents for Coding
Main entry point for the multi-agent coding system
"""

import asyncio
import logging
import argparse
import sys
import json
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

# Import core components
from core.observer_coordinator import (
    ObserverCoordinator, LoggingObserver, MetricsObserver, 
    Agent, AgentState, Priority, Task
)

# Import specialized agents
from agents.perception_agent import PerceptionAgent
from agents.memory_agent import MemoryAgent
from agents.data_agent import DataAgent
from agents.analysis_agent import AnalysisAgent
from agents.visualization_agent import VisualizationAgent, VisualizationConfig

class AIMultiAgentSystem:
    """Main system orchestrating all AI agents"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_config(config_path)
        
        # Initialize coordinator
        self.coordinator = ObserverCoordinator(
            max_concurrent_tasks=self.config.get('max_concurrent_tasks', 10)
        )
        
        # Initialize agents
        self.agents = {}
        self._initialize_agents()
        
        # Initialize observers
        self._initialize_observers()
        
        # System state
        self.running = False
        self.start_time = None
        
        # Setup logging
        logging.basicConfig(
            level=getattr(logging, self.config.get('log_level', 'INFO')),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load system configuration"""
        default_config = {
            'max_concurrent_tasks': 10,
            'log_level': 'INFO',
            'agents': {
                'perception': {'enabled': True},
                'memory': {'enabled': True, 'db_path': 'memory.db'},
                'data': {'enabled': True},
                'analysis': {'enabled': True},
                'visualization': {'enabled': True}
            },
            'visualization': {
                'theme': 'dark',
                'output_format': 'html',
                'width': 1200,
                'height': 800
            }
        }
        
        if config_path and Path(config_path).exists():
            try:
                with open(config_path, 'r') as f:
                    user_config = json.load(f)
                    default_config.update(user_config)
            except Exception as e:
                print(f"Warning: Could not load config from {config_path}: {e}")
        
        return default_config
    
    def _initialize_agents(self):
        """Initialize all AI agents"""
        
        # Perception Agent
        if self.config['agents']['perception']['enabled']:
            perception_agent = AIMULGENTAgent(
                agent_id='perception_agent',
                agent_type='perception',
                capabilities=['visual_analysis', 'code_structure', 'pattern_recognition'],
                core_agent=PerceptionAgent()
            )
            self.agents['perception'] = perception_agent
        
        # Memory Agent
        if self.config['agents']['memory']['enabled']:
            memory_agent = AIMULGENTAgent(
                agent_id='memory_agent',
                agent_type='memory',
                capabilities=['episodic_memory', 'semantic_memory', 'procedural_memory'],
                core_agent=MemoryAgent(
                    db_path=self.config['agents']['memory'].get('db_path', 'memory.db')
                )
            )
            self.agents['memory'] = memory_agent
        
        # Data Agent
        if self.config['agents']['data']['enabled']:
            data_agent = AIMULGENTAgent(
                agent_id='data_agent',
                agent_type='data',
                capabilities=['data_processing', 'schema_analysis', 'pipeline_execution'],
                core_agent=DataAgent()
            )
            self.agents['data'] = data_agent
        
        # Analysis Agent
        if self.config['agents']['analysis']['enabled']:
            analysis_agent = AIMULGENTAgent(
                agent_id='analysis_agent',
                agent_type='analysis',
                capabilities=['code_analysis', 'security_analysis', 'quality_assessment'],
                core_agent=AnalysisAgent()
            )
            self.agents['analysis'] = analysis_agent
        
        # Visualization Agent
        if self.config['agents']['visualization']['enabled']:
            visualization_agent = AIMULGENTAgent(
                agent_id='visualization_agent',
                agent_type='visualization',
                capabilities=['code_visualization', 'data_visualization', 'report_generation'],
                core_agent=VisualizationAgent()
            )
            self.agents['visualization'] = visualization_agent
        
        self.logger.info(f"Initialized {len(self.agents)} agents")
    
    def _initialize_observers(self):
        """Initialize system observers"""
        
        # Logging observer
        logging_observer = LoggingObserver(
            log_level=getattr(logging, self.config.get('log_level', 'INFO'))
        )
        self.coordinator.add_observer(logging_observer)
        
        # Metrics observer
        self.metrics_observer = MetricsObserver()
        self.coordinator.add_observer(self.metrics_observer)
        
        self.logger.info("Initialized system observers")
    
    async def start(self):
        """Start the multi-agent system"""
        if self.running:
            self.logger.warning("System is already running")
            return
        
        self.logger.info("Starting AI Multiple Agents system...")
        self.start_time = datetime.now()
        
        # Start coordinator
        await self.coordinator.start()
        
        # Register all agents
        for agent in self.agents.values():
            await agent.register_with_coordinator(self.coordinator)
        
        self.running = True
        self.logger.info("AI Multiple Agents system started successfully")
    
    async def stop(self):
        """Stop the multi-agent system"""
        if not self.running:
            return
        
        self.logger.info("Stopping AI Multiple Agents system...")
        
        # Stop coordinator
        await self.coordinator.stop()
        
        self.running = False
        self.logger.info("AI Multiple Agents system stopped")
    
    async def analyze_code(self, 
                          code: str, 
                          file_path: Optional[str] = None,
                          include_visualization: bool = True) -> Dict[str, Any]:
        """Comprehensive code analysis using multiple agents"""
        
        if not self.running:
            raise RuntimeError("System is not running. Call start() first.")
        
        self.logger.info(f"Starting comprehensive code analysis for {file_path or 'code snippet'}")
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'file_path': file_path,
            'analysis_results': {}
        }
        
        task_ids = []
        
        # Submit tasks to different agents
        if 'perception' in self.agents:
            task_id = await self.coordinator.submit_task(
                task_type='visual_analysis',
                input_data={'code': code, 'file_path': file_path},
                priority=Priority.HIGH
            )
            task_ids.append(('perception', task_id))
        
        if 'analysis' in self.agents:
            task_id = await self.coordinator.submit_task(
                task_type='code_analysis',
                input_data={'code': code, 'file_path': file_path},
                priority=Priority.HIGH
            )
            task_ids.append(('analysis', task_id))
        
        if 'memory' in self.agents:
            task_id = await self.coordinator.submit_task(
                task_type='episodic_memory',
                input_data={'content': f'Analyzed code from {file_path or "snippet"}', 'context': {'type': 'code_analysis'}},
                priority=Priority.MEDIUM
            )
            task_ids.append(('memory', task_id))
        
        # Collect results
        for agent_type, task_id in task_ids:
            try:
                result = await self.coordinator.get_task_result(task_id, timeout=30.0)
                results['analysis_results'][agent_type] = result
            except Exception as e:
                self.logger.error(f"Failed to get result from {agent_type} agent: {e}")
                results['analysis_results'][agent_type] = {'error': str(e)}
        
        # Generate visualization if requested
        if include_visualization and 'visualization' in self.agents:
            try:
                viz_task_id = await self.coordinator.submit_task(
                    task_type='code_visualization',
                    input_data={
                        'analysis_results': results['analysis_results'],
                        'config': self.config['visualization']
                    },
                    priority=Priority.MEDIUM
                )
                
                viz_result = await self.coordinator.get_task_result(viz_task_id, timeout=15.0)
                results['visualization'] = viz_result
                
            except Exception as e:
                self.logger.error(f"Failed to generate visualization: {e}")
                results['visualization'] = {'error': str(e)}
        
        return results
    
    async def process_data_pipeline(self, 
                                  pipeline_config: Dict[str, Any]) -> Dict[str, Any]:
        """Execute data processing pipeline"""
        
        if not self.running:
            raise RuntimeError("System is not running. Call start() first.")
        
        if 'data' not in self.agents:
            raise RuntimeError("Data agent not available")
        
        task_id = await self.coordinator.submit_task(
            task_type='pipeline_execution',
            input_data=pipeline_config,
            priority=Priority.HIGH
        )
        
        result = await self.coordinator.get_task_result(task_id, timeout=60.0)
        return result
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        
        status = self.coordinator.get_system_status()
        
        # Add system-specific information
        status.update({
            'system_name': 'AIMULGENT',
            'version': '1.0.0',
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'running': self.running,
            'configuration': self.config
        })
        
        # Add metrics if available
        if hasattr(self, 'metrics_observer'):
            status['metrics'] = self.metrics_observer.get_metrics()
        
        return status
    
    async def emergency_stop(self):
        """Emergency stop all operations"""
        self.logger.warning("Emergency stop initiated")
        await self.coordinator.emergency_stop()
        self.running = False

class AIMULGENTAgent(Agent):
    """Wrapper class for AIMULGENT agents to work with coordinator"""
    
    def __init__(self, agent_id: str, agent_type: str, capabilities: List[str], core_agent: Any):
        super().__init__(agent_id, agent_type, capabilities)
        self.core_agent = core_agent
    
    async def execute_task(self, task: Task) -> Any:
        """Execute task using the core agent"""
        
        try:
            # Update state
            self.state = AgentState.BUSY
            await self.notify_coordinator('agent_state_changed', {'new_state': 'busy'})
            
            # Execute based on task type
            if task.task_type == 'visual_analysis':
                result = self.core_agent.perceive(task.input_data['code'])
            elif task.task_type == 'code_analysis':
                result = self.core_agent.analyze_code(task.input_data['code'])
            elif task.task_type == 'episodic_memory':
                result = self.core_agent.store_episodic_memory(
                    task.input_data['content'],
                    task.input_data.get('context', {})
                )
            elif task.task_type == 'pipeline_execution':
                result = await self.core_agent.process_data_pipeline(task.input_data)
            elif task.task_type == 'code_visualization':
                config = VisualizationConfig(**task.input_data.get('config', {}))
                result = self.core_agent.visualize_code_quality(
                    task.input_data['analysis_results'],
                    config
                )
            else:
                raise ValueError(f"Unknown task type: {task.task_type}")
            
            # Update state
            self.state = AgentState.IDLE
            await self.notify_coordinator('agent_state_changed', {'new_state': 'idle'})
            
            return result
            
        except Exception as e:
            self.state = AgentState.ERROR
            await self.notify_coordinator('agent_state_changed', {'new_state': 'error', 'error': str(e)})
            raise
    
    async def get_status(self) -> Dict[str, Any]:
        """Get agent status"""
        return {
            'agent_id': self.agent_id,
            'agent_type': self.agent_type,
            'state': self.state.value,
            'capabilities': self.capabilities,
            'core_agent_type': type(self.core_agent).__name__
        }

async def main():
    """Main entry point"""
    
    parser = argparse.ArgumentParser(description='AIMULGENT - AI Multiple Agents for Coding')
    parser.add_argument('--config', type=str, help='Configuration file path')
    parser.add_argument('--analyze', type=str, help='File to analyze')
    parser.add_argument('--demo', action='store_true', help='Run demonstration')
    parser.add_argument('--status', action='store_true', help='Show system status')
    
    args = parser.parse_args()
    
    # Initialize system
    system = AIMultiAgentSystem(config_path=args.config)
    
    try:
        # Start system
        await system.start()
        
        if args.analyze:
            # Analyze file
            file_path = Path(args.analyze)
            if not file_path.exists():
                print(f"Error: File {args.analyze} not found")
                return
            
            with open(file_path, 'r') as f:
                code = f.read()
            
            print(f"Analyzing {args.analyze}...")
            results = await system.analyze_code(code, str(file_path))
            
            # Save results
            output_file = file_path.stem + '_analysis.json'
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            print(f"Analysis complete. Results saved to {output_file}")
        
        elif args.demo:
            # Run demonstration
            await run_demonstration(system)
        
        elif args.status:
            # Show system status
            status = system.get_system_status()
            print(json.dumps(status, indent=2, default=str))
        
        else:
            # Interactive mode
            print("AIMULGENT - AI Multiple Agents for Coding")
            print("System started. Type 'help' for commands, 'quit' to exit.")
            
            while True:
                try:
                    command = input("\naimulgent> ").strip().lower()
                    
                    if command == 'quit' or command == 'exit':
                        break
                    elif command == 'help':
                        print("Commands:")
                        print("  status  - Show system status")
                        print("  agents  - List all agents")
                        print("  analyze <file> - Analyze a code file")
                        print("  demo    - Run demonstration")
                        print("  quit    - Exit system")
                    elif command == 'status':
                        status = system.get_system_status()
                        print(json.dumps(status, indent=2, default=str))
                    elif command == 'agents':
                        print("Registered agents:")
                        for agent_id, agent in system.agents.items():
                            status = await agent.get_status()
                            print(f"  {agent_id}: {status}")
                    elif command.startswith('analyze '):
                        file_path = command[8:].strip()
                        if Path(file_path).exists():
                            with open(file_path, 'r') as f:
                                code = f.read()
                            print(f"Analyzing {file_path}...")
                            results = await system.analyze_code(code, file_path)
                            print(f"Analysis complete. Quality score: {results.get('quality_score', 'N/A')}")
                        else:
                            print(f"File not found: {file_path}")
                    elif command == 'demo':
                        await run_demonstration(system)
                    else:
                        print("Unknown command. Type 'help' for available commands.")
                
                except KeyboardInterrupt:
                    break
                except Exception as e:
                    print(f"Error: {e}")
    
    finally:
        # Stop system
        await system.stop()
        print("System stopped.")

async def run_demonstration(system: AIMultiAgentSystem):
    """Run system demonstration"""
    
    print("\n🤖 AIMULGENT Demonstration")
    print("=" * 50)
    
    # Sample code for analysis
    sample_code = '''
import os
import sys
from typing import List, Dict

class Calculator:
    def __init__(self):
        self.history = []
        self.secret_key = "hardcoded_password123"  # Security issue
    
    def add(self, a: float, b: float) -> float:
        """Add two numbers"""
        result = a + b
        self.history.append(f"{a} + {b} = {result}")
        return result
    
    def divide(self, a: float, b: float) -> float:
        # No error handling - potential issue
        return a / b
    
    def complex_calculation(self, data: List[Dict]):
        results = []
        for item in data:
            if item['type'] == 'A':
                if item['value'] > 100:
                    if item['category'] == 'premium':
                        if item['status'] == 'active':
                            # Deep nesting - code smell
                            processed = self.process_item(item)
                            results.append(processed)
        return results
    
    def process_item(self, item):
        return item['value'] * 42  # Magic number
    '''
    
    print("\n1. Analyzing sample code...")
    results = await system.analyze_code(sample_code, "sample_calculator.py")
    
    print(f"✅ Analysis completed")
    print(f"   - Agents involved: {list(results['analysis_results'].keys())}")
    
    if 'analysis' in results['analysis_results']:
        analysis_result = results['analysis_results']['analysis']
        if hasattr(analysis_result, 'quality_score'):
            print(f"   - Quality score: {analysis_result.quality_score:.2f}")
            print(f"   - Security issues: {len(analysis_result.security_issues)}")
            print(f"   - Code smells: {len(analysis_result.code_smells)}")
    
    print("\n2. System status:")
    status = system.get_system_status()
    print(f"   - Active agents: {status['active_agents']}")
    print(f"   - Tasks completed: {status['coordination_stats']['tasks_completed']}")
    print(f"   - Events processed: {status['coordination_stats']['events_processed']}")
    
    if 'metrics' in status:
        metrics = status['metrics']
        print(f"   - Total events: {metrics['total_events']}")
    
    print("\n3. Memory system demonstration...")
    if 'memory' in system.agents:
        memory_task = await system.coordinator.submit_task(
            task_type='episodic_memory',
            input_data={
                'content': 'Demonstrated AIMULGENT capabilities with sample calculator code',
                'context': {'demo_session': True, 'timestamp': datetime.now().isoformat()}
            }
        )
        
        await system.coordinator.get_task_result(memory_task)
        print("   ✅ Experience stored in memory system")
    
    print("\n🎉 Demonstration completed!")
    print("The system successfully coordinated multiple specialized agents to:")
    print("   • Analyze code structure and patterns (Perception Agent)")
    print("   • Perform deep code analysis and quality assessment (Analysis Agent)")  
    print("   • Store the analysis experience (Memory Agent)")
    print("   • Generate visual representations (Visualization Agent)")
    print("   • Process and manage data flows (Data Agent)")
    print("   • Coordinate all operations seamlessly (Observer/Coordinator)")

if __name__ == "__main__":
    asyncio.run(main())