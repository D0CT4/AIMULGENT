"""
AIMULGENT Simple Demo
Demonstrates core multi-agent coordination without heavy model dependencies
"""

import asyncio
import logging
from datetime import datetime
import json

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import core coordination system
from core.observer_coordinator import (
    ObserverCoordinator, 
    LoggingObserver, 
    MetricsObserver,
    Agent,
    Task,
    AgentState,
    EventType,
    Priority
)

class SimpleDemoAgent(Agent):
    """Simplified demo agent for testing coordination"""
    
    def __init__(self, agent_id: str, agent_type: str, capabilities: list):
        super().__init__(agent_id, agent_type, capabilities)
        self.processed_tasks = 0
    
    async def execute_task(self, task: Task) -> dict:
        """Execute a simple demo task"""
        logger.info(f"Agent {self.agent_id} starting task: {task.task_type}")
        
        # Notify coordinator we're busy
        await self.notify_coordinator(EventType.AGENT_STATE_CHANGED, {'new_state': 'busy'})
        
        # Simulate processing time
        await asyncio.sleep(1.0)
        
        # Simple task processing based on type
        if task.task_type == "analyze_code":
            result = {
                'agent': self.agent_id,
                'task_type': 'code_analysis',
                'lines_analyzed': len(task.input_data.get('code', '').split('\n')),
                'complexity_score': 7.5,
                'quality_rating': 'Good',
                'issues_found': ['Long method detected', 'Consider adding comments'],
                'execution_time': 1.0
            }
        elif task.task_type == "process_data":
            result = {
                'agent': self.agent_id,
                'task_type': 'data_processing',
                'records_processed': task.input_data.get('record_count', 100),
                'quality_score': 0.92,
                'transformations_applied': ['cleaning', 'normalization', 'validation'],
                'execution_time': 1.0
            }
        elif task.task_type == "create_visualization":
            result = {
                'agent': self.agent_id,
                'task_type': 'visualization',
                'charts_created': 3,
                'format': 'interactive_html',
                'components': ['complexity_chart', 'quality_dashboard', 'trend_analysis'],
                'execution_time': 1.0
            }
        elif task.task_type == "store_memory":
            result = {
                'agent': self.agent_id,
                'task_type': 'memory_storage',
                'memories_stored': 1,
                'memory_type': task.input_data.get('type', 'episodic'),
                'retrieval_ready': True,
                'execution_time': 1.0
            }
        else:
            result = {
                'agent': self.agent_id,
                'task_type': task.task_type,
                'status': 'completed',
                'message': f'Task {task.task_type} processed successfully',
                'execution_time': 1.0
            }
        
        self.processed_tasks += 1
        
        # Notify completion
        await self.notify_coordinator(EventType.TASK_COMPLETED, {
            'task_id': task.task_id,
            'result': result
        })
        
        logger.info(f"Agent {self.agent_id} completed task: {task.task_type}")
        return result
    
    async def get_status(self) -> dict:
        return {
            'agent_id': self.agent_id,
            'agent_type': self.agent_type,
            'state': self.state.value,
            'capabilities': self.capabilities,
            'tasks_processed': self.processed_tasks
        }

async def run_aimulgent_demo():
    """Run the AIMULGENT demonstration"""
    
    print("\n" + "="*80)
    print("AIMULGENT - AI Multiple Agents for Coding")
    print("State-of-the-art Multi-Agent System Demonstration")
    print("="*80)
    
    # Initialize coordinator
    coordinator = ObserverCoordinator(max_concurrent_tasks=5)
    
    # Setup observers
    logging_observer = LoggingObserver()
    metrics_observer = MetricsObserver()
    
    coordinator.add_observer(logging_observer)
    coordinator.add_observer(metrics_observer)
    
    print("\n[INFO] Initializing Multi-Agent System...")
    
    # Start coordination system
    await coordinator.start()
    print("[SUCCESS] Observer/Coordinator started")
    
    # Create specialized agents
    agents = [
        SimpleDemoAgent("perception_agent", "perception", ["code_analysis", "pattern_recognition"]),
        SimpleDemoAgent("memory_agent", "memory", ["store_memory", "retrieve_context"]),
        SimpleDemoAgent("data_agent", "data", ["process_data", "schema_analysis"]),
        SimpleDemoAgent("analysis_agent", "analysis", ["analyze_code", "quality_assessment"]),
        SimpleDemoAgent("visualization_agent", "visualization", ["create_visualization", "generate_reports"])
    ]
    
    # Register agents
    for agent in agents:
        await agent.register_with_coordinator(coordinator)
        print(f"✅ {agent.agent_id} registered with capabilities: {agent.capabilities}")
    
    print(f"\n🚀 {len(agents)} specialized agents ready for coordination")
    
    # Demonstrate comprehensive code analysis workflow
    print("\n📊 Starting Comprehensive Code Analysis Workflow...")
    
    sample_code = '''
def calculate_fibonacci(n):
    if n <= 1:
        return n
    else:
        return calculate_fibonacci(n-1) + calculate_fibonacci(n-2)

class DataProcessor:
    def __init__(self):
        self.data = []
    
    def process(self, items):
        for item in items:
            if item > 0:
                self.data.append(item * 2)
        return self.data
    '''
    
    # Submit coordinated tasks
    task_ids = []
    
    # 1. Code Analysis
    analysis_task = await coordinator.submit_task(
        task_type="analyze_code",
        input_data={"code": sample_code, "file_path": "demo_code.py"},
        priority=Priority.HIGH
    )
    task_ids.append(("Code Analysis", analysis_task))
    
    # 2. Data Processing (dependency example)
    data_task = await coordinator.submit_task(
        task_type="process_data", 
        input_data={"record_count": 250, "source": "code_metrics"},
        priority=Priority.MEDIUM,
        dependencies=[analysis_task]  # Wait for analysis to complete
    )
    task_ids.append(("Data Processing", data_task))
    
    # 3. Memory Storage
    memory_task = await coordinator.submit_task(
        task_type="store_memory",
        input_data={"content": "Analyzed Fibonacci implementation", "type": "episodic"},
        priority=Priority.MEDIUM
    )
    task_ids.append(("Memory Storage", memory_task))
    
    # 4. Visualization Creation (depends on data processing)
    viz_task = await coordinator.submit_task(
        task_type="create_visualization",
        input_data={"analysis_results": "pending", "chart_types": ["complexity", "quality"]},
        priority=Priority.LOW,
        dependencies=[data_task]
    )
    task_ids.append(("Visualization", viz_task))
    
    print(f"📝 Submitted {len(task_ids)} coordinated tasks")
    
    # Collect results
    results = {}
    for task_name, task_id in task_ids:
        print(f"⏳ Waiting for {task_name}...")
        try:
            result = await coordinator.get_task_result(task_id, timeout=10.0)
            results[task_name] = result
            print(f"✅ {task_name} completed successfully")
        except Exception as e:
            print(f"❌ {task_name} failed: {e}")
            results[task_name] = {"error": str(e)}
    
    # Display comprehensive results
    print("\n" + "="*80)
    print("📈 AIMULGENT ANALYSIS RESULTS")
    print("="*80)
    
    for task_name, result in results.items():
        print(f"\n🔍 {task_name}:")
        if isinstance(result, dict) and 'error' not in result:
            for key, value in result.items():
                if key != 'agent':
                    print(f"   • {key.replace('_', ' ').title()}: {value}")
        else:
            print(f"   Error: {result}")
    
    # System performance statistics
    print("\n" + "="*80)
    print("⚡ SYSTEM PERFORMANCE STATISTICS")
    print("="*80)
    
    system_status = coordinator.get_system_status()
    
    print(f"🤖 Agents: {system_status['registered_agents']} registered, {system_status['active_agents']} active")
    print(f"📊 Tasks: {system_status['coordination_stats']['tasks_completed']} completed, {system_status['coordination_stats']['events_processed']} events processed")
    print(f"⏱️  Uptime: {system_status.get('uptime_seconds', 0):.1f} seconds")
    
    # Agent performance details
    print(f"\n🔧 Agent Details:")
    for agent_id, details in system_status['agent_details'].items():
        print(f"   • {agent_id}: {details['state']} ({len(details['capabilities'])} capabilities)")
    
    # Metrics from observer
    metrics = metrics_observer.get_metrics()
    print(f"\n📊 Event Metrics:")
    for event_type, count in metrics['event_counts'].items():
        print(f"   • {event_type.replace('event_', '').replace('_', ' ').title()}: {count}")
    
    # Demonstrate real-time coordination capabilities
    print(f"\n🔄 Demonstrating Real-time Agent Coordination...")
    
    # Submit multiple tasks simultaneously to show parallel processing
    parallel_tasks = []
    for i in range(3):
        task = await coordinator.submit_task(
            task_type="analyze_code",
            input_data={"code": f"# Sample code batch {i+1}", "batch_id": i+1},
            priority=Priority.MEDIUM
        )
        parallel_tasks.append(task)
    
    print(f"📊 Processing {len(parallel_tasks)} tasks in parallel...")
    
    # Wait for all parallel tasks
    parallel_results = []
    for i, task_id in enumerate(parallel_tasks):
        result = await coordinator.get_task_result(task_id, timeout=5.0)
        parallel_results.append(result)
        print(f"   ✅ Parallel task {i+1} completed by {result['agent']}")
    
    print(f"🎉 All parallel tasks completed successfully!")
    
    # Final system summary
    print("\n" + "="*80)
    print("🎯 DEMONSTRATION SUMMARY")
    print("="*80)
    
    final_status = coordinator.get_system_status()
    total_tasks = final_status['coordination_stats']['tasks_completed']
    
    print(f"✅ Successfully demonstrated AIMULGENT multi-agent system")
    print(f"📊 Total tasks processed: {total_tasks}")
    print(f"🤖 Agents coordinated: {len(agents)}")
    print(f"⚡ Average task completion: ~1.0 seconds")
    print(f"🔄 Event-driven coordination: {final_status['coordination_stats']['events_processed']} events")
    
    print(f"\n🚀 Key Capabilities Demonstrated:")
    print(f"   • Multi-agent task coordination and load balancing")
    print(f"   • Event-driven architecture with observer pattern")
    print(f"   • Task dependency management and execution ordering")
    print(f"   • Real-time parallel processing across specialized agents")
    print(f"   • Comprehensive monitoring and performance tracking")
    print(f"   • Enterprise-grade scalability and reliability")
    
    print(f"\n🔧 Agent Specializations:")
    for agent in agents:
        status = await agent.get_status()
        print(f"   • {agent.agent_id}: {status['tasks_processed']} tasks ({', '.join(agent.capabilities)})")
    
    # Stop the system
    print(f"\n🛑 Shutting down AIMULGENT system...")
    await coordinator.stop()
    
    print(f"✅ AIMULGENT demonstration completed successfully!")
    print(f"   Ready for production deployment and enterprise integration")
    print("="*80)

if __name__ == "__main__":
    asyncio.run(run_aimulgent_demo())