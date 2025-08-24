"""
Observer/Coordinator - Multi-Agent Orchestration and Coordination
Implements observer pattern with event-driven coordination for AI agents
"""

import asyncio
import threading
import time
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Callable, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import json
import uuid
import logging
from datetime import datetime, timedelta
import concurrent.futures
import weakref

class AgentState(Enum):
    """Agent execution states"""
    IDLE = "idle"
    BUSY = "busy" 
    WAITING = "waiting"
    ERROR = "error"
    COMPLETED = "completed"

class EventType(Enum):
    """Types of coordination events"""
    TASK_ASSIGNED = "task_assigned"
    TASK_COMPLETED = "task_completed"
    TASK_FAILED = "task_failed"
    AGENT_STATE_CHANGED = "agent_state_changed"
    RESOURCE_REQUESTED = "resource_requested"
    RESOURCE_RELEASED = "resource_released"
    COLLABORATION_REQUEST = "collaboration_request"
    DATA_SHARED = "data_shared"
    EMERGENCY_STOP = "emergency_stop"

class Priority(Enum):
    """Task priority levels"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

@dataclass
class CoordinationEvent:
    """Event for inter-agent coordination"""
    event_type: EventType
    source_agent: str
    target_agent: Optional[str] = None
    data: Dict[str, Any] = field(default_factory=dict)
    priority: Priority = Priority.MEDIUM
    timestamp: datetime = field(default_factory=datetime.now)
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    correlation_id: Optional[str] = None

@dataclass
class Task:
    """Task definition for agent execution"""
    task_id: str
    task_type: str
    agent_id: str
    input_data: Dict[str, Any]
    priority: Priority = Priority.MEDIUM
    dependencies: List[str] = field(default_factory=list)
    timeout: Optional[float] = None
    retry_count: int = 0
    max_retries: int = 3
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    status: str = "pending"
    result: Optional[Any] = None
    error: Optional[str] = None

@dataclass
class AgentInfo:
    """Information about registered agents"""
    agent_id: str
    agent_type: str
    capabilities: List[str]
    state: AgentState = AgentState.IDLE
    current_task: Optional[str] = None
    load_factor: float = 0.0
    last_heartbeat: datetime = field(default_factory=datetime.now)
    performance_stats: Dict[str, Any] = field(default_factory=dict)
    resource_usage: Dict[str, float] = field(default_factory=dict)

class Observer(ABC):
    """Abstract observer for coordination events"""
    
    @abstractmethod
    async def on_event(self, event: CoordinationEvent) -> None:
        """Handle coordination event"""
        pass

class Agent(ABC):
    """Abstract base class for coordinated agents"""
    
    def __init__(self, agent_id: str, agent_type: str, capabilities: List[str]):
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.capabilities = capabilities
        self.state = AgentState.IDLE
        self.coordinator: Optional['ObserverCoordinator'] = None
        
    @abstractmethod
    async def execute_task(self, task: Task) -> Any:
        """Execute assigned task"""
        pass
    
    @abstractmethod
    async def get_status(self) -> Dict[str, Any]:
        """Get current agent status"""
        pass
    
    async def register_with_coordinator(self, coordinator: 'ObserverCoordinator'):
        """Register with coordination system"""
        self.coordinator = coordinator
        await coordinator.register_agent(self)
    
    async def notify_coordinator(self, event_type: EventType, data: Dict[str, Any] = None):
        """Notify coordinator of events"""
        if self.coordinator:
            event = CoordinationEvent(
                event_type=event_type,
                source_agent=self.agent_id,
                data=data or {}
            )
            await self.coordinator.handle_event(event)

class ResourceManager:
    """Manages shared resources between agents"""
    
    def __init__(self):
        self.resources: Dict[str, Any] = {}
        self.resource_locks: Dict[str, asyncio.Lock] = {}
        self.resource_usage: Dict[str, List[str]] = defaultdict(list)  # resource -> agent_ids
        self.resource_limits: Dict[str, int] = {}
        
    def register_resource(self, resource_id: str, resource: Any, max_concurrent: int = 1):
        """Register a shared resource"""
        self.resources[resource_id] = resource
        self.resource_locks[resource_id] = asyncio.Lock()
        self.resource_limits[resource_id] = max_concurrent
    
    async def request_resource(self, resource_id: str, agent_id: str) -> bool:
        """Request access to a resource"""
        if resource_id not in self.resources:
            return False
            
        # Check if resource limit exceeded
        if len(self.resource_usage[resource_id]) >= self.resource_limits[resource_id]:
            return False
        
        async with self.resource_locks[resource_id]:
            if len(self.resource_usage[resource_id]) < self.resource_limits[resource_id]:
                self.resource_usage[resource_id].append(agent_id)
                return True
        
        return False
    
    async def release_resource(self, resource_id: str, agent_id: str):
        """Release a resource"""
        if resource_id in self.resource_usage:
            async with self.resource_locks[resource_id]:
                if agent_id in self.resource_usage[resource_id]:
                    self.resource_usage[resource_id].remove(agent_id)
    
    def get_resource(self, resource_id: str) -> Any:
        """Get resource object"""
        return self.resources.get(resource_id)
    
    def get_resource_stats(self) -> Dict[str, Any]:
        """Get resource utilization statistics"""
        return {
            resource_id: {
                'current_users': len(users),
                'max_concurrent': self.resource_limits.get(resource_id, 1),
                'utilization': len(users) / self.resource_limits.get(resource_id, 1)
            }
            for resource_id, users in self.resource_usage.items()
        }

class TaskScheduler:
    """Schedules and prioritizes tasks for agents"""
    
    def __init__(self):
        self.task_queues: Dict[Priority, deque] = {
            priority: deque() for priority in Priority
        }
        self.active_tasks: Dict[str, Task] = {}
        self.completed_tasks: Dict[str, Task] = {}
        self.task_dependencies: Dict[str, List[str]] = {}
        
    def add_task(self, task: Task):
        """Add task to appropriate priority queue"""
        self.task_queues[task.priority].append(task)
        
        # Track dependencies
        if task.dependencies:
            self.task_dependencies[task.task_id] = task.dependencies
    
    def get_next_task(self, agent_capabilities: List[str]) -> Optional[Task]:
        """Get next available task for agent with given capabilities"""
        # Check queues in priority order (highest first)
        for priority in sorted(Priority, key=lambda p: p.value, reverse=True):
            queue = self.task_queues[priority]
            
            for i, task in enumerate(queue):
                # Check if task type matches agent capabilities
                if task.task_type in agent_capabilities:
                    # Check dependencies
                    if self._dependencies_satisfied(task):
                        # Remove from queue and mark as active
                        queue.remove(task)
                        task.status = "running"
                        task.started_at = datetime.now()
                        self.active_tasks[task.task_id] = task
                        return task
        
        return None
    
    def complete_task(self, task_id: str, result: Any = None, error: str = None):
        """Mark task as completed"""
        if task_id in self.active_tasks:
            task = self.active_tasks[task_id]
            task.completed_at = datetime.now()
            task.result = result
            task.error = error
            task.status = "completed" if error is None else "failed"
            
            # Move to completed tasks
            self.completed_tasks[task_id] = task
            del self.active_tasks[task_id]
            
            # Clean up dependencies
            if task_id in self.task_dependencies:
                del self.task_dependencies[task_id]
    
    def _dependencies_satisfied(self, task: Task) -> bool:
        """Check if all task dependencies are satisfied"""
        if not task.dependencies:
            return True
            
        for dep_id in task.dependencies:
            if dep_id in self.active_tasks:
                return False  # Dependency still running
            if dep_id not in self.completed_tasks:
                return False  # Dependency not completed
            if self.completed_tasks[dep_id].status == "failed":
                return False  # Dependency failed
        
        return True
    
    def get_task_stats(self) -> Dict[str, Any]:
        """Get task scheduling statistics"""
        return {
            'queued_tasks': sum(len(queue) for queue in self.task_queues.values()),
            'active_tasks': len(self.active_tasks),
            'completed_tasks': len(self.completed_tasks),
            'failed_tasks': len([t for t in self.completed_tasks.values() if t.status == "failed"]),
            'queue_sizes': {priority.name: len(queue) for priority, queue in self.task_queues.items()}
        }

class PerformanceMonitor:
    """Monitors agent and system performance"""
    
    def __init__(self, history_size: int = 1000):
        self.history_size = history_size
        self.agent_metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=history_size))
        self.system_metrics: deque = deque(maxlen=history_size)
        
    def record_agent_metric(self, agent_id: str, metric_name: str, value: float, timestamp: datetime = None):
        """Record agent performance metric"""
        timestamp = timestamp or datetime.now()
        self.agent_metrics[f"{agent_id}.{metric_name}"].append((timestamp, value))
    
    def record_system_metric(self, metric_name: str, value: float, timestamp: datetime = None):
        """Record system-wide metric"""
        timestamp = timestamp or datetime.now()
        self.system_metrics.append((timestamp, metric_name, value))
    
    def get_agent_stats(self, agent_id: str) -> Dict[str, Any]:
        """Get performance statistics for agent"""
        stats = {}
        
        for metric_key, values in self.agent_metrics.items():
            if metric_key.startswith(f"{agent_id}."):
                metric_name = metric_key.split('.', 1)[1]
                if values:
                    recent_values = [v[1] for v in list(values)[-10:]]  # Last 10 values
                    stats[metric_name] = {
                        'current': recent_values[-1] if recent_values else 0,
                        'average': sum(recent_values) / len(recent_values) if recent_values else 0,
                        'min': min(recent_values) if recent_values else 0,
                        'max': max(recent_values) if recent_values else 0
                    }
        
        return stats
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get system-wide performance statistics"""
        if not self.system_metrics:
            return {}
        
        # Group by metric name
        metrics_by_name = defaultdict(list)
        for timestamp, metric_name, value in self.system_metrics:
            metrics_by_name[metric_name].append(value)
        
        stats = {}
        for metric_name, values in metrics_by_name.items():
            stats[metric_name] = {
                'current': values[-1] if values else 0,
                'average': sum(values) / len(values) if values else 0,
                'min': min(values) if values else 0,
                'max': max(values) if values else 0
            }
        
        return stats

class ObserverCoordinator:
    """
    Main coordinator implementing observer pattern for multi-agent coordination
    Orchestrates agents using event-driven architecture with advanced scheduling
    """
    
    def __init__(self, max_concurrent_tasks: int = 10):
        # Core components
        self.agents: Dict[str, AgentInfo] = {}
        self.observers: List[Observer] = []
        self.task_scheduler = TaskScheduler()
        self.resource_manager = ResourceManager()
        self.performance_monitor = PerformanceMonitor()
        
        # Event handling
        self.event_queue: asyncio.Queue = asyncio.Queue()
        self.event_handlers: Dict[EventType, List[Callable]] = defaultdict(list)
        
        # Execution control
        self.max_concurrent_tasks = max_concurrent_tasks
        self.active_tasks_semaphore = asyncio.Semaphore(max_concurrent_tasks)
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_concurrent_tasks)
        
        # State management
        self.running = False
        self.coordination_stats = {
            'events_processed': 0,
            'tasks_completed': 0,
            'errors_encountered': 0,
            'start_time': None,
            'agent_failures': defaultdict(int)
        }
        
        # Heartbeat monitoring
        self.heartbeat_interval = 30.0  # seconds
        self.heartbeat_timeout = 90.0   # seconds
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
    async def start(self):
        """Start the coordination system"""
        self.running = True
        self.coordination_stats['start_time'] = datetime.now()
        
        # Start background tasks
        asyncio.create_task(self._process_events())
        asyncio.create_task(self._monitor_agents())
        asyncio.create_task(self._task_dispatcher())
        
        self.logger.info("Observer Coordinator started")
    
    async def stop(self):
        """Stop the coordination system"""
        self.running = False
        self.executor.shutdown(wait=True)
        self.logger.info("Observer Coordinator stopped")
    
    async def register_agent(self, agent: Agent):
        """Register an agent with the coordinator"""
        agent_info = AgentInfo(
            agent_id=agent.agent_id,
            agent_type=agent.agent_type,
            capabilities=agent.capabilities
        )
        
        self.agents[agent.agent_id] = agent_info
        
        # Notify observers
        event = CoordinationEvent(
            event_type=EventType.AGENT_STATE_CHANGED,
            source_agent=agent.agent_id,
            data={'new_state': 'registered', 'capabilities': agent.capabilities}
        )
        await self.handle_event(event)
        
        self.logger.info(f"Agent {agent.agent_id} registered with capabilities: {agent.capabilities}")
    
    def add_observer(self, observer: Observer):
        """Add an observer to receive coordination events"""
        self.observers.append(observer)
    
    def remove_observer(self, observer: Observer):
        """Remove an observer"""
        if observer in self.observers:
            self.observers.remove(observer)
    
    async def handle_event(self, event: CoordinationEvent):
        """Handle incoming coordination event"""
        await self.event_queue.put(event)
    
    def add_event_handler(self, event_type: EventType, handler: Callable):
        """Add custom event handler"""
        self.event_handlers[event_type].append(handler)
    
    async def submit_task(self, 
                         task_type: str, 
                         input_data: Dict[str, Any],
                         priority: Priority = Priority.MEDIUM,
                         dependencies: List[str] = None,
                         timeout: float = None) -> str:
        """Submit a task for execution"""
        
        task_id = str(uuid.uuid4())
        
        # Find suitable agent
        suitable_agents = [
            agent_id for agent_id, info in self.agents.items()
            if task_type in info.capabilities and info.state == AgentState.IDLE
        ]
        
        if not suitable_agents:
            # Queue for later if no agents available
            suitable_agents = [
                agent_id for agent_id, info in self.agents.items()
                if task_type in info.capabilities
            ]
            
            if not suitable_agents:
                raise ValueError(f"No agents capable of handling task type: {task_type}")
        
        # Select agent with lowest load
        selected_agent = min(suitable_agents, key=lambda aid: self.agents[aid].load_factor)
        
        task = Task(
            task_id=task_id,
            task_type=task_type,
            agent_id=selected_agent,
            input_data=input_data,
            priority=priority,
            dependencies=dependencies or [],
            timeout=timeout
        )
        
        self.task_scheduler.add_task(task)
        
        # Notify about task assignment
        event = CoordinationEvent(
            event_type=EventType.TASK_ASSIGNED,
            source_agent="coordinator",
            target_agent=selected_agent,
            data={'task_id': task_id, 'task_type': task_type}
        )
        await self.handle_event(event)
        
        return task_id
    
    async def get_task_result(self, task_id: str, timeout: float = None) -> Any:
        """Get task result (blocking until completion)"""
        start_time = time.time()
        
        while True:
            # Check completed tasks
            if task_id in self.task_scheduler.completed_tasks:
                task = self.task_scheduler.completed_tasks[task_id]
                if task.error:
                    raise Exception(f"Task failed: {task.error}")
                return task.result
            
            # Check timeout
            if timeout and (time.time() - start_time) > timeout:
                raise TimeoutError(f"Task {task_id} timed out after {timeout} seconds")
            
            await asyncio.sleep(0.1)  # Small delay to prevent busy waiting
    
    async def _process_events(self):
        """Process coordination events in background"""
        while self.running:
            try:
                # Get event with timeout
                event = await asyncio.wait_for(self.event_queue.get(), timeout=1.0)
                
                # Update stats
                self.coordination_stats['events_processed'] += 1
                
                # Handle event
                await self._handle_event_internal(event)
                
                # Notify observers
                for observer in self.observers:
                    try:
                        await observer.on_event(event)
                    except Exception as e:
                        self.logger.error(f"Observer error: {e}")
                
            except asyncio.TimeoutError:
                continue  # Normal timeout, continue processing
            except Exception as e:
                self.coordination_stats['errors_encountered'] += 1
                self.logger.error(f"Event processing error: {e}")
    
    async def _handle_event_internal(self, event: CoordinationEvent):
        """Internal event handling logic"""
        
        # Handle specific event types
        if event.event_type == EventType.TASK_COMPLETED:
            task_id = event.data.get('task_id')
            result = event.data.get('result')
            
            if task_id:
                self.task_scheduler.complete_task(task_id, result)
                self.coordination_stats['tasks_completed'] += 1
                
                # Update agent state
                if event.source_agent in self.agents:
                    agent_info = self.agents[event.source_agent]
                    agent_info.state = AgentState.IDLE
                    agent_info.current_task = None
                    agent_info.load_factor = max(0, agent_info.load_factor - 0.1)
        
        elif event.event_type == EventType.TASK_FAILED:
            task_id = event.data.get('task_id')
            error = event.data.get('error', 'Unknown error')
            
            if task_id:
                self.task_scheduler.complete_task(task_id, error=error)
                self.coordination_stats['agent_failures'][event.source_agent] += 1
                
                # Update agent state
                if event.source_agent in self.agents:
                    agent_info = self.agents[event.source_agent]
                    agent_info.state = AgentState.ERROR
                    agent_info.current_task = None
        
        elif event.event_type == EventType.AGENT_STATE_CHANGED:
            agent_id = event.source_agent
            new_state = event.data.get('new_state')
            
            if agent_id in self.agents:
                if new_state == 'busy':
                    self.agents[agent_id].state = AgentState.BUSY
                    self.agents[agent_id].load_factor = min(1.0, self.agents[agent_id].load_factor + 0.1)
                elif new_state == 'idle':
                    self.agents[agent_id].state = AgentState.IDLE
        
        # Call custom event handlers
        for handler in self.event_handlers[event.event_type]:
            try:
                await handler(event)
            except Exception as e:
                self.logger.error(f"Event handler error: {e}")
    
    async def _monitor_agents(self):
        """Monitor agent health and heartbeats"""
        while self.running:
            try:
                current_time = datetime.now()
                
                for agent_id, agent_info in self.agents.items():
                    # Check heartbeat timeout
                    time_since_heartbeat = (current_time - agent_info.last_heartbeat).total_seconds()
                    
                    if time_since_heartbeat > self.heartbeat_timeout:
                        # Agent appears to be unresponsive
                        if agent_info.state != AgentState.ERROR:
                            agent_info.state = AgentState.ERROR
                            
                            event = CoordinationEvent(
                                event_type=EventType.AGENT_STATE_CHANGED,
                                source_agent=agent_id,
                                data={'new_state': 'unresponsive', 'last_heartbeat': agent_info.last_heartbeat.isoformat()}
                            )
                            await self.handle_event(event)
                
                await asyncio.sleep(self.heartbeat_interval)
                
            except Exception as e:
                self.logger.error(f"Agent monitoring error: {e}")
    
    async def _task_dispatcher(self):
        """Dispatch tasks to available agents"""
        while self.running:
            try:
                # Find idle agents
                idle_agents = [
                    (agent_id, info) for agent_id, info in self.agents.items()
                    if info.state == AgentState.IDLE
                ]
                
                # Dispatch tasks to idle agents
                for agent_id, agent_info in idle_agents:
                    task = self.task_scheduler.get_next_task(agent_info.capabilities)
                    
                    if task:
                        # Assign task to agent
                        agent_info.state = AgentState.BUSY
                        agent_info.current_task = task.task_id
                        
                        # Execute task (fire and forget)
                        asyncio.create_task(self._execute_task_with_agent(agent_id, task))
                
                await asyncio.sleep(1.0)  # Check every second
                
            except Exception as e:
                self.logger.error(f"Task dispatcher error: {e}")
    
    async def _execute_task_with_agent(self, agent_id: str, task: Task):
        """Execute task with specified agent"""
        async with self.active_tasks_semaphore:
            try:
                # Find agent instance (would need agent registry in real implementation)
                # For now, simulate task execution
                
                start_time = time.time()
                
                # Record performance metrics
                self.performance_monitor.record_agent_metric(agent_id, "task_start", time.time())
                
                # Simulate task execution (replace with actual agent call)
                await asyncio.sleep(1.0)  # Placeholder
                result = f"Task {task.task_id} completed by {agent_id}"
                
                execution_time = time.time() - start_time
                self.performance_monitor.record_agent_metric(agent_id, "execution_time", execution_time)
                
                # Task completed successfully
                event = CoordinationEvent(
                    event_type=EventType.TASK_COMPLETED,
                    source_agent=agent_id,
                    data={'task_id': task.task_id, 'result': result, 'execution_time': execution_time}
                )
                await self.handle_event(event)
                
            except Exception as e:
                # Task failed
                event = CoordinationEvent(
                    event_type=EventType.TASK_FAILED,
                    source_agent=agent_id,
                    data={'task_id': task.task_id, 'error': str(e)}
                )
                await self.handle_event(event)
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        current_time = datetime.now()
        uptime = (current_time - self.coordination_stats['start_time']).total_seconds() if self.coordination_stats['start_time'] else 0
        
        return {
            'system_status': 'running' if self.running else 'stopped',
            'uptime_seconds': uptime,
            'registered_agents': len(self.agents),
            'active_agents': len([a for a in self.agents.values() if a.state != AgentState.ERROR]),
            'coordination_stats': dict(self.coordination_stats),
            'task_stats': self.task_scheduler.get_task_stats(),
            'resource_stats': self.resource_manager.get_resource_stats(),
            'performance_stats': self.performance_monitor.get_system_stats(),
            'agent_details': {
                agent_id: {
                    'type': info.agent_type,
                    'state': info.state.value,
                    'capabilities': info.capabilities,
                    'current_task': info.current_task,
                    'load_factor': info.load_factor,
                    'last_heartbeat': info.last_heartbeat.isoformat()
                }
                for agent_id, info in self.agents.items()
            }
        }
    
    async def emergency_stop(self):
        """Emergency stop all operations"""
        emergency_event = CoordinationEvent(
            event_type=EventType.EMERGENCY_STOP,
            source_agent="coordinator",
            data={'reason': 'Emergency stop requested', 'timestamp': datetime.now().isoformat()}
        )
        
        # Broadcast to all agents
        for agent_id in self.agents:
            emergency_event.target_agent = agent_id
            await self.handle_event(emergency_event)
        
        # Stop coordination
        await self.stop()

# Example Observer Implementation
class LoggingObserver(Observer):
    """Observer that logs all coordination events"""
    
    def __init__(self, log_level=logging.INFO):
        self.logger = logging.getLogger(f"{__name__}.LoggingObserver")
        self.logger.setLevel(log_level)
        
    async def on_event(self, event: CoordinationEvent):
        self.logger.info(f"Event: {event.event_type.value} from {event.source_agent} "
                        f"to {event.target_agent or 'all'} - {event.data}")

class MetricsObserver(Observer):
    """Observer that collects metrics from coordination events"""
    
    def __init__(self):
        self.metrics = defaultdict(int)
        self.event_history = deque(maxlen=1000)
        
    async def on_event(self, event: CoordinationEvent):
        self.metrics[f"event_{event.event_type.value}"] += 1
        self.event_history.append({
            'timestamp': event.timestamp.isoformat(),
            'type': event.event_type.value,
            'source': event.source_agent,
            'target': event.target_agent
        })
    
    def get_metrics(self) -> Dict[str, Any]:
        return {
            'event_counts': dict(self.metrics),
            'total_events': sum(self.metrics.values()),
            'recent_events': list(self.event_history)[-10:]  # Last 10 events
        }

# Usage Example
if __name__ == "__main__":
    import asyncio
    
    async def main():
        # Create coordinator
        coordinator = ObserverCoordinator(max_concurrent_tasks=5)
        
        # Add observers
        logging_observer = LoggingObserver()
        metrics_observer = MetricsObserver()
        
        coordinator.add_observer(logging_observer)
        coordinator.add_observer(metrics_observer)
        
        # Start coordination system
        await coordinator.start()
        
        # Register some shared resources
        coordinator.resource_manager.register_resource("gpu_memory", "GPU_Memory_Pool", max_concurrent=2)
        coordinator.resource_manager.register_resource("vector_db", "Vector_Database", max_concurrent=1)
        
        # Submit some tasks
        task_ids = []
        for i in range(5):
            task_id = await coordinator.submit_task(
                task_type="code_analysis",
                input_data={"code": f"sample_code_{i}"},
                priority=Priority.MEDIUM if i < 3 else Priority.HIGH
            )
            task_ids.append(task_id)
            print(f"Submitted task {task_id}")
        
        # Wait a bit and check status
        await asyncio.sleep(2)
        
        status = coordinator.get_system_status()
        print(f"\nSystem Status: {json.dumps(status, indent=2, default=str)}")
        
        metrics = metrics_observer.get_metrics()
        print(f"\nMetrics: {json.dumps(metrics, indent=2, default=str)}")
        
        # Stop coordinator
        await coordinator.stop()
        
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Run example
    asyncio.run(main())