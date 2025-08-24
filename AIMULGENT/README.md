# AIMULGENT - AI Multiple Agents for Coding

> State-of-the-art multi-agent system for comprehensive code analysis

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

## Overview

AIMULGENT is a production-ready multi-agent system that provides comprehensive code analysis using specialized AI agents. Built following KISS principles with a focus on maintainability and reliability.

### Key Features

- **Multi-Agent Architecture**: Specialized agents for different analysis tasks
- **Real-time Coordination**: Event-driven system with efficient task distribution  
- **Code Quality Assessment**: Comprehensive metrics and recommendations
- **Security Analysis**: Detection of common security vulnerabilities
- **Simple CLI Interface**: Easy-to-use command-line tools
- **Production Ready**: Following enterprise development best practices

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/aimulgent/aimulgent.git
cd aimulgent

# Create virtual environment  
uv venv
uv sync

# Install in development mode
uv pip install -e .
```

### Basic Usage

```bash
# Analyze a Python file
uv run aimulgent analyze my_code.py

# Show system status
uv run aimulgent status

# Get help
uv run aimulgent --help
```

### Python API

```python
import asyncio
from aimulgent import AIMULGENTSystem

async def main():
    # Initialize system
    system = AIMULGENTSystem()
    await system.start()
    
    try:
        # Analyze code
        code = '''
def hello_world():
    print("Hello, World!")
        '''
        
        result = await system.analyze_code(code, "hello.py")
        
        print(f"Quality Score: {result['analysis']['quality_score']}/10")
        print(f"Rating: {result['analysis']['rating']}")
        
        # Show recommendations
        for rec in result['analysis']['recommendations']:
            print(f"- {rec}")
            
    finally:
        await system.stop()

asyncio.run(main())
```

## Architecture

AIMULGENT follows a clean, modular architecture:

```
aimulgent/
├── core/                   # Core system components
│   ├── system.py          # Main system orchestrator
│   ├── coordinator.py     # Agent coordination  
│   └── config.py          # Configuration management
├── agents/                # Specialized agents
│   ├── base.py           # Abstract base agent
│   └── analysis.py       # Code analysis agent
└── main.py               # CLI interface
```

### Design Principles

- **KISS**: Keep implementations simple and focused
- **Single Responsibility**: Each component has one clear purpose
- **Fail Fast**: Early error detection and proper exception handling
- **Under 500 Lines**: All files kept under 500 lines for maintainability

## Agents

### Analysis Agent

The core agent providing:

- **Code Structure Analysis**: AST-based parsing and metrics
- **Complexity Calculation**: Cyclomatic complexity scoring
- **Security Scanning**: Detection of common vulnerabilities
- **Quality Assessment**: Overall code quality scoring (0-10)

#### Supported Checks

- SQL injection patterns
- Hardcoded secrets and passwords
- Command injection risks
- Code complexity metrics
- Function and class structure analysis

## Development

### Setup Development Environment

```bash
# Install development dependencies
uv sync --dev

# Install pre-commit hooks
uv run pre-commit install
```

### Running Tests

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=aimulgent --cov-report=html

# Run specific tests
uv run pytest tests/test_system.py -v
```

### Code Quality

```bash
# Format code
uv run ruff format .

# Check linting
uv run ruff check .

# Fix linting issues
uv run ruff check --fix .

# Type checking
uv run mypy aimulgent/
```

### Project Standards

This project follows the guidelines in [CLAUDE.md](CLAUDE.md):

- **Line length**: 100 characters maximum
- **Type hints**: Required for all public functions
- **Docstrings**: Google-style for all public APIs
- **Testing**: TDD approach with comprehensive test coverage
- **Dependencies**: Managed with UV package manager

## Configuration

Create a `.env` file for custom settings:

```env
# System settings
AIMULGENT_DEBUG=false
AIMULGENT_LOG_LEVEL=INFO

# Database
AIMULGENT_DATABASE_URL=sqlite:///./aimulgent.db

# Agent settings
AIMULGENT_MAX_CONCURRENT_TASKS=10
```

## API Reference

### Core Classes

#### `AIMULGENTSystem`

Main system class for managing agents and coordination.

**Methods:**

- `async start()` - Initialize and start the system
- `async stop()` - Stop the system and cleanup resources  
- `async analyze_code(code: str, file_path: str = None)` - Analyze code and return results
- `async get_system_status()` - Get comprehensive system status

#### `Settings`

Configuration management with Pydantic validation.

**Key Settings:**

- `app_name: str` - Application name
- `debug: bool` - Debug mode flag
- `log_level: str` - Logging level
- `coordinator.max_concurrent_tasks: int` - Maximum concurrent tasks

## Examples

### Analyzing Multiple Files

```python
import asyncio
from pathlib import Path
from aimulgent import AIMULGENTSystem

async def analyze_project(project_path: Path):
    system = AIMULGENTSystem()
    await system.start()
    
    try:
        python_files = list(project_path.rglob("*.py"))
        
        for file_path in python_files:
            with open(file_path, 'r') as f:
                code = f.read()
            
            result = await system.analyze_code(code, str(file_path))
            
            print(f"{file_path}: {result['analysis']['quality_score']}/10")
            
    finally:
        await system.stop()
```

### Custom Configuration

```python
from aimulgent import AIMULGENTSystem, Settings
from aimulgent.core.config import AgentConfig

# Custom settings
settings = Settings(
    debug=True,
    log_level="DEBUG",
    agents={
        "analysis": AgentConfig(
            enabled=True,
            max_concurrent_tasks=5,
            timeout_seconds=120
        )
    }
)

system = AIMULGENTSystem(settings)
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Follow the development guidelines in [CLAUDE.md](CLAUDE.md)
4. Write tests for your changes
5. Ensure all tests pass and code is properly formatted
6. Commit your changes (`git commit -m 'Add amazing feature'`)
7. Push to the branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

### Commit Message Format

```
<type>(<scope>): <subject>

<body>

<footer>
```

Types: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`

## Troubleshooting

### Common Issues

**ImportError**: Make sure all dependencies are installed with `uv sync`

**Permission Errors**: Ensure proper file permissions for data directory

**Timeout Errors**: Increase timeout settings in configuration for large files

### Performance Tips

- Use async methods for better concurrency
- Process files in batches for large projects
- Adjust `max_concurrent_tasks` based on system resources

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for version history and changes.

## Support

- **Issues**: [GitHub Issues](https://github.com/aimulgent/aimulgent/issues)
- **Discussions**: [GitHub Discussions](https://github.com/aimulgent/aimulgent/discussions)
- **Email**: research@aimulgent.ai

---

Built with ❤️ following modern Python best practices and enterprise development standards.