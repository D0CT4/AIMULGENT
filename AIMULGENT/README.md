# AIMULGENT - AI Multiple Agents for Coding

> Multi-agent system 
## Overview

AIMULGENT is a production-ready multi-agent system that provides 

- **Multi-Agent Architecture**: Specialized agents for different analysis tasks
- **Real-time Coordination**: Event-driven system with efficient task distribution  
- **Code Quality Assessment**: Comprehensive metrics and recommendations
- **Security Analysis**: Detection of common security vulnerabilities
- **Simple CLI Interface**: Easy-to-use command-line tools
- **Production Ready**: Following enterprise development best practices

## Quick Start

### Installation

# Clone the repository
git clone https://github.com/aimulgent/aimulgent.git
cd aimulgent

# Create virtual environment  
uv venv
uv sync

# Install in development mode
uv pip install -e 

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



### Core Classes

#### `AIMULGENTSystem`

Main system class for managing agents and coordination.

**Methods:**

- `async start()` - Initialize and start the system
- `async stop()` - Stop the system and cleanup resources  
- `async analyze_code(code: str, file_path: str = None)` - Analyze code and return results
- `async get_system_status()` - Get comprehensive system status



## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
4. Write tests for your changes
5. Ensure all tests pass and code is properly formatted
6. Commit your changes (`git commit -m 'Add amazing feature'`)
7. Push to the branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request


## Troubleshooting

### Common Issues

**ImportError**: Make sure all dependencies are installed with `uv sync`

**Permission Errors**: Ensure proper file permissions for data directory


## License

This project is licensed under the MIT License - see the [LICENSE] file for details.

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for version history and changes.

## Support

- **Issues**: [GitHub Issues](https://github.com/aimulgent/aimulgent/issues)
- **Discussions**: [GitHub Discussions](https://github.com/aimulgent/aimulgent/discussions)


---

Built with ❤️ following modern Python best practices and enterprise development standards.