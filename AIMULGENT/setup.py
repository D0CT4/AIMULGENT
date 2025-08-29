"""
AIMULGENT Setup Configuration
Setup script for AI Multiple Agents for Coding system
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding='utf-8')

# Read requirements
requirements = []
requirements_file = this_directory / "requirements.txt"
if requirements_file.exists():
    with open(requirements_file, 'r', encoding='utf-8') as f:
        requirements = [
            line.strip() 
            for line in f 
            if line.strip() and not line.startswith('#') and not line.startswith('--')
        ]

setup(
    name="aimulgent",
    version="1.0.0",
    author="AI Research Team",
    author_email="research@aimulgent.ai",
    description="AI Multiple Agents for Coding - Advanced multi-agent system for comprehensive code analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/aimulgent/aimulgent",
    project_urls={
        "Bug Tracker": "https://github.com/aimulgent/aimulgent/issues",
        "Documentation": "https://docs.aimulgent.ai",
        "Source Code": "https://github.com/aimulgent/aimulgent",
    },
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Software Development :: Code Generators",
        "Topic :: Software Development :: Quality Assurance", 
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.2.0",
            "pytest-asyncio>=0.20.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
            "pre-commit>=3.0.0",
        ],
        "docs": [
            "sphinx>=6.0.0",
            "sphinx-rtd-theme>=1.2.0",
            "myst-parser>=0.19.0",
            "sphinxcontrib-mermaid>=0.8.0",
        ],
        "gpu": [
            "torch[gpu]>=2.0.0",
            "faiss-gpu>=1.7.3",
            "cupy-cuda11x>=11.0.0",
        ],
        "visualization": [
            "graphviz>=0.20.0",
            "pygraphviz>=1.10.0",
            "dash>=2.8.0",
            "streamlit>=1.20.0",
        ],
        "enterprise": [
            "redis>=4.5.0",
            "celery>=5.2.0",
            "prometheus-client>=0.15.0",
            "opentelemetry-api>=1.15.0",
            "kubernetes>=25.3.0",
        ],
        "research": [
            "langchain>=0.1.0",
            "crewai>=0.1.0", 
            "autogen-agentchat>=0.2.0",
            "diffusers>=0.21.0",
            "accelerate>=0.21.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "aimulgent=aimulgent.main:main",
            "aimulgent-cli=aimulgent.main:main",
        ],
    },
    include_package_data=True,
    package_data={
        "aimulgent": [
            "config/*.json",
            "config/*.yaml", 
            "templates/*.html",
            "templates/*.css",
            "templates/*.js",
            "models/*.pt",
            "models/*.pkl",
            "data/*.json",
            "schemas/*.json",
        ],
    },
    zip_safe=False,
    keywords=[
        "artificial intelligence",
        "multi-agent systems", 
        "code analysis",
        "neural networks",
        "deep learning",
        "software engineering",
        "code quality",
        "static analysis",
        "visualization",
        "coordination patterns",
        "observer pattern",
        "CrewAI",
        "AutoGen",
        "LangGraph",
        "neuro-symbolic AI",
        "predictive coding",
        "agent orchestration",
        "code understanding",
        "software metrics",
        "technical debt",
        "security analysis",
    ],
    platforms=["any"],
    license="MIT",
    license_files=["LICENSE"],
)

# Custom commands for development
import sys
import subprocess
import os

class DevelopmentCommands:
    """Custom development commands"""
    
    @staticmethod
    def install_dev():
        """Install development dependencies"""
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-e", ".[dev,docs,visualization]"
        ])
        print("✅ Development environment installed")
    
    @staticmethod
    def install_gpu():
        """Install GPU-optimized dependencies"""
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-e", ".[gpu,research]"
        ])
        print("✅ GPU environment installed")
    
    @staticmethod
    def setup_pre_commit():
        """Setup pre-commit hooks"""
        try:
            subprocess.check_call(["pre-commit", "install"])
            print("✅ Pre-commit hooks installed")
        except subprocess.CalledProcessError:
            print("❌ Failed to install pre-commit hooks. Install pre-commit first:")
            print("pip install pre-commit")
    
    @staticmethod
    def run_tests():
        """Run test suite"""
        subprocess.check_call([
            sys.executable, "-m", "pytest", "tests/", "-v", "--cov=aimulgent"
        ])
        print("✅ Tests completed")
    
    @staticmethod
    def format_code():
        """Format code with black and isort"""
        subprocess.check_call(["black", "."])
        subprocess.check_call(["isort", "."])
        print("✅ Code formatted")
    
    @staticmethod
    def lint_code():
        """Lint code with flake8 and mypy"""
        try:
            subprocess.check_call(["flake8", ".", "--max-line-length=100"])
            subprocess.check_call(["mypy", ".", "--ignore-missing-imports"])
            print("✅ Code linting passed")
        except subprocess.CalledProcessError:
            print("❌ Linting issues found")
            sys.exit(1)
    
    @staticmethod
    def build_docs():
        """Build documentation"""
        docs_dir = Path("docs")
        if not docs_dir.exists():
            docs_dir.mkdir()
            # Initialize sphinx
            subprocess.check_call([
                "sphinx-quickstart", "-q", "--project=AIMULGENT", 
                "--author=AI Research Team", "--release=1.0.0", 
                "--language=en", "--extensions=sphinx.ext.autodoc,sphinx.ext.viewcode,myst_parser",
                str(docs_dir)
            ])
        
        subprocess.check_call(["sphinx-build", "-b", "html", "docs", "docs/_build"])
        print("✅ Documentation built in docs/_build")

# Add custom commands to setup
if __name__ == "__main__":
    import argparse
    
    if len(sys.argv) > 1 and sys.argv[1] in ['install', 'develop', 'bdist_wheel', 'sdist']:
        # Standard setup.py commands
        setup()
    else:
        # Custom development commands
        parser = argparse.ArgumentParser(description="AIMULGENT Development Commands")
        parser.add_argument("command", choices=[
            "install-dev", "install-gpu", "setup-pre-commit", 
            "test", "format", "lint", "docs"
        ], help="Development command to run")
        
        args = parser.parse_args()
        
        commands = DevelopmentCommands()
        
        if args.command == "install-dev":
            commands.install_dev()
        elif args.command == "install-gpu":
            commands.install_gpu()
        elif args.command == "setup-pre-commit":
            commands.setup_pre_commit()
        elif args.command == "test":
            commands.run_tests()
        elif args.command == "format":
            commands.format_code()
        elif args.command == "lint":
            commands.lint_code()
        elif args.command == "docs":
            commands.build_docs()
        else:
            parser.print_help()