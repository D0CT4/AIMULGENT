"""
AIMULGENT Main Entry Point
CLI interface for the AIMULGENT system.
"""

import asyncio
import json
import sys
from pathlib import Path
from typing import Optional

import click

from aimulgent.core.config import get_settings
from aimulgent.core.system import AIMULGENTSystem


@click.group()
@click.version_option(version="1.0.0", prog_name="AIMULGENT")
def cli():
    """AIMULGENT - AI Multiple Agents for Coding"""
    pass


@cli.command()
@click.argument("file_path", type=click.Path(exists=True, path_type=Path))
@click.option("--output", "-o", type=click.Path(path_type=Path), help="Output file for results")
@click.option("--format", "output_format", type=click.Choice(["json", "text"]), default="json")
def analyze(file_path: Path, output: Optional[Path], output_format: str):
    """Analyze a code file."""
    
    async def run_analysis():
        # Read the code file
        with open(file_path, 'r', encoding='utf-8') as f:
            code = f.read()
        
        # Initialize system
        settings = get_settings()
        system = AIMULGENTSystem(settings)
        
        try:
            await system.start()
            
            # Analyze the code
            click.echo(f"Analyzing {file_path}...")
            result = await system.analyze_code(code, str(file_path))
            
            # Format output
            if output_format == "json":
                output_data = json.dumps(result, indent=2, default=str)
            else:
                # Text format
                analysis = result["analysis"]
                output_data = f"""
AIMULGENT Analysis Results
=========================
File: {file_path}
Quality Score: {analysis.get('quality_score', 'N/A')}/10
Rating: {analysis.get('rating', 'N/A')}

Metrics:
- Lines of Code: {analysis.get('metrics', {}).get('lines_of_code', 'N/A')}
- Functions: {analysis.get('metrics', {}).get('function_count', 'N/A')}
- Classes: {analysis.get('metrics', {}).get('class_count', 'N/A')}
- Complexity: {analysis.get('metrics', {}).get('complexity', 'N/A')}
- Security Issues: {analysis.get('metrics', {}).get('security_issues', 'N/A')}

Recommendations:
{chr(10).join('- ' + rec for rec in analysis.get('recommendations', []))}
                """.strip()
            
            # Output results
            if output:
                with open(output, 'w', encoding='utf-8') as f:
                    f.write(output_data)
                click.echo(f"Results saved to {output}")
            else:
                click.echo(output_data)
                
        finally:
            await system.stop()
    
    asyncio.run(run_analysis())


@cli.command()
def status():
    """Show system status."""
    
    async def show_status():
        settings = get_settings()
        system = AIMULGENTSystem(settings)
        
        try:
            await system.start()
            status_info = await system.get_system_status()
            
            click.echo("AIMULGENT System Status")
            click.echo("======================")
            click.echo(f"System: {status_info['system']['name']} v{status_info['system']['version']}")
            click.echo(f"Running: {status_info['system']['running']}")
            click.echo(f"Agents: {status_info['coordinator']['agents']['total']}")
            click.echo(f"Tasks Completed: {status_info['coordinator']['tasks']['completed']}")
            
        finally:
            await system.stop()
    
    asyncio.run(show_status())


@cli.command()
@click.option("--host", default="127.0.0.1", help="Host to bind to")
@click.option("--port", default=8000, help="Port to bind to") 
def serve(host: str, port: int):
    """Start AIMULGENT as a web service."""
    
    try:
        import uvicorn
        from aimulgent.api.app import create_app
        
        app = create_app()
        uvicorn.run(app, host=host, port=port)
        
    except ImportError:
        click.echo("Web service requires 'uvicorn' and 'fastapi'. Install with:")
        click.echo("pip install 'aimulgent[api]'")
        sys.exit(1)


def main():
    """Main entry point."""
    cli()


if __name__ == "__main__":
    main()