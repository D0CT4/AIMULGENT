"""
Magical Interactive Interface for AIMULGENT
A beautiful, animated terminal interface with Rich effects
"""

import asyncio
import random
import time
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime

from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    Progress, 
    SpinnerColumn, 
    BarColumn, 
    TextColumn, 
    MofNCompleteColumn,
    TimeElapsedColumn
)
from rich.live import Live
from rich.table import Table
from rich.text import Text
from rich.align import Align
from rich.layout import Layout
from rich.tree import Tree
from rich.syntax import Syntax
from rich.columns import Columns
from rich.prompt import Prompt, Confirm
from rich.rule import Rule

from aimulgent.core.system import AIMULGENTSystem
from aimulgent.core.config import Settings


class MagicalInterface:
    """Magical interactive interface with Rich animations and effects."""
    
    def __init__(self):
        self.console = Console()
        self.system: Optional[AIMULGENTSystem] = None
        self.analysis_history: List[Dict] = []
        
        # Magical symbols and colors
        self.magic_symbols = ["✨", "🌟", "⭐", "💫", "🔮", "🪄", "🎯", "🚀"]
        self.agent_emojis = {
            "analysis": "🧠",
            "security": "🛡️", 
            "quality": "💎",
            "complexity": "🧮",
            "coordinator": "🎭"
        }
        
        # Color themes
        self.colors = {
            "primary": "bright_magenta",
            "secondary": "bright_cyan", 
            "success": "bright_green",
            "warning": "bright_yellow",
            "danger": "bright_red",
            "info": "bright_blue",
            "magic": "magenta"
        }
    
    def _create_sparkles(self, count: int = 5) -> str:
        """Generate random sparkle effects."""
        return " ".join(random.choices(self.magic_symbols, k=count))
    
    def _create_title_panel(self, title: str, subtitle: str = "") -> Panel:
        """Create a magical title panel with effects."""
        sparkles = self._create_sparkles(3)
        title_text = Text()
        title_text.append(f"{sparkles} ", style=self.colors["magic"])
        title_text.append(title, style=f"bold {self.colors['primary']}")
        title_text.append(f" {sparkles}", style=self.colors["magic"])
        
        if subtitle:
            title_text.append(f"\n{subtitle}", style=self.colors["secondary"])
        
        return Panel(
            Align.center(title_text),
            border_style=self.colors["magic"],
            padding=(1, 2)
        )
    
    def _create_agent_status_table(self, system_status: Dict) -> Table:
        """Create animated agent status display."""
        table = Table(
            title="🎭 Agent Status Dashboard", 
            border_style=self.colors["primary"],
            show_header=True,
            header_style="bold bright_white"
        )
        
        table.add_column("Agent", style=self.colors["info"], min_width=12)
        table.add_column("Status", justify="center", min_width=10)
        table.add_column("Capabilities", style=self.colors["secondary"], min_width=25)
        table.add_column("Tasks", justify="right", min_width=8)
        
        # Add agents from status
        agents_info = system_status.get("agents", {})
        coordinator_info = system_status.get("coordinator", {})
        
        for agent_id, info in agents_info.items():
            emoji = self.agent_emojis.get(agent_id, "🤖")
            status_emoji = "🟢" if info.get("status") == "idle" else "🔄"
            capabilities = ", ".join(info.get("capabilities", []))
            tasks_done = coordinator_info.get("tasks", {}).get("completed", 0)
            
            table.add_row(
                f"{emoji} {agent_id.title()}",
                f"{status_emoji} Active",
                capabilities,
                str(tasks_done)
            )
        
        return table
    
    def _create_analysis_results_panel(self, result: Dict[str, Any]) -> Panel:
        """Create beautiful analysis results display."""
        analysis = result.get("analysis", {})
        
        # Create main metrics display
        metrics_text = Text()
        
        # Quality score with visual indicator
        quality_score = analysis.get("quality_score", 0)
        quality_stars = "⭐" * min(int(quality_score), 10)
        metrics_text.append("🎯 Quality Score: ", style="bold")
        metrics_text.append(f"{quality_score}/10 {quality_stars}\n", 
                           style=self.colors["success"] if quality_score >= 7 else self.colors["warning"])
        
        # Issues found
        issues = analysis.get("issues_found", 0)
        issue_color = self.colors["success"] if issues == 0 else self.colors["danger"]
        metrics_text.append("🔍 Issues Found: ", style="bold")
        metrics_text.append(f"{issues}\n", style=issue_color)
        
        # Complexity
        complexity = analysis.get("complexity_score", 0)
        complexity_emoji = "🟢" if complexity < 5 else "🟡" if complexity < 8 else "🔴"
        metrics_text.append("🧮 Complexity: ", style="bold")
        metrics_text.append(f"{complexity}/10 {complexity_emoji}\n", style=self.colors["info"])
        
        # Recommendations
        recommendations = analysis.get("recommendations", [])
        if recommendations:
            metrics_text.append("\n💡 Recommendations:\n", style="bold bright_yellow")
            for i, rec in enumerate(recommendations[:3], 1):
                metrics_text.append(f"  {i}. {rec}\n", style=self.colors["secondary"])
        
        return Panel(
            metrics_text,
            title="📊 Analysis Results",
            border_style=self.colors["success"],
            padding=(1, 2)
        )
    
    def _show_animated_startup(self):
        """Show magical startup animation."""
        startup_steps = [
            ("🌟", "Initializing magical systems..."),
            ("🧠", "Awakening AI agents..."),
            ("🔮", "Calibrating neural networks..."),
            ("⚡", "Charging analysis engines..."),
            ("✨", "Ready for magical code analysis!")
        ]
        
        with Progress(
            SpinnerColumn("dots"),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            console=self.console
        ) as progress:
            
            task = progress.add_task("Starting AIMULGENT...", total=len(startup_steps))
            
            for emoji, description in startup_steps:
                progress.update(task, description=f"{emoji} {description}")
                time.sleep(0.8)
                progress.advance(task)
        
        # Final magical effect
        self.console.print("\n" + "✨" * 50, style=self.colors["magic"])
        self.console.print(
            Align.center("🪄 AIMULGENT MAGICAL INTERFACE ACTIVATED 🪄"),
            style=f"bold {self.colors['primary']}"
        )
        self.console.print("✨" * 50, style=self.colors["magic"])
    
    async def start_system(self):
        """Start the AIMULGENT system with magical effects."""
        self._show_animated_startup()
        
        settings = Settings(debug=False)
        self.system = AIMULGENTSystem(settings)
        await self.system.start()
        
        self.console.print(f"\n🚀 System online! {random.choice(self.magic_symbols)}", 
                          style=f"bold {self.colors['success']}")
    
    async def stop_system(self):
        """Stop the system gracefully."""
        if self.system:
            await self.system.stop()
            self.console.print("🌙 System shutdown complete. Sweet dreams!", 
                              style=f"bold {self.colors['info']}")
    
    def show_main_menu(self) -> str:
        """Display main menu and get user choice."""
        self.console.clear()
        
        # Title
        title_panel = self._create_title_panel(
            "AIMULGENT MAGICAL INTERFACE",
            "AI Multiple Agents for Coding Analysis"
        )
        self.console.print(title_panel)
        
        # Menu options
        menu_table = Table(show_header=False, box=None, padding=(0, 2))
        menu_table.add_column("Option", style=f"bold {self.colors['primary']}")
        menu_table.add_column("Description", style=self.colors['secondary'])
        
        options = [
            ("1", "🔍 Analyze Code File", "Perform magical code analysis"),
            ("2", "📊 View System Status", "Check agent health and statistics"),
            ("3", "📚 Analysis History", "Review previous analysis results"),
            ("4", "🎯 Interactive Analysis", "Real-time code analysis mode"),
            ("5", "⚙️ Settings", "Configure magical parameters"),
            ("6", "❌ Exit", "Close the magical interface")
        ]
        
        for num, emoji_desc, desc in options:
            menu_table.add_row(f"[{num}]", f"{emoji_desc} - {desc}")
        
        self.console.print("\n")
        self.console.print(Panel(menu_table, title="🎭 Choose Your Adventure", 
                                border_style=self.colors["magic"]))
        
        choice = Prompt.ask(
            "\n✨ Enter your magical choice",
            choices=["1", "2", "3", "4", "5", "6"],
            default="1"
        )
        
        return choice
    
    async def analyze_file_interactive(self):
        """Interactive file analysis with magical effects."""
        file_path = Prompt.ask("🔮 Enter the path to your code file")
        
        if not Path(file_path).exists():
            self.console.print(f"❌ File not found: {file_path}", 
                              style=f"bold {self.colors['danger']}")
            return
        
        # Read file content
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                code_content = f.read()
        except Exception as e:
            self.console.print(f"❌ Error reading file: {e}", 
                              style=f"bold {self.colors['danger']}")
            return
        
        # Show file preview
        if len(code_content) < 1000:
            syntax = Syntax(code_content, "python", theme="monokai", line_numbers=True)
            self.console.print(Panel(syntax, title=f"📄 {Path(file_path).name}", 
                                    border_style=self.colors["info"]))
        
        # Animated analysis
        self.console.print(f"\n🪄 Beginning magical analysis of {Path(file_path).name}...")
        
        analysis_steps = [
            ("🔍", "Scanning code structure..."),
            ("🛡️", "Detecting security vulnerabilities..."),
            ("🧮", "Calculating complexity metrics..."),
            ("💎", "Assessing code quality..."),
            ("📊", "Generating recommendations...")
        ]
        
        with Progress(
            SpinnerColumn("aesthetic"),
            TextColumn("[bold cyan]{task.description}"),
            BarColumn(bar_width=None),
            console=self.console
        ) as progress:
            
            analysis_task = progress.add_task("Analyzing...", total=len(analysis_steps))
            
            # Start actual analysis
            result_future = asyncio.create_task(
                self.system.analyze_code(code_content, file_path)
            )
            
            # Show animated progress
            for emoji, description in analysis_steps:
                progress.update(analysis_task, description=f"{emoji} {description}")
                await asyncio.sleep(0.5)
                progress.advance(analysis_task)
            
            # Wait for analysis to complete
            result = await result_future
        
        # Store in history
        self.analysis_history.append({
            "file_path": file_path,
            "timestamp": datetime.now().isoformat(),
            "result": result
        })
        
        # Display results
        self.console.print(f"\n{random.choice(self.magic_symbols)} Analysis Complete! {random.choice(self.magic_symbols)}")
        results_panel = self._create_analysis_results_panel(result)
        self.console.print(results_panel)
        
        # Ask if user wants to save results
        save_results = Confirm.ask("💾 Save results to file?", default=False)
        if save_results:
            output_file = f"analysis_{Path(file_path).stem}_{int(time.time())}.json"
            with open(output_file, 'w') as f:
                import json
                json.dump(result, f, indent=2)
            self.console.print(f"✅ Results saved to {output_file}", 
                              style=f"bold {self.colors['success']}")
    
    async def show_system_status(self):
        """Display animated system status."""
        if not self.system:
            self.console.print("❌ System not started!", style=f"bold {self.colors['danger']}")
            return
        
        self.console.print("🔄 Fetching system status...", style=self.colors["info"])
        
        status = await self.system.get_system_status()
        
        # Create status display
        self.console.clear()
        
        # System info panel
        system_info = status.get("system", {})
        info_text = Text()
        info_text.append("🚀 System: ", style="bold")
        info_text.append(f"{system_info.get('name', 'AIMULGENT')} v{system_info.get('version', '1.0.0')}\n")
        info_text.append("⚡ Status: ", style="bold")
        info_text.append("🟢 Running\n", style=self.colors["success"])
        info_text.append("🕐 Uptime: ", style="bold")
        info_text.append(f"{system_info.get('uptime', 'Unknown')}\n")
        
        info_panel = Panel(info_text, title="📋 System Information", 
                          border_style=self.colors["primary"])
        
        # Agent status table
        agent_table = self._create_agent_status_table(status)
        
        # Layout
        layout = Layout()
        layout.split_row(
            Layout(info_panel, name="info"),
            Layout(agent_table, name="agents")
        )
        
        self.console.print(layout)
    
    def show_analysis_history(self):
        """Display analysis history with magical formatting."""
        if not self.analysis_history:
            self.console.print("📭 No analysis history found.", 
                              style=self.colors["warning"])
            return
        
        self.console.clear()
        
        history_table = Table(
            title="📚 Analysis History",
            border_style=self.colors["magic"],
            show_header=True,
            header_style="bold bright_white"
        )
        
        history_table.add_column("📅 Date", style=self.colors["secondary"])
        history_table.add_column("📄 File", style=self.colors["info"])
        history_table.add_column("🎯 Score", justify="center")
        history_table.add_column("🔍 Issues", justify="center")
        history_table.add_column("⭐ Rating", justify="center")
        
        for entry in self.analysis_history[-10:]:  # Show last 10
            timestamp = datetime.fromisoformat(entry["timestamp"])
            file_name = Path(entry["file_path"]).name
            analysis = entry["result"].get("analysis", {})
            
            score = analysis.get("quality_score", 0)
            issues = analysis.get("issues_found", 0)
            rating = analysis.get("rating", "Unknown")
            
            score_color = self.colors["success"] if score >= 7 else self.colors["warning"]
            issue_color = self.colors["success"] if issues == 0 else self.colors["danger"]
            
            history_table.add_row(
                timestamp.strftime("%m/%d %H:%M"),
                file_name,
                f"[{score_color}]{score}/10[/]",
                f"[{issue_color}]{issues}[/]",
                rating
            )
        
        self.console.print(history_table)
    
    async def run_interactive_mode(self):
        """Main interactive loop with magical interface."""
        try:
            await self.start_system()
            
            while True:
                choice = self.show_main_menu()
                
                if choice == "1":
                    await self.analyze_file_interactive()
                elif choice == "2":
                    await self.show_system_status()
                elif choice == "3":
                    self.show_analysis_history()
                elif choice == "4":
                    self.console.print("🚧 Interactive mode coming soon!", 
                                      style=self.colors["warning"])
                elif choice == "5":
                    self.console.print("⚙️ Settings panel coming soon!", 
                                      style=self.colors["warning"])
                elif choice == "6":
                    confirm_exit = Confirm.ask("🌙 Are you sure you want to exit?", default=False)
                    if confirm_exit:
                        break
                
                if choice in ["1", "2", "3", "4", "5"]:
                    Prompt.ask("\n✨ Press Enter to continue")
            
        except KeyboardInterrupt:
            self.console.print("\n🌟 Interrupted by user", style=self.colors["warning"])
        except Exception as e:
            self.console.print(f"\n❌ Error: {e}", style=f"bold {self.colors['danger']}")
        finally:
            await self.stop_system()