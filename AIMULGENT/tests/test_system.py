"""
Tests for AIMULGENT System
Following TDD principles from CLAUDE.md
"""

import pytest
import asyncio
from pathlib import Path

from aimulgent.core.system import AIMULGENTSystem
from aimulgent.core.config import Settings


class TestAIMULGENTSystem:
    """Test suite for the main AIMULGENT system."""
    
    @pytest.fixture
    async def system(self):
        """Provide a test system instance."""
        settings = Settings(debug=True, data_dir=Path("test_data"))
        system = AIMULGENTSystem(settings)
        await system.start()
        yield system
        await system.stop()
    
    async def test_system_starts_and_stops(self):
        """Test that system can start and stop properly."""
        settings = Settings(debug=True, data_dir=Path("test_data"))
        system = AIMULGENTSystem(settings)
        
        # System should not be running initially
        assert not system.running
        
        # Start system
        await system.start()
        assert system.running
        
        # Stop system  
        await system.stop()
        assert not system.running
    
    async def test_code_analysis_with_valid_code(self, system):
        """Test code analysis with valid Python code."""
        code = '''
def hello_world():
    """A simple function."""
    print("Hello, World!")
    return "Hello, World!"
        '''
        
        result = await system.analyze_code(code, "test.py")
        
        # Check result structure
        assert "file_path" in result
        assert "analysis" in result
        assert "timestamp" in result
        
        # Check analysis content
        analysis = result["analysis"]
        assert "quality_score" in analysis
        assert "rating" in analysis
        assert "metrics" in analysis
        
        # Quality score should be reasonable for simple, clean code
        assert 0 <= analysis["quality_score"] <= 10
    
    async def test_code_analysis_with_syntax_error(self, system):
        """Test code analysis with invalid syntax."""
        code = '''
def broken_function(
    # Missing closing parenthesis and proper syntax
        '''
        
        result = await system.analyze_code(code, "broken.py")
        analysis = result["analysis"]
        
        # Should handle syntax errors gracefully
        assert "error" in analysis or analysis["quality_score"] == 0
    
    async def test_system_status(self, system):
        """Test system status reporting."""
        status = await system.get_system_status()
        
        # Check status structure
        assert "system" in status
        assert "coordinator" in status  
        assert "agents" in status
        
        # Check system info
        system_info = status["system"]
        assert system_info["name"] == "AIMULGENT"
        assert system_info["running"] is True
        
        # Check coordinator info
        coordinator_info = status["coordinator"]
        assert "agents" in coordinator_info
        assert "tasks" in coordinator_info
    
    async def test_concurrent_analysis_tasks(self, system):
        """Test that system can handle concurrent analysis tasks."""
        code_samples = [
            'def func1(): pass',
            'def func2(): return 42',
            'class TestClass: pass'
        ]
        
        # Submit multiple tasks concurrently
        tasks = [
            system.analyze_code(code, f"test_{i}.py") 
            for i, code in enumerate(code_samples)
        ]
        
        results = await asyncio.gather(*tasks)
        
        # All tasks should complete successfully
        assert len(results) == 3
        for result in results:
            assert "analysis" in result
            assert "quality_score" in result["analysis"]


class TestAnalysisQuality:
    """Test suite for analysis quality and accuracy."""
    
    @pytest.fixture
    async def system(self):
        """Provide a test system instance."""
        settings = Settings(debug=True)
        system = AIMULGENTSystem(settings)
        await system.start()
        yield system
        await system.stop()
    
    async def test_security_issue_detection(self, system):
        """Test that security issues are properly detected."""
        vulnerable_code = '''
import os
password = "hardcoded_secret_123"
user_input = input("Enter command: ")
os.system(user_input)  # Command injection risk
        '''
        
        result = await system.analyze_code(vulnerable_code)
        analysis = result["analysis"]
        
        # Should detect security issues
        security = analysis.get("security", {})
        assert security.get("issues_found", 0) > 0
        
        # Quality score should be lower due to security issues
        assert analysis["quality_score"] < 8.0
    
    async def test_complexity_scoring(self, system):
        """Test that high complexity is properly scored."""
        complex_code = '''
def complex_function(x, y, z, a, b, c, d, e):
    if x > 0:
        if y > 0:
            if z > 0:
                if a > 0:
                    if b > 0:
                        if c > 0:
                            if d > 0:
                                if e > 0:
                                    return x + y + z + a + b + c + d + e
                                else:
                                    return -1
                            else:
                                return -2
                        else:
                            return -3
                    else:
                        return -4
                else:
                    return -5
            else:
                return -6
        else:
            return -7
    else:
        return -8
        '''
        
        result = await system.analyze_code(complex_code)
        analysis = result["analysis"]
        
        # Should detect high complexity
        metrics = analysis.get("metrics", {})
        assert metrics.get("complexity", 0) > 8
        
        # Should recommend refactoring
        recommendations = analysis.get("recommendations", [])
        assert any("complex" in rec.lower() for rec in recommendations)


@pytest.mark.asyncio
async def test_system_integration():
    """Integration test for complete system workflow."""
    # Create system
    settings = Settings(debug=True, data_dir=Path("integration_test"))
    system = AIMULGENTSystem(settings)
    
    try:
        # Start system
        await system.start()
        
        # Analyze a realistic code sample
        code = '''
class Calculator:
    """A simple calculator class."""
    
    def __init__(self):
        self.history = []
    
    def add(self, a, b):
        """Add two numbers."""
        result = a + b
        self.history.append(f"{a} + {b} = {result}")
        return result
    
    def divide(self, a, b):
        """Divide two numbers."""
        if b == 0:
            raise ValueError("Cannot divide by zero")
        result = a / b
        self.history.append(f"{a} / {b} = {result}")
        return result
    
    def get_history(self):
        """Get calculation history."""
        return self.history.copy()
        '''
        
        # Perform analysis
        result = await system.analyze_code(code, "calculator.py")
        
        # Verify comprehensive analysis
        analysis = result["analysis"]
        assert "quality_score" in analysis
        assert "metrics" in analysis
        assert "recommendations" in analysis
        
        # Check metrics
        metrics = analysis["metrics"]
        assert metrics["function_count"] == 4  # __init__, add, divide, get_history
        assert metrics["class_count"] == 1
        assert metrics["lines_of_code"] > 20
        
        # Quality should be good for this clean code
        assert analysis["quality_score"] >= 7.0
        
        # Get system status
        status = await system.get_system_status()
        assert status["system"]["running"]
        assert status["coordinator"]["tasks"]["completed"] >= 1
        
    finally:
        await system.stop()