"""
Analysis Agent Implementation
Handles code analysis tasks for the AIMULGENT system.
"""

import ast
import re
import logging
from typing import Any, Dict, List

from .base import BaseAgent

logger = logging.getLogger(__name__)


class AnalysisAgent(BaseAgent):
    """
    Agent specialized in code analysis tasks.
    
    Provides comprehensive code quality assessment following KISS principles.
    """
    
    def __init__(self):
        super().__init__("analysis")
        self.capabilities = [
            "code_analysis",
            "security_analysis", 
            "quality_assessment",
            "complexity_measurement"
        ]
    
    async def process_task(self, task_type: str, input_data: Dict[str, Any]) -> Any:
        """Process analysis tasks."""
        
        if task_type == "code_analysis":
            return await self._analyze_code(
                input_data.get("code", ""),
                input_data.get("file_path")
            )
        else:
            raise ValueError(f"Unsupported task type: {task_type}")
    
    async def _analyze_code(self, code: str, file_path: str = None) -> Dict[str, Any]:
        """
        Perform comprehensive code analysis.
        
        Args:
            code: Source code to analyze
            file_path: Optional file path for context
            
        Returns:
            Analysis results with metrics and recommendations
        """
        
        if not code.strip():
            return {
                "quality_score": 0,
                "rating": "No Code",
                "metrics": {},
                "security_issues": [],
                "recommendations": ["No code provided for analysis"]
            }
        
        try:
            # Parse AST for structural analysis
            tree = ast.parse(code)
            
            # Calculate metrics
            metrics = self._calculate_metrics(code, tree)
            
            # Security analysis
            security_issues = self._analyze_security(code)
            
            # Generate quality score and recommendations
            quality_score = self._calculate_quality_score(metrics, security_issues)
            recommendations = self._generate_recommendations(metrics, security_issues)
            
            return {
                "quality_score": quality_score,
                "rating": self._get_quality_rating(quality_score),
                "metrics": metrics,
                "security_issues": security_issues,
                "recommendations": recommendations,
                "file_path": file_path
            }
            
        except SyntaxError as e:
            return {
                "quality_score": 0,
                "rating": "Syntax Error",
                "metrics": {"syntax_error": str(e)},
                "security_issues": [],
                "recommendations": [f"Fix syntax error: {e}"]
            }
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            return {
                "quality_score": 0,
                "rating": "Analysis Failed",
                "metrics": {"error": str(e)},
                "security_issues": [],
                "recommendations": ["Unable to analyze code due to error"]
            }
    
    def _calculate_metrics(self, code: str, tree: ast.AST) -> Dict[str, Any]:
        """Calculate code metrics."""
        
        lines = code.split('\n')
        
        # Basic metrics
        lines_of_code = len([line for line in lines if line.strip() and not line.strip().startswith('#')])
        total_lines = len(lines)
        comment_lines = len([line for line in lines if line.strip().startswith('#')])
        
        # AST-based metrics
        function_count = len([node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)])
        class_count = len([node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)])
        
        # Complexity estimate (simplified)
        complexity = self._estimate_complexity(tree)
        
        return {
            "lines_of_code": lines_of_code,
            "total_lines": total_lines,
            "comment_lines": comment_lines,
            "function_count": function_count,
            "class_count": class_count,
            "complexity": complexity,
            "comment_ratio": comment_lines / max(total_lines, 1)
        }
    
    def _estimate_complexity(self, tree: ast.AST) -> int:
        """Estimate cyclomatic complexity."""
        
        complexity = 1  # Base complexity
        
        for node in ast.walk(tree):
            # Control flow increases complexity
            if isinstance(node, (ast.If, ast.While, ast.For, ast.Try)):
                complexity += 1
            elif isinstance(node, ast.ExceptHandler):
                complexity += 1
            elif isinstance(node, (ast.And, ast.Or)):
                complexity += 1
        
        return complexity
    
    def _analyze_security(self, code: str) -> List[Dict[str, str]]:
        """Analyze code for security issues."""
        
        issues = []
        
        # Check for hardcoded secrets
        secret_patterns = [
            (r'password\s*=\s*["\'][^"\']+["\']', "Hardcoded password detected"),
            (r'api_key\s*=\s*["\'][^"\']+["\']', "Hardcoded API key detected"),
            (r'secret\s*=\s*["\'][^"\']+["\']', "Hardcoded secret detected"),
            (r'token\s*=\s*["\'][^"\']+["\']', "Hardcoded token detected")
        ]
        
        for pattern, message in secret_patterns:
            if re.search(pattern, code, re.IGNORECASE):
                issues.append({
                    "type": "security",
                    "severity": "high",
                    "message": message
                })
        
        # Check for SQL injection risks
        sql_patterns = [
            (r'execute\([^)]*%', "Potential SQL injection via string formatting"),
            (r'cursor\.execute\([^)]*\+', "Potential SQL injection via string concatenation")
        ]
        
        for pattern, message in sql_patterns:
            if re.search(pattern, code):
                issues.append({
                    "type": "security",
                    "severity": "high", 
                    "message": message
                })
        
        # Check for command injection risks
        if re.search(r'os\.system\([^)]*\+', code):
            issues.append({
                "type": "security",
                "severity": "high",
                "message": "Potential command injection via os.system"
            })
        
        return issues
    
    def _calculate_quality_score(self, metrics: Dict[str, Any], security_issues: List[Dict]) -> float:
        """Calculate overall quality score (0-10)."""
        
        score = 10.0
        
        # Penalize high complexity
        complexity = metrics.get("complexity", 0)
        if complexity > 20:
            score -= 3.0
        elif complexity > 10:
            score -= 1.5
        
        # Penalize lack of comments
        comment_ratio = metrics.get("comment_ratio", 0)
        if comment_ratio < 0.1:
            score -= 1.0
        
        # Penalize security issues
        high_severity_issues = len([issue for issue in security_issues if issue.get("severity") == "high"])
        score -= high_severity_issues * 2.0
        
        # Ensure score is in valid range
        return max(0.0, min(10.0, score))
    
    def _get_quality_rating(self, score: float) -> str:
        """Convert numeric score to rating."""
        
        if score >= 9.0:
            return "Excellent"
        elif score >= 7.0:
            return "Good"
        elif score >= 5.0:
            return "Fair"
        elif score >= 3.0:
            return "Poor"
        else:
            return "Critical"
    
    def _generate_recommendations(self, metrics: Dict[str, Any], security_issues: List[Dict]) -> List[str]:
        """Generate improvement recommendations."""
        
        recommendations = []
        
        # Complexity recommendations
        complexity = metrics.get("complexity", 0)
        if complexity > 20:
            recommendations.append("Consider breaking down complex functions into smaller ones")
        elif complexity > 10:
            recommendations.append("Review function complexity and consider refactoring")
        
        # Comment recommendations
        comment_ratio = metrics.get("comment_ratio", 0)
        if comment_ratio < 0.1:
            recommendations.append("Add more comments to improve code documentation")
        
        # Security recommendations
        if security_issues:
            recommendations.append("Address security issues found in the code")
            for issue in security_issues[:3]:  # Limit to top 3 issues
                recommendations.append(f"- {issue['message']}")
        
        # Function/class structure recommendations
        if metrics.get("function_count", 0) == 0 and metrics.get("lines_of_code", 0) > 10:
            recommendations.append("Consider organizing code into functions")
        
        if not recommendations:
            recommendations.append("Code quality looks good!")
        
        return recommendations
