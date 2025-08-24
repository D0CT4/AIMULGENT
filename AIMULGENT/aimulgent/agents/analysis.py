"""
AIMULGENT Analysis Agent
Code analysis agent focusing on quality assessment and metrics.
Follows KISS principle with simplified implementation under 500 lines.
"""

import ast
import re
from typing import Any, Dict, List, Optional

from aimulgent.agents.base import BaseAgent


class AnalysisAgent(BaseAgent):
    """
    Code analysis agent for quality assessment and metrics calculation.
    
    Capabilities:
    - Code structure analysis
    - Complexity calculation
    - Basic security issue detection
    - Code quality scoring
    """
    
    def __init__(self):
        super().__init__(
            agent_id="analysis",
            capabilities=["code_analysis", "security_analysis", "quality_assessment"]
        )
        
        # Security patterns to detect
        self.security_patterns = {
            "sql_injection": [
                r'execute\s*\(\s*["\'].*%.*["\']',
                r'cursor\.execute\s*\([^)]*%[^)]*\)',
            ],
            "hardcoded_secrets": [
                r'password\s*=\s*["\'][^"\']{8,}["\']',
                r'api_key\s*=\s*["\'][^"\']{16,}["\']',
            ],
            "command_injection": [
                r'os\.system\s*\(',
                r'subprocess\.call\s*\(',
            ]
        }
    
    async def process_task(
        self, 
        task_type: str, 
        input_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process analysis tasks."""
        
        if not self.can_handle(task_type):
            raise ValueError(f"Cannot handle task type: {task_type}")
        
        code = input_data.get("code", "")
        file_path = input_data.get("file_path")
        
        if task_type == "code_analysis":
            result = self._analyze_code_structure(code)
        elif task_type == "security_analysis":
            result = self._analyze_security(code)
        elif task_type == "quality_assessment":
            result = self._assess_quality(code)
        else:
            raise ValueError(f"Unknown task type: {task_type}")
        
        self.tasks_processed += 1
        self.logger.info(f"Processed {task_type} task for {file_path or 'code snippet'}")
        
        return result
    
    def _analyze_code_structure(self, code: str) -> Dict[str, Any]:
        """Analyze code structure using AST."""
        
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return {
                "error": f"Syntax error: {e}",
                "valid": False
            }
        
        # Basic structure analysis
        structure = {
            "valid": True,
            "lines_of_code": len(code.split('\n')),
            "functions": [],
            "classes": [],
            "imports": [],
            "complexity_score": 0
        }
        
        # Walk through AST
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                func_info = {
                    "name": node.name,
                    "line_number": node.lineno,
                    "parameters": len(node.args.args),
                    "complexity": self._calculate_complexity(node)
                }
                structure["functions"].append(func_info)
                structure["complexity_score"] += func_info["complexity"]
            
            elif isinstance(node, ast.ClassDef):
                class_info = {
                    "name": node.name,
                    "line_number": node.lineno,
                    "methods": len([n for n in node.body if isinstance(n, ast.FunctionDef)])
                }
                structure["classes"].append(class_info)
            
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        structure["imports"].append(alias.name)
                else:
                    module = node.module or ""
                    for alias in node.names:
                        structure["imports"].append(f"{module}.{alias.name}")
        
        # Calculate average complexity
        func_count = len(structure["functions"])
        if func_count > 0:
            structure["average_complexity"] = structure["complexity_score"] / func_count
        else:
            structure["average_complexity"] = 0
        
        return structure
    
    def _calculate_complexity(self, node: ast.FunctionDef) -> int:
        """Calculate cyclomatic complexity for a function."""
        
        complexity = 1  # Base complexity
        
        for child in ast.walk(node):
            # Decision points increase complexity
            if isinstance(child, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1
        
        return complexity
    
    def _analyze_security(self, code: str) -> Dict[str, Any]:
        """Analyze code for basic security issues."""
        
        issues = []
        
        lines = code.split('\n')
        for i, line in enumerate(lines, 1):
            for issue_type, patterns in self.security_patterns.items():
                for pattern in patterns:
                    if re.search(pattern, line, re.IGNORECASE):
                        issues.append({
                            "type": issue_type,
                            "line": i,
                            "description": f"Potential {issue_type.replace('_', ' ')} detected",
                            "severity": self._get_severity(issue_type),
                            "line_content": line.strip()
                        })
        
        return {
            "issues_found": len(issues),
            "issues": issues,
            "overall_risk": self._calculate_risk_level(issues)
        }
    
    def _assess_quality(self, code: str) -> Dict[str, Any]:
        """Assess overall code quality."""
        
        # Get structure analysis
        structure = self._analyze_code_structure(code)
        
        if not structure.get("valid", False):
            return {"quality_score": 0.0, "error": structure.get("error")}
        
        # Get security analysis  
        security = self._analyze_security(code)
        
        # Calculate quality metrics
        metrics = {
            "lines_of_code": structure["lines_of_code"],
            "complexity": structure.get("average_complexity", 0),
            "function_count": len(structure["functions"]),
            "class_count": len(structure["classes"]),
            "security_issues": security["issues_found"]
        }
        
        # Calculate overall quality score (0-10)
        quality_score = 10.0
        
        # Penalize high complexity
        if metrics["complexity"] > 10:
            quality_score -= 2.0
        elif metrics["complexity"] > 5:
            quality_score -= 1.0
        
        # Penalize security issues
        quality_score -= min(security["issues_found"] * 0.5, 3.0)
        
        # Penalize very long files
        if metrics["lines_of_code"] > 500:
            quality_score -= 1.0
        
        # Ensure score is within bounds
        quality_score = max(0.0, min(10.0, quality_score))
        
        # Generate recommendations
        recommendations = self._generate_recommendations(metrics, security)
        
        return {
            "quality_score": round(quality_score, 1),
            "metrics": metrics,
            "security": security,
            "recommendations": recommendations,
            "rating": self._get_quality_rating(quality_score)
        }
    
    def _get_severity(self, issue_type: str) -> str:
        """Get severity level for security issue type."""
        high_severity = ["sql_injection", "command_injection"]
        medium_severity = ["hardcoded_secrets"]
        
        if issue_type in high_severity:
            return "high"
        elif issue_type in medium_severity:
            return "medium"
        return "low"
    
    def _calculate_risk_level(self, issues: List[Dict[str, Any]]) -> str:
        """Calculate overall risk level based on issues."""
        if not issues:
            return "low"
        
        high_count = sum(1 for issue in issues if issue["severity"] == "high")
        medium_count = sum(1 for issue in issues if issue["severity"] == "medium")
        
        if high_count > 0:
            return "high"
        elif medium_count > 2:
            return "high"
        elif medium_count > 0:
            return "medium"
        return "low"
    
    def _generate_recommendations(
        self, 
        metrics: Dict[str, Any], 
        security: Dict[str, Any]
    ) -> List[str]:
        """Generate code improvement recommendations."""
        
        recommendations = []
        
        # Complexity recommendations
        if metrics["complexity"] > 10:
            recommendations.append("Consider breaking down complex functions into smaller ones")
        
        # File length recommendations
        if metrics["lines_of_code"] > 500:
            recommendations.append("File is quite long - consider splitting into multiple modules")
        
        # Security recommendations
        if security["issues_found"] > 0:
            recommendations.append(f"Address {security['issues_found']} security issues found")
        
        # General recommendations
        if metrics["function_count"] == 0:
            recommendations.append("Consider organizing code into functions for better structure")
        
        if not recommendations:
            recommendations.append("Code structure looks good - consider adding tests if not present")
        
        return recommendations
    
    def _get_quality_rating(self, score: float) -> str:
        """Convert numeric score to quality rating."""
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