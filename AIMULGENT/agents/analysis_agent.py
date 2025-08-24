"""
Analysis Agent - Deep Code Analysis and Understanding
Uses Code-specialized Transformers (CodeT5/CodeBERT) for comprehensive code analysis
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
import ast
import re
from pathlib import Path
import subprocess
import json
from transformers import AutoTokenizer, AutoModel, RobertaTokenizer, T5ForConditionalGeneration
import tree_sitter
from tree_sitter import Language, Parser
import networkx as nx
from collections import defaultdict, deque
import radon.complexity as cc
from radon.metrics import h_visit, mi_visit
import bandit
from bandit.core import manager as bandit_manager
import pylint.lint
from io import StringIO
import sys

@dataclass
class CodeMetrics:
    """Comprehensive code metrics"""
    cyclomatic_complexity: float
    cognitive_complexity: float
    halstead_metrics: Dict[str, float]
    maintainability_index: float
    lines_of_code: int
    logical_lines: int
    comment_ratio: float
    duplication_ratio: float

@dataclass
class SecurityIssue:
    """Security vulnerability information"""
    severity: str  # high, medium, low
    issue_type: str
    line_number: int
    description: str
    recommendation: str
    confidence: float

@dataclass
class CodeSmell:
    """Code smell detection result"""
    smell_type: str
    location: str
    severity: str
    description: str
    suggestion: str
    impact: str

@dataclass
class AnalysisResult:
    """Complete analysis results"""
    metrics: CodeMetrics
    security_issues: List[SecurityIssue]
    code_smells: List[CodeSmell]
    architecture_patterns: List[str]
    dependencies: Dict[str, List[str]]
    complexity_hotspots: List[Dict[str, Any]]
    quality_score: float
    recommendations: List[str]

class CodeTransformer(nn.Module):
    """Code-specialized transformer for deep code understanding"""
    
    def __init__(self, model_name="microsoft/codebert-base", max_length=512):
        super().__init__()
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name)
        self.max_length = max_length
        
        # Task-specific heads
        self.complexity_head = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        
        self.vulnerability_head = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 10)  # 10 common vulnerability types
        )
        
        self.quality_head = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        
    def forward(self, code_text: str):
        # Tokenize
        inputs = self.tokenizer(
            code_text,
            max_length=self.max_length,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
        
        # Encode
        with torch.no_grad():
            outputs = self.encoder(**inputs)
            pooled_output = outputs.pooler_output
        
        # Task predictions
        complexity_score = self.complexity_head(pooled_output)
        vulnerability_logits = self.vulnerability_head(pooled_output)
        quality_score = self.quality_head(pooled_output)
        
        return {
            'embedding': pooled_output,
            'complexity': complexity_score,
            'vulnerabilities': vulnerability_logits,
            'quality': quality_score
        }

class ASTAnalyzer:
    """Advanced AST-based code analysis"""
    
    def __init__(self):
        self.node_visitors = {
            ast.FunctionDef: self._visit_function,
            ast.ClassDef: self._visit_class,
            ast.Import: self._visit_import,
            ast.ImportFrom: self._visit_import_from,
            ast.For: self._visit_loop,
            ast.While: self._visit_loop,
            ast.If: self._visit_conditional,
            ast.Try: self._visit_exception_handler
        }
        
    def analyze_ast(self, code: str) -> Dict[str, Any]:
        """Comprehensive AST analysis"""
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return {'error': f'Syntax error: {e}', 'valid': False}
        
        analysis = {
            'valid': True,
            'functions': [],
            'classes': [],
            'imports': [],
            'control_structures': {'loops': 0, 'conditionals': 0, 'try_blocks': 0},
            'nesting_depth': 0,
            'dependencies': defaultdict(list),
            'call_graph': nx.DiGraph()
        }
        
        # Walk through AST
        for node in ast.walk(tree):
            node_type = type(node)
            if node_type in self.node_visitors:
                self.node_visitors[node_type](node, analysis)
        
        # Calculate additional metrics
        analysis['total_nodes'] = len(list(ast.walk(tree)))
        analysis['max_nesting'] = self._calculate_max_nesting(tree)
        
        return analysis
    
    def _visit_function(self, node: ast.FunctionDef, analysis: Dict):
        """Analyze function definitions"""
        func_info = {
            'name': node.name,
            'line_number': node.lineno,
            'parameters': len(node.args.args),
            'return_annotation': ast.unparse(node.returns) if node.returns else None,
            'decorators': [ast.unparse(d) for d in node.decorator_list],
            'docstring': ast.get_docstring(node),
            'complexity': self._calculate_function_complexity(node)
        }
        analysis['functions'].append(func_info)
        
        # Add to call graph
        analysis['call_graph'].add_node(node.name, type='function')
    
    def _visit_class(self, node: ast.ClassDef, analysis: Dict):
        """Analyze class definitions"""
        class_info = {
            'name': node.name,
            'line_number': node.lineno,
            'bases': [ast.unparse(base) for base in node.bases],
            'methods': [n.name for n in node.body if isinstance(n, ast.FunctionDef)],
            'attributes': self._extract_attributes(node),
            'decorators': [ast.unparse(d) for d in node.decorator_list],
            'docstring': ast.get_docstring(node)
        }
        analysis['classes'].append(class_info)
        
        # Add to call graph
        analysis['call_graph'].add_node(node.name, type='class')
    
    def _visit_import(self, node: ast.Import, analysis: Dict):
        """Analyze import statements"""
        for alias in node.names:
            import_info = {
                'module': alias.name,
                'alias': alias.asname,
                'line_number': node.lineno
            }
            analysis['imports'].append(import_info)
            analysis['dependencies']['imports'].append(alias.name)
    
    def _visit_import_from(self, node: ast.ImportFrom, analysis: Dict):
        """Analyze from-import statements"""
        module = node.module or ""
        for alias in node.names:
            import_info = {
                'module': f"{module}.{alias.name}",
                'alias': alias.asname,
                'line_number': node.lineno,
                'from_module': module
            }
            analysis['imports'].append(import_info)
            analysis['dependencies']['from_imports'].append(f"{module}.{alias.name}")
    
    def _visit_loop(self, node: Union[ast.For, ast.While], analysis: Dict):
        """Analyze loop structures"""
        analysis['control_structures']['loops'] += 1
    
    def _visit_conditional(self, node: ast.If, analysis: Dict):
        """Analyze conditional statements"""
        analysis['control_structures']['conditionals'] += 1
    
    def _visit_exception_handler(self, node: ast.Try, analysis: Dict):
        """Analyze try-except blocks"""
        analysis['control_structures']['try_blocks'] += 1
    
    def _calculate_function_complexity(self, node: ast.FunctionDef) -> int:
        """Calculate cyclomatic complexity for function"""
        complexity = 1  # Base complexity
        
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For)):
                complexity += 1
            elif isinstance(child, ast.ExceptHandler):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1
            elif isinstance(child, ast.AsyncFor):
                complexity += 1
        
        return complexity
    
    def _extract_attributes(self, node: ast.ClassDef) -> List[str]:
        """Extract class attributes"""
        attributes = []
        
        for child in node.body:
            if isinstance(child, ast.Assign):
                for target in child.targets:
                    if isinstance(target, ast.Name):
                        attributes.append(target.id)
        
        return attributes
    
    def _calculate_max_nesting(self, tree: ast.AST) -> int:
        """Calculate maximum nesting depth"""
        def visit_node(node, depth=0):
            max_depth = depth
            nesting_nodes = (ast.If, ast.For, ast.While, ast.With, ast.Try, 
                           ast.FunctionDef, ast.ClassDef, ast.AsyncWith, ast.AsyncFor)
            
            for child in ast.iter_child_nodes(node):
                if isinstance(child, nesting_nodes):
                    child_depth = visit_node(child, depth + 1)
                    max_depth = max(max_depth, child_depth)
                else:
                    child_depth = visit_node(child, depth)
                    max_depth = max(max_depth, child_depth)
            
            return max_depth
        
        return visit_node(tree)

class SecurityAnalyzer:
    """Security vulnerability analysis"""
    
    def __init__(self):
        self.vulnerability_patterns = {
            'sql_injection': [
                r'execute\s*\(\s*["\'].*%.*["\']',
                r'cursor\.execute\s*\([^)]*%[^)]*\)',
                r'query\s*=.*%.*'
            ],
            'xss': [
                r'innerHTML\s*=.*\+',
                r'document\.write\s*\(',
                r'eval\s*\('
            ],
            'path_traversal': [
                r'open\s*\([^)]*\.\.[^)]*\)',
                r'file\s*=.*\.\.'
            ],
            'hardcoded_secrets': [
                r'password\s*=\s*["\'][^"\']{8,}["\']',
                r'api_key\s*=\s*["\'][^"\']{16,}["\']',
                r'secret\s*=\s*["\'][^"\']{16,}["\']'
            ],
            'weak_crypto': [
                r'md5\s*\(',
                r'sha1\s*\(',
                r'DES\s*\('
            ]
        }
        
    def analyze_security(self, code: str) -> List[SecurityIssue]:
        """Analyze code for security vulnerabilities"""
        issues = []
        
        lines = code.split('\n')
        
        for i, line in enumerate(lines, 1):
            for vuln_type, patterns in self.vulnerability_patterns.items():
                for pattern in patterns:
                    if re.search(pattern, line, re.IGNORECASE):
                        severity = self._determine_severity(vuln_type)
                        
                        issue = SecurityIssue(
                            severity=severity,
                            issue_type=vuln_type,
                            line_number=i,
                            description=self._get_vulnerability_description(vuln_type),
                            recommendation=self._get_security_recommendation(vuln_type),
                            confidence=0.8  # Pattern-based detection
                        )
                        issues.append(issue)
        
        return issues
    
    def _determine_severity(self, vuln_type: str) -> str:
        """Determine vulnerability severity"""
        high_severity = ['sql_injection', 'xss', 'hardcoded_secrets']
        medium_severity = ['path_traversal', 'weak_crypto']
        
        if vuln_type in high_severity:
            return 'high'
        elif vuln_type in medium_severity:
            return 'medium'
        else:
            return 'low'
    
    def _get_vulnerability_description(self, vuln_type: str) -> str:
        """Get vulnerability description"""
        descriptions = {
            'sql_injection': 'Potential SQL injection vulnerability detected',
            'xss': 'Potential cross-site scripting vulnerability detected',
            'path_traversal': 'Potential path traversal vulnerability detected',
            'hardcoded_secrets': 'Hardcoded credentials or secrets detected',
            'weak_crypto': 'Weak cryptographic algorithm detected'
        }
        return descriptions.get(vuln_type, 'Security issue detected')
    
    def _get_security_recommendation(self, vuln_type: str) -> str:
        """Get security recommendation"""
        recommendations = {
            'sql_injection': 'Use parameterized queries or prepared statements',
            'xss': 'Sanitize user input and use safe DOM manipulation methods',
            'path_traversal': 'Validate and sanitize file paths, use whitelist approach',
            'hardcoded_secrets': 'Use environment variables or secure key management',
            'weak_crypto': 'Use strong cryptographic algorithms (AES, SHA-256+)'
        }
        return recommendations.get(vuln_type, 'Review and fix security issue')

class CodeSmellDetector:
    """Detect code smells and anti-patterns"""
    
    def __init__(self):
        self.smell_detectors = {
            'long_method': self._detect_long_method,
            'large_class': self._detect_large_class,
            'long_parameter_list': self._detect_long_parameter_list,
            'duplicate_code': self._detect_duplicate_code,
            'dead_code': self._detect_dead_code,
            'magic_numbers': self._detect_magic_numbers,
            'deep_nesting': self._detect_deep_nesting
        }
    
    def detect_smells(self, code: str, ast_analysis: Dict[str, Any]) -> List[CodeSmell]:
        """Detect various code smells"""
        smells = []
        
        for smell_type, detector in self.smell_detectors.items():
            try:
                detected_smells = detector(code, ast_analysis)
                smells.extend(detected_smells)
            except Exception as e:
                print(f"Error detecting {smell_type}: {e}")
        
        return smells
    
    def _detect_long_method(self, code: str, ast_analysis: Dict) -> List[CodeSmell]:
        """Detect methods that are too long"""
        smells = []
        
        for func in ast_analysis.get('functions', []):
            # Estimate function length (simple heuristic)
            if func.get('complexity', 0) > 10:
                smell = CodeSmell(
                    smell_type='long_method',
                    location=f"Function '{func['name']}' at line {func['line_number']}",
                    severity='medium',
                    description=f"Method '{func['name']}' has high complexity ({func.get('complexity', 0)})",
                    suggestion='Break down into smaller, more focused methods',
                    impact='Reduced readability and maintainability'
                )
                smells.append(smell)
        
        return smells
    
    def _detect_large_class(self, code: str, ast_analysis: Dict) -> List[CodeSmell]:
        """Detect classes that are too large"""
        smells = []
        
        for cls in ast_analysis.get('classes', []):
            method_count = len(cls.get('methods', []))
            if method_count > 20:
                smell = CodeSmell(
                    smell_type='large_class',
                    location=f"Class '{cls['name']}' at line {cls['line_number']}",
                    severity='high',
                    description=f"Class '{cls['name']}' has too many methods ({method_count})",
                    suggestion='Consider splitting into multiple classes or using composition',
                    impact='Violates single responsibility principle'
                )
                smells.append(smell)
        
        return smells
    
    def _detect_long_parameter_list(self, code: str, ast_analysis: Dict) -> List[CodeSmell]:
        """Detect functions with too many parameters"""
        smells = []
        
        for func in ast_analysis.get('functions', []):
            param_count = func.get('parameters', 0)
            if param_count > 5:
                smell = CodeSmell(
                    smell_type='long_parameter_list',
                    location=f"Function '{func['name']}' at line {func['line_number']}",
                    severity='medium',
                    description=f"Function has too many parameters ({param_count})",
                    suggestion='Use parameter objects or configuration dictionaries',
                    impact='Decreased function usability and increased coupling'
                )
                smells.append(smell)
        
        return smells
    
    def _detect_duplicate_code(self, code: str, ast_analysis: Dict) -> List[CodeSmell]:
        """Detect duplicate code blocks"""
        smells = []
        lines = code.split('\n')
        
        # Simple duplicate detection (can be enhanced)
        line_hashes = {}
        duplicates = []
        
        for i, line in enumerate(lines):
            stripped = line.strip()
            if len(stripped) > 10:  # Ignore short lines
                if stripped in line_hashes:
                    duplicates.append((i + 1, line_hashes[stripped]))
                else:
                    line_hashes[stripped] = i + 1
        
        if len(duplicates) > 5:  # Threshold for duplicate concern
            smell = CodeSmell(
                smell_type='duplicate_code',
                location='Multiple locations',
                severity='medium',
                description=f'Detected {len(duplicates)} potentially duplicate lines',
                suggestion='Extract common code into reusable functions',
                impact='Increased maintenance burden and inconsistency risk'
            )
            smells.append(smell)
        
        return smells
    
    def _detect_dead_code(self, code: str, ast_analysis: Dict) -> List[CodeSmell]:
        """Detect potentially unused code"""
        smells = []
        
        # Simple heuristic: functions not called in call graph
        call_graph = ast_analysis.get('call_graph', nx.DiGraph())
        
        for func in ast_analysis.get('functions', []):
            func_name = func['name']
            if func_name.startswith('_'):  # Private methods might be intentionally unused
                continue
                
            # Check if function has incoming edges in call graph
            if call_graph.in_degree(func_name) == 0:
                smell = CodeSmell(
                    smell_type='dead_code',
                    location=f"Function '{func_name}' at line {func['line_number']}",
                    severity='low',
                    description=f"Function '{func_name}' appears to be unused",
                    suggestion='Remove unused code or verify if it should be called',
                    impact='Code bloat and maintenance overhead'
                )
                smells.append(smell)
        
        return smells
    
    def _detect_magic_numbers(self, code: str, ast_analysis: Dict) -> List[CodeSmell]:
        """Detect magic numbers in code"""
        smells = []
        
        # Pattern to find numeric literals (excluding common ones like 0, 1)
        magic_number_pattern = r'\b(?<![\w.])\d{2,}(?![\w.])\b'
        lines = code.split('\n')
        
        magic_numbers = []
        for i, line in enumerate(lines, 1):
            matches = re.findall(magic_number_pattern, line)
            for match in matches:
                if int(match) not in [0, 1, 10, 100, 1000]:  # Exclude common values
                    magic_numbers.append((i, match))
        
        if len(magic_numbers) > 3:
            smell = CodeSmell(
                smell_type='magic_numbers',
                location='Multiple locations',
                severity='low',
                description=f'Detected {len(magic_numbers)} potential magic numbers',
                suggestion='Replace magic numbers with named constants',
                impact='Reduced code readability and maintainability'
            )
            smells.append(smell)
        
        return smells
    
    def _detect_deep_nesting(self, code: str, ast_analysis: Dict) -> List[CodeSmell]:
        """Detect deeply nested code structures"""
        smells = []
        
        max_nesting = ast_analysis.get('max_nesting', 0)
        if max_nesting > 4:
            smell = CodeSmell(
                smell_type='deep_nesting',
                location='Code structure',
                severity='medium',
                description=f'Maximum nesting depth is {max_nesting} levels',
                suggestion='Refactor nested code using early returns or helper methods',
                impact='Reduced readability and increased cognitive load'
            )
            smells.append(smell)
        
        return smells

class AnalysisAgent:
    """
    Advanced Analysis Agent for comprehensive code analysis
    Combines multiple analysis techniques with deep learning
    """
    
    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        
        # Initialize components
        self.code_transformer = CodeTransformer().to(device)
        self.ast_analyzer = ASTAnalyzer()
        self.security_analyzer = SecurityAnalyzer()
        self.smell_detector = CodeSmellDetector()
        
        # Architecture pattern detection
        self.architecture_patterns = {
            'mvc': ['model', 'view', 'controller'],
            'mvp': ['model', 'view', 'presenter'],
            'mvvm': ['model', 'view', 'viewmodel'],
            'singleton': ['instance', 'singleton'],
            'factory': ['factory', 'create'],
            'observer': ['observer', 'notify', 'subscribe'],
            'strategy': ['strategy', 'algorithm'],
            'decorator': ['decorator', 'wrap']
        }
        
        # Quality thresholds
        self.quality_thresholds = {
            'complexity': 10,
            'nesting': 4,
            'parameters': 5,
            'method_length': 50,
            'class_methods': 20
        }
    
    def analyze_code(self, code: str, file_path: Optional[str] = None) -> AnalysisResult:
        """Comprehensive code analysis"""
        
        # AST Analysis
        ast_analysis = self.ast_analyzer.analyze_ast(code)
        
        if not ast_analysis.get('valid', False):
            return AnalysisResult(
                metrics=CodeMetrics(0, 0, {}, 0, 0, 0, 0, 0),
                security_issues=[],
                code_smells=[],
                architecture_patterns=[],
                dependencies={},
                complexity_hotspots=[],
                quality_score=0.0,
                recommendations=['Fix syntax errors before analysis']
            )
        
        # Neural network analysis
        nn_results = self.code_transformer(code)
        
        # Calculate metrics
        metrics = self._calculate_code_metrics(code, ast_analysis)
        
        # Security analysis
        security_issues = self.security_analyzer.analyze_security(code)
        
        # Code smell detection
        code_smells = self.smell_detector.detect_smells(code, ast_analysis)
        
        # Architecture pattern detection
        patterns = self._detect_architecture_patterns(code, ast_analysis)
        
        # Dependency analysis
        dependencies = self._analyze_dependencies(ast_analysis)
        
        # Complexity hotspots
        hotspots = self._identify_complexity_hotspots(ast_analysis)
        
        # Overall quality score
        quality_score = self._calculate_quality_score(
            metrics, security_issues, code_smells, nn_results
        )
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            metrics, security_issues, code_smells, quality_score
        )
        
        return AnalysisResult(
            metrics=metrics,
            security_issues=security_issues,
            code_smells=code_smells,
            architecture_patterns=patterns,
            dependencies=dependencies,
            complexity_hotspots=hotspots,
            quality_score=quality_score,
            recommendations=recommendations
        )
    
    def _calculate_code_metrics(self, code: str, ast_analysis: Dict) -> CodeMetrics:
        """Calculate comprehensive code metrics"""
        
        lines = code.split('\n')
        total_lines = len(lines)
        logical_lines = len([line for line in lines if line.strip() and not line.strip().startswith('#')])
        comment_lines = len([line for line in lines if line.strip().startswith('#')])
        
        # Cyclomatic complexity
        total_complexity = sum(func.get('complexity', 1) for func in ast_analysis.get('functions', []))
        avg_complexity = total_complexity / max(len(ast_analysis.get('functions', [1])), 1)
        
        # Cognitive complexity (simplified)
        cognitive_complexity = avg_complexity * 1.2  # Approximation
        
        # Halstead metrics (simplified)
        halstead_metrics = {
            'vocabulary': len(set(re.findall(r'\b\w+\b', code))),
            'length': len(re.findall(r'\b\w+\b', code)),
            'volume': 0,  # Would need proper calculation
            'difficulty': 0,
            'effort': 0
        }
        
        # Comment ratio
        comment_ratio = comment_lines / max(total_lines, 1)
        
        # Duplication ratio (simplified)
        unique_lines = len(set(line.strip() for line in lines if line.strip()))
        duplication_ratio = 1 - (unique_lines / max(logical_lines, 1))
        
        # Maintainability index (simplified)
        maintainability = max(0, 171 - 5.2 * np.log(max(logical_lines, 1)) - 0.23 * avg_complexity)
        
        return CodeMetrics(
            cyclomatic_complexity=avg_complexity,
            cognitive_complexity=cognitive_complexity,
            halstead_metrics=halstead_metrics,
            maintainability_index=maintainability,
            lines_of_code=total_lines,
            logical_lines=logical_lines,
            comment_ratio=comment_ratio,
            duplication_ratio=duplication_ratio
        )
    
    def _detect_architecture_patterns(self, code: str, ast_analysis: Dict) -> List[str]:
        """Detect architecture patterns in code"""
        detected_patterns = []
        code_lower = code.lower()
        
        for pattern, keywords in self.architecture_patterns.items():
            matches = sum(1 for keyword in keywords if keyword in code_lower)
            if matches >= len(keywords) // 2:  # At least half the keywords present
                detected_patterns.append(pattern)
        
        # Additional pattern detection based on structure
        classes = ast_analysis.get('classes', [])
        if len(classes) > 0:
            class_names = [cls['name'].lower() for cls in classes]
            
            # MVC pattern
            if any('controller' in name for name in class_names) and \
               any('model' in name for name in class_names):
                detected_patterns.append('mvc')
            
            # Singleton pattern
            for cls in classes:
                methods = cls.get('methods', [])
                if 'get_instance' in methods or '__new__' in methods:
                    detected_patterns.append('singleton')
        
        return list(set(detected_patterns))
    
    def _analyze_dependencies(self, ast_analysis: Dict) -> Dict[str, List[str]]:
        """Analyze code dependencies"""
        dependencies = {
            'internal': [],
            'external': [],
            'standard_library': []
        }
        
        # Standard library modules (partial list)
        standard_modules = {
            'os', 'sys', 'json', 're', 'datetime', 'collections', 'itertools',
            'functools', 'pathlib', 'urllib', 'http', 'socket', 'threading',
            'multiprocessing', 'asyncio', 'logging', 'unittest', 'sqlite3'
        }
        
        for import_info in ast_analysis.get('imports', []):
            module = import_info['module'].split('.')[0]
            
            if module in standard_modules:
                dependencies['standard_library'].append(import_info['module'])
            elif module.startswith('.') or not any(char in module for char in '.-'):
                dependencies['internal'].append(import_info['module'])
            else:
                dependencies['external'].append(import_info['module'])
        
        return dependencies
    
    def _identify_complexity_hotspots(self, ast_analysis: Dict) -> List[Dict[str, Any]]:
        """Identify complexity hotspots in code"""
        hotspots = []
        
        # Function complexity hotspots
        for func in ast_analysis.get('functions', []):
            complexity = func.get('complexity', 1)
            if complexity > self.quality_thresholds['complexity']:
                hotspots.append({
                    'type': 'function',
                    'name': func['name'],
                    'line_number': func['line_number'],
                    'complexity': complexity,
                    'severity': 'high' if complexity > 15 else 'medium',
                    'description': f"Function '{func['name']}' has high complexity ({complexity})"
                })
        
        # Class size hotspots
        for cls in ast_analysis.get('classes', []):
            method_count = len(cls.get('methods', []))
            if method_count > self.quality_thresholds['class_methods']:
                hotspots.append({
                    'type': 'class',
                    'name': cls['name'],
                    'line_number': cls['line_number'],
                    'method_count': method_count,
                    'severity': 'high' if method_count > 30 else 'medium',
                    'description': f"Class '{cls['name']}' has many methods ({method_count})"
                })
        
        # Nesting hotspots
        max_nesting = ast_analysis.get('max_nesting', 0)
        if max_nesting > self.quality_thresholds['nesting']:
            hotspots.append({
                'type': 'nesting',
                'name': 'Code structure',
                'nesting_depth': max_nesting,
                'severity': 'high' if max_nesting > 6 else 'medium',
                'description': f'Deep nesting detected (max depth: {max_nesting})'
            })
        
        return hotspots
    
    def _calculate_quality_score(self, 
                               metrics: CodeMetrics,
                               security_issues: List[SecurityIssue],
                               code_smells: List[CodeSmell],
                               nn_results: Dict) -> float:
        """Calculate overall code quality score (0-1)"""
        
        scores = []
        
        # Complexity score (lower is better)
        complexity_score = 1.0 / (1.0 + metrics.cyclomatic_complexity / 10.0)
        scores.append(complexity_score * 0.25)
        
        # Maintainability score
        maintainability_score = min(metrics.maintainability_index / 171.0, 1.0)
        scores.append(maintainability_score * 0.20)
        
        # Security score (fewer issues is better)
        high_security_issues = len([issue for issue in security_issues if issue.severity == 'high'])
        security_score = 1.0 / (1.0 + high_security_issues)
        scores.append(security_score * 0.25)
        
        # Code smell score
        critical_smells = len([smell for smell in code_smells if smell.severity in ['high', 'medium']])
        smell_score = 1.0 / (1.0 + critical_smells * 0.1)
        scores.append(smell_score * 0.15)
        
        # Comment ratio score
        comment_score = min(metrics.comment_ratio * 5, 1.0)  # Optimal around 20%
        scores.append(comment_score * 0.10)
        
        # Neural network quality prediction
        nn_quality = float(nn_results.get('quality', torch.tensor(0.5)).item())
        scores.append(nn_quality * 0.05)
        
        return sum(scores)
    
    def _generate_recommendations(self,
                                metrics: CodeMetrics,
                                security_issues: List[SecurityIssue],
                                code_smells: List[CodeSmell],
                                quality_score: float) -> List[str]:
        """Generate actionable recommendations"""
        
        recommendations = []
        
        # Quality-based recommendations
        if quality_score < 0.6:
            recommendations.append("Overall code quality is below average - consider comprehensive refactoring")
        
        # Complexity recommendations
        if metrics.cyclomatic_complexity > 10:
            recommendations.append("Reduce cyclomatic complexity by breaking down large functions")
        
        # Maintainability recommendations
        if metrics.maintainability_index < 50:
            recommendations.append("Improve maintainability by reducing complexity and adding documentation")
        
        # Security recommendations
        high_security = [issue for issue in security_issues if issue.severity == 'high']
        if high_security:
            recommendations.append(f"Address {len(high_security)} high-severity security issues immediately")
        
        # Code smell recommendations
        critical_smells = [smell for smell in code_smells if smell.severity in ['high', 'medium']]
        if len(critical_smells) > 5:
            recommendations.append("Refactor code to address multiple code smells detected")
        
        # Comment recommendations
        if metrics.comment_ratio < 0.1:
            recommendations.append("Add more comments and documentation to improve code readability")
        
        # Duplication recommendations
        if metrics.duplication_ratio > 0.2:
            recommendations.append("Eliminate duplicate code by extracting common functionality")
        
        return recommendations

# Usage example
if __name__ == "__main__":
    agent = AnalysisAgent()
    
    sample_code = '''
import os
import sys
from typing import List, Dict

class DataProcessor:
    def __init__(self):
        self.data = []
        self.config = {}
    
    def process_data(self, input_data: List[Dict], threshold: int = 10):
        results = []
        for item in input_data:
            if item['value'] > threshold:
                # This is a complex processing step
                if item['type'] == 'A':
                    if item['priority'] == 'high':
                        if item['status'] == 'active':
                            processed = self.complex_calculation(item)
                            results.append(processed)
        return results
    
    def complex_calculation(self, item):
        # Magic number usage
        return item['value'] * 42 + 123
    
    def get_connection_string(self):
        # Hardcoded secret
        return "postgresql://user:password123@localhost/db"
    
    def execute_query(self, user_input):
        # SQL injection vulnerability
        query = "SELECT * FROM users WHERE id = '%s'" % user_input
        return query
    '''
    
    result = agent.analyze_code(sample_code)
    
    print(f"Quality Score: {result.quality_score:.2f}")
    print(f"Cyclomatic Complexity: {result.metrics.cyclomatic_complexity:.2f}")
    print(f"Security Issues: {len(result.security_issues)}")
    print(f"Code Smells: {len(result.code_smells)}")
    print(f"Architecture Patterns: {result.architecture_patterns}")
    print(f"Recommendations: {len(result.recommendations)}")
    
    for rec in result.recommendations[:3]:
        print(f"- {rec}")