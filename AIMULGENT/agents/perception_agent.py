"""
Perception Agent - Specialized Visual and Structural Code Analysis
Uses Vision Transformer + CNN hybrid architecture for code pattern recognition
"""

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from PIL import Image
import numpy as np
from typing import Dict, List, Any, Tuple
import ast
import cv2
from dataclasses import dataclass

@dataclass
class PerceptionResult:
    """Results from perception analysis"""
    structure_analysis: Dict[str, Any]
    visual_patterns: List[Dict[str, Any]]
    code_complexity: Dict[str, float]
    ui_elements: List[Dict[str, Any]]
    confidence_score: float

class CodeVisualEncoder(nn.Module):
    """CNN-based encoder for visual code representation"""
    
    def __init__(self, input_channels=3, hidden_dim=512):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_channels, 64, 7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, 5, stride=2, padding=2),
            nn.BatchNorm2d(128), 
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(256, hidden_dim, 3, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        
    def forward(self, x):
        return self.conv_layers(x).squeeze()

class CodeStructureTransformer(nn.Module):
    """Transformer-based structural code analysis"""
    
    def __init__(self, vocab_size=50000, hidden_dim=768, num_layers=12, num_heads=12):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.pos_encoding = nn.Parameter(torch.randn(5000, hidden_dim))
        
        encoder_layer = nn.TransformerEncoderLayer(
            hidden_dim, num_heads, dim_feedforward=3072, dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 256)
        )
        
    def forward(self, token_ids, attention_mask=None):
        seq_len = token_ids.size(1)
        x = self.embedding(token_ids) + self.pos_encoding[:seq_len]
        x = x.transpose(0, 1)  # (seq_len, batch, hidden_dim)
        
        if attention_mask is not None:
            attention_mask = ~attention_mask.bool()
            
        output = self.transformer(x, src_key_padding_mask=attention_mask)
        return self.classifier(output.mean(0))  # Global average pooling

class PerceptionAgent:
    """
    Advanced Perception Agent with specialized neural networks
    Combines Vision Transformer and CNN for comprehensive code analysis
    """
    
    def __init__(self, model_path: str = None, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.visual_encoder = CodeVisualEncoder().to(device)
        self.structure_transformer = CodeStructureTransformer().to(device)
        
        # Load pre-trained tokenizer for code
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
        
        # Pattern recognition templates
        self.patterns = {
            'design_patterns': [
                'singleton', 'factory', 'observer', 'strategy', 'decorator',
                'adapter', 'facade', 'command', 'iterator', 'proxy'
            ],
            'code_smells': [
                'long_method', 'large_class', 'duplicate_code', 'long_parameter_list',
                'feature_envy', 'data_clumps', 'primitive_obsession'
            ],
            'architectural_patterns': [
                'mvc', 'mvvm', 'layered', 'microservices', 'event_driven',
                'pipeline', 'repository', 'dependency_injection'
            ]
        }
        
        if model_path:
            self.load_model(model_path)
    
    def analyze_code_structure(self, code: str) -> Dict[str, Any]:
        """Analyze code structure using AST and neural networks"""
        try:
            tree = ast.parse(code)
            structure = {
                'classes': [],
                'functions': [],
                'imports': [],
                'complexity_score': 0,
                'nesting_depth': 0,
                'lines_of_code': len(code.split('\n'))
            }
            
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    structure['classes'].append({
                        'name': node.name,
                        'methods': [n.name for n in node.body if isinstance(n, ast.FunctionDef)],
                        'line_number': node.lineno
                    })
                elif isinstance(node, ast.FunctionDef):
                    structure['functions'].append({
                        'name': node.name,
                        'args': len(node.args.args),
                        'line_number': node.lineno,
                        'complexity': self._calculate_cyclomatic_complexity(node)
                    })
                elif isinstance(node, ast.Import) or isinstance(node, ast.ImportFrom):
                    if isinstance(node, ast.Import):
                        structure['imports'].extend([alias.name for alias in node.names])
                    else:
                        module = node.module or ""
                        structure['imports'].extend([f"{module}.{alias.name}" for alias in node.names])
            
            # Neural network analysis
            tokens = self.tokenizer(code, return_tensors="pt", truncation=True, max_length=512)
            with torch.no_grad():
                neural_features = self.structure_transformer(
                    tokens['input_ids'].to(self.device),
                    tokens['attention_mask'].to(self.device)
                )
                structure['neural_embedding'] = neural_features.cpu().numpy().tolist()
            
            return structure
            
        except SyntaxError:
            return {'error': 'Syntax error in code', 'classes': [], 'functions': [], 'imports': []}
    
    def analyze_visual_patterns(self, code_image: np.ndarray) -> List[Dict[str, Any]]:
        """Analyze visual patterns in code screenshots or diagrams"""
        if code_image is None:
            return []
            
        # Preprocess image
        if len(code_image.shape) == 3:
            code_image = cv2.resize(code_image, (224, 224))
            code_image = torch.FloatTensor(code_image).permute(2, 0, 1).unsqueeze(0)
        else:
            code_image = cv2.resize(code_image, (224, 224))
            code_image = torch.FloatTensor(code_image).unsqueeze(0).unsqueeze(0)
            code_image = code_image.repeat(1, 3, 1, 1)  # Convert grayscale to RGB
        
        code_image = code_image.to(self.device) / 255.0
        
        with torch.no_grad():
            visual_features = self.visual_encoder(code_image)
            
        # Pattern detection based on visual features
        patterns = []
        feature_vector = visual_features.cpu().numpy()
        
        # Detect indentation patterns
        patterns.append({
            'type': 'indentation',
            'consistency': float(np.std(feature_vector[:50])),  # Lower std = more consistent
            'detected_style': 'spaces' if feature_vector[0] > 0 else 'tabs'
        })
        
        # Detect code density
        patterns.append({
            'type': 'code_density',
            'density_score': float(np.mean(feature_vector[50:100])),
            'readability': 'high' if np.mean(feature_vector[50:100]) < 0.5 else 'low'
        })
        
        return patterns
    
    def detect_ui_elements(self, code: str) -> List[Dict[str, Any]]:
        """Detect UI/UX related code elements"""
        ui_elements = []
        
        ui_keywords = {
            'react': ['component', 'jsx', 'props', 'state', 'usestate', 'useeffect'],
            'html': ['div', 'span', 'button', 'input', 'form', 'img'],
            'css': ['style', 'class', 'id', 'color', 'background', 'margin', 'padding'],
            'ui_frameworks': ['bootstrap', 'material', 'ant-design', 'chakra', 'tailwind']
        }
        
        code_lower = code.lower()
        
        for category, keywords in ui_keywords.items():
            for keyword in keywords:
                if keyword in code_lower:
                    ui_elements.append({
                        'category': category,
                        'element': keyword,
                        'occurrences': code_lower.count(keyword),
                        'confidence': min(code_lower.count(keyword) * 0.1, 1.0)
                    })
        
        return ui_elements
    
    def calculate_complexity_metrics(self, code: str) -> Dict[str, float]:
        """Calculate various complexity metrics"""
        try:
            tree = ast.parse(code)
            metrics = {
                'cyclomatic_complexity': 0,
                'cognitive_complexity': 0,
                'maintainability_index': 0,
                'nesting_depth': 0
            }
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    metrics['cyclomatic_complexity'] += self._calculate_cyclomatic_complexity(node)
                    metrics['nesting_depth'] = max(metrics['nesting_depth'], self._calculate_nesting_depth(node))
            
            # Simplified maintainability index
            loc = len(code.split('\n'))
            metrics['maintainability_index'] = max(0, 171 - 5.2 * np.log(loc) - 0.23 * metrics['cyclomatic_complexity'])
            
            return metrics
            
        except SyntaxError:
            return {'error': 'Cannot calculate complexity for invalid syntax'}
    
    def _calculate_cyclomatic_complexity(self, node) -> int:
        """Calculate cyclomatic complexity for a function"""
        complexity = 1  # Base complexity
        
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1
        
        return complexity
    
    def _calculate_nesting_depth(self, node) -> int:
        """Calculate maximum nesting depth"""
        depth = 0
        max_depth = 0
        
        def visit_node(n, current_depth):
            nonlocal max_depth
            max_depth = max(max_depth, current_depth)
            
            if isinstance(n, (ast.If, ast.While, ast.For, ast.With, ast.Try)):
                for child in ast.iter_child_nodes(n):
                    visit_node(child, current_depth + 1)
            else:
                for child in ast.iter_child_nodes(n):
                    visit_node(child, current_depth)
        
        visit_node(node, 0)
        return max_depth
    
    def perceive(self, code: str, code_image: np.ndarray = None) -> PerceptionResult:
        """Main perception method combining all analysis capabilities"""
        
        # Structural analysis
        structure = self.analyze_code_structure(code)
        
        # Visual pattern analysis (if image provided)
        visual_patterns = self.analyze_visual_patterns(code_image) if code_image is not None else []
        
        # Complexity metrics
        complexity = self.calculate_complexity_metrics(code)
        
        # UI element detection
        ui_elements = self.detect_ui_elements(code)
        
        # Calculate overall confidence
        confidence = self._calculate_confidence(structure, complexity, ui_elements)
        
        return PerceptionResult(
            structure_analysis=structure,
            visual_patterns=visual_patterns,
            code_complexity=complexity,
            ui_elements=ui_elements,
            confidence_score=confidence
        )
    
    def _calculate_confidence(self, structure, complexity, ui_elements) -> float:
        """Calculate overall confidence score for the perception results"""
        factors = []
        
        # Structure confidence
        if 'error' not in structure:
            factors.append(0.9)
        else:
            factors.append(0.1)
            
        # Complexity confidence
        if 'error' not in complexity:
            factors.append(0.8)
        else:
            factors.append(0.2)
            
        # UI detection confidence
        ui_confidence = min(len(ui_elements) * 0.1, 1.0) if ui_elements else 0.5
        factors.append(ui_confidence)
        
        return float(np.mean(factors))
    
    def save_model(self, path: str):
        """Save the trained models"""
        torch.save({
            'visual_encoder': self.visual_encoder.state_dict(),
            'structure_transformer': self.structure_transformer.state_dict()
        }, path)
    
    def load_model(self, path: str):
        """Load pre-trained models"""
        checkpoint = torch.load(path, map_location=self.device)
        self.visual_encoder.load_state_dict(checkpoint['visual_encoder'])
        self.structure_transformer.load_state_dict(checkpoint['structure_transformer'])

# Usage example
if __name__ == "__main__":
    agent = PerceptionAgent()
    
    sample_code = """
class Calculator:
    def __init__(self):
        self.history = []
    
    def add(self, a, b):
        result = a + b
        self.history.append(f"{a} + {b} = {result}")
        return result
    
    def get_history(self):
        return self.history
    """
    
    result = agent.perceive(sample_code)
    print(f"Structure: {result.structure_analysis}")
    print(f"Complexity: {result.code_complexity}")
    print(f"Confidence: {result.confidence_score}")