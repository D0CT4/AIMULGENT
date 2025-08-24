"""
Data Agent - Specialized Data Processing and Management
Uses Graph Neural Networks and embedding models for data flow analysis
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
import pandas as pd
import networkx as nx
import json
import sqlite3
import requests
from abc import ABC, abstractmethod
import asyncio
import aiohttp
from sqlalchemy import create_engine, MetaData, inspect
from pymongo import MongoClient
import redis
from elasticsearch import Elasticsearch

@dataclass
class DataFlowNode:
    """Represents a node in data flow graph"""
    id: str
    node_type: str  # source, transformation, sink, computation
    data_schema: Dict[str, str]
    processing_time: float
    dependencies: List[str]
    metadata: Dict[str, Any]

@dataclass
class DataProcessingResult:
    """Results from data processing operations"""
    processed_data: Any
    schema_analysis: Dict[str, Any]
    quality_metrics: Dict[str, float]
    performance_stats: Dict[str, Any]
    recommendations: List[str]

class GraphNeuralNetwork(nn.Module):
    """Graph Neural Network for data relationship modeling"""
    
    def __init__(self, node_features=128, edge_features=64, hidden_dim=256, num_layers=3):
        super().__init__()
        
        self.node_embedding = nn.Linear(node_features, hidden_dim)
        self.edge_embedding = nn.Linear(edge_features, hidden_dim)
        
        # Graph convolution layers
        self.gnn_layers = nn.ModuleList([
            GraphConvLayer(hidden_dim, hidden_dim) for _ in range(num_layers)
        ])
        
        # Output layers
        self.node_classifier = nn.Linear(hidden_dim, 10)  # 10 node types
        self.edge_predictor = nn.Linear(hidden_dim * 2, 5)  # 5 relationship types
        
    def forward(self, node_features, edge_features, edge_index):
        # Embed nodes and edges
        node_emb = self.node_embedding(node_features)
        edge_emb = self.edge_embedding(edge_features)
        
        # Apply GNN layers
        for layer in self.gnn_layers:
            node_emb = layer(node_emb, edge_emb, edge_index)
        
        # Predictions
        node_pred = self.node_classifier(node_emb)
        
        # Edge predictions (concatenate connected nodes)
        edge_pred = self.edge_predictor(
            torch.cat([node_emb[edge_index[0]], node_emb[edge_index[1]]], dim=1)
        )
        
        return node_pred, edge_pred

class GraphConvLayer(nn.Module):
    """Graph Convolution Layer with attention mechanism"""
    
    def __init__(self, in_dim, out_dim):
        super().__init__()
        
        self.message_net = nn.Sequential(
            nn.Linear(in_dim * 2, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim)
        )
        
        self.attention = nn.MultiheadAttention(in_dim, num_heads=8)
        self.norm = nn.LayerNorm(out_dim)
        self.update_net = nn.Sequential(
            nn.Linear(in_dim + out_dim, out_dim),
            nn.ReLU()
        )
        
    def forward(self, node_features, edge_features, edge_index):
        row, col = edge_index
        
        # Message passing
        messages = self.message_net(
            torch.cat([node_features[row], node_features[col]], dim=1)
        )
        
        # Aggregate messages with attention
        aggregated = torch.zeros_like(node_features)
        for i in range(node_features.size(0)):
            mask = (col == i)
            if mask.sum() > 0:
                neighbor_messages = messages[mask]
                if neighbor_messages.size(0) > 0:
                    attended, _ = self.attention(
                        neighbor_messages.unsqueeze(1),
                        neighbor_messages.unsqueeze(1),
                        neighbor_messages.unsqueeze(1)
                    )
                    aggregated[i] = attended.squeeze(1).mean(0)
        
        # Update node features
        updated = self.update_net(torch.cat([node_features, aggregated], dim=1))
        return self.norm(updated)

class DataConnector(ABC):
    """Abstract base class for data connectors"""
    
    @abstractmethod
    async def connect(self) -> bool:
        pass
    
    @abstractmethod
    async def read_data(self, query: str) -> pd.DataFrame:
        pass
    
    @abstractmethod
    async def write_data(self, data: pd.DataFrame, destination: str) -> bool:
        pass
    
    @abstractmethod
    def get_schema(self) -> Dict[str, str]:
        pass

class SQLConnector(DataConnector):
    """SQL Database Connector"""
    
    def __init__(self, connection_string: str):
        self.connection_string = connection_string
        self.engine = None
        
    async def connect(self) -> bool:
        try:
            self.engine = create_engine(self.connection_string)
            return True
        except Exception as e:
            print(f"SQL connection failed: {e}")
            return False
    
    async def read_data(self, query: str) -> pd.DataFrame:
        return pd.read_sql(query, self.engine)
    
    async def write_data(self, data: pd.DataFrame, destination: str) -> bool:
        try:
            data.to_sql(destination, self.engine, if_exists='append', index=False)
            return True
        except Exception as e:
            print(f"SQL write failed: {e}")
            return False
    
    def get_schema(self) -> Dict[str, str]:
        inspector = inspect(self.engine)
        schema = {}
        for table_name in inspector.get_table_names():
            columns = inspector.get_columns(table_name)
            schema[table_name] = {col['name']: str(col['type']) for col in columns}
        return schema

class NoSQLConnector(DataConnector):
    """NoSQL Database Connector (MongoDB)"""
    
    def __init__(self, connection_string: str, database: str):
        self.connection_string = connection_string
        self.database = database
        self.client = None
        self.db = None
        
    async def connect(self) -> bool:
        try:
            self.client = MongoClient(self.connection_string)
            self.db = self.client[self.database]
            return True
        except Exception as e:
            print(f"NoSQL connection failed: {e}")
            return False
    
    async def read_data(self, query: str) -> pd.DataFrame:
        # Convert string query to MongoDB query
        mongo_query = json.loads(query) if isinstance(query, str) else query
        collection_name = mongo_query.pop('collection')
        collection = self.db[collection_name]
        
        cursor = collection.find(mongo_query)
        return pd.DataFrame(list(cursor))
    
    async def write_data(self, data: pd.DataFrame, destination: str) -> bool:
        try:
            collection = self.db[destination]
            records = data.to_dict('records')
            collection.insert_many(records)
            return True
        except Exception as e:
            print(f"NoSQL write failed: {e}")
            return False
    
    def get_schema(self) -> Dict[str, str]:
        schema = {}
        for collection_name in self.db.list_collection_names():
            # Sample documents to infer schema
            sample = list(self.db[collection_name].find().limit(100))
            if sample:
                fields = {}
                for doc in sample:
                    for key, value in doc.items():
                        if key not in fields:
                            fields[key] = type(value).__name__
                schema[collection_name] = fields
        return schema

class APIConnector(DataConnector):
    """REST API Connector"""
    
    def __init__(self, base_url: str, headers: Dict[str, str] = None):
        self.base_url = base_url
        self.headers = headers or {}
        self.session = None
        
    async def connect(self) -> bool:
        self.session = aiohttp.ClientSession(headers=self.headers)
        return True
    
    async def read_data(self, endpoint: str) -> pd.DataFrame:
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        async with self.session.get(url) as response:
            if response.status == 200:
                data = await response.json()
                return pd.json_normalize(data)
            else:
                raise Exception(f"API request failed: {response.status}")
    
    async def write_data(self, data: pd.DataFrame, endpoint: str) -> bool:
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        payload = data.to_dict('records')
        
        async with self.session.post(url, json=payload) as response:
            return response.status in [200, 201, 202]
    
    def get_schema(self) -> Dict[str, str]:
        # Would need OpenAPI/Swagger spec or sample endpoints
        return {"api_endpoints": "various"}

class DataAgent:
    """
    Advanced Data Agent for comprehensive data processing and management
    Uses Graph Neural Networks for data relationship modeling
    """
    
    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        
        # Initialize neural networks
        self.gnn = GraphNeuralNetwork().to(device)
        
        # Data connectors
        self.connectors: Dict[str, DataConnector] = {}
        
        # Data flow graph
        self.data_graph = nx.DiGraph()
        
        # Processing cache
        self.processing_cache = {}
        
        # Quality metrics
        self.quality_metrics = {
            'completeness': self._calculate_completeness,
            'consistency': self._calculate_consistency,
            'accuracy': self._calculate_accuracy,
            'timeliness': self._calculate_timeliness,
            'validity': self._calculate_validity
        }
        
        # Schema registry
        self.schema_registry = {}
        
        # Performance stats
        self.performance_stats = {
            'operations_count': 0,
            'total_processing_time': 0.0,
            'cache_hits': 0,
            'cache_misses': 0,
            'error_count': 0
        }
    
    def add_connector(self, name: str, connector: DataConnector):
        """Add a data connector"""
        self.connectors[name] = connector
    
    async def analyze_data_schema(self, data: Union[pd.DataFrame, Dict, str]) -> Dict[str, Any]:
        """Analyze data schema and structure"""
        
        if isinstance(data, str):
            # Try to parse as JSON
            try:
                data = json.loads(data)
            except:
                # Treat as text data
                return {'type': 'text', 'length': len(data), 'encoding': 'utf-8'}
        
        if isinstance(data, pd.DataFrame):
            schema_analysis = {
                'type': 'dataframe',
                'shape': data.shape,
                'columns': list(data.columns),
                'dtypes': data.dtypes.to_dict(),
                'null_counts': data.isnull().sum().to_dict(),
                'memory_usage': data.memory_usage(deep=True).sum(),
                'categorical_columns': list(data.select_dtypes(include=['object']).columns),
                'numerical_columns': list(data.select_dtypes(include=[np.number]).columns)
            }
            
            # Statistical analysis
            numeric_data = data.select_dtypes(include=[np.number])
            if not numeric_data.empty:
                schema_analysis['statistics'] = numeric_data.describe().to_dict()
            
            return schema_analysis
            
        elif isinstance(data, dict):
            return {
                'type': 'dictionary',
                'keys': list(data.keys()),
                'key_count': len(data),
                'nested_levels': self._calculate_dict_depth(data),
                'value_types': {k: type(v).__name__ for k, v in data.items()}
            }
        
        return {'type': 'unknown', 'raw_type': type(data).__name__}
    
    def build_data_flow_graph(self, sources: List[str], transformations: List[Dict], sinks: List[str]):
        """Build data flow graph for processing pipeline"""
        
        # Clear existing graph
        self.data_graph.clear()
        
        # Add source nodes
        for source in sources:
            self.data_graph.add_node(source, node_type='source')
        
        # Add transformation nodes
        for i, transform in enumerate(transformations):
            transform_id = f"transform_{i}"
            self.data_graph.add_node(transform_id, node_type='transformation', **transform)
            
            # Connect to dependencies
            if 'inputs' in transform:
                for input_node in transform['inputs']:
                    self.data_graph.add_edge(input_node, transform_id)
        
        # Add sink nodes
        for sink in sinks:
            self.data_graph.add_node(sink, node_type='sink')
        
        return self.data_graph
    
    async def process_data_pipeline(self, pipeline_config: Dict[str, Any]) -> DataProcessingResult:
        """Execute data processing pipeline"""
        
        start_time = torch.cuda.Event(enable_timing=True) if self.device == 'cuda' else None
        end_time = torch.cuda.Event(enable_timing=True) if self.device == 'cuda' else None
        
        if start_time:
            start_time.record()
        
        try:
            results = {}
            performance_metrics = {}
            
            # Process nodes in topological order
            processing_order = list(nx.topological_sort(self.data_graph))
            
            for node_id in processing_order:
                node_data = self.data_graph.nodes[node_id]
                node_type = node_data['node_type']
                
                if node_type == 'source':
                    # Load data from source
                    connector_name = node_data.get('connector')
                    query = node_data.get('query')
                    
                    if connector_name in self.connectors:
                        results[node_id] = await self.connectors[connector_name].read_data(query)
                    
                elif node_type == 'transformation':
                    # Apply transformation
                    inputs = [results[pred] for pred in self.data_graph.predecessors(node_id)]
                    transform_func = node_data.get('function')
                    params = node_data.get('parameters', {})
                    
                    if transform_func:
                        results[node_id] = await self._apply_transformation(
                            inputs, transform_func, params
                        )
                
                elif node_type == 'sink':
                    # Save results
                    input_data = [results[pred] for pred in self.data_graph.predecessors(node_id)]
                    connector_name = node_data.get('connector')
                    destination = node_data.get('destination')
                    
                    if connector_name in self.connectors and input_data:
                        success = await self.connectors[connector_name].write_data(
                            input_data[0], destination
                        )
                        results[node_id] = success
            
            # Calculate quality metrics
            final_data = results.get(processing_order[-1])
            quality_metrics = await self._calculate_all_quality_metrics(final_data)
            
            # Schema analysis
            schema_analysis = await self.analyze_data_schema(final_data)
            
            if end_time:
                end_time.record()
                torch.cuda.synchronize()
                processing_time = start_time.elapsed_time(end_time) / 1000.0
            else:
                processing_time = 0.0
            
            performance_stats = {
                'processing_time': processing_time,
                'nodes_processed': len(processing_order),
                'memory_peak': torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
            }
            
            # Generate recommendations
            recommendations = self._generate_recommendations(
                quality_metrics, performance_stats, schema_analysis
            )
            
            self.performance_stats['operations_count'] += 1
            self.performance_stats['total_processing_time'] += processing_time
            
            return DataProcessingResult(
                processed_data=final_data,
                schema_analysis=schema_analysis,
                quality_metrics=quality_metrics,
                performance_stats=performance_stats,
                recommendations=recommendations
            )
            
        except Exception as e:
            self.performance_stats['error_count'] += 1
            raise Exception(f"Data processing failed: {e}")
    
    async def _apply_transformation(self, inputs: List[Any], function: str, params: Dict) -> Any:
        """Apply transformation function to input data"""
        
        # Common transformation functions
        transformations = {
            'merge': lambda dfs: pd.concat(dfs, ignore_index=True),
            'filter': lambda df, condition: df.query(condition),
            'aggregate': lambda df, group_by, agg_func: df.groupby(group_by).agg(agg_func),
            'pivot': lambda df, index, columns, values: df.pivot_table(
                index=index, columns=columns, values=values, fill_value=0
            ),
            'normalize': lambda df: (df - df.mean()) / df.std(),
            'deduplicate': lambda df: df.drop_duplicates(),
            'fill_missing': lambda df, method: df.fillna(method=method) if method else df.dropna()
        }
        
        if function in transformations and inputs:
            if len(inputs) == 1:
                return transformations[function](inputs[0], **params)
            else:
                return transformations[function](inputs, **params)
        
        return inputs[0] if inputs else None
    
    async def _calculate_all_quality_metrics(self, data: Any) -> Dict[str, float]:
        """Calculate all data quality metrics"""
        
        if not isinstance(data, pd.DataFrame):
            return {'overall_quality': 0.5}
        
        metrics = {}
        
        for metric_name, metric_func in self.quality_metrics.items():
            try:
                metrics[metric_name] = metric_func(data)
            except:
                metrics[metric_name] = 0.0
        
        # Overall quality score
        metrics['overall_quality'] = np.mean(list(metrics.values()))
        
        return metrics
    
    def _calculate_completeness(self, data: pd.DataFrame) -> float:
        """Calculate data completeness (% of non-null values)"""
        total_cells = data.size
        non_null_cells = data.count().sum()
        return float(non_null_cells / total_cells) if total_cells > 0 else 0.0
    
    def _calculate_consistency(self, data: pd.DataFrame) -> float:
        """Calculate data consistency score"""
        consistency_scores = []
        
        for column in data.columns:
            if data[column].dtype == 'object':
                # Check string consistency (similar formats)
                unique_patterns = set()
                for value in data[column].dropna():
                    if isinstance(value, str):
                        pattern = ''.join(['A' if c.isalpha() else 'N' if c.isdigit() else 'S' for c in value])
                        unique_patterns.add(pattern)
                
                consistency = 1.0 / len(unique_patterns) if unique_patterns else 1.0
                consistency_scores.append(consistency)
            else:
                # Numerical consistency (low variance in ranges)
                if not data[column].empty:
                    cv = data[column].std() / data[column].mean() if data[column].mean() != 0 else 0
                    consistency = 1.0 / (1.0 + cv)
                    consistency_scores.append(consistency)
        
        return float(np.mean(consistency_scores)) if consistency_scores else 1.0
    
    def _calculate_accuracy(self, data: pd.DataFrame) -> float:
        """Estimate data accuracy based on outliers and anomalies"""
        accuracy_scores = []
        
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        
        for column in numeric_columns:
            if not data[column].empty:
                Q1 = data[column].quantile(0.25)
                Q3 = data[column].quantile(0.75)
                IQR = Q3 - Q1
                
                # Count outliers
                outliers = data[
                    (data[column] < Q1 - 1.5 * IQR) | 
                    (data[column] > Q3 + 1.5 * IQR)
                ][column].count()
                
                accuracy = 1.0 - (outliers / len(data))
                accuracy_scores.append(accuracy)
        
        return float(np.mean(accuracy_scores)) if accuracy_scores else 0.8
    
    def _calculate_timeliness(self, data: pd.DataFrame) -> float:
        """Calculate timeliness score based on timestamp columns"""
        
        # Look for timestamp columns
        timestamp_columns = []
        for column in data.columns:
            if 'time' in column.lower() or 'date' in column.lower():
                try:
                    pd.to_datetime(data[column].iloc[0])
                    timestamp_columns.append(column)
                except:
                    continue
        
        if not timestamp_columns:
            return 0.7  # Neutral score if no timestamps
        
        # Calculate how recent the data is
        timeliness_scores = []
        current_time = pd.Timestamp.now()
        
        for column in timestamp_columns:
            try:
                timestamps = pd.to_datetime(data[column])
                avg_age = (current_time - timestamps.max()).total_seconds() / (24 * 3600)  # days
                timeliness = 1.0 / (1.0 + avg_age / 30)  # Decay over 30 days
                timeliness_scores.append(timeliness)
            except:
                continue
        
        return float(np.mean(timeliness_scores)) if timeliness_scores else 0.7
    
    def _calculate_validity(self, data: pd.DataFrame) -> float:
        """Calculate validity score based on data format compliance"""
        validity_scores = []
        
        for column in data.columns:
            if data[column].dtype == 'object':
                # Check for common patterns (emails, phones, etc.)
                valid_count = 0
                total_count = data[column].dropna().count()
                
                if 'email' in column.lower():
                    # Simple email validation
                    valid_count = data[column].dropna().str.contains(r'^[^@]+@[^@]+\.[^@]+$').sum()
                elif 'phone' in column.lower():
                    # Simple phone validation
                    valid_count = data[column].dropna().str.contains(r'^\+?[\d\s\-\(\)]+$').sum()
                else:
                    # General string validity (non-empty, reasonable length)
                    valid_strings = data[column].dropna()
                    valid_count = valid_strings[(valid_strings.str.len() > 0) & (valid_strings.str.len() < 1000)].count()
                
                validity = valid_count / total_count if total_count > 0 else 1.0
                validity_scores.append(validity)
        
        return float(np.mean(validity_scores)) if validity_scores else 0.8
    
    def _generate_recommendations(self, 
                                quality_metrics: Dict[str, float], 
                                performance_stats: Dict[str, Any],
                                schema_analysis: Dict[str, Any]) -> List[str]:
        """Generate data processing recommendations"""
        
        recommendations = []
        
        # Quality-based recommendations
        if quality_metrics.get('completeness', 0) < 0.8:
            recommendations.append("Consider data imputation or collection improvement for missing values")
        
        if quality_metrics.get('consistency', 0) < 0.7:
            recommendations.append("Implement data standardization and validation rules")
        
        if quality_metrics.get('accuracy', 0) < 0.75:
            recommendations.append("Review and clean outliers, implement anomaly detection")
        
        # Performance-based recommendations
        if performance_stats.get('processing_time', 0) > 10.0:
            recommendations.append("Consider data partitioning or parallel processing")
        
        # Schema-based recommendations
        if isinstance(schema_analysis, dict) and 'shape' in schema_analysis:
            rows, cols = schema_analysis['shape']
            if cols > 100:
                recommendations.append("Consider feature selection or dimensionality reduction")
            if rows > 1000000:
                recommendations.append("Consider data sampling or streaming processing")
        
        return recommendations
    
    def _calculate_dict_depth(self, d: dict, depth: int = 0) -> int:
        """Calculate maximum depth of nested dictionary"""
        if not isinstance(d, dict) or not d:
            return depth
        return max(self._calculate_dict_depth(value, depth + 1) if isinstance(value, dict) else depth + 1 
                  for value in d.values())
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics"""
        avg_processing_time = (
            self.performance_stats['total_processing_time'] / 
            self.performance_stats['operations_count']
        ) if self.performance_stats['operations_count'] > 0 else 0.0
        
        return {
            **self.performance_stats,
            'average_processing_time': avg_processing_time,
            'cache_hit_rate': (
                self.performance_stats['cache_hits'] / 
                (self.performance_stats['cache_hits'] + self.performance_stats['cache_misses'])
            ) if (self.performance_stats['cache_hits'] + self.performance_stats['cache_misses']) > 0 else 0.0
        }

# Usage example
if __name__ == "__main__":
    import asyncio
    
    async def main():
        agent = DataAgent()
        
        # Add connectors
        sql_connector = SQLConnector("sqlite:///example.db")
        await sql_connector.connect()
        agent.add_connector("database", sql_connector)
        
        # Example data processing
        sample_data = pd.DataFrame({
            'id': range(100),
            'name': [f'User_{i}' for i in range(100)],
            'score': np.random.normal(75, 15, 100),
            'category': np.random.choice(['A', 'B', 'C'], 100)
        })
        
        # Analyze schema
        schema = await agent.analyze_data_schema(sample_data)
        print(f"Schema analysis: {schema}")
        
        # Build simple pipeline
        pipeline_config = {
            'sources': ['input_data'],
            'transformations': [
                {'function': 'filter', 'parameters': {'condition': 'score > 60'}, 'inputs': ['input_data']},
                {'function': 'aggregate', 'parameters': {'group_by': 'category', 'agg_func': 'mean'}, 'inputs': ['filter_0']}
            ],
            'sinks': ['output']
        }
        
        print(f"Performance stats: {agent.get_performance_stats()}")
    
    asyncio.run(main())