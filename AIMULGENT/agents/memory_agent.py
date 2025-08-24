"""
Memory Agent - Episodic and Semantic Memory for Code Context
Uses Transformer with specialized memory architectures and vector databases
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import json
import sqlite3
import faiss
from sentence_transformers import SentenceTransformer
from datetime import datetime
import hashlib
import pickle

@dataclass
class MemoryEntry:
    """Individual memory entry with metadata"""
    id: str
    content: str
    memory_type: str  # episodic, semantic, procedural
    timestamp: str
    context: Dict[str, Any]
    embedding: Optional[np.ndarray] = None
    access_count: int = 0
    importance_score: float = 0.0
    tags: List[str] = None

@dataclass
class MemorySearchResult:
    """Results from memory search"""
    entries: List[MemoryEntry]
    similarity_scores: List[float]
    total_results: int
    search_time: float

class EpisodicMemoryNetwork(nn.Module):
    """Neural network for episodic memory encoding and retrieval"""
    
    def __init__(self, embedding_dim=768, hidden_dim=512, memory_size=10000):
        super().__init__()
        
        # Encoder for new experiences
        self.experience_encoder = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        # Memory consolidation network
        self.consolidation_network = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid()
        )
        
        # Retrieval attention mechanism
        self.retrieval_attention = nn.MultiheadAttention(
            hidden_dim, num_heads=8, dropout=0.1
        )
        
        # Memory buffer
        self.register_buffer('memory_bank', torch.zeros(memory_size, hidden_dim))
        self.register_buffer('memory_timestamps', torch.zeros(memory_size))
        self.register_buffer('memory_importance', torch.zeros(memory_size))
        
        self.memory_size = memory_size
        self.current_idx = 0
        
    def encode_experience(self, embedding: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """Encode new experience into memory representation"""
        experience = self.experience_encoder(embedding)
        
        # Consolidate with context
        if context is not None:
            consolidated = self.consolidation_network(torch.cat([experience, context], dim=-1))
            experience = experience * consolidated
            
        return experience
    
    def store_memory(self, encoded_memory: torch.Tensor, timestamp: float, importance: float):
        """Store encoded memory in memory bank"""
        idx = self.current_idx % self.memory_size
        
        self.memory_bank[idx] = encoded_memory.detach()
        self.memory_timestamps[idx] = timestamp
        self.memory_importance[idx] = importance
        
        self.current_idx += 1
    
    def retrieve_memories(self, query: torch.Tensor, k: int = 10) -> Tuple[torch.Tensor, torch.Tensor]:
        """Retrieve most relevant memories using attention mechanism"""
        query = query.unsqueeze(0)  # Add batch dimension
        memory_bank = self.memory_bank[:min(self.current_idx, self.memory_size)].unsqueeze(0)
        
        attended_memories, attention_weights = self.retrieval_attention(
            query, memory_bank, memory_bank
        )
        
        # Get top-k memories
        _, top_indices = attention_weights.squeeze(0).squeeze(0).topk(k)
        
        return self.memory_bank[top_indices], attention_weights.squeeze(0).squeeze(0)[top_indices]

class SemanticMemoryGraph(nn.Module):
    """Graph neural network for semantic relationships in code"""
    
    def __init__(self, node_dim=768, edge_dim=128, num_relations=50):
        super().__init__()
        
        # Node transformation
        self.node_transform = nn.Sequential(
            nn.Linear(node_dim, node_dim),
            nn.ReLU(),
            nn.Linear(node_dim, node_dim)
        )
        
        # Relation embeddings
        self.relation_embeddings = nn.Embedding(num_relations, edge_dim)
        
        # Message passing
        self.message_net = nn.Sequential(
            nn.Linear(node_dim * 2 + edge_dim, node_dim),
            nn.ReLU(),
            nn.Linear(node_dim, node_dim)
        )
        
        # Aggregation
        self.aggregation = nn.Sequential(
            nn.Linear(node_dim * 2, node_dim),
            nn.ReLU(),
            nn.Linear(node_dim, node_dim)
        )
        
    def forward(self, nodes, edge_index, edge_type):
        """Forward pass through semantic graph"""
        transformed_nodes = self.node_transform(nodes)
        
        # Message passing
        row, col = edge_index
        edge_embeddings = self.relation_embeddings(edge_type)
        
        # Compute messages
        messages = self.message_net(
            torch.cat([nodes[row], nodes[col], edge_embeddings], dim=1)
        )
        
        # Aggregate messages
        aggregated = torch.zeros_like(nodes)
        aggregated = aggregated.scatter_add(0, row.unsqueeze(1).expand(-1, nodes.size(1)), messages)
        
        # Update nodes
        updated_nodes = self.aggregation(torch.cat([nodes, aggregated], dim=1))
        
        return updated_nodes

class MemoryAgent:
    """
    Advanced Memory Agent with episodic and semantic memory capabilities
    Combines neural networks with vector databases for efficient storage and retrieval
    """
    
    def __init__(self, 
                 db_path: str = "memory.db",
                 embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        
        self.device = device
        self.db_path = db_path
        
        # Initialize neural networks
        self.episodic_net = EpisodicMemoryNetwork().to(device)
        self.semantic_net = SemanticMemoryGraph().to(device)
        
        # Initialize embedding model
        self.embedding_model = SentenceTransformer(embedding_model)
        
        # Initialize vector database
        self.vector_db = faiss.IndexFlatIP(384)  # Cosine similarity
        
        # Initialize SQLite database
        self._init_database()
        
        # Memory statistics
        self.stats = {
            'total_memories': 0,
            'episodic_memories': 0,
            'semantic_memories': 0,
            'procedural_memories': 0,
            'average_retrieval_time': 0.0,
            'cache_hits': 0,
            'cache_misses': 0
        }
        
        # LRU Cache for frequently accessed memories
        self.memory_cache = {}
        self.cache_order = []
        self.max_cache_size = 1000
        
    def _init_database(self):
        """Initialize SQLite database for persistent storage"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                memory_type TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                context TEXT,
                access_count INTEGER DEFAULT 0,
                importance_score REAL DEFAULT 0.0,
                tags TEXT,
                embedding BLOB
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS semantic_relationships (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_id TEXT,
                target_id TEXT,
                relationship_type TEXT,
                strength REAL,
                FOREIGN KEY (source_id) REFERENCES memories (id),
                FOREIGN KEY (target_id) REFERENCES memories (id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def _generate_memory_id(self, content: str) -> str:
        """Generate unique ID for memory entry"""
        return hashlib.sha256(f"{content}{datetime.now().isoformat()}".encode()).hexdigest()[:16]
    
    def store_episodic_memory(self, 
                            content: str, 
                            context: Dict[str, Any], 
                            importance: float = 0.5,
                            tags: List[str] = None) -> str:
        """Store episodic memory (specific experiences/interactions)"""
        
        memory_id = self._generate_memory_id(content)
        embedding = self.embedding_model.encode(content)
        
        # Neural encoding
        embedding_tensor = torch.FloatTensor(embedding).to(self.device)
        context_tensor = self._encode_context(context)
        
        encoded_memory = self.episodic_net.encode_experience(embedding_tensor, context_tensor)
        self.episodic_net.store_memory(
            encoded_memory, 
            datetime.now().timestamp(), 
            importance
        )
        
        # Store in vector database
        self.vector_db.add(embedding.reshape(1, -1))
        
        # Store in persistent database
        memory_entry = MemoryEntry(
            id=memory_id,
            content=content,
            memory_type="episodic",
            timestamp=datetime.now().isoformat(),
            context=context,
            embedding=embedding,
            importance_score=importance,
            tags=tags or []
        )
        
        self._persist_memory(memory_entry)
        self.stats['episodic_memories'] += 1
        self.stats['total_memories'] += 1
        
        return memory_id
    
    def store_semantic_memory(self, 
                            content: str, 
                            concept: str,
                            relationships: List[Tuple[str, str, float]] = None,
                            tags: List[str] = None) -> str:
        """Store semantic memory (conceptual knowledge)"""
        
        memory_id = self._generate_memory_id(content)
        embedding = self.embedding_model.encode(content)
        
        # Store relationships if provided
        if relationships:
            for target_concept, relation_type, strength in relationships:
                self._store_semantic_relationship(concept, target_concept, relation_type, strength)
        
        # Store in databases
        self.vector_db.add(embedding.reshape(1, -1))
        
        memory_entry = MemoryEntry(
            id=memory_id,
            content=content,
            memory_type="semantic",
            timestamp=datetime.now().isoformat(),
            context={"concept": concept},
            embedding=embedding,
            tags=tags or []
        )
        
        self._persist_memory(memory_entry)
        self.stats['semantic_memories'] += 1
        self.stats['total_memories'] += 1
        
        return memory_id
    
    def store_procedural_memory(self, 
                              content: str, 
                              procedure_type: str,
                              steps: List[str],
                              conditions: Dict[str, Any] = None,
                              tags: List[str] = None) -> str:
        """Store procedural memory (how-to knowledge)"""
        
        memory_id = self._generate_memory_id(content)
        embedding = self.embedding_model.encode(content)
        
        context = {
            "procedure_type": procedure_type,
            "steps": steps,
            "conditions": conditions or {}
        }
        
        self.vector_db.add(embedding.reshape(1, -1))
        
        memory_entry = MemoryEntry(
            id=memory_id,
            content=content,
            memory_type="procedural",
            timestamp=datetime.now().isoformat(),
            context=context,
            embedding=embedding,
            tags=tags or []
        )
        
        self._persist_memory(memory_entry)
        self.stats['procedural_memories'] += 1
        self.stats['total_memories'] += 1
        
        return memory_id
    
    def retrieve_memories(self, 
                         query: str, 
                         memory_type: Optional[str] = None,
                         k: int = 10,
                         similarity_threshold: float = 0.7) -> MemorySearchResult:
        """Retrieve relevant memories based on query"""
        
        start_time = datetime.now()
        
        # Check cache first
        cache_key = f"{query}_{memory_type}_{k}_{similarity_threshold}"
        if cache_key in self.memory_cache:
            self.stats['cache_hits'] += 1
            self._update_cache_order(cache_key)
            return self.memory_cache[cache_key]
        
        self.stats['cache_misses'] += 1
        
        # Encode query
        query_embedding = self.embedding_model.encode(query)
        
        # Vector similarity search
        distances, indices = self.vector_db.search(query_embedding.reshape(1, -1), k * 2)
        
        # Retrieve from database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        query_sql = "SELECT * FROM memories"
        params = []
        
        if memory_type:
            query_sql += " WHERE memory_type = ?"
            params.append(memory_type)
        
        cursor.execute(query_sql, params)
        all_memories = cursor.fetchall()
        conn.close()
        
        # Filter and rank results
        results = []
        scores = []
        
        for i, distance in enumerate(distances[0][:k]):
            if distance >= similarity_threshold and i < len(all_memories):
                memory_data = all_memories[i]
                
                memory_entry = MemoryEntry(
                    id=memory_data[0],
                    content=memory_data[1],
                    memory_type=memory_data[2],
                    timestamp=memory_data[3],
                    context=json.loads(memory_data[4]) if memory_data[4] else {},
                    access_count=memory_data[5],
                    importance_score=memory_data[6],
                    tags=json.loads(memory_data[7]) if memory_data[7] else [],
                    embedding=pickle.loads(memory_data[8]) if memory_data[8] else None
                )
                
                results.append(memory_entry)
                scores.append(float(distance))
                
                # Update access count
                self._update_access_count(memory_entry.id)
        
        search_time = (datetime.now() - start_time).total_seconds()
        self.stats['average_retrieval_time'] = (
            self.stats['average_retrieval_time'] + search_time
        ) / 2
        
        search_result = MemorySearchResult(
            entries=results,
            similarity_scores=scores,
            total_results=len(results),
            search_time=search_time
        )
        
        # Cache result
        self._cache_result(cache_key, search_result)
        
        return search_result
    
    def consolidate_memories(self, similarity_threshold: float = 0.95):
        """Consolidate similar memories to avoid redundancy"""
        
        # Get all memories
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM memories")
        memories = cursor.fetchall()
        
        consolidated_ids = set()
        
        for i, memory1 in enumerate(memories):
            if memory1[0] in consolidated_ids:
                continue
                
            embedding1 = pickle.loads(memory1[8]) if memory1[8] else None
            if embedding1 is None:
                continue
            
            similar_memories = []
            
            for j, memory2 in enumerate(memories[i+1:], i+1):
                if memory2[0] in consolidated_ids:
                    continue
                    
                embedding2 = pickle.loads(memory2[8]) if memory2[8] else None
                if embedding2 is None:
                    continue
                
                # Calculate similarity
                similarity = np.dot(embedding1, embedding2) / (
                    np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
                )
                
                if similarity >= similarity_threshold:
                    similar_memories.append(memory2)
                    consolidated_ids.add(memory2[0])
            
            # Consolidate similar memories
            if similar_memories:
                self._merge_memories(memory1, similar_memories)
        
        conn.close()
    
    def forget_memories(self, 
                       age_threshold_days: int = 30,
                       access_threshold: int = 1,
                       importance_threshold: float = 0.1):
        """Implement forgetting mechanism for old, unimportant memories"""
        
        cutoff_date = datetime.now().timestamp() - (age_threshold_days * 24 * 3600)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Find memories to forget
        cursor.execute("""
            SELECT id FROM memories 
            WHERE timestamp < ? 
            AND access_count <= ? 
            AND importance_score <= ?
        """, (cutoff_date, access_threshold, importance_threshold))
        
        memories_to_forget = cursor.fetchall()
        
        # Delete forgotten memories
        for memory_id in memories_to_forget:
            cursor.execute("DELETE FROM memories WHERE id = ?", (memory_id[0],))
            cursor.execute("DELETE FROM semantic_relationships WHERE source_id = ? OR target_id = ?", 
                          (memory_id[0], memory_id[0]))
        
        conn.commit()
        conn.close()
        
        return len(memories_to_forget)
    
    def _encode_context(self, context: Dict[str, Any]) -> torch.Tensor:
        """Encode context dictionary into tensor representation"""
        # Simple context encoding - can be enhanced
        context_str = json.dumps(context, sort_keys=True)
        context_embedding = self.embedding_model.encode(context_str)
        return torch.FloatTensor(context_embedding).to(self.device)
    
    def _persist_memory(self, memory: MemoryEntry):
        """Persist memory to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO memories 
            (id, content, memory_type, timestamp, context, access_count, importance_score, tags, embedding)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            memory.id,
            memory.content,
            memory.memory_type,
            memory.timestamp,
            json.dumps(memory.context),
            memory.access_count,
            memory.importance_score,
            json.dumps(memory.tags),
            pickle.dumps(memory.embedding) if memory.embedding is not None else None
        ))
        
        conn.commit()
        conn.close()
    
    def _store_semantic_relationship(self, source: str, target: str, relation_type: str, strength: float):
        """Store semantic relationship between concepts"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO semantic_relationships (source_id, target_id, relationship_type, strength)
            VALUES (?, ?, ?, ?)
        """, (source, target, relation_type, strength))
        
        conn.commit()
        conn.close()
    
    def _update_access_count(self, memory_id: str):
        """Update access count for memory"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            UPDATE memories SET access_count = access_count + 1 WHERE id = ?
        """, (memory_id,))
        
        conn.commit()
        conn.close()
    
    def _cache_result(self, key: str, result: MemorySearchResult):
        """Cache search result with LRU eviction"""
        if len(self.memory_cache) >= self.max_cache_size:
            # Remove least recently used
            lru_key = self.cache_order.pop(0)
            del self.memory_cache[lru_key]
        
        self.memory_cache[key] = result
        self.cache_order.append(key)
    
    def _update_cache_order(self, key: str):
        """Update cache order for LRU"""
        if key in self.cache_order:
            self.cache_order.remove(key)
            self.cache_order.append(key)
    
    def _merge_memories(self, primary_memory, similar_memories):
        """Merge similar memories into one consolidated memory"""
        # Implementation for memory consolidation
        pass
    
    def get_memory_statistics(self) -> Dict[str, Any]:
        """Get comprehensive memory statistics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get type distribution
        cursor.execute("SELECT memory_type, COUNT(*) FROM memories GROUP BY memory_type")
        type_distribution = dict(cursor.fetchall())
        
        # Get average importance
        cursor.execute("SELECT AVG(importance_score) FROM memories")
        avg_importance = cursor.fetchone()[0] or 0.0
        
        # Get most accessed memories
        cursor.execute("SELECT content, access_count FROM memories ORDER BY access_count DESC LIMIT 5")
        most_accessed = cursor.fetchall()
        
        conn.close()
        
        return {
            **self.stats,
            'type_distribution': type_distribution,
            'average_importance': avg_importance,
            'most_accessed_memories': most_accessed,
            'cache_size': len(self.memory_cache)
        }

# Usage example
if __name__ == "__main__":
    agent = MemoryAgent()
    
    # Store different types of memories
    episode_id = agent.store_episodic_memory(
        "User asked about implementing a binary search algorithm",
        {"user_id": "user123", "session": "session456", "difficulty": "medium"},
        importance=0.8,
        tags=["algorithm", "search", "binary"]
    )
    
    semantic_id = agent.store_semantic_memory(
        "Binary search is a divide-and-conquer algorithm",
        "binary_search",
        relationships=[
            ("divide_and_conquer", "algorithm_paradigm", 0.9),
            ("logarithmic_complexity", "time_complexity", 0.8)
        ],
        tags=["algorithm", "complexity", "search"]
    )
    
    # Retrieve memories
    results = agent.retrieve_memories("binary search algorithm", k=5)
    print(f"Found {results.total_results} relevant memories")
    for entry in results.entries:
        print(f"- {entry.content} (score: {results.similarity_scores[results.entries.index(entry)]:.3f})")
    
    # Get statistics
    stats = agent.get_memory_statistics()
    print(f"Memory statistics: {stats}")