import os
import json
import logging
import numpy as np
from typing import Dict, List, Tuple, Optional
import pickle

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logger.warning("FAISS not available. Install with: pip install faiss-cpu")


class FAISSVectorStore:
    def __init__(self, embedding_dim: int = 768, use_gpu: bool = False):
        if not FAISS_AVAILABLE:
            raise ImportError("FAISS not available. Install with: pip install faiss-cpu")
        
        self.embedding_dim = embedding_dim
        self.use_gpu = use_gpu
        self.index = None
        self.author_names = []
        self.metadata = {}
        
        self._initialize_index()
    
    def _initialize_index(self):
        self.index = faiss.IndexFlatIP(self.embedding_dim)
        
        if self.use_gpu and faiss.get_num_gpus() > 0:
            logger.info("Using GPU for FAISS")
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
        
        logger.info(f"Initialized FAISS index with dimension {self.embedding_dim}")
    
    def add_embeddings(self, embeddings: np.ndarray, author_names: List[str], metadata: Optional[Dict] = None):
        if embeddings.shape[1] != self.embedding_dim:
            raise ValueError(f"Embedding dimension mismatch: expected {self.embedding_dim}, got {embeddings.shape[1]}")
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)
        self.author_names.extend(author_names)
        
        if metadata:
            self.metadata.update(metadata)
        
        logger.info(f"Added {len(author_names)} embeddings to FAISS index. Total: {len(self.author_names)}")
    
    def search(self, query_embedding: np.ndarray, top_k: int = 10) -> List[Tuple[str, float]]:
        if len(query_embedding.shape) == 1:
            query_embedding = query_embedding.reshape(1, -1)
        faiss.normalize_L2(query_embedding)
        scores, indices = self.index.search(query_embedding, min(top_k, len(self.author_names)))
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < len(self.author_names):  # Valid index
                author_name = self.author_names[idx]
                results.append((author_name, float(score)))
        
        return results
    
    def save(self, directory: str, index_name: str = "index"):
        os.makedirs(directory, exist_ok=True)
        index_path = os.path.join(directory, f"{index_name}.faiss")
        if self.use_gpu and faiss.get_num_gpus() > 0:
            cpu_index = faiss.index_gpu_to_cpu(self.index)
            faiss.write_index(cpu_index, index_path)
        else:
            faiss.write_index(self.index, index_path)
        metadata_path = os.path.join(directory, f"{index_name}_metadata.json")
        metadata = {
            'author_names': self.author_names,
            'embedding_dim': self.embedding_dim,
            'num_vectors': len(self.author_names),
            'metadata': self.metadata
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Saved FAISS index to {index_path}")
        logger.info(f"Saved metadata to {metadata_path}")
    
    def load(self, directory: str, index_name: str = "index"):
        index_path = os.path.join(directory, f"{index_name}.faiss")
        
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"FAISS index not found: {index_path}")
        
        self.index = faiss.read_index(index_path)
        if self.use_gpu and faiss.get_num_gpus() > 0:
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
        metadata_path = os.path.join(directory, f"{index_name}_metadata.json")
        
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            self.author_names = metadata['author_names']
            self.embedding_dim = metadata['embedding_dim']
            self.metadata = metadata.get('metadata', {})
        
        logger.info(f"Loaded FAISS index from {index_path}")
        logger.info(f"Index contains {len(self.author_names)} vectors")
    
    def get_stats(self) -> Dict:
        return {
            'num_vectors': len(self.author_names),
            'embedding_dim': self.embedding_dim,
            'index_type': type(self.index).__name__,
            'is_trained': self.index.is_trained,
            'use_gpu': self.use_gpu
        }


class MultiIndexFAISSStore:
    def __init__(self, cache_dir: str = "cache"):
        self.cache_dir = cache_dir
        self.indices = {}
        os.makedirs(cache_dir, exist_ok=True)
    
    def add_index(self, name: str, embedding_dim: int = 768):
        if FAISS_AVAILABLE:
            self.indices[name] = FAISSVectorStore(embedding_dim)
            logger.info(f"Added FAISS index: {name}")
        else:
            logger.warning(f"FAISS not available, skipping index: {name}")
    
    def save_all(self):
        for name, store in self.indices.items():
            store.save(self.cache_dir, index_name=name)
        
        logger.info(f"Saved {len(self.indices)} FAISS indices")
    
    def load_all(self):
        faiss_files = [f for f in os.listdir(self.cache_dir) if f.endswith('.faiss')]
        
        for faiss_file in faiss_files:
            index_name = faiss_file.replace('.faiss', '')
            metadata_path = os.path.join(self.cache_dir, f"{index_name}_metadata.json")
            
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                embedding_dim = metadata['embedding_dim']
            else:
                embedding_dim = 768  
            
            if FAISS_AVAILABLE:
                store = FAISSVectorStore(embedding_dim)
                store.load(self.cache_dir, index_name)
                self.indices[index_name] = store
        
        logger.info(f"Loaded {len(self.indices)} FAISS indices")
    
    def get_index(self, name: str) -> Optional[FAISSVectorStore]:
        return self.indices.get(name)


if __name__ == "__main__":
    print("Testing FAISS Vector Store")
    print("="*80)
    
    if not FAISS_AVAILABLE:
        print("FAISS not available. Install with: pip install faiss-cpu")
    else:
        n_authors = 10
        embedding_dim = 384
        
        embeddings = np.random.randn(n_authors, embedding_dim).astype('float32')
        author_names = [f"Author_{i}" for i in range(n_authors)]
        store = FAISSVectorStore(embedding_dim)
        store.add_embeddings(embeddings, author_names)
        query = np.random.randn(embedding_dim).astype('float32')
        results = store.search(query, top_k=5)
        
        print("\nTop 5 similar authors:")
        for author, score in results:
            print(f"  {author}: {score:.4f}")

        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            store.save(tmpdir, "test_index")

            new_store = FAISSVectorStore(embedding_dim)
            new_store.load(tmpdir, "test_index")
            results2 = new_store.search(query, top_k=5)
            
            print("\nResults after save/load:")
            for author, score in results2:
                print(f"  {author}: {score:.4f}")
        
        print("\n FAISS tests passed!")

