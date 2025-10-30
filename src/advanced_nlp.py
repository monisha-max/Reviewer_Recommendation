import os
import numpy as np
import pickle
import logging
from typing import Dict, List, Tuple, Optional

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF
from sklearn.metrics.pairwise import cosine_similarity
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
try:
    from faiss_store import FAISSVectorStore
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logger.warning("FAISS not available. Using slower numpy-based search.")


class TopicModelingSimilarity:
    def __init__(self, n_topics: int = 20, method: str = 'lda', 
                 max_features: int = 5000):
        self.n_topics = n_topics
        self.method = method
        self.max_features = max_features
        if method == 'lda':
            self.vectorizer = CountVectorizer(
                max_features=max_features,
                stop_words='english',
                min_df=2,
                max_df=0.8
            )
            self.model = LatentDirichletAllocation(
                n_components=n_topics,
                random_state=42,
                max_iter=20,
                learning_method='online'
            )
        else:  # nmf
            self.vectorizer = TfidfVectorizer(
                max_features=max_features,
                stop_words='english',
                min_df=2,
                max_df=0.8
            )
            self.model = NMF(
                n_components=n_topics,
                random_state=42,
                max_iter=200
            )
        self.document_topics = None
        self.document_names = None
        self.feature_names = None
    
    def fit(self, documents: Dict[str, str]):
        self.document_names = list(documents.keys())
        texts = [documents[name] for name in self.document_names]
        doc_term_matrix = self.vectorizer.fit_transform(texts)
        self.feature_names = self.vectorizer.get_feature_names_out()
        self.document_topics = self.model.fit_transform(doc_term_matrix)
        
        logger.info(f"{self.method.upper()} topic model fitted with {self.n_topics} topics "
                   f"on {len(documents)} documents")
    
    def calculate_similarity(self, query_text: str, top_k: int = 10) -> List[Tuple[str, float]]:
        query_vector = self.vectorizer.transform([query_text])
        query_topics = self.model.transform(query_vector)
        similarities = cosine_similarity(query_topics, self.document_topics)[0]
        top_indices = np.argsort(similarities)[::-1][:top_k]
        results = [(self.document_names[idx], similarities[idx]) 
                  for idx in top_indices]
        
        return results
    
    def get_top_words_per_topic(self, n_words: int = 10) -> Dict[int, List[str]]:
        topics = {}
        for topic_idx, topic in enumerate(self.model.components_):
            top_indices = topic.argsort()[-n_words:][::-1]
            top_words = [self.feature_names[i] for i in top_indices]
            topics[topic_idx] = top_words
        
        return topics
    
    def get_document_topics(self, author_name: str) -> np.ndarray:
        if author_name in self.document_names:
            idx = self.document_names.index(author_name)
            return self.document_topics[idx]
        return None


class Doc2VecSimilarity:
    def __init__(self, vector_size: int = 100, window: int = 5, min_count: int = 2, epochs: int = 20):
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.epochs = epochs
        self.model = None
        self.doc_vectors = {}
        self.doc_keys: List[str] = []
    
    def fit(self, documents: Dict[str, str]):
        try:
            from gensim.models.doc2vec import Doc2Vec, TaggedDocument
        except Exception as e:
            logger.error(f"gensim Doc2Vec not available: {e}")
            return
        texts = []
        self.doc_keys = list(documents.keys())
        for key, text in documents.items():
            words = text.split()
            texts.append(TaggedDocument(words=words, tags=[key]))
        self.model = Doc2Vec(vector_size=self.vector_size, window=self.window, min_count=self.min_count, workers=2, epochs=self.epochs)
        self.model.build_vocab(texts)
        self.model.train(texts, total_examples=self.model.corpus_count, epochs=self.epochs)
        self.doc_vectors = {key: self.model.infer_vector(documents[key].split()) for key in self.doc_keys}
    
    def calculate_similarity(self, query_text: str, top_k: int = 10) -> List[Tuple[str, float]]:
        if not self.model:
            return []
        qv = self.model.infer_vector(query_text.split())
        sims = []
        import numpy as np
        for key in self.doc_keys:
            dv = self.doc_vectors.get(key)
            if dv is None:
                continue
            denom = np.linalg.norm(qv) * np.linalg.norm(dv) + 1e-10
            sim = float(np.dot(qv, dv) / denom) if denom > 0 else 0.0
            sims.append((key, sim))
        sims.sort(key=lambda x: x[1], reverse=True)
        return sims[:top_k]


class CrossEncoderReranker:
    def __init__(self, model_name: str = 'cross-encoder/ms-marco-MiniLM-L-6-v2'):
        self.model_name = model_name
        self.model = None
        self._load_model()
    
    def _load_model(self):
        try:
            from sentence_transformers import CrossEncoder
            self.model = CrossEncoder(self.model_name)
            logger.info(f"Loaded CrossEncoder model: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to load CrossEncoder {self.model_name}: {e}")
            self.model = None
    
    def rerank(self, query_text: str, candidates: Dict[str, str], top_k: int = 50) -> List[Tuple[str, float]]:
        if not self.model or not candidates:
            return []
        pairs = [(query_text, candidates[name][:2000]) for name in candidates.keys()]
        try:
            scores = self.model.predict(pairs, convert_to_numpy=True)
        except Exception as e:
            logger.error(f"CrossEncoder prediction failed: {e}")
            return []
        names = list(candidates.keys())
        results = list(zip(names, [float(s) for s in scores]))
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]


class E5Similarity:
    def __init__(self, model_name: str = 'intfloat/e5-base'):
        self.model_name = model_name
        self.model = None
        self.document_embeddings = {}
        self.document_names: List[str] = []
        self._load_model()
    
    def _load_model(self):
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.model_name)
            logger.info(f"Loaded E5 model: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to load E5 model {self.model_name}: {e}")
            self.model = None
    
    def _encode_passage(self, text: str):
        if not self.model:
            return None
        return self.model.encode(f"passage: {text}", show_progress_bar=False, normalize_embeddings=True)
    
    def _encode_query(self, text: str):
        if not self.model:
            return None
        return self.model.encode(f"query: {text}", show_progress_bar=False, normalize_embeddings=True)
    
    def fit(self, documents: Dict[str, str]):
        if not self.model:
            return
        self.document_names = list(documents.keys())
        for name in self.document_names:
            text = documents[name][:500000]
            emb = self._encode_passage(text)
            self.document_embeddings[name] = emb
        logger.info(f"E5 embeddings generated for {len(self.document_names)} documents")
    
    def calculate_similarity(self, query_text: str, top_k: int = 10) -> List[Tuple[str, float]]:
        if not self.model:
            return []
        q = self._encode_query(query_text)
        if q is None:
            return []
        sims = []
        import numpy as np
        for name in self.document_names:
            dv = self.document_embeddings.get(name)
            if dv is None:
                continue
            sim = float(np.dot(q, dv))  
            sims.append((name, sim))
        sims.sort(key=lambda x: x[1], reverse=True)
        return sims[:top_k]

class SentenceBERTSimilarity:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', use_faiss: bool = True):
        self.model_name = model_name
        self.model = None
        self.document_embeddings = {}
        self.document_names = []
        self.use_faiss = use_faiss and FAISS_AVAILABLE
        self.faiss_store = None
        self._load_model()
    
    def _load_model(self):
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.model_name)
            logger.info(f"Loaded Sentence-BERT model: {self.model_name}")
        except ImportError:
            logger.error("sentence-transformers not installed")
            self.model = None
        except Exception as e:
            logger.error(f"Failed to load Sentence-BERT model: {str(e)}")
            self.model = None
    
    def fit(self, documents: Dict[str, str]):
        if self.model is None:
            logger.error("Model not loaded. Cannot fit.")
            return
        
        self.document_names = list(documents.keys())
        
        logger.info(f"Generating Sentence-BERT embeddings for {len(documents)} documents...")
        for name in self.document_names:
            text = documents[name]
            text = text[:500000] 
            embedding = self.model.encode(text, show_progress_bar=False)
            self.document_embeddings[name] = embedding
        
        logger.info(f"Sentence-BERT embeddings generated")
        if self.use_faiss and len(self.document_embeddings) > 0:
            self._build_faiss_index()
    
    def _build_faiss_index(self):
        try:
            first_embedding = next(iter(self.document_embeddings.values()))
            embedding_dim = len(first_embedding)
            self.faiss_store = FAISSVectorStore(embedding_dim)
            embeddings = np.array([self.document_embeddings[name] for name in self.document_names]).astype('float32')
            self.faiss_store.add_embeddings(embeddings, self.document_names)   
            logger.info(f"FAISS index built for Sentence-BERT (dimension={embedding_dim})")
        except (Exception, SystemError, OSError) as e:
            logger.warning(f"FAISS not available on this system ({e}). Using numpy-based search instead.")
            self.use_faiss = False
            self.faiss_store = None
    
    def calculate_similarity(self, query_text: str, top_k: int = 10) -> List[Tuple[str, float]]:
        if self.model is None:
            logger.error("Model not loaded. Cannot calculate similarity.")
            return []
        query_embedding = self.model.encode(query_text, show_progress_bar=False)
        if self.use_faiss and self.faiss_store is not None:
            return self.faiss_store.search(query_embedding, top_k)
        similarities = []
        
        for name in self.document_names:
            doc_embedding = self.document_embeddings[name]
            similarity = np.dot(query_embedding, doc_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding) + 1e-10
            )
            similarities.append((name, float(similarity)))
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:top_k]


class BERTEmbeddingSimilarity: 
    def __init__(self, model_name: str = 'bert-base-uncased', use_faiss: bool = True):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.document_embeddings = {}
        self.document_names = []
        self.use_faiss = use_faiss and FAISS_AVAILABLE
        self.faiss_store = None
        self._load_model()
    
    def _load_model(self):
        try:
            from transformers import AutoTokenizer, AutoModel
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name)
            self.model.eval()
            logger.info(f"Loaded BERT model: {self.model_name}")
        except ImportError:
            logger.error("transformers not installed")
            self.model = None
        except Exception as e:
            logger.error(f"Failed to load BERT model: {str(e)}")
            self.model = None
    
    def _get_bert_embedding(self, text: str, max_length: int = 512) -> np.ndarray:
        if self.model is None:
            return np.zeros(768) 
        inputs = self.tokenizer(
            text,
            return_tensors='pt',
            max_length=max_length,
            truncation=True,
            padding=True
        )
        with torch.no_grad():
            outputs = self.model(**inputs)
            embedding = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
        return embedding
    
    def fit(self, documents: Dict[str, str]):
        if self.model is None:
            logger.error("Model not loaded. Cannot fit.")
            return
        
        self.document_names = list(documents.keys())
        
        logger.info(f"Generating BERT embeddings for {len(documents)} documents...")
        for name in self.document_names:
            text = documents[name]
            text = text[:5000]
            embedding = self._get_bert_embedding(text)
            self.document_embeddings[name] = embedding
        logger.info(f"BERT embeddings generated")
        if self.use_faiss and len(self.document_embeddings) > 0:
            self._build_faiss_index()
    
    def _build_faiss_index(self):
        try:
            first_embedding = next(iter(self.document_embeddings.values()))
            embedding_dim = len(first_embedding)
            self.faiss_store = FAISSVectorStore(embedding_dim)
            embeddings = np.array([self.document_embeddings[name] for name in self.document_names]).astype('float32')
            self.faiss_store.add_embeddings(embeddings, self.document_names)   
            logger.info(f"FAISS index built for BERT (dimension={embedding_dim})")
        except (Exception, SystemError, OSError) as e:
            logger.warning(f"FAISS not available on this system ({e}). Using numpy-based search instead.")
            self.use_faiss = False
            self.faiss_store = None
    
    def calculate_similarity(self, query_text: str, top_k: int = 10) -> List[Tuple[str, float]]:
        if self.model is None:
            logger.error("Model not loaded. Cannot calculate similarity.")
            return []
        query_embedding = self._get_bert_embedding(query_text)
        if self.use_faiss and self.faiss_store is not None:
            return self.faiss_store.search(query_embedding, top_k)
        similarities = []
        
        for name in self.document_names:
            doc_embedding = self.document_embeddings[name]
            similarity = np.dot(query_embedding, doc_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding) + 1e-10
            )
            similarities.append((name, float(similarity)))
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:top_k]


class SciBERTSimilarity:
    def __init__(self, use_faiss: bool = True):
        self.bert_similarity = BERTEmbeddingSimilarity(
            model_name='allenai/scibert_scivocab_uncased',
            use_faiss=use_faiss
        )
    def fit(self, documents: Dict[str, str]):

        self.bert_similarity.fit(documents)
    
    def calculate_similarity(self, query_text: str, top_k: int = 10) -> List[Tuple[str, float]]:
        return self.bert_similarity.calculate_similarity(query_text, top_k)


class AdvancedNLPCalculator:
    def __init__(self, cache_dir: str = "cache"):
        self.cache_dir = cache_dir
        self.methods = {}
        os.makedirs(cache_dir, exist_ok=True)
    
    def add_method(self, name: str, method):
        self.methods[name] = method
        logger.info(f"Added advanced NLP method: {name}")
    
    def fit_all(self, documents: Dict[str, str], save_cache: bool = True):
        if self._load_cache():
            logger.info("Loaded advanced NLP models from cache (instant!)")
            return
        logger.info(f"Cache not found. Fitting {len(self.methods)} advanced NLP methods...")
        
        for name, method in self.methods.items():
            logger.info(f"Fitting {name}...")
            try:
                method.fit(documents)
            except Exception as e:
                logger.error(f"Failed to fit {name}: {str(e)}")
        
        if save_cache:
            self._save_cache()
    
    def calculate_all_similarities(self, query_text: str, top_k: int = 10) -> Dict[str, List[Tuple[str, float]]]:
        results = {}
        
        for name, method in self.methods.items():
            logger.info(f"Calculating similarity using {name}...")
            try:
                results[name] = method.calculate_similarity(query_text, top_k)
            except Exception as e:
                logger.error(f"Failed to calculate similarity using {name}: {str(e)}")
                results[name] = []
        
        return results
    
    def _save_cache(self):
        cache_path = os.path.join(self.cache_dir, "advanced_nlp_models.pkl")
        
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(self.methods, f)
            
            logger.info(f"Advanced NLP models saved to {cache_path}")
        except Exception as e:
            logger.error(f"Failed to save cache: {str(e)}")
    
    def _load_cache(self) -> bool:
        cache_path = os.path.join(self.cache_dir, "advanced_nlp_models.pkl")
        
        if not os.path.exists(cache_path):
            return False
        
        try:
            with open(cache_path, 'rb') as f:
                self.methods = pickle.load(f)
            
            logger.info(f"Loaded {len(self.methods)} advanced NLP models from cache")
            return True
        except Exception as e:
            logger.error(f"Failed to load cache: {str(e)}")
            return False


if __name__ == "__main__":
    documents = {
        'Author1': 'machine learning deep learning neural networks artificial intelligence computer vision',
        'Author2': 'medical research healthcare diagnosis treatment clinical trials patient outcomes',
        'Author3': 'machine learning algorithms classification regression supervised learning optimization'
    }
    
    query = 'deep learning convolutional neural networks image classification computer vision'
    
    print("Testing Advanced NLP Methods")
    print("="*80)
    print("\n1. LDA Topic Modeling")
    lda = TopicModelingSimilarity(n_topics=5, method='lda')
    lda.fit(documents)
    results = lda.calculate_similarity(query, top_k=3)
    for author, score in results:
        print(f"   {author}: {score:.4f}")
    print("\n2. NMF Topic Modeling")
    nmf = TopicModelingSimilarity(n_topics=5, method='nmf')
    nmf.fit(documents)
    results = nmf.calculate_similarity(query, top_k=3)
    for author, score in results:
        print(f"   {author}: {score:.4f}")
    print("\n3. Sentence-BERT")
    try:
        sbert = SentenceBERTSimilarity()
        if sbert.model is not None:
            sbert.fit(documents)
            results = sbert.calculate_similarity(query, top_k=3)
            for author, score in results:
                print(f"   {author}: {score:.4f}")
        else:
            print("   Sentence-BERT not available")
    except Exception as e:
        print(f"   Error: {str(e)}")

