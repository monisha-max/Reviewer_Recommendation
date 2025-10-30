import os
import numpy as np
import pickle
import logging
from typing import Dict, List, Tuple, Optional
from collections import Counter
import re

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import LatentDirichletAllocation, NMF
from scipy.spatial.distance import cdist
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
try:
    from rank_bm25 import BM25Okapi
    _HAS_BM25 = True
except Exception:
    _HAS_BM25 = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)


class TextPreprocessor:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        
    def preprocess(self, text: str, remove_stopwords: bool = True, 
                   min_word_length: int = 3) -> str:
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        tokens = word_tokenize(text)
        tokens = [token for token in tokens 
                 if len(token) >= min_word_length]
        if remove_stopwords:
            tokens = [token for token in tokens 
                     if token not in self.stop_words]
        
        return ' '.join(tokens)
    
    def extract_keywords(self, text: str, top_n: int = 50) -> List[str]:
        preprocessed = self.preprocess(text)
        tokens = word_tokenize(preprocessed)
        freq_dist = Counter(tokens)
        keywords = [word for word, count in freq_dist.most_common(top_n)]
        
        return keywords


class TFIDFCosineSimilarity:
    def __init__(self, max_features: int = 5000, ngram_range: Tuple = (1, 2)):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            stop_words='english',
            min_df=2,
            max_df=0.8
        )
        self.document_vectors = None
        self.document_names = None
    
    def fit(self, documents: Dict[str, str]):
        self.document_names = list(documents.keys())
        texts = [documents[name] for name in self.document_names]
        self.document_vectors = self.vectorizer.fit_transform(texts)
        
        logger.info(f"TF-IDF fitted on {len(documents)} documents with "
                   f"{self.document_vectors.shape[1]} features")
    
    def calculate_similarity(self, query_text: str, top_k: int = 10) -> List[Tuple[str, float]]:
        query_vector = self.vectorizer.transform([query_text])
        similarities = cosine_similarity(query_vector, self.document_vectors)[0]

        top_indices = np.argsort(similarities)[::-1][:top_k] 
        results = [(self.document_names[idx], similarities[idx]) 
                  for idx in top_indices]
        
        return results


class JaccardSimilarity:
    def __init__(self, ngram_size: int = 2, use_words: bool = True):
        self.ngram_size = ngram_size
        self.use_words = use_words
        self.document_sets = {}
        self.preprocessor = TextPreprocessor()
    
    def _create_ngrams(self, text: str) -> set:
        if self.use_words:
            words = word_tokenize(text.lower())
            words = [w for w in words if len(w) > 2]
            
            if self.ngram_size == 1:
                return set(words)
            else:
                ngrams = []
                for i in range(len(words) - self.ngram_size + 1):
                    ngram = ' '.join(words[i:i+self.ngram_size])
                    ngrams.append(ngram)
                return set(ngrams)
        else:
            text = ''.join(text.split()).lower()
            ngrams = [text[i:i+self.ngram_size] 
                     for i in range(len(text) - self.ngram_size + 1)]
            return set(ngrams)
    
    def fit(self, documents: Dict[str, str]):
        for name, text in documents.items():
            preprocessed = self.preprocessor.preprocess(text)
            self.document_sets[name] = self._create_ngrams(preprocessed)
        
        logger.info(f"Jaccard similarity prepared for {len(documents)} documents")
    
    def calculate_similarity(self, query_text: str, top_k: int = 10) -> List[Tuple[str, float]]:
        preprocessed = self.preprocessor.preprocess(query_text)
        query_set = self._create_ngrams(preprocessed)
        
        similarities = []
        
        for name, doc_set in self.document_sets.items():
            # Jaccard similarity = |A ∩ B| / |A ∪ B|
            intersection = len(query_set & doc_set)
            union = len(query_set | doc_set)
            
            if union > 0:
                similarity = intersection / union
            else:
                similarity = 0.0
            
            similarities.append((name, similarity))
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:top_k]


class WordMoversDistance:
    def __init__(self, embedding_model='glove', embedding_dim: int = 100):
        self.embedding_model = embedding_model
        self.embedding_dim = embedding_dim
        self.word_vectors = {}
        self.document_embeddings = {}
        self.preprocessor = TextPreprocessor()
        self._load_embeddings()
    
    def _load_embeddings(self):
        try:
            import gensim.downloader as api
            logger.info("Loading GloVe embeddings (glove-wiki-gigaword-50)...")
            self.word_vectors = api.load("glove-wiki-gigaword-50")
            self.embedding_dim = self.word_vectors.vector_size
            logger.info("Loaded GloVe embeddings.")
        except Exception as e:
            logger.warning(f"Failed to load GloVe embeddings ({str(e)}). Falling back to random vectors.")
            self.word_vectors = {}
    
    def _get_word_embedding(self, word: str) -> np.ndarray:
        try:
            if hasattr(self.word_vectors, "__contains__") and word in self.word_vectors:
                return self.word_vectors[word]
        except Exception:
            pass
        embedding = np.random.randn(self.embedding_dim)
        return embedding
    
    def _text_to_embedding(self, text: str) -> np.ndarray:
        preprocessed = self.preprocessor.preprocess(text)
        words = word_tokenize(preprocessed)
        
        if not words:
            return np.zeros(self.embedding_dim)
        
        embeddings = [self._get_word_embedding(word) for word in words]
        return np.mean(embeddings, axis=0)
    
    def fit(self, documents: Dict[str, str]):
        for name, text in documents.items():
            self.document_embeddings[name] = self._text_to_embedding(text)
        
        logger.info(f"WMD prepared for {len(documents)} documents")
    
    def calculate_similarity(self, query_text: str, top_k: int = 10) -> List[Tuple[str, float]]:
        similarities = []
        use_true_wmd = False
        try:
            from gensim.similarities import WmdSimilarity
            import gensim
            if hasattr(self.word_vectors, "__contains__"):
                use_true_wmd = True
        except Exception:
            use_true_wmd = False

        preprocessed_docs = {name: self.preprocessor.preprocess(text).split()
                             for name, text in self.document_texts.items()}
        query_tokens = self.preprocessor.preprocess(query_text).split()

        if use_true_wmd:
            try:
                corpus = list(preprocessed_docs.values())
                instance = WmdSimilarity(corpus, self.word_vectors, num_best=top_k)
                wmd_results = instance[query_tokens]
                name_by_index = list(preprocessed_docs.keys())
                for idx, distance in wmd_results:
                    name = name_by_index[idx]
                    sim = 1.0 / (1.0 + float(distance))
                    similarities.append((name, sim))
                similarities.sort(key=lambda x: x[1], reverse=True)
                return similarities[:top_k]
            except Exception as e:
                logger.warning(f"True WMD failed ({str(e)}). Falling back to cosine-based approximation.")
        query_embedding = self._text_to_embedding(query_text)
        for name, doc_embedding in self.document_embeddings.items():
            denom = (np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding) + 1e-10)
            similarity = float(np.dot(query_embedding, doc_embedding) / denom) if denom > 0 else 0.0
            similarities.append((name, similarity))
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]

    def fit(self, documents: Dict[str, str]):
        self.document_texts = documents
        for name, text in documents.items():
            self.document_embeddings[name] = self._text_to_embedding(text)
        logger.info(f"WMD prepared for {len(documents)} documents")


class KeywordBasedMatching:
    def __init__(self, top_keywords: int = 100):
        self.top_keywords = top_keywords
        self.document_keywords = {}
        self.preprocessor = TextPreprocessor()
    
    def fit(self, documents: Dict[str, str]):
        for name, text in documents.items():
            keywords = self.preprocessor.extract_keywords(text, self.top_keywords)
            self.document_keywords[name] = set(keywords)
        
        logger.info(f"Keyword matching prepared for {len(documents)} documents")
    
    def calculate_similarity(self, query_text: str, top_k: int = 10) -> List[Tuple[str, float]]:
        query_keywords = set(self.preprocessor.extract_keywords(query_text, self.top_keywords))
        
        similarities = []
        
        for name, doc_keywords in self.document_keywords.items():
            intersection = len(query_keywords & doc_keywords)
            union = len(query_keywords | doc_keywords)
            
            if union > 0:
                similarity = intersection / union
            else:
                similarity = 0.0
            
            similarities.append((name, similarity))
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:top_k]


class BM25Similarity:
    def __init__(self):
        self.tokenized_corpus = []
        self.document_names: List[str] = []
        self.bm25 = None
        self.preprocessor = TextPreprocessor()
        if not _HAS_BM25:
            logger.warning("rank_bm25 not available; BM25Similarity will be disabled.")
    
    def fit(self, documents: Dict[str, str]):
        self.document_names = list(documents.keys())
        texts = [documents[name] for name in self.document_names]
        # Light tokenization; BM25 works on terms, we reuse our preprocessor
        self.tokenized_corpus = [word_tokenize(self.preprocessor.preprocess(t)) for t in texts]
        if _HAS_BM25:
            self.bm25 = BM25Okapi(self.tokenized_corpus)
        logger.info(f"BM25 prepared for {len(self.document_names)} documents")
    
    def calculate_similarity(self, query_text: str, top_k: int = 10) -> List[Tuple[str, float]]:
        if not _HAS_BM25 or self.bm25 is None:
            return [(name, 0.0) for name in self.document_names[:top_k]]
        query_tokens = word_tokenize(self.preprocessor.preprocess(query_text))
        scores = self.bm25.get_scores(query_tokens)
        indices = np.argsort(scores)[::-1][:top_k]
        return [(self.document_names[i], float(scores[i])) for i in indices]


class EntityBasedMatching:
    def __init__(self, model: str = "en_core_web_sm"):
        self.model_name = model
        self.nlp = None
        self.doc_entities: Dict[str, Dict[str, set]] = {}
        self.preprocessor = TextPreprocessor()
        self.entity_labels = ["PERSON", "ORG", "GPE", "NORP", "FAC", "LOC", "PRODUCT", "EVENT", "WORK_OF_ART"]
        self.weights = {
            "PERSON": 3.0,
            "ORG": 2.0,
            "GPE": 1.5,
            "NORP": 1.2
        }

    def _ensure_nlp(self):
        if self.nlp is None:
            try:
                import spacy
                self.nlp = spacy.load(self.model_name)
            except Exception:
                try:
                    import spacy
                    from spacy.cli import download
                    download(self.model_name)
                    self.nlp = spacy.load(self.model_name)
                except Exception:
                    self.nlp = None
                    logger.warning("spaCy model not available; EntityBasedMatching disabled.")

    def _extract_entities(self, text: str) -> Dict[str, set]:
        self._ensure_nlp()
        if not self.nlp or not text:
            return {label: set() for label in self.entity_labels}
        doc = self.nlp(text)
        entities: Dict[str, set] = {label: set() for label in self.entity_labels}
        for ent in doc.ents:
            if ent.label_ in entities:
                entities[ent.label_].add(ent.text.strip().lower())
        return entities

    def fit(self, documents: Dict[str, str]):
        self.doc_entities = {}
        for name, text in documents.items():
            self.doc_entities[name] = self._extract_entities(text)
        logger.info(f"Entity-based features prepared for {len(documents)} documents")

    def calculate_similarity(self, query_text: str, top_k: int = 10) -> List[Tuple[str, float]]:
        query_entities = self._extract_entities(query_text)
        similarities: List[Tuple[str, float]] = []
        
        for name, ents in self.doc_entities.items():
            score = 0.0
            total_weight = 0.0
            for label in self.entity_labels:
                qset = query_entities.get(label, set())
                dset = ents.get(label, set())
                if not qset and not dset:
                    continue
                inter = len(qset & dset)
                union = len(qset | dset)
                jacc = inter / union if union > 0 else 0.0
                weight = self.weights.get(label, 1.0)
                score += weight * jacc
                total_weight += weight
            similarities.append((name, score / total_weight if total_weight > 0 else 0.0))

        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]


class SimilarityCalculator:
    def __init__(self, cache_dir: str = "cache"):
        self.cache_dir = cache_dir
        self.methods = {}
        os.makedirs(cache_dir, exist_ok=True)
    
    def add_method(self, name: str, method):
        self.methods[name] = method
        logger.info(f"Added similarity method: {name}")
    
    def fit_all(self, documents: Dict[str, str], save_cache: bool = True):
        if self._load_cache():
            logger.info("✓ Loaded similarity models from cache (instant!)")
            return
        logger.info(f"Cache not found. Fitting {len(self.methods)} similarity methods...")
        
        for name, method in self.methods.items():
            logger.info(f"Fitting {name}...")
            method.fit(documents)
        
        if save_cache:
            self._save_cache()
    
    def calculate_all_similarities(self, query_text: str, top_k: int = 10) -> Dict[str, List[Tuple[str, float]]]:
        results = {}
        
        for name, method in self.methods.items():
            logger.info(f"Calculating similarity using {name}...")
            results[name] = method.calculate_similarity(query_text, top_k)
        
        return results
    
    def _save_cache(self):
        cache_path = os.path.join(self.cache_dir, "similarity_models.pkl")
        
        with open(cache_path, 'wb') as f:
            pickle.dump(self.methods, f)
        
        logger.info(f"Similarity models saved to {cache_path}")
    
    def _load_cache(self) -> bool:
        cache_path = os.path.join(self.cache_dir, "similarity_models.pkl")
        
        if not os.path.exists(cache_path):
            return False
        
        try:
            with open(cache_path, 'rb') as f:
                self.methods = pickle.load(f)
            
            logger.info(f"Loaded {len(self.methods)} similarity models from cache")
            return True
        except Exception as e:
            logger.error(f"Failed to load cache: {str(e)}")
            return False


if __name__ == "__main__":
    documents = {
        'Author1': 'machine learning deep learning neural networks artificial intelligence',
        'Author2': 'medical research healthcare diagnosis treatment clinical trials',
        'Author3': 'machine learning algorithms classification regression supervised learning'
    }
    
    query = 'deep learning convolutional neural networks image classification'
    
    print("Testing Similarity Methods")
    print("="*80)
    
    # TF-IDF Cosine
    print("\n1. TF-IDF + Cosine Similarity")
    tfidf = TFIDFCosineSimilarity()
    tfidf.fit(documents)
    results = tfidf.calculate_similarity(query, top_k=3)
    for author, score in results:
        print(f"   {author}: {score:.4f}")
    
    # Jaccard
    print("\n2. Jaccard Similarity")
    jaccard = JaccardSimilarity()
    jaccard.fit(documents)
    results = jaccard.calculate_similarity(query, top_k=3)
    for author, score in results:
        print(f"   {author}: {score:.4f}")
    
    # Keyword
    print("\n3. Keyword-Based Matching")
    keyword = KeywordBasedMatching()
    keyword.fit(documents)
    results = keyword.calculate_similarity(query, top_k=3)
    for author, score in results:
        print(f"   {author}: {score:.4f}")

