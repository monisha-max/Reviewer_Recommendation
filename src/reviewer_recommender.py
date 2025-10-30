import os
import json
import logging
from typing import Dict, List, Tuple, Optional
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from pdf_parser import PDFParser
from author_profile import AuthorProfileBuilder, AuthorProfile
from similarity_calculator import (
    SimilarityCalculator,
    TFIDFCosineSimilarity,
    JaccardSimilarity,
    WordMoversDistance,
    KeywordBasedMatching,
    EntityBasedMatching,
    BM25Similarity
)
from advanced_nlp import (
    AdvancedNLPCalculator,
    TopicModelingSimilarity,
    SentenceBERTSimilarity,
    BERTEmbeddingSimilarity,
    SciBERTSimilarity,
    Doc2VecSimilarity,
    CrossEncoderReranker
)
from evaluator import StrategyEvaluator

try:
    from structural_similarity import StructuralSimilarity
    STRUCTURAL_AVAILABLE = True
except Exception as e:
    STRUCTURAL_AVAILABLE = False
    logger.warning(f"Structural similarity not available ({e})")

try:
    from visual_similarity import VisualSimilarity
    VISUAL_AVAILABLE = True
except Exception as e:
    VISUAL_AVAILABLE = False
    logger.warning(f"Visual similarity not available ({e})")

try:
    from citation_metrics import CitationMetrics, CitationSimilarity
    CITATION_AVAILABLE = True
except ImportError:
    CITATION_AVAILABLE = False
    logger.warning("Citation metrics not available")
class ConflictOfInterestAnalyzer:
    def __init__(self, author_profiles: Dict[str, AuthorProfile]):
        self.author_profiles = author_profiles
        self.author_names = set(author_profiles.keys())
        self.affiliations = {} 
        try:
            import spacy
            self.nlp = spacy.load("en_core_web_sm")
        except Exception:
            self.nlp = None
    
    def find_conflicts(self, paper_text: str) -> set:
        conflicts = set()
        text_l = paper_text.lower()
        for name in self.author_names:
            if name and name.lower() in text_l:
                conflicts.add(name)
        if self.nlp:
            try:
                doc = self.nlp(paper_text)
                orgs = {ent.text.strip().lower() for ent in doc.ents if ent.label_ == 'ORG'}
                for name, aff in self.affiliations.items():
                    if aff and aff.lower() in orgs:
                        conflicts.add(name)
            except Exception:
                pass
        return conflicts


class ReviewerRecommender:
    
    def __init__(self, 
                 dataset_path: str,
                 cache_dir: str = "cache",
                 use_advanced_models: bool = True,
                 use_advanced_features: bool = False):
        self.dataset_path = dataset_path
        self.cache_dir = cache_dir
        self.use_advanced_models = use_advanced_models
        self.use_advanced_features = use_advanced_features
        os.makedirs(cache_dir, exist_ok=True)
        self.pdf_parser = PDFParser()
        self.author_builder = AuthorProfileBuilder(dataset_path, cache_dir)
        self.author_profiles = {}
        self.similarity_calculator = SimilarityCalculator(cache_dir)
        self.advanced_nlp_calculator = AdvancedNLPCalculator(cache_dir)
        self.cross_encoder = CrossEncoderReranker()
        
        if use_advanced_features:
            if STRUCTURAL_AVAILABLE:
                self.structural_similarity = StructuralSimilarity()
            if VISUAL_AVAILABLE:
                self.visual_similarity = VisualSimilarity()
            if CITATION_AVAILABLE:
                self.citation_metrics = CitationMetrics(cache_dir=cache_dir)
        
        self.evaluator = StrategyEvaluator()
        self.coi_analyzer = None
        self.advanced_ready = False
        self.is_initialized = False
    
    def initialize(self, force_rebuild: bool = False):
        logger.info("Initializing Reviewer Recommendation System...")
        logger.info("Building author profiles...")
        self.author_profiles = self.author_builder.build_profiles(force_rebuild)
        
        if not self.author_profiles:
            raise ValueError("No author profiles found. Check dataset path.")
        documents = {
            name: profile.aggregated_text 
            for name, profile in self.author_profiles.items()
            if profile.aggregated_text
        }
        
        logger.info(f"Preparing {len(documents)} author documents for similarity calculation...")
        logger.info("Initializing basic similarity methods...")
        cache_loaded = self.similarity_calculator._load_cache()
        
        if not cache_loaded:
            self.similarity_calculator.add_method(
                'TF-IDF + Cosine',
                TFIDFCosineSimilarity(max_features=5000, ngram_range=(1, 2))
            )

            try:
                self.similarity_calculator.add_method(
                    'BM25',
                    BM25Similarity()
                )
            except Exception as e:
                logger.warning(f"BM25 unavailable: {e}")
            
            self.similarity_calculator.add_method(
                'Jaccard (Bigrams)',
                JaccardSimilarity(ngram_size=2, use_words=True)
            )
            
            self.similarity_calculator.add_method(
                'Keyword Matching',
                KeywordBasedMatching(top_keywords=100)
            )
            try:
                self.similarity_calculator.add_method(
                    'Entity Matching (NER)',
                    EntityBasedMatching(model='en_core_web_sm')
                )
            except Exception as e:
                logger.warning(f"Entity-based matching unavailable: {e}")
            self.similarity_calculator.fit_all(documents, save_cache=True)
        else:
            logger.info("✓ Basic similarity methods loaded from cache!")
        logger.info("Initializing advanced NLP methods...")
        advanced_cache_loaded = self.advanced_nlp_calculator._load_cache()
        
        if not advanced_cache_loaded:
            self.advanced_nlp_calculator.add_method(
                'LDA Topic Model',
                TopicModelingSimilarity(n_topics=20, method='lda')
            )
            
            self.advanced_nlp_calculator.add_method(
                'NMF Topic Model',
                TopicModelingSimilarity(n_topics=20, method='nmf')
            )
            
            if self.use_advanced_models:
                logger.info("Loading transformer-based models (this may take a while)...")
                
                try:
                    self.advanced_nlp_calculator.add_method(
                        'Sentence-BERT',
                        SentenceBERTSimilarity(model_name='all-MiniLM-L6-v2')
                    )
                except Exception as e:
                    logger.warning(f"Failed to load Sentence-BERT: {str(e)}")
                
                try:
                    self.advanced_nlp_calculator.add_method(
                        'SciBERT',
                        SciBERTSimilarity()
                    )
                except Exception as e:
                    logger.warning(f"Failed to load SciBERT: {str(e)}")
            
            try:
                self.advanced_nlp_calculator.add_method(
                    'Doc2Vec',
                    Doc2VecSimilarity()
                )
            except Exception as e:
                logger.warning(f"Failed to add Doc2Vec: {str(e)}")
            self.advanced_nlp_calculator.fit_all(documents, save_cache=True)
        else:
            logger.info("✓ Advanced NLP methods loaded from cache!")
            if 'E5 Embeddings' not in self.advanced_nlp_calculator.methods:
                try:
                    from advanced_nlp import E5Similarity
                    e5 = E5Similarity()
                    self.advanced_nlp_calculator.add_method('E5 Embeddings', e5)
                    e5.fit(documents)
                    self.advanced_nlp_calculator._save_cache()
                    logger.info("Cached E5 embeddings added to advanced models cache")
                except Exception as e:
                    logger.warning(f"E5 not available: {e}")
        self.is_initialized = True
        logger.info("Core system initialization complete!")
        if self.use_advanced_features:
            try:
                import threading
                def _warmup():
                    try:
                        logger.info("Initializing advanced features (structural, visual, citation)...")
                        if STRUCTURAL_AVAILABLE and hasattr(self, 'structural_similarity'):
                            logger.info("Fitting structural similarity...")
                            author_pdfs = {
                                name: [pub['pdf_path'] for pub in profile.publications]
                                for name, profile in self.author_profiles.items()
                            }
                            try:
                                self.structural_similarity.set_author_pdfs(author_pdfs)
                            except Exception:
                                pass
                            if not self.structural_similarity.load_cache(self.cache_dir):
                                self.structural_similarity.fit(documents)
                                self.structural_similarity.save_cache(self.cache_dir)
                        if VISUAL_AVAILABLE and hasattr(self, 'visual_similarity'):
                            logger.info("Fitting visual similarity...")
                            try:
                                self.visual_similarity.author_pdfs = {
                                    name: [pub['pdf_path'] for pub in profile.publications]
                                    for name, profile in self.author_profiles.items()
                                }
                            except Exception:
                                pass
                            if not self.visual_similarity.load_cache(self.cache_dir):
                                self.visual_similarity.fit(documents)
                                self.visual_similarity.save_cache(self.cache_dir)
                        if CITATION_AVAILABLE and hasattr(self, 'citation_metrics'):
                            logger.info("Fetching citation metrics (this may take time)...")
                            author_names = list(self.author_profiles.keys())
                            self.citation_metrics.fetch_metrics_for_authors(author_names)
                        self.advanced_ready = True
                        logger.info("Advanced features ready!")
                    except Exception as e:
                        logger.warning(f"Advanced features warmup failed: {e}")
                threading.Thread(target=_warmup, daemon=True).start()
            except Exception as e:
                logger.warning(f"Advanced features async not started: {e}")

        if self.coi_analyzer is None:
            try:
                self.coi_analyzer = ConflictOfInterestAnalyzer(self.author_profiles)
            except Exception as e:
                logger.warning(f"COI analyzer unavailable: {e}")
    
    def recommend_reviewers(self, 
                           paper_path: str,
                           top_k: int = 10,
                           methods: Optional[List[str]] = None) -> Dict[str, List[Tuple[str, float]]]:
        if not self.is_initialized:
            raise ValueError("System not initialized. Call initialize() first.")

        logger.info(f"Parsing paper: {paper_path}")
        result = self.pdf_parser.parse(paper_path)
        
        if not result['success']:
            raise ValueError(f"Failed to parse paper: {paper_path}")
        
        paper_text = result['text']
        recommendations = {}
        basic_results = self.similarity_calculator.calculate_all_similarities(
            paper_text, top_k
        )
        recommendations.update(basic_results)
        advanced_results = self.advanced_nlp_calculator.calculate_all_similarities(
            paper_text, top_k
        )
        recommendations.update(advanced_results)
        if self.use_advanced_features:
            if STRUCTURAL_AVAILABLE and hasattr(self, 'structural_similarity'):
                try:
                    structural_results = []
                    try:
                        structural_results = self.structural_similarity.calculate_similarity_from_pdf(paper_path, top_k)
                    except Exception:
                        structural_results = []
                    if not structural_results:
                        structural_results = self.structural_similarity.calculate_similarity(paper_text, top_k)
                    if structural_results: 
                        recommendations['Structural Similarity'] = structural_results
                except Exception as e:
                    logger.warning(f"Structural similarity failed: {e}")
            
            if VISUAL_AVAILABLE and hasattr(self, 'visual_similarity'):
                try:
                    visual_results = []
                    visual_label = 'Visual Similarity (images)'
                    try:
                        visual_results = self.visual_similarity.calculate_image_similarity_pipeline(paper_path, top_k)
                    except Exception:
                        visual_results = []
                    if not visual_results:
                        visual_results = self.visual_similarity.calculate_similarity(paper_text, top_k)
                        visual_label = 'Visual Similarity (text)'
                    if visual_results: 
                        recommendations[visual_label] = visual_results
                except Exception as e:
                    logger.warning(f"Visual similarity failed: {e}")
            
            if CITATION_AVAILABLE and hasattr(self, 'citation_metrics'):
                try:
                    citation_results = self.citation_metrics.calculate_citation_based_similarity(paper_text, top_k)
                    if citation_results: 
                        recommendations['Citation Impact'] = citation_results
                except Exception as e:
                    logger.warning(f"Citation similarity failed: {e}")
        if methods:
            recommendations = {
                method: results 
                for method, results in recommendations.items() 
                if method in methods
            }
        
        return recommendations
    
    def recommend_reviewers_from_text(self,
                                     paper_text: str,
                                     top_k: int = 10,
                                     methods: Optional[List[str]] = None) -> Dict[str, List[Tuple[str, float]]]:
        if not self.is_initialized:
            raise ValueError("System not initialized. Call initialize() first.")
        recommendations = {}
        basic_results = self.similarity_calculator.calculate_all_similarities(
            paper_text, top_k
        )
        recommendations.update(basic_results)
        advanced_results = self.advanced_nlp_calculator.calculate_all_similarities(
            paper_text, top_k
        )
        recommendations.update(advanced_results)
        if self.use_advanced_features:
            if STRUCTURAL_AVAILABLE and hasattr(self, 'structural_similarity'):
                try:
                    structural_results = self.structural_similarity.calculate_similarity(paper_text, top_k)
                    if structural_results:  
                        recommendations['Structural Similarity'] = structural_results
                except Exception as e:
                    logger.warning(f"Structural similarity failed: {e}")
            
            if VISUAL_AVAILABLE and hasattr(self, 'visual_similarity'):
                try:
                    visual_results = self.visual_similarity.calculate_similarity(paper_text, top_k)
                    if visual_results:  
                        recommendations['Visual Similarity'] = visual_results
                except Exception as e:
                    logger.warning(f"Visual similarity failed: {e}")
            
            if CITATION_AVAILABLE and hasattr(self, 'citation_metrics'):
                try:
                    citation_results = self.citation_metrics.calculate_citation_based_similarity(paper_text, top_k)
                    if citation_results:  
                        recommendations['Citation Impact'] = citation_results
                except Exception as e:
                    logger.warning(f"Citation similarity failed: {e}")
        if methods:
            recommendations = {
                method: results 
                for method, results in recommendations.items() 
                if method in methods
            }
        
        return recommendations
    
    def get_ensemble_recommendations(self,
                                    paper_path: str,
                                    top_k: int = 10,
                                    weights: Optional[Dict[str, float]] = None) -> List[Tuple[str, float, Dict]]:
        all_recommendations = self.recommend_reviewers(paper_path, top_k * 2)
        author_scores = {}
        author_reasoning = {}
        
        for method_name, recommendations in all_recommendations.items():
            if not recommendations:
                continue
                
            weight = weights.get(method_name, 1.0) if weights else 1.0
            
            for author, score in recommendations:
                if author not in author_scores:
                    author_scores[author] = 0.0
                    author_reasoning[author] = {}
                
                author_scores[author] += score * weight
                author_reasoning[author][method_name] = score

        #Reciprocal Rank Fusion (RRF) 
        try:
            k_rrf = 60.0
            rrf_scores = {}
            for method_name, recs in all_recommendations.items():
                for rank_idx, (author, _score) in enumerate(recs, 1):
                    rrf_scores[author] = rrf_scores.get(author, 0.0) + 1.0 / (k_rrf + rank_idx)
            if rrf_scores:
                max_rrf = max(rrf_scores.values()) if rrf_scores else 1.0
                for author, rrf in rrf_scores.items():
                    if author not in author_scores:
                        author_scores[author] = 0.0
                        author_reasoning[author] = {}
                    rrf_norm = float(rrf) / max_rrf
                    author_scores[author] += rrf_norm * (weights.get('RRF Fusion', 1.0) if weights else 1.0)
                    author_reasoning[author]['RRF Fusion'] = rrf_norm
        except Exception as e:
            logger.warning(f"RRF fusion skipped: {e}")
        max_score = max(author_scores.values()) if author_scores else 1.0
        normalized_scores = [
            (author, score / max_score, author_reasoning[author]) 
            for author, score in author_scores.items()
        ]
        try:
            pool = {author: self.author_profiles[author].aggregated_text for author, _, _ in normalized_scores}
            parsed = self.pdf_parser.parse(paper_path)
            query_text = parsed['text'] if parsed.get('success') else ''
            reranked = self.cross_encoder.rerank(query_text=query_text, candidates=pool, top_k=max(top_k, 50))
            if reranked:
                ce_scores = {name: score for name, score in reranked}
                max_ce = max(ce_scores.values()) if ce_scores else 1.0
                enriched_tmp = []
                for author, score, reasoning in normalized_scores:
                    ce = ce_scores.get(author, None)
                    if ce is not None and max_ce > 0:
                        ce_norm = float(ce) / max_ce
                        blended = 0.8 * score + 0.2 * ce_norm
                        reasoning = dict(reasoning)
                        reasoning['Cross-Encoder'] = ce_norm
                        enriched_tmp.append((author, blended, reasoning))
                    else:
                        enriched_tmp.append((author, score, reasoning))
                normalized_scores = enriched_tmp
        except Exception as e:
            logger.warning(f"Cross-encoder reranking skipped: {e}")
        try:
            parsed = self.pdf_parser.parse(paper_path)
            paper_text = parsed['text'] if parsed.get('success') else ''
            if self.coi_analyzer and paper_text:
                conflicts = self.coi_analyzer.find_conflicts(paper_text)
            else:
                conflicts = set()
        except Exception:
            conflicts = set()
        
        enriched = []
        for author, score, reasoning in normalized_scores:
            reasoning = dict(reasoning)
            reasoning['COI'] = (author in conflicts)
            enriched.append((author, score, reasoning))
        enriched.sort(key=lambda x: x[1], reverse=True)
        
        top_list = enriched[:top_k]
        try:
            parsed = self.pdf_parser.parse(paper_path)
            query_text = parsed['text'] if parsed.get('success') else ''
            if query_text:
                for idx in range(len(top_list)):
                    author, score, reasoning = top_list[idx]
                    profile = self.author_profiles.get(author)
                    if not profile or not profile.publications:
                        continue
                    papers = profile.publications[:10]
                    best_paper = None
                    best_score = -1.0
                    try:
                        from advanced_nlp import E5Similarity
                        e5 = E5Similarity()
                        docs = {p['metadata']['filename'] or p['pdf_path']: p['text'] for p in papers}
                        e5.fit(docs)
                        paper_scores = e5.calculate_similarity(query_text, top_k=3)
                    except Exception:
                        tfidf = TFIDFCosineSimilarity(max_features=5000, ngram_range=(1, 2))
                        docs = {p['metadata']['filename'] or p['pdf_path']: p['text'] for p in papers}
                        tfidf.fit(docs)
                        paper_scores = tfidf.calculate_similarity(query_text, top_k=3)
                    if paper_scores:
                        best_paper, best_score = paper_scores[0]
                    reasoning = dict(reasoning)
                    reasoning['Best Paper'] = best_paper if best_paper else 'N/A'
                    reasoning['Best Paper Score'] = float(best_score) if best_score >= 0 else 0.0
                    top_list[idx] = (author, score, reasoning)
        except Exception as e:
            logger.warning(f"Per-paper drill-down failed: {e}")
        
        return top_list
    
    def evaluate_methods(self,
                        test_papers: List[str],
                        ground_truth: Dict[str, List[str]],
                        k_values: List[int] = [1, 3, 5, 10]) -> str:
        logger.info(f"Evaluating methods on {len(test_papers)} test papers...")
        
        all_comparisons = []
        
        for paper_path in test_papers:
            logger.info(f"Evaluating paper: {paper_path}")
            recommendations = self.recommend_reviewers(paper_path, top_k=max(k_values))
            relevant = ground_truth.get(paper_path, [])
            
            if not relevant:
                logger.warning(f"No ground truth for {paper_path}")
                continue
            comparison = self.evaluator.compare_strategies(
                recommendations, relevant, k_values
            )
            
            all_comparisons.append(comparison)
        if all_comparisons:
            import pandas as pd
            aggregated = pd.concat(all_comparisons).groupby('Strategy').mean().reset_index()
            query_info = {
                'Test Papers': len(test_papers),
                'Evaluation Date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            report_path = self.evaluator.generate_report(
                aggregated, query_info, 'method_evaluation'
            )
            
            return report_path
        
        return None
    
    def get_author_info(self, author_name: str) -> Optional[Dict]:
        profile = self.author_profiles.get(author_name)
        
        if profile:
            return profile.get_summary()
        
        return None
    
    def get_statistics(self) -> Dict:
        return self.author_builder.get_statistics()
    
    def get_available_methods(self) -> List[str]:
        methods = []
        methods.extend(self.similarity_calculator.methods.keys())
        methods.extend(self.advanced_nlp_calculator.methods.keys())
        return methods
    
    def save_recommendations(self,
                            recommendations: Dict[str, List[Tuple[str, float]]],
                            paper_path: str,
                            output_path: str):
        output = {
            'paper': paper_path,
            'paper_name': os.path.basename(paper_path),
            'recommendations': {}
        }
        
        for method_name, results in recommendations.items():
            output['recommendations'][method_name] = [
                {'author': author, 'score': float(score)}
                for author, score in results
            ]
        
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)
        
        logger.info(f"Recommendations saved to {output_path}")


class ReviewerReviewerSimilarity:
    
    def __init__(self, author_profiles: Dict[str, AuthorProfile]):
        self.author_profiles = author_profiles
        self.similarity_matrix = None
    
    def calculate_similarity_matrix(self, method: str = 'tfidf') -> Dict:
        logger.info("Calculating reviewer-reviewer similarity matrix...")
        
        authors = list(self.author_profiles.keys())
        n = len(authors)
        documents = {
            name: profile.aggregated_text 
            for name, profile in self.author_profiles.items()
        }
        calculator = TFIDFCosineSimilarity()
        calculator.fit(documents)
        import numpy as np
        similarity_matrix = np.zeros((n, n))
        
        for i, author1 in enumerate(authors):
            text1 = documents[author1]
            results = calculator.calculate_similarity(text1, top_k=n)
            
            for author2, score in results:
                j = authors.index(author2)
                similarity_matrix[i, j] = score
        
        return {
            'authors': authors,
            'matrix': similarity_matrix.tolist()
        }
    
    def find_similar_reviewers(self, 
                               author_name: str,
                               top_k: int = 10) -> List[Tuple[str, float]]:
        if author_name not in self.author_profiles:
            return []
        
        profile = self.author_profiles[author_name]
        text = profile.aggregated_text
        documents = {
            name: p.aggregated_text 
            for name, p in self.author_profiles.items()
        }
        
        calculator = TFIDFCosineSimilarity()
        calculator.fit(documents)
        
        results = calculator.calculate_similarity(text, top_k + 1)
        results = [(author, score) for author, score in results if author != author_name]
        
        return results[:top_k]


if __name__ == "__main__":
    dataset_path = "/Users/lrao/Desktop/AppliedAI/Dataset"
    cache_dir = "/Users/lrao/Desktop/AppliedAI/cache"
    
    print("="*80)
    print("REVIEWER RECOMMENDATION SYSTEM - TEST")
    print("="*80)
    recommender = ReviewerRecommender(
        dataset_path=dataset_path,
        cache_dir=cache_dir,
        use_advanced_models=False 
    )
    
    print("\nInitializing system...")
    recommender.initialize(force_rebuild=False)
    stats = recommender.get_statistics()
    print(f"\nSystem Statistics:")
    print(f"  Total Authors: {stats['num_authors']}")
    print(f"  Total Publications: {stats['total_publications']}")
    print(f"  Avg Publications per Author: {stats['avg_publications_per_author']:.2f}")
    methods = recommender.get_available_methods()
    print(f"\nAvailable Methods: {', '.join(methods)}")
    
    print("\n" + "="*80)
    print("System ready for reviewer recommendations!")
    print("="*80)

