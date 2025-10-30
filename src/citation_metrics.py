import os
import json
import logging
import time
from typing import Dict, List, Tuple, Optional
import requests
from collections import defaultdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CitationMetrics:
    
    def __init__(self, api_key: Optional[str] = None, cache_dir: str = "cache"):
        self.api_key = api_key
        self.cache_dir = cache_dir
        self.author_metrics = {}
        self.rate_limit_delay = 1.0 
        self.semantic_scholar_base = "https://api.semanticscholar.org/graph/v1"
        self.openalex_base = "https://api.openalex.org"
        os.makedirs(cache_dir, exist_ok=True)
        self._load_from_cache()
        
    def fetch_author_metrics(self, author_name: str) -> Dict:
        try:
            metrics = self._fetch_from_semantic_scholar(author_name)
            
            if not metrics:
                metrics = self._fetch_from_openalex(author_name)
            if not metrics or not isinstance(metrics, dict):
                return self._get_default_metrics()
            
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to fetch metrics for {author_name}: {str(e)}")
            return self._get_default_metrics()
    
    def _fetch_from_semantic_scholar(self, author_name: str) -> Optional[Dict]:
        try:
            search_url = f"{self.semantic_scholar_base}/author/search"
            params = {'query': author_name, 'limit': 1}
            
            headers = {}
            if self.api_key:
                headers['x-api-key'] = self.api_key
            
            response = requests.get(search_url, params=params, headers=headers, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                if data.get('data') and len(data['data']) > 0:
                    author_data = data['data'][0]
                    author_id = author_data.get('authorId')
                    time.sleep(self.rate_limit_delay)
                    author_url = f"{self.semantic_scholar_base}/author/{author_id}"
                    author_response = requests.get(
                        author_url,
                        params={'fields': 'name,citationCount,hIndex,paperCount,papers.citationCount'},
                        headers=headers,
                        timeout=10
                    )
                    
                    if author_response.status_code == 200:
                        author_info = author_response.json()
                        
                        metrics = {
                            'name': author_info.get('name', author_name),
                            'total_citations': author_info.get('citationCount', 0),
                            'h_index': author_info.get('hIndex', 0),
                            'paper_count': author_info.get('paperCount', 0),
                            'citations_per_paper': 0,
                            'top_papers_citations': [],
                            'source': 'Semantic Scholar'
                        }
                        papers = author_info.get('papers', [])
                        citations = [p.get('citationCount', 0) for p in papers if p.get('citationCount')]
                        if citations:
                            metrics['top_papers_citations'] = sorted(citations, reverse=True)[:5]
                            metrics['citations_per_paper'] = sum(citations) / len(citations)
                        
                        logger.info(f"Fetched metrics for {author_name} from Semantic Scholar")
                        return metrics
            
            return None
            
        except Exception as e:
            logger.debug(f"Semantic Scholar fetch failed: {str(e)}")
            return None
    
    def _fetch_from_openalex(self, author_name: str) -> Optional[Dict]:
        try:
            search_url = f"{self.openalex_base}/authors"
            params = {
                'search': author_name,
                'per_page': 1
            }
            
            response = requests.get(search_url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                if data.get('results') and len(data['results']) > 0:
                    author_data = data['results'][0]
                    metrics = {
                        'name': author_data.get('display_name', author_name),
                        'total_citations': author_data.get('cited_by_count', 0),
                        'h_index': author_data.get('summary_stats', {}).get('h_index', 0),
                        'paper_count': author_data.get('works_count', 0),
                        'citations_per_paper': 0,
                        'i10_index': author_data.get('summary_stats', {}).get('i10_index', 0),
                        'source': 'OpenAlex'
                    }
                    if metrics['paper_count'] > 0:
                        metrics['citations_per_paper'] = metrics['total_citations'] / metrics['paper_count']
                    
                    logger.info(f"Fetched metrics for {author_name} from OpenAlex")
                    return metrics
            
            return None
            
        except Exception as e:
            logger.debug(f"OpenAlex fetch failed: {str(e)}")
            return None
    
    def _get_default_metrics(self) -> Dict:
        return {
            'name': 'Unknown',
            'total_citations': 0,
            'h_index': 0,
            'paper_count': 0,
            'citations_per_paper': 0,
            'source': 'None'
        }
    
    def fetch_metrics_for_authors(self, author_names: List[str]) -> Dict[str, Dict]:
        authors_to_fetch = [name for name in author_names if name not in self.author_metrics]
        
        if not authors_to_fetch:
            logger.info(f"All {len(author_names)} authors loaded from cache!")
            return self.author_metrics
        
        logger.info(f"Fetching citation metrics for {len(authors_to_fetch)} authors...")
        logger.info(f"({len(author_names) - len(authors_to_fetch)} already in cache)")
        
        for author_name in authors_to_fetch:
            metrics = self.fetch_author_metrics(author_name)
            if metrics and isinstance(metrics, dict):
                self.author_metrics[author_name] = metrics
            else:
                self.author_metrics[author_name] = self._get_default_metrics()
            time.sleep(self.rate_limit_delay)
        
        logger.info(f"Fetched metrics for {len(self.author_metrics)} authors")
        self._save_to_cache()
        
        return self.author_metrics
    
    def calculate_citation_based_similarity(self, 
                                           query_topic: str,
                                           top_k: int = 10) -> List[Tuple[str, float]]:
        if not self.author_metrics:
            logger.warning("No citation metrics available")
            return []
        scores = []
        max_citations = max((m.get('total_citations', 0) for m in self.author_metrics.values()), default=1)
        max_h_index = max((m.get('h_index', 0) for m in self.author_metrics.values()), default=1)
        max_papers = max((m.get('paper_count', 0) for m in self.author_metrics.values()), default=1)
        
        for author_name, metrics in self.author_metrics.items():
            score = (
                0.4 * (metrics.get('h_index', 0) / max_h_index) +
                0.3 * (metrics.get('total_citations', 0) / max_citations) +
                0.2 * (metrics.get('paper_count', 0) / max_papers) +
                0.1 * min(metrics.get('citations_per_paper', 0) / 100, 1.0)
            )
            
            scores.append((author_name, float(score)))
        scores.sort(key=lambda x: x[1], reverse=True)
        
        return scores[:top_k]
    
    def get_author_rank(self, author_name: str) -> Optional[Dict]:
        metrics = self.author_metrics.get(author_name)
        
        if not metrics:
            return None
        all_h_indices = [m.get('h_index', 0) for m in self.author_metrics.values()]
        all_citations = [m.get('total_citations', 0) for m in self.author_metrics.values()]
        
        h_index = metrics.get('h_index', 0)
        citations = metrics.get('total_citations', 0)
        
        h_index_percentile = sum(1 for h in all_h_indices if h < h_index) / len(all_h_indices) * 100
        citation_percentile = sum(1 for c in all_citations if c < citations) / len(all_citations) * 100
        
        return {
            'author': author_name,
            'h_index': h_index,
            'h_index_percentile': h_index_percentile,
            'total_citations': citations,
            'citation_percentile': citation_percentile,
            'impact_level': self._get_impact_level(h_index, citations)
        }
    
    def _get_impact_level(self, h_index: int, citations: int) -> str:
        if h_index >= 20 or citations >= 5000:
            return 'High Impact'
        elif h_index >= 10 or citations >= 1000:
            return 'Medium Impact'
        elif h_index >= 5 or citations >= 100:
            return 'Emerging'
        else:
            return 'Early Career'
    
    def _load_from_cache(self):
        cache_file = os.path.join(self.cache_dir, "citation_metrics.json")
        
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r') as f:
                    loaded_data = json.load(f)
                if isinstance(loaded_data, dict):
                    self.author_metrics = {
                        k: v for k, v in loaded_data.items() 
                        if isinstance(v, dict)
                    }
                    logger.info(f"Loaded citation metrics for {len(self.author_metrics)} authors from cache")
                else:
                    logger.warning("Invalid cache format, starting fresh")
                    self.author_metrics = {}
            except Exception as e:
                logger.warning(f"Failed to load citation metrics from cache: {e}")
                self.author_metrics = {}
        else:
            logger.info("No citation metrics cache found")
    
    def _save_to_cache(self):
        cache_file = os.path.join(self.cache_dir, "citation_metrics.json")
        
        try:
            with open(cache_file, 'w') as f:
                json.dump(self.author_metrics, f, indent=2)
            logger.info(f"Saved citation metrics for {len(self.author_metrics)} authors to cache")
        except Exception as e:
            logger.error(f"Failed to save citation metrics to cache: {e}")


class CitationSimilarity:
    def __init__(self, citation_metrics: CitationMetrics):
        self.citation_metrics = citation_metrics
    
    def fit(self, author_names: List[str]):
        self.citation_metrics.fetch_metrics_for_authors(author_names)
    
    def calculate_similarity(self, query_text: str, top_k: int = 10) -> List[Tuple[str, float]]:
        return self.citation_metrics.calculate_citation_based_similarity(query_text, top_k)


if __name__ == "__main__":
    print("Testing Citation Metrics")
    print("="*80)
    metrics = CitationMetrics()
    test_authors = ["Geoffrey Hinton", "Yann LeCun"]
    
    print("\nFetching citation metrics (this may take a few seconds)...")
    author_metrics = metrics.fetch_metrics_for_authors(test_authors[:1]) 
    
    for author, data in author_metrics.items():
        print(f"\n{author}:")
        print(f"  H-index: {data.get('h_index', 'N/A')}")
        print(f"  Total Citations: {data.get('total_citations', 'N/A')}")
        print(f"  Papers: {data.get('paper_count', 'N/A')}")
        print(f"  Citations/Paper: {data.get('citations_per_paper', 0):.1f}")
        print(f"  Source: {data.get('source', 'Unknown')}")

