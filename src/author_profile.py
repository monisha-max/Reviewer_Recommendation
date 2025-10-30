import os
import json
import pickle
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict
import logging

from tqdm import tqdm
from pdf_parser import PDFParser, get_all_pdfs

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AuthorProfile:
    def __init__(self, author_name: str):
        self.author_name = author_name
        self.publications = []
        self.aggregated_text = ""
        self.num_publications = 0
        self.total_text_length = 0
    
    def add_publication(self, pdf_path: str, text: str, metadata: Dict):
        clean_metadata = {
            'filename': str(metadata.get('filename', '')),
            'path': str(metadata.get('path', '')),
            'num_pages': int(metadata.get('num_pages', 0)),
            'file_size': int(metadata.get('file_size', 0)),
            'title': str(metadata.get('title', '')),
            'author': str(metadata.get('author', '')),
            'subject': str(metadata.get('subject', '')),
            'creator': str(metadata.get('creator', ''))
        }
        
        self.publications.append({
            'pdf_path': str(pdf_path),
            'text': str(text),
            'metadata': clean_metadata,
            'text_length': len(text)
        })
        self.num_publications += 1
        self.total_text_length += len(text)
    
    def aggregate_texts(self):
        texts = [pub['text'] for pub in self.publications if pub['text']]
        self.aggregated_text = ' '.join(texts)
    
    def get_summary(self) -> Dict:
        return {
            'author_name': self.author_name,
            'num_publications': self.num_publications,
            'total_text_length': self.total_text_length,
            'avg_text_length': self.total_text_length / self.num_publications if self.num_publications > 0 else 0,
            'publication_files': [pub['metadata']['filename'] for pub in self.publications]
        }
    
    def to_dict(self) -> Dict:
        return {
            'author_name': self.author_name,
            'publications': self.publications,
            'aggregated_text': self.aggregated_text,
            'num_publications': self.num_publications,
            'total_text_length': self.total_text_length
        }
    
    @classmethod
    def from_dict(cls, data: Dict):
        profile = cls(data['author_name'])
        profile.publications = data['publications']
        profile.aggregated_text = data['aggregated_text']
        profile.num_publications = data['num_publications']
        profile.total_text_length = data['total_text_length']
        return profile


class AuthorProfileBuilder:
    def __init__(self, dataset_path: str, cache_dir: str = "cache"):
        self.dataset_path = dataset_path
        self.cache_dir = cache_dir
        self.profiles: Dict[str, AuthorProfile] = {}
        self.pdf_parser = PDFParser(use_tika=False)
        os.makedirs(cache_dir, exist_ok=True)
        
        self.cache_file = os.path.join(cache_dir, "author_profiles.pkl")
        self.metadata_file = os.path.join(cache_dir, "author_metadata.json")
    
    def build_profiles(self, force_rebuild: bool = False) -> Dict[str, AuthorProfile]:
        if not force_rebuild and os.path.exists(self.cache_file):
            logger.info("Loading author profiles from cache...")
            return self.load_profiles()
        
        logger.info("Building author profiles from dataset...")
        author_dirs = [d for d in os.listdir(self.dataset_path) 
                      if os.path.isdir(os.path.join(self.dataset_path, d))]
        
        logger.info(f"Found {len(author_dirs)} authors in dataset")
        for author_name in tqdm(author_dirs, desc="Processing authors"):
            author_dir = os.path.join(self.dataset_path, author_name)
            profile = AuthorProfile(author_name)
            pdf_files = get_all_pdfs(author_dir, recursive=False)
            for pdf_path in pdf_files:
                try:
                    result = self.pdf_parser.parse(pdf_path)
                    
                    if result['success']:
                        profile.add_publication(
                            pdf_path=pdf_path,
                            text=result['text'],
                            metadata=result['metadata']
                        )
                    else:
                        logger.warning(f"Failed to parse {pdf_path}")
                        
                except Exception as e:
                    logger.error(f"Error processing {pdf_path}: {str(e)}")
            if profile.num_publications > 0:
                profile.aggregate_texts()
                self.profiles[author_name] = profile
                logger.info(f"Created profile for {author_name}: {profile.num_publications} publications")
            else:
                logger.warning(f"No valid publications found for {author_name}")
        self.save_profiles()
        
        return self.profiles
    
    def save_profiles(self):
        logger.info(f"Saving {len(self.profiles)} author profiles to cache...")
        with open(self.cache_file, 'wb') as f:
            pickle.dump(self.profiles, f)
        metadata = {
            'num_authors': len(self.profiles),
            'authors': {name: profile.get_summary() 
                       for name, profile in self.profiles.items()}
        }
        
        with open(self.metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Profiles saved to {self.cache_file}")
    
    def load_profiles(self) -> Dict[str, AuthorProfile]:
        try:
            with open(self.cache_file, 'rb') as f:
                self.profiles = pickle.load(f)
            
            logger.info(f"Loaded {len(self.profiles)} author profiles from cache")
            return self.profiles
            
        except Exception as e:
            logger.error(f"Failed to load profiles from cache: {str(e)}")
            return {}
    
    def get_profile(self, author_name: str) -> AuthorProfile:
        return self.profiles.get(author_name)
    
    def get_all_profiles(self) -> Dict[str, AuthorProfile]:
        return self.profiles
    
    def get_statistics(self) -> Dict:
        if not self.profiles:
            return {}
        
        total_publications = sum(p.num_publications for p in self.profiles.values())
        total_text_length = sum(p.total_text_length for p in self.profiles.values())
        
        return {
            'num_authors': len(self.profiles),
            'total_publications': total_publications,
            'avg_publications_per_author': total_publications / len(self.profiles),
            'total_text_length': total_text_length,
            'avg_text_per_author': total_text_length / len(self.profiles),
            'authors': list(self.profiles.keys())
        }
    
    def search_authors(self, query: str) -> List[str]:
        query = query.lower()
        return [name for name in self.profiles.keys() if query in name.lower()]


def parse_author_name_from_path(pdf_path: str, dataset_path: str) -> str:
    rel_path = os.path.relpath(pdf_path, dataset_path)
    author_name = rel_path.split(os.sep)[0]
    return author_name


if __name__ == "__main__":
    dataset_path = "/Users/lrao/Desktop/AppliedAI/Dataset"
    cache_dir = "/Users/lrao/Desktop/AppliedAI/cache"
    builder = AuthorProfileBuilder(dataset_path, cache_dir)
    profiles = builder.build_profiles(force_rebuild=False)
    stats = builder.get_statistics()
    
    print("\n" + "="*80)
    print("AUTHOR PROFILE STATISTICS")
    print("="*80)
    print(f"Total Authors: {stats['num_authors']}")
    print(f"Total Publications: {stats['total_publications']}")
    print(f"Average Publications per Author: {stats['avg_publications_per_author']:.2f}")
    print(f"Total Text Length: {stats['total_text_length']:,} characters")
    print(f"Average Text per Author: {stats['avg_text_per_author']:,.0f} characters")
    
    print("\n" + "="*80)
    print("SAMPLE AUTHOR PROFILES")
    print("="*80)
    for i, (author_name, profile) in enumerate(list(profiles.items())[:5]):
        summary = profile.get_summary()
        print(f"\n{i+1}. {author_name}")
        print(f"   Publications: {summary['num_publications']}")
        print(f"   Total Text: {summary['total_text_length']:,} characters")
        print(f"   Files: {', '.join(summary['publication_files'][:3])}")
        if len(summary['publication_files']) > 3:
            print(f"          ... and {len(summary['publication_files']) - 3} more")

