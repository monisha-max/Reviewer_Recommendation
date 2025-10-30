import os
import re
import json
import logging
from typing import Dict, List, Tuple, Optional
from collections import Counter
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

import pdfplumber
from difflib import SequenceMatcher
import fitz  # PyMuPDF

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DocumentStructure:
    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
        self.headings = []
        self.sections = []
        self.subsections = []
        self.structure_tree = {}
        self.page_structure = []
        
    def extract_structure(self):
        try:
            with pdfplumber.open(self.pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    #extract text with formatting
                    text = page.extract_text()
                    if not text:
                        continue
                    
                    #Analyze structure
                    page_structure = self._analyze_page_structure(text, page_num)
                    self.page_structure.append(page_structure)
                    
                    # Extract headings(lines that are all caps or numbered)
                    lines = text.split('\n')
                    for line in lines:
                        line = line.strip()
                        if self._is_heading(line):
                            self.headings.append({
                                'text': line,
                                'page': page_num,
                                'level': self._get_heading_level(line)
                            })
            
            self._build_structure_tree()
            logger.info(f"Extracted {len(self.headings)} headings from {self.pdf_path}")
            
        except Exception as e:
            logger.error(f"Failed to extract structure from {self.pdf_path}: {str(e)}")
    
    def _is_heading(self, line: str) -> bool:
        """Determine if a line is likely a heading"""
        if len(line) < 3 or len(line) > 200:
            return False
        
        # Check for common heading patterns
        patterns = [
            r'^\d+\.?\s+[A-Z]',  # "1. Introduction" or "1 INTRODUCTION"
            r'^[A-Z][A-Z\s]{3,}$',  # All caps (min 3 chars)
            r'^\d+\.\d+\s+',  # "1.1 Subsection"
            r'^[IVX]+\.\s+',  # Roman numerals
            r'^(Introduction|Abstract|Conclusion|References|Methods|Results|Discussion)',
        ]
        
        for pattern in patterns:
            if re.match(pattern, line, re.IGNORECASE):
                return True
        
        return False
    
    def _get_heading_level(self, heading: str) -> int:
        if re.match(r'^\d+\.\d+\.\d+', heading):
            return 3
        elif re.match(r'^\d+\.\d+', heading):
            return 2
        elif re.match(r'^\d+\.', heading):
            return 1
        if heading.isupper():
            return 1
        
        return 2
    
    def _analyze_page_structure(self, text: str, page_num: int) -> Dict:
        lines = text.split('\n')
        
        structure = {
            'page': page_num,
            'num_lines': len(lines),
            'num_paragraphs': len([l for l in lines if len(l.strip()) > 50]),
            'has_equations': bool(re.search(r'[∫∑∏∂∆√]|\\[a-z]+\{', text)),
            'has_citations': bool(re.search(r'\[\d+\]|\(\d{4}\)', text)),
            'has_lists': bool(re.search(r'^\s*[-•]\s+', text, re.MULTILINE)),
        }
        
        return structure
    
    def _build_structure_tree(self):
        self.structure_tree = {
            'headings': self.headings,
            'total_headings': len(self.headings),
            'sections': len([h for h in self.headings if h['level'] == 1]),
            'subsections': len([h for h in self.headings if h['level'] == 2]),
        }
    
    def get_structure_vector(self) -> np.ndarray:
        features = [
            len(self.headings),
            len([h for h in self.headings if h['level'] == 1]),
            len([h for h in self.headings if h['level'] == 2]),
            len([h for h in self.headings if h['level'] == 3]),
            len(self.page_structure),
            sum(p.get('has_equations', 0) for p in self.page_structure),
            sum(p.get('has_citations', 0) for p in self.page_structure),
            sum(p.get('has_lists', 0) for p in self.page_structure),
        ]
        
        return np.array(features, dtype=float)


class StructuralSimilarity:
    def __init__(self):
        self.document_structures = {}
        self.structure_vectors = {}
        self.author_pdfs = {}
        self._cache_loaded = False
    
    def set_author_pdfs(self, author_pdfs: Dict[str, List[str]]):
        self.author_pdfs = author_pdfs or {}
    
    def fit(self, documents: Dict[str, str]):
        logger.info("Extracting document structures...")
        
        use_pdf = bool(self.author_pdfs)
        for author_name, text in documents.items():
            if use_pdf and author_name in self.author_pdfs and self.author_pdfs[author_name]:
                # Aggregate structure across author's PDFs
                agg = {
                    'num_sections': 0,
                    'num_paragraphs': 0,
                    'has_equations': False,
                    'has_citations': 0,
                    'has_lists': 0,
                    'has_tables': False,
                    'has_figures': False,
                    'avg_sentence_length': 0.0,
                }
                total_docs = 0
                for pdf_path in self.author_pdfs[author_name][:3]:
                    try:
                        s = self._extract_pdf_structure_fitz(pdf_path, max_pages=5)
                        agg['num_sections'] += s.get('num_sections', 0)
                        agg['num_paragraphs'] += s.get('num_paragraphs', 0)
                        agg['has_equations'] = agg['has_equations'] or bool(s.get('has_equations', False))
                        agg['has_citations'] += s.get('has_citations', 0)
                        agg['has_lists'] += s.get('has_lists', 0)
                        agg['has_tables'] = agg['has_tables'] or bool(s.get('has_tables', False))
                        agg['has_figures'] = agg['has_figures'] or bool(s.get('has_figures', False))
                        agg['avg_sentence_length'] += s.get('avg_sentence_length', 0.0)
                        total_docs += 1
                    except Exception as e:
                        logger.debug(f"PDF structure parse failed for {pdf_path}: {e}")
                if total_docs > 0:
                    agg['avg_sentence_length'] = agg['avg_sentence_length'] / total_docs
                structure = agg
            else:
                structure = self._extract_text_structure(text)
            self.document_structures[author_name] = structure
            self.structure_vectors[author_name] = self._text_to_vector(structure)
        
        logger.info(f"Extracted structure for {len(documents)} authors")
    
    def _extract_text_structure(self, text: str) -> Dict:
        lines = text.split('\n')
        structure = {
            'num_sections': len(re.findall(r'\n[A-Z][A-Z\s]{5,}\n', text)),
            'num_paragraphs': len([l for l in lines if len(l.strip()) > 100]),
            'has_equations': bool(re.search(r'equation|formula|∫|∑|∏', text, re.IGNORECASE)),
            'has_citations': len(re.findall(r'\[\d+\]|\(\d{4}\)', text)),
            'has_lists': len(re.findall(r'^\s*[-•]\s+', text, re.MULTILINE)),
            'has_tables': bool(re.search(r'table|column|row', text, re.IGNORECASE)),
            'has_figures': bool(re.search(r'figure|fig\.|image', text, re.IGNORECASE)),
            'avg_sentence_length': np.mean([len(s.split()) for s in re.split(r'[.!?]', text) if len(s.split()) > 3])
        }
        
        return structure
    
    def _text_to_vector(self, structure: Dict) -> np.ndarray:
        features = [
            structure.get('num_sections', 0),
            structure.get('num_paragraphs', 0),
            int(structure.get('has_equations', False)),
            structure.get('has_citations', 0),
            structure.get('has_lists', 0),
            int(structure.get('has_tables', False)),
            int(structure.get('has_figures', False)),
            structure.get('avg_sentence_length', 0),
        ]
        
        return np.array(features, dtype=float)
    
    def calculate_similarity(self, query_text: str, top_k: int = 10) -> List[Tuple[str, float]]:
        query_structure = self._extract_text_structure(query_text)
        query_vector = self._text_to_vector(query_structure).reshape(1, -1)
        
        similarities = []
        
        for author_name, doc_vector in self.structure_vectors.items():
            doc_vector_reshaped = doc_vector.reshape(1, -1)
            if np.linalg.norm(query_vector) > 0 and np.linalg.norm(doc_vector_reshaped) > 0:
                similarity = cosine_similarity(query_vector, doc_vector_reshaped)[0][0]
            else:
                similarity = 0.0
            
            similarities.append((author_name, float(similarity)))
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:top_k]

    def calculate_similarity_from_pdf(self, pdf_path: str, top_k: int = 10) -> List[Tuple[str, float]]:
        try:
            query_structure = self._extract_pdf_structure_fitz(pdf_path)
            query_vector = self._text_to_vector(query_structure).reshape(1, -1)
        except Exception as e:
            logger.warning(f"Failed PDF structural parse for query ({e}). Falling back to text heuristics.")
            return []
        similarities = []
        for author_name, doc_vector in self.structure_vectors.items():
            doc_vector_reshaped = doc_vector.reshape(1, -1)
            if np.linalg.norm(query_vector) > 0 and np.linalg.norm(doc_vector_reshaped) > 0:
                sim = cosine_similarity(query_vector, doc_vector_reshaped)[0][0]
            else:
                sim = 0.0
            similarities.append((author_name, float(sim)))
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]

    def save_cache(self, cache_dir: str):
        try:
            os.makedirs(cache_dir, exist_ok=True)
            path = os.path.join(cache_dir, 'structural_cache.json')
            serializable = {k: v.tolist() if hasattr(v, 'tolist') else list(v) for k, v in self.structure_vectors.items()}
            with open(path, 'w') as f:
                json.dump({'structure_vectors': serializable}, f)
            logger.info(f"Structural cache saved to {path}")
        except Exception as e:
            logger.warning(f"Failed to save structural cache: {e}")

    def load_cache(self, cache_dir: str) -> bool:
        try:
            path = os.path.join(cache_dir, 'structural_cache.json')
            if not os.path.exists(path):
                return False
            with open(path, 'r') as f:
                data = json.load(f)
            vectors = data.get('structure_vectors', {})
            import numpy as np
            self.structure_vectors = {k: np.array(v, dtype=float) for k, v in vectors.items()}
            self._cache_loaded = True
            logger.info(f"Loaded structural cache from {path} ({len(self.structure_vectors)} authors)")
            return True
        except Exception as e:
            logger.warning(f"Failed to load structural cache: {e}")
            return False

    def _extract_pdf_structure_fitz(self, pdf_path: str, max_pages: int = 5) -> Dict:
        doc = fitz.open(pdf_path)
        heading_spans = []
        sizes = []
        total_lines = 0
        long_lines = 0
        num_lists = 0
        num_citations = 0
        has_tables = False
        has_figures = False
        import re
        for i, page in enumerate(doc):
            if i >= max_pages:
                break
            data = page.get_text("dict")
            for block in data.get('blocks', []):
                for line in block.get('lines', []):
                    total_lines += 1
                    spans = line.get('spans', [])
                    if not spans:
                        continue
                    # Concatenate text in spans for this line
                    line_text = ''.join(span.get('text', '') for span in spans).strip()
                    if len(line_text.split()) > 15:
                        long_lines += 1
                    # Detect lists
                    if re.match(r'^\s*([\-•\*]|\d+\.|[a-z]\))\s+', line_text.lower()):
                        num_lists += 1
                    # Citations patterns
                    num_citations += len(re.findall(r'\[(?:\d+|[A-Za-z]+\s?\d{4})\]|\(\d{4}\)', line_text))
                    if re.search(r'table\s*\d+', line_text, re.IGNORECASE):
                        has_tables = True
                    if re.search(r'fig(?:ure)?\s*\d+', line_text, re.IGNORECASE):
                        has_figures = True
                    for span in spans:
                        size = span.get('size', 0)
                        sizes.append(size)
                        if size > 0:
                            is_numbered = bool(re.match(r'^(\d+(?:\.\d+)*)\s+', line_text))
                            is_upper = (line_text.isupper() and len(line_text) > 3)
                            heading_spans.append({
                                'text': line_text[:200],
                                'size': size,
                                'is_numbered': is_numbered,
                                'is_upper': is_upper
                            })
        size_threshold = np.percentile(sizes, 75) if sizes else 0
        headings = []
        for h in heading_spans:
            if h['size'] >= size_threshold or h['is_numbered'] or h['is_upper']:
                level = 1
                m = re.match(r'^(\d+(?:\.(\d+))*)\s+', h['text'])
                if m:
                    dots = m.group(1).count('.')
                    level = min(3, 1 + dots)
                headings.append({'text': h['text'], 'level': level})
        structure = {
            'headings': headings,
            'num_sections': len([h for h in headings if h['level'] == 1]),
            'num_paragraphs': long_lines,
            'has_equations': False,  
            'has_citations': num_citations,
            'has_lists': num_lists,
            'has_tables': has_tables,
            'has_figures': has_figures,
            'avg_sentence_length': 0.0  
        }
        return structure
    
    def calculate_tree_edit_distance(self, struct1: Dict, struct2: Dict) -> float:
        headings1 = struct1.get('headings', [])
        headings2 = struct2.get('headings', [])
        
        seq1 = ' '.join([h.get('text', '')[:20] for h in headings1])
        seq2 = ' '.join([h.get('text', '')[:20] for h in headings2])
        similarity = SequenceMatcher(None, seq1, seq2).ratio()
        
        return 1.0 - similarity 


if __name__ == "__main__":
    print("Testing Structural Similarity")
    print("="*80)
    
    documents = {
        'Author1': 'Introduction\n\nMachine learning methods.\n\nMethodology\n\nWe use neural networks.\n\nResults\n\nOur approach achieves 95% accuracy.\n\nConclusion\n\nThis work demonstrates...',
        'Author2': 'Abstract\n\nMedical diagnosis system.\n\nMethods\n\nClinical data analysis.\n\nResults\n\nPatient outcomes improved.\n\nDiscussion\n\nFuture work...',
    }
    
    structural = StructuralSimilarity()
    structural.fit(documents)
    
    query = 'Introduction\n\nDeep learning for computer vision.\n\nMethodology\n\nConvolutional neural networks.\n\nExperiments\n\nResults on ImageNet.\n\nConclusion\n\nState-of-the-art performance.'
    
    results = structural.calculate_similarity(query, top_k=2)
    
    print("\nStructural Similarity Results:")
    for author, score in results:
        print(f"  {author}: {score:.4f}")

