import os
import logging
from typing import Dict, List, Tuple, Optional
import numpy as np
from PIL import Image
import io
import fitz  
from skimage.metrics import structural_similarity as ssim

import pdfplumber

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VisualSimilarity: 
    def __init__(self):
        self.document_visual_features = {}
        self.has_pillow = True
        self.author_images = {}  # author -> List[Image.Image]
        self.author_phashes = {}  # author -> List[str]
        self.author_ssim_vectors = {}  # author -> List[List[int]] (grayscale small images)
        try:
            import imagehash
            self.has_imagehash = True
            self.imagehash = imagehash
        except ImportError:
            self.has_imagehash = False
        
        try:
            from PIL import Image
        except ImportError:
            logger.warning("PIL not available. Visual similarity will use basic features only.")
            self.has_pillow = False
    
    def fit(self, documents: Dict[str, str]):
        logger.info("Extracting visual features...")
        
        for author_name, text in documents.items():

            features = self._extract_visual_markers(text)
            self.document_visual_features[author_name] = features
            if hasattr(self, 'author_pdfs') and self.author_pdfs.get(author_name):
                images = []
                for pdf_path in self.author_pdfs[author_name]:
                    images.extend(self.extract_images_from_pdf(pdf_path))
                images = images[:10]
                self.author_images[author_name] = images
                if images:
                    self.author_phashes[author_name] = [
                        str(self.imagehash.phash(img)) if self.has_imagehash else ""
                    for img in images]
                    import numpy as _np
                    self.author_ssim_vectors[author_name] = [
                        _np.array(img.convert('L').resize((64, 64))).astype(int).tolist() for img in images
                    ]
        
        logger.info(f"Extracted visual features for {len(documents)} authors")
    
    def _extract_visual_markers(self, text: str) -> Dict:
        import re
        
        features = {
            'num_figures': len(re.findall(r'fig(?:ure)?\s*\d+|fig\s*\d+', text, re.IGNORECASE)),
            'num_tables': len(re.findall(r'table\s*\d+', text, re.IGNORECASE)),
            'num_equations': len(re.findall(r'equation\s*\d+|eq\s*\d+', text, re.IGNORECASE)),
            'has_charts': bool(re.search(r'chart|graph|plot|diagram', text, re.IGNORECASE)),
            'has_images': bool(re.search(r'image|photograph|picture', text, re.IGNORECASE)),
            'has_diagrams': bool(re.search(r'diagram|flowchart|schematic', text, re.IGNORECASE)),
            'visualization_density': self._calculate_viz_density(text),
        }
        
        return features
    
    def _calculate_viz_density(self, text: str) -> float:
        import re
        
        visual_keywords = [
            'figure', 'table', 'chart', 'graph', 'plot', 'diagram',
            'image', 'illustration', 'visualization', 'shown', 'depicted'
        ]
        
        word_count = len(text.split())
        if word_count == 0:
            return 0.0
        
        visual_count = sum(len(re.findall(rf'\b{kw}\b', text, re.IGNORECASE)) 
                          for kw in visual_keywords)
        
        return visual_count / word_count * 1000 
    
    def calculate_similarity(self, query_text: str, top_k: int = 10) -> List[Tuple[str, float]]:
        query_features = self._extract_visual_markers(query_text)
        similarities = []
        
        for author_name, doc_features in self.document_visual_features.items():
            similarity = self._calculate_feature_similarity(query_features, doc_features)
            similarities.append((author_name, similarity))
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:top_k]

    def calculate_image_similarity_pipeline(self, query_pdf_path: str, top_k: int = 10) -> List[Tuple[str, float]]:
        query_images = self.extract_images_from_pdf(query_pdf_path)
        if not query_images:
            return []
        scores: List[Tuple[str, float]] = []
        for author, images in self.author_images.items():
            if not images:
                continue
            phash_score = self._phash_set_similarity(query_images, images)
            ssim_score = self._ssim_set_similarity(query_images, images)
            score = 0.5 * phash_score + 0.5 * ssim_score
            scores.append((author, float(score)))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    def extract_images_from_pdf(self, pdf_path: str) -> List[Image.Image]:
        images: List[Image.Image] = []
        try:
            doc = fitz.open(pdf_path)
            for page_index in range(len(doc)):
                page = doc[page_index]
                for img in page.get_images(full=True):
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image.get("image")
                    if image_bytes:
                        try:
                            pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                            images.append(pil_img)
                        except Exception as e:
                            logger.debug(f"Image decode failed: {e}")
        except Exception as e:
            logger.error(f"Failed to extract images from {pdf_path}: {str(e)}")
        return images

    def _phash_set_similarity(self, imgs1: List[Image.Image], imgs2: List[Image.Image]) -> float:
        if not self.has_imagehash or not imgs1 or not imgs2:
            return 0.0
        try:
            hashes1 = [self.imagehash.phash(img) for img in imgs1[:5]]
            hashes2 = [self.imagehash.phash(img) for img in imgs2[:10]]
            def best_match(h, hs):
                diffs = [h - h2 for h2 in hs]
                best = min(diffs) if diffs else 64
                return 1.0 - best / 64.0
            sims = [max(best_match(h, hashes2), 0.0) for h in hashes1]
            return float(sum(sims) / len(sims)) if sims else 0.0
        except Exception:
            return 0.0

    def _ssim_set_similarity(self, imgs1: List[Image.Image], imgs2: List[Image.Image]) -> float:
        if not imgs1 or not imgs2:
            return 0.0
        import numpy as np
        sims = []
        for img1 in imgs1[:5]:
            g1 = np.array(img1.convert('L').resize((256, 256)))
            best = 0.0
            for img2 in imgs2[:10]:
                g2 = np.array(img2.convert('L').resize((256, 256)))
                try:
                    val = ssim(g1, g2, data_range=255)
                except Exception:
                    val = 0.0
                if val > best:
                    best = val
            sims.append(best)
        return float(sum(sims) / len(sims)) if sims else 0.0

    def save_cache(self, cache_dir: str):
        import json, os
        os.makedirs(cache_dir, exist_ok=True)
        path = os.path.join(cache_dir, 'visual_cache.json')
        data = {
            'document_visual_features': self.document_visual_features,
            'author_phashes': self.author_phashes,
            'author_ssim_vectors': self.author_ssim_vectors
        }
        with open(path, 'w') as f:
            json.dump(data, f)
        logger.info(f"Visual cache saved to {path}")

    def load_cache(self, cache_dir: str) -> bool:
        import json, os
        path = os.path.join(cache_dir, 'visual_cache.json')
        if not os.path.exists(path):
            return False
        try:
            with open(path, 'r') as f:
                data = json.load(f)
            self.document_visual_features = data.get('document_visual_features', {})
            self.author_phashes = data.get('author_phashes', {})
            self.author_ssim_vectors = data.get('author_ssim_vectors', {})
            self.author_images = {}
            return True
        except Exception as e:
            logger.warning(f"Failed to load visual cache: {e}")
            return False
    
    def _calculate_feature_similarity(self, features1: Dict, features2: Dict) -> float:
        score = 0.0
        total_weight = 0.0
        numeric_features = ['num_figures', 'num_tables', 'num_equations', 'visualization_density']
        
        for feature in numeric_features:
            val1 = features1.get(feature, 0)
            val2 = features2.get(feature, 0)
            max_val = max(val1, val2, 1)
            min_val = min(val1, val2)
            
            feature_sim = min_val / max_val
            score += feature_sim
            total_weight += 1.0
        boolean_features = ['has_charts', 'has_images', 'has_diagrams']
        
        for feature in boolean_features:
            if features1.get(feature, False) == features2.get(feature, False):
                score += 1.0
            total_weight += 1.0
        
        return score / total_weight if total_weight > 0 else 0.0


class ImageHashSimilarity:
    def __init__(self):
        try:
            import imagehash
            self.has_imagehash = True
            self.imagehash = imagehash
        except ImportError:
            logger.warning("imagehash not available. Install with: pip install imagehash")
            self.has_imagehash = False
    
    def extract_images_from_pdf(self, pdf_path: str) -> List[Image.Image]:
        images: List[Image.Image] = []
        try:
            doc = fitz.open(pdf_path)
            for page_index in range(len(doc)):
                page = doc[page_index]
                for img in page.get_images(full=True):
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image.get("image")
                    if image_bytes:
                        try:
                            pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                            images.append(pil_img)
                        except Exception as e:
                            logger.debug(f"Image decode failed: {e}")
        except Exception as e:
            logger.error(f"Failed to extract images from {pdf_path}: {str(e)}")
        return images
    
    def calculate_image_similarity(self, img1: Image.Image, img2: Image.Image) -> float:
        if not self.has_imagehash:
            return 0.0
        
        try:
            hash1 = self.imagehash.phash(img1)
            hash2 = self.imagehash.phash(img2)
            diff = hash1 - hash2
            max_diff = 64  
            similarity = 1.0 - (diff / max_diff)
            
            return max(0.0, similarity)
        except Exception as e:
            logger.error(f"Failed to calculate image similarity: {str(e)}")
            return 0.0


class FigureExtractor:
    def __init__(self):
        self.figure_metadata = {}
    
    def extract_figures(self, pdf_path: str) -> List[Dict]:
        figures = []
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    text = page.extract_text()
                    if not text:
                        continue
                    import re
                    figure_refs = re.finditer(
                        r'(fig(?:ure)?\s*\d+[a-z]?)[:\s]+(.*?)(?=\n|fig(?:ure)?|\Z)',
                        text,
                        re.IGNORECASE | re.DOTALL
                    )
                    
                    for match in figure_refs:
                        figure_id = match.group(1)
                        caption = match.group(2)[:200]
                        
                        figures.append({
                            'id': figure_id,
                            'caption': caption.strip(),
                            'page': page_num,
                            'type': self._classify_figure_type(caption)
                        })
        
        except Exception as e:
            logger.error(f"Failed to extract figures from {pdf_path}: {str(e)}")
        
        return figures
    
    def _classify_figure_type(self, caption: str) -> str:
        caption_lower = caption.lower()
        
        if any(word in caption_lower for word in ['bar chart', 'histogram', 'bar graph']):
            return 'bar_chart'
        elif any(word in caption_lower for word in ['line graph', 'plot', 'curve']):
            return 'line_plot'
        elif any(word in caption_lower for word in ['scatter', 'distribution']):
            return 'scatter_plot'
        elif any(word in caption_lower for word in ['flowchart', 'flow chart', 'diagram']):
            return 'flowchart'
        elif any(word in caption_lower for word in ['architecture', 'model', 'network']):
            return 'architecture'
        elif any(word in caption_lower for word in ['screenshot', 'interface']):
            return 'screenshot'
        else:
            return 'other'


if __name__ == "__main__":
    print("Testing Visual Similarity")
    print("="*80)
    
    documents = {
        'Author1': 'Figure 1 shows the neural network architecture. Table 1 presents results. As depicted in Figure 2, the model performs well.',
        'Author2': 'The diagram illustrates the process. Chart 1 compares methods. Image analysis revealed patterns.',
    }
    
    visual = VisualSimilarity()
    visual.fit(documents)
    
    query = 'Figure 1 depicts the CNN architecture. Table 1 shows accuracy. The plot in Figure 2 demonstrates convergence.'
    
    results = visual.calculate_similarity(query, top_k=2)
    
    print("\nVisual Similarity Results:")
    for author, score in results:
        print(f"  {author}: {score:.4f}")

