import os
import re
import logging
from typing import Optional, Dict, List
from pathlib import Path


import PyPDF2
import pdfplumber
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PDFParser:
    
    def __init__(self, use_tika: bool = False):
        self.use_tika = use_tika
        if use_tika:
            try:
                from tika import parser as tika_parser
                self.tika_parser = tika_parser
                logger.info("Tika parser enabled")
            except ImportError:
                logger.warning("Tika not available, falling back to other parsers")
                self.use_tika = False
    
    def extract_text_pypdf2(self, pdf_path: str) -> Optional[str]:
        try:
            text = []
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text.append(page_text)
            
            extracted_text = '\n'.join(text)
            logger.debug(f"PyPDF2 extracted {len(extracted_text)} characters from {pdf_path}")
            return extracted_text if extracted_text.strip() else None
            
        except Exception as e:
            logger.error(f"PyPDF2 failed for {pdf_path}: {str(e)}")
            return None
    
    def extract_text_pdfplumber(self, pdf_path: str) -> Optional[str]:
        try:
            text = []
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text.append(page_text)
            
            extracted_text = '\n'.join(text)
            logger.debug(f"pdfplumber extracted {len(extracted_text)} characters from {pdf_path}")
            return extracted_text if extracted_text.strip() else None
            
        except Exception as e:
            logger.error(f"pdfplumber failed for {pdf_path}: {str(e)}")
            return None
    
    def extract_text_tika(self, pdf_path: str) -> Optional[str]:
        if not self.use_tika:
            return None
        
        try:
            parsed = self.tika_parser.from_file(pdf_path)
            text = parsed.get('content', '')
            logger.debug(f"Tika extracted {len(text)} characters from {pdf_path}")
            return text if text and text.strip() else None
            
        except Exception as e:
            logger.error(f"Tika failed for {pdf_path}: {str(e)}")
            return None
    
    def clean_text(self, text: str) -> str:
        if not text:
            return ""

        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s\.\,\;\:\!\?\-\(\)]', ' ', text)
        text = re.sub(r' +', ' ', text)
        
        return text.strip()
    
    def extract_metadata(self, pdf_path: str) -> Dict[str, any]:
        metadata = {
            'filename': os.path.basename(pdf_path),
            'path': pdf_path,
            'num_pages': 0,
            'file_size': os.path.getsize(pdf_path) if os.path.exists(pdf_path) else 0
        }
        
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                metadata['num_pages'] = len(pdf_reader.pages)
                if pdf_reader.metadata:
                    metadata['title'] = str(pdf_reader.metadata.get('/Title', ''))
                    metadata['author'] = str(pdf_reader.metadata.get('/Author', ''))
                    metadata['subject'] = str(pdf_reader.metadata.get('/Subject', ''))
                    metadata['creator'] = str(pdf_reader.metadata.get('/Creator', ''))
                    
        except Exception as e:
            logger.error(f"Failed to extract metadata from {pdf_path}: {str(e)}")
        
        return metadata
    
    def parse(self, pdf_path: str, clean: bool = True) -> Dict[str, any]:
        if not os.path.exists(pdf_path):
            logger.error(f"PDF file not found: {pdf_path}")
            return {'text': '', 'metadata': {}, 'parser_used': None, 'success': False}
        text = None
        parser_used = None
        text = self.extract_text_pdfplumber(pdf_path)
        if text and len(text.strip()) > 20:  
            parser_used = 'pdfplumber'
        if not text:
            text = self.extract_text_pypdf2(pdf_path)
            if text and len(text.strip()) > 20:
                parser_used = 'PyPDF2'
        if not text and self.use_tika:
            text = self.extract_text_tika(pdf_path)
            if text and len(text.strip()) > 20:
                parser_used = 'Tika'
        if text and clean:
            text = self.clean_text(text)
        metadata = self.extract_metadata(pdf_path)
        
        success = text is not None and len(text.strip()) > 20
        
        if not success:
            logger.warning(f"Failed to extract meaningful text from {pdf_path}")
        else:
            logger.info(f"Successfully parsed {pdf_path} using {parser_used}")
        
        return {
            'text': text or '',
            'metadata': metadata,
            'parser_used': parser_used,
            'success': success,
            'text_length': len(text) if text else 0
        }
    
    def batch_parse(self, pdf_paths: List[str], clean: bool = True) -> List[Dict[str, any]]:
        results = []
        for pdf_path in pdf_paths:
            result = self.parse(pdf_path, clean=clean)
            results.append(result)
        
        return results


def get_all_pdfs(directory: str, recursive: bool = True) -> List[str]:
    pdf_files = []
    
    if recursive:
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.lower().endswith('.pdf'):
                    pdf_files.append(os.path.join(root, file))
    else:
        for file in os.listdir(directory):
            if file.lower().endswith('.pdf'):
                pdf_files.append(os.path.join(directory, file))
    
    return sorted(pdf_files)


if __name__ == "__main__":
    parser = PDFParser(use_tika=False)
    dataset_path = "/Users/lrao/Desktop/AppliedAI/Dataset"
    sample_pdfs = get_all_pdfs(dataset_path)[:3]
    
    print(f"Found {len(sample_pdfs)} PDFs in dataset")
    print("\nTesting parser on sample PDFs:")
    
    for pdf_path in sample_pdfs:
        result = parser.parse(pdf_path)
        print(f"\nFile: {result['metadata']['filename']}")
        print(f"Parser used: {result['parser_used']}")
        print(f"Success: {result['success']}")
        print(f"Text length: {result['text_length']}")
        print(f"Preview: {result['text'][:200]}...")

