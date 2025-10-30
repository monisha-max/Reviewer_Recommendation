# Reviewer Recommendation System 
Find top-k reviewers for a research paper using NLP over 700+ PDFs (70+ authors).

## 1) What each file does

- app.py: Streamlit app (UI + orchestration)
- requirements.txt: Python dependencies
- Dataset/: All authors’ PDFs (folder-per-author)
- cache/: All caches (profiles, models, citations, structural/visual)

Source (src/)
- pdf_parser.py: Extracts text/metadata from PDFs (PyPDF2/pdfplumber/Tika)
- author_profile.py: Builds author profiles (aggregated text across papers)
- similarity_calculator.py: Classic text similarities (TF‑IDF+Cosine, BM25, Jaccard, Keywords, WMD)
- advanced_nlp.py: Advanced models (LDA/NMF, Sentence‑BERT, SciBERT, E5, Cross‑Encoder)
- faiss_store.py: FAISS vector index (fast similarity search)
- structural_similarity.py: Structural features from PDFs; structural vectors + similarity
- visual_similarity.py: Visual features; image extraction + pHash/SSIM similarity
- citation_metrics.py: H‑index/citation counts with JSON caching
- reviewer_recommender.py: Main engine (initialization, ensemble, per‑paper drill‑down)
- evaluator.py: Precision@K, Recall@K, NDCG@K, MRR (optional evaluation)

## 2) How to set up

2. Install deps
```bash
python3 -m pip install -r requirements.txt
```

3. Run the app
```bash
streamlit run app.py
```
4. Use
- Upload a PDF → get top‑k reviewers with explanations.

## 3) NLP/ML techniques used (concise)

Text-based
- TF‑IDF + Cosine: lexical similarity
- BM25: probabilistic term relevance baseline
- Jaccard (bigrams): token overlap
- Keyword matching: keyword overlap
- WMD (true/fallback): semantic distance over word embeddings

Topic/Embeddings
- LDA, NMF: topic distributions
- Sentence‑BERT, SciBERT, E5: dense semantic embeddings (FAISS‑accelerated)
- Cross‑Encoder reranker: reranks top candidates with joint attention

Structure/Visual/Citations
- Structural similarity: section/paragraph/list/citation/table/figure features
- Visual similarity (images): image extraction + pHash + SSIM
- Visual similarity (text): fallback via textual visual markers
- Citation impact: h‑index and citation counts (cached)

Ensemble
- Weighted blend of all signals; Reciprocal Rank Fusion (RRF) adds a small consensus boost across method rankings.
- Optional per‑paper drill‑down for shortlisted authors (shows “Best Matching Paper”).
 
## 4) Everything else we included (concise)
 
- Reviewer–Reviewer Similarity: explore similar experts to any reviewer (useful for alternates/COI).
- Evaluation tab: upload ground‑truth JSON → compute Precision@K/Recall@K/NDCG/MRR and download a report.
- Cross‑Encoder Reranker: improves precision by re‑scoring the shortlist with a cross‑encoder.
- Per‑Paper Drill‑Down: for shortlisted authors, finds the best matching paper and shows it as evidence.
- Conflict‑of‑Interest (COI): flags likely conflicts (name/affiliation cues) without boosting scores.
- FAISS Acceleration: fast dense retrieval for SciBERT/E5/SBERT; indices persist to disk.
- Caching Everywhere: author profiles, classical models, advanced models, citation metrics, structural/visual (JSON/PKL/FAISS).
- Async Warmup: structural/visual/citations load in the background so the UI is responsive immediately.

