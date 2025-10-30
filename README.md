# Reviewer Recommendation System

**Project Goal:** To identify the **top-k relevant reviewers** for a submitted research paper by applying advanced Ensemble NLP techniques across a corpus of over 600 PDF papers authored by 70+ researchers.

---

## Getting Started

This section outlines how to set up and run the application.

### Prerequisites

### Installation and Execution

1.  **Install Dependencies:** Install all required Python packages from `requirements.txt`.

    ```bash
    python3 -m pip install -r requirements.txt
    ```

2.  **Run the Application:** Start the Streamlit web interface.

    ```bash
    streamlit run app.py
    ```

3.  **Usage:** Upload a new paper's PDF through the web interface to instantly receive a list of **top-k recommended reviewers** along with matching explanations.

---


## NLP/ML techniques used 

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

---

## Project File Structure

| File/Folder | Description |
| :--- | :--- |
| `app.py` | **Main application.** Contains the Streamlit UI and high-level orchestration. |
| `requirements.txt` | Python dependency list. |
| `Dataset/` | **Input Data.** Contains author-specific folders holding their PDFs. |
| `cache/` | **Caching Directory.** Persists author profiles, models, citations, and features. |
| **`src/`** | **Source Code Directory.** |
| `src/pdf_parser.py` | Extracts text and metadata from PDFs (using `PyPDF2`, `pdfplumber`, and `Tika`). |
| `src/author_profile.py` | Aggregates text from all an author's papers to build a comprehensive profile. |
| `src/similarity_calculator.py` | Implements classical text similarities (**BM25**, TF-IDF, Jaccard, WMD). |
| `src/advanced_nlp.py` | Implements advanced models (LDA/NMF, S-BERT, SciBERT, E5, Cross-Encoder). |
| `src/faiss_store.py` | Manages the **FAISS vector index** for fast dense similarity search. |
| `src/structural_similarity.py` | Computes structural vectors and similarity from PDF features. |
| `src/visual_similarity.py` | Handles image extraction, pHash, and SSIM-based visual similarity. |
| `src/citation_metrics.py` | Calculates and caches citation-based metrics (H-index). |
| `src/reviewer_recommender.py` | The **main engine** for initialization, ensemble scoring, and drill-down. |
| `src/evaluator.py` | Implements evaluation metrics: Precision@K, Recall@K, NDCG@K, MRR. |

---

## Team Members

| Name | Roll Number |
| :--- | :--- |
| **Monisha Kollipara** | SE22UARI098 |
| **Mullangi Guna Shritha** | SE22UARI102 |
| **Laxmi Sowmya Niharika Peddireddi** | SE22UARI081 |
| **Kuruva Bhuvana Vijaya** | SE22UARI080 |

---

### And other elements included

* **Per-Paper Drill-Down:** For shortlisted reviewers, the system identifies and displays their single **Best Matching Paper** as supporting evidence.
* **Conflict-of-Interest (COI) Flagging:** Flags likely conflicts based on name or affiliation cues (does not influence the score).
* **Reviewer–Reviewer Similarity:** Allows exploration of experts similar to a chosen reviewer, useful for finding alternates.
* **Evaluation Tab:** Enables the computation and reporting of Precision@K, Recall@K, NDCG@K, and MRR using uploaded ground-truth JSON.
* **FAISS Acceleration:** fast dense retrieval for SciBERT/E5/SBERT; indices persist to disk.


## Visual Demonstration:
![WhatsApp Image 2025-10-30 at 20 28 01_06ff692e](https://github.com/user-attachments/assets/6fe193fe-d127-417d-ad1d-6e869b3bd539)
![WhatsApp Image 2025-10-30 at 20 30 18_93061d0c](https://github.com/user-attachments/assets/fa44617e-96e5-4e44-9928-d6c9f530f22a)
![WhatsApp Image 2025-10-30 at 20 30 34_6c1308ee](https://github.com/user-attachments/assets/86ec5de5-e95a-4c45-8a88-918d526704ff)
![WhatsApp Image 2025-10-28 at 17 46 24_f159acbd](https://github.com/user-attachments/assets/374fa11a-15fc-4ba5-8122-52bc863b700a)
![WhatsApp Image 2025-10-28 at 17 56 20_422d8248](https://github.com/user-attachments/assets/8351e231-e9f0-4302-9c00-fcb2ccce39b4)





