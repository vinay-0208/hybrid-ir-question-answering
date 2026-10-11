# Hybrid Transformer-Based Information Retrieval and Question Answering System

**Course:** CSI4001 — Natural Language Processing and Computational Linguistics  
**Assessment:** Case Study 2 (Question 29)  
**Author:** Vinay Vishwanathan  
**Register Number:** 23MID0429  
**Repository URL:** [https://github.com/vinay-0208/hybrid-ir-question-answering](https://github.com/vinay-0208/hybrid-ir-question-answering)

---

## 1. Project Objective

This project implements a complete, local-first **Hybrid Information Retrieval (IR) and Question Answering (QA)** web application using Python and Streamlit. It demonstrates how traditional lexical search and modern Transformer-based semantic search can be combined, fused, reranked, and grounded to answer technical natural-language questions accurately while abstaining on unanswerable queries:

1. **Heuristic ReAct Query Rewriting & HyDE:** Implements a bounded **2-step Heuristic ReAct-style (`Thought -> Action -> Observation`) query rewriter** for deterministic domain terminology expansion and weighted RRF candidate fusion, alongside **Hypothetical Document Embeddings (HyDE)** using template-based + top-1 corpus pseudo-relevance feedback (PRF) and configurable query-document vector blending ($\alpha=0.6$).
2. **First-Stage Candidate Retrieval (BM25 + FAISS ANN):** Combines traditional lexical retrieval (**Okapi BM25**) and dense semantic retrieval (**Sentence-BERT** bi-encoder indexed via **FAISS `IndexHNSWFlat` / `IndexFlatIP`**) using **Reciprocal Rank Fusion (RRF, $k=60$)** to address the vocabulary mismatch problem without distorting incompatible score scales.
3. **Second-Stage Precision Reranking:** Applies a Transformer cross-encoder (**MS MARCO MiniLM**) to capture deep token-level cross-attention over fused candidates.
4. **Extractive QA & Multi-Passage Grounded Citation Synthesis:** Employs a local **DistilBERT SQuAD** model for extractive span selection alongside a **Multi-Passage Extractive Answer Synthesizer** (`GroundedAnswerSynthesizer`) that combines verified corpus sentences with inline bracketed provenance citations (`[1: passage_id]`) and enforces strict abstention policies (*"Insufficient evidence in the knowledge base to answer this question reliably."*) across all five modes on out-of-domain queries.
5. **Standard IR, QA & RAGAS-Inspired Proxy Evaluation Dashboard:** Measures **Precision@k**, **Recall@k**, **Mean Reciprocal Rank (MRR)**, **NDCG@k**, **Exact Match (EM)**, **Token F1**, and **local RAGAS-inspired heuristic proxy metrics** (**Faithfulness Proxy**, **Answer Relevance Proxy**, **Context Precision**, **Context Recall**) across all five retrieval modalities with normalization strictly bounded in $[0.0, 1.0]$.

---

## 2. Key Features & The Five Retrieval Modes

### Five Supported Retrieval Modes
1. **Mode 1 — `bm25_only` (Lexical Retrieval Only):** Uses Okapi BM25 ($k_1=1.5, b=0.75$) over tokenized passages for fast exact keyword matching, paired with DistilBERT extractive QA and multi-passage sentence synthesis.
2. **Mode 2 — `dense_only` (Dense Semantic Retrieval via FAISS ANN):** Encodes passages and queries into 384-dimensional L2-normalized vectors using `sentence-transformers/all-MiniLM-L6-v2` and searches a FAISS `IndexHNSWFlat` ($M=16$, `METRIC_INNER_PRODUCT`) index.
3. **Mode 3 — `hybrid_rrf` (Hybrid BM25 + Dense with Reciprocal Rank Fusion):** Combines lexical BM25 and dense FAISS rankings via unsupervised Reciprocal Rank Fusion ($\text{RRF}(d) = \sum_{m} \frac{1}{k + r_m(d)}, k=60$) or optional Min-Max Convex Combination.
4. **Mode 4 — `hybrid_rerank` (Hybrid RRF + Cross-Encoder Reranking):** Passes top RRF-fused candidates into a second-stage Transformer cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`) for joint query-passage attention re-scoring.
5. **Mode 5 — `agentic_hyde_rerank` (Heuristic ReAct + HyDE + Hybrid Rerank):** Executes 2-step heuristic ReAct query expansion (`0.65` original + `0.35` expanded BM25 RRF) and template/PRF HyDE vector blending ($\mathbf{v}_{\text{blend}} = 0.6 \mathbf{v}_q + 0.4 \mathbf{v}_{\text{hyde}}$) prior to RRF fusion and Cross-Encoder reranking.

### Additional Application Features
- **Multi-Modal Document Ingestion:** Supports `.txt` and `.pdf` uploads, built-in sample knowledge base (6 documents, 13 passages), and configurable sliding-window passage segmentation (`chunk_size=150` words, `overlap=30` words).
- **Interactive 4-Tab Streamlit Dashboard:**
  - **Tab 1 (`Ask a Question`):** Live query execution, Heuristic ReAct trace, HyDE document preview, multi-passage cited answer, DistilBERT span, RAGAS proxy metrics, latency breakdown, and JSON/CSV export.
  - **Tab 2 (`Side-by-Side Comparison`):** Simultaneous execution and comparison across all 5 retrieval modes on any user query.
  - **Tab 3 (`Document Ingestion & Corpus`):** Upload `.txt`/`.pdf` files, inspect passage chunks, and rebuild BM25 + FAISS indexes.
  - **Tab 4 (`Evaluation Dashboard`):** Run the 10-query benchmark across all 5 modes with visual comparison charts.
- **Offline / Lexical Fallback Mode:** Dedicated sidebar toggle enabling 100% offline pure-Python lexical execution without neural weights.

---

## 3. Technology Stack

| Component | Library / Model | Role in Pipeline |
| :--- | :--- | :--- |
| **Frontend Dashboard** | `streamlit>=1.35.0` | Interactive 4-tab web interface for querying, comparison, ingestion, and evaluation |
| **Lexical Search** | `rank-bm25>=0.2.2` (+ pure-Python fallback) | Okapi BM25 inverted index ($k_1=1.5, b=0.75$) |
| **Dense Bi-Encoder** | `sentence-transformers/all-MiniLM-L6-v2` | 384-dimensional L2-normalized semantic embeddings |
| **ANN Vector Index** | `faiss-cpu>=1.8.0` (`IndexHNSWFlat` / `IndexFlatIP`) | Approximate nearest-neighbor inner-product search with NumPy fallback |
| **Cross-Encoder Reranker** | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Joint query-passage cross-attention reranking |
| **Extractive Reader** | `distilbert-base-cased-distilled-squad` | Token-level start/end span extraction |
| **Document & Report Tools** | `pypdf>=4.2.0`, `python-docx>=1.1.0`, `matplotlib>=3.8.0` | PDF text extraction, Word report generation, and evaluation plotting |
| **Testing Framework** | `pytest>=8.0.0` | 49 automated unit and integration tests |

---

## 4. Architecture and Workflow

```text
[OFFLINE INDEXING STAGE]
Raw Documents (.txt, .pdf)
       │
       ▼
Text Extraction & Cleaning
       │
       ▼
Sliding-Window Passage Segmentation (Chunk: 150w, Overlap: 30w)
       │
       ├─────────────────────────────────────────┐
       ▼                                         ▼
Inverted Index (Okapi BM25)             FAISS HNSW / FlatIP Vector Index
                                        (all-MiniLM-L6-v2, 384-d L2-norm)

─────────────────────────────────────────────────────────────────────────────

[ONLINE QUERY STAGE]
User Query
       │
       ├─────────────────────────────────────────┐
       ▼                                         ▼
Heuristic ReAct Query Rewriting        HyDE Hypothetical Passage Synthesis
(Thought -> Action -> Observation)      (Alpha=0.6 Query-HyDE Vector Blend)
       │                                         │
       ▼                                         ▼
BM25 Candidate Retrieval              FAISS ANN Semantic Search
       │                                         │
       └────────────────────┬────────────────────┘
                            ▼
                 Candidate Fusion (RRF: k=60)
                            │
                            ▼
              Stage 2: Cross-Encoder Reranking
              (cross-encoder/ms-marco-MiniLM-L-6-v2)
                            │
                            ▼
              Evidence Selection & Grounding
                            │
       ┌────────────────────┴────────────────────┐
       ▼                                         ▼
DistilBERT SQuAD Extractive QA         Multi-Passage Extractive Synthesizer
(Concise Span Extraction)              (Verified Corpus Sentences + [1: doc_id])
       │                                         │
       └────────────────────┬────────────────────┘
                            ▼
              Abstention & RAGAS Proxy Verification
                            │
                            ▼
              Final Answer + Evidence + Citations
```

### Directory Structure

```text
hybrid-ir-question-answering/
├── app/
│   ├── __init__.py
│   ├── engine.py             # Central orchestrator and 5-mode pipeline facade
│   ├── ingestion.py          # Document parsing, cleaning, and sliding chunking
│   ├── query_transform.py    # Heuristic ReAct query rewriter and template/PRF HyDE generator
│   ├── retrieval.py          # BM25, FAISS HNSW/FlatIP dense semantic, and RRF fusion
│   ├── reranking.py          # Cross-Encoder reranker with fallback
│   ├── answering.py          # Evidence selection, DistilBERT QA, Extractive Synthesis, abstention
│   ├── metrics.py            # Standard IR/QA metrics + RAGAS-inspired heuristic proxy metrics
│   └── streamlit_app.py      # Streamlit web user interface
├── sample_docs/              # Built-in knowledge base (6 technical NLP documents, 13 passages)
├── tests/                    # Pytest suite (49 automated tests)
│   ├── test_ingestion.py
│   ├── test_query_transform.py
│   ├── test_retrieval.py
│   ├── test_reranking.py
│   ├── test_answering.py
│   ├── test_metrics.py       # Comprehensive IR, QA, and RAGAS proxy edge case tests
│   └── test_integration.py
├── reports/
│   ├── Implementation_Report_23MID0429.docx   # Formal Word submission report
│   ├── architecture_diagram.png               # High-res architecture flow diagram
│   ├── benchmark_summary.csv                  # 5-mode IR, QA, and RAGAS proxy metrics table
│   ├── benchmark_summary.json                 # Query-by-query benchmark output
│   └── screenshot_*.png                       # Live captured UI screenshots
├── assets/                   # Architecture diagram and UI screenshots
├── data/
│   └── evaluation_dataset.json                # 10 labelled benchmark test queries with passage IDs
├── generate_architecture_diagram.py
├── generate_docx_report.py
├── run_benchmark_eval.py
├── capture_app_screenshots.py
├── requirements.txt
├── run_app.bat               # Windows one-click batch launcher
└── README.md
```

---

## 5. Installation and Setup

### Prerequisites
- Python 3.10, 3.11, or 3.12 (tested on Python 3.11.15 on Windows)
- Git

### Step-by-Step Setup

1. **Clone the repository and navigate to the project directory:**
   ```bash
   git clone https://github.com/vinay-0208/hybrid-ir-question-answering.git
   cd hybrid-ir-question-answering
   ```
   *(Or if working in the local project folder on Windows:* `cd "C:\Fall Semister\2026\nlp\case study-2"`*)*

2. **Create and activate a virtual environment:**
   - **Windows (PowerShell):**
     ```powershell
     python -m venv .venv
     .\.venv\Scripts\Activate.ps1
     ```
   - **Windows (Command Prompt):**
     ```cmd
     python -m venv .venv
     .\.venv\Scripts\activate.bat
     ```
   - **Linux / macOS:**
     ```bash
     python3 -m venv .venv
     source .venv/bin/activate
     ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## 6. Launching the Streamlit Dashboard

### Option A: Direct Streamlit Command (Recommended)
With the virtual environment activated:
```bash
streamlit run app/streamlit_app.py
```
Or directly via the virtual environment binary on Windows:
```powershell
.\.venv\Scripts\streamlit.exe run app/streamlit_app.py
```
Then open `http://localhost:8501` in your browser.

### Option B: Windows One-Click Batch Launcher
Double-click `run_app.bat` or run from Command Prompt/PowerShell:
```cmd
.\run_app.bat
```

---

## 7. Running Automated Tests and Benchmarks

### Run the Automated Pytest Suite (49 Tests)
```powershell
.\.venv\Scripts\pytest.exe tests/ -v
```

**Verified Test Output:**
```text
============================= 49 passed in 34.74s =============================
tests/test_answering.py ......................... [ 7 PASSED]
tests/test_ingestion.py ......................... [ 4 PASSED]
tests/test_integration.py ....................... [11 PASSED]
tests/test_metrics.py ........................... [13 PASSED]
tests/test_query_transform.py ................... [ 5 PASSED]
tests/test_reranking.py ......................... [ 2 PASSED]
tests/test_retrieval.py ......................... [ 7 PASSED]
```

### Run the 5-Mode Evaluation Benchmark
```powershell
.\.venv\Scripts\python.exe run_benchmark_eval.py
```
This script warms up all Transformer models, evaluates all 5 retrieval modes on `data/evaluation_dataset.json`, and writes updated results to `reports/benchmark_summary.csv` and `reports/benchmark_summary.json`.

---

## 8. Evaluation Metrics & Benchmark Results ($k=5$)

> **Important Methodological Note:**  
> 1. **Compact In-Domain Benchmark:** The evaluation dataset (`data/evaluation_dataset.json`) consists of **10 curated technical queries** evaluated over **13 passages** (segmented from 6 sample NLP documents at `chunk_size=150`, `overlap=30`). Because each question maps to 1 primary ground-truth passage, the mathematical upper bound for `Precision@5` is $1/5 = 0.200$.  
> 2. **RAGAS-Inspired Heuristic Proxy Metrics:** To run 100% locally without external LLM API keys or paid judge calls, `Faithfulness`, `Answer Relevance`, `Context Precision`, and `Context Recall` in `app/metrics.py` are implemented as **deterministic local heuristic proxy metrics** (using sentence-level lexical/negation grounding, bi-encoder cosine + lexical query-answer similarity, rank-weighted context precision, and reference claim coverage) rather than official LLM-as-a-judge RAGAS calls.

| Retrieval Mode | P@5 | R@5 | MRR | NDCG@5 | EM | Token F1 | Faithfulness (Proxy) | Ans Rel (Proxy) | Ctx Prec | Ctx Rec | Avg Latency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. BM25 Only** | 0.180 | 0.900 | 0.850 | 0.863 | 0.200 | 0.433 | 1.000 | **0.674** | 0.850 | 0.949 | **222.07 ms** |
| **2. Dense Only (FAISS)** | 0.180 | 0.900 | 0.770 | 0.802 | 0.200 | 0.373 | 1.000 | 0.629 | 0.770 | 0.902 | 224.09 ms |
| **3. Hybrid (RRF)** | 0.160 | 0.800 | 0.750 | 0.763 | 0.200 | 0.373 | 1.000 | 0.662 | 0.750 | 0.882 | 239.55 ms |
| **4. Hybrid + Cross-Encoder** | **0.200** | **1.000** | 0.875 | 0.906 | 0.200 | 0.383 | 1.000 | 0.667 | 0.875 | **0.969** | 627.86 ms |
| **5. Heuristic ReAct + HyDE + Rerank** | **0.200** | **1.000** | **0.950** | **0.963** | **0.300** | **0.543** | **1.000** | 0.658 | **0.950** | **0.969** | 674.65 ms |

**Key Findings:**
- **Heuristic ReAct + HyDE + Rerank (Mode 5):** Combining 2-step heuristic ReAct query expansion (`0.65` original + `0.35` expanded RRF) with template/PRF HyDE vector blending ($\alpha=0.6$) and Cross-Encoder reranking achieved **Recall@5 = 1.000**, **MRR = 0.950**, and **NDCG@5 = 0.963**, placing the target passage at **Rank 1 on 9 of 10 queries** (and Rank 2 on `q6`) and improving **Exact Match to 0.300** and **Token F1 to 0.543** over Mode 4 (`MRR = 0.875`, `NDCG@5 = 0.906`, `EM = 0.200`, `Token F1 = 0.383`).
- **Faithfulness Proxy (`1.000`):** Because `GroundedAnswerSynthesizer` extracts sentences verbatim from the top retrieved passages (`[1: passage_id]`), every synthesized statement is directly supported by the retrieved context whenever the system does not abstain.

---

## 9. Known Limitations

1. **Deterministic Heuristic ReAct & Template/PRF HyDE (No Generative LLM):** Because this system targets CPU-only execution (`CUDA=False`, ~1.66 GB available free RAM), `AgenticQueryRewriter` uses deterministic rule-based acronym/concept expansion and a fixed 2-step RRF loop rather than an autonomous LLM agent. Similarly, `HyDEGenerator` constructs hypothetical passages using deterministic templates and top-1 BM25 pseudo-relevance feedback rather than generative LLM decoding.
2. **Extractive Multi-Passage Synthesis:** `GroundedAnswerSynthesizer` performs extractive sentence selection and deduplication with inline citations (`[1: passage_id]`) rather than abstractive free-form text generation.
3. **Local Heuristic RAGAS Proxies:** The RAGAS-inspired metrics in `app/metrics.py` are deterministic local approximations and should not be conflated with LLM-judged RAGAS scores.
4. **Small Sample Corpus & FAISS ANN Scale:** The built-in corpus contains 6 documents (13 passages) and 10 evaluation queries. At $N=13$ vectors, FAISS `IndexHNSWFlat` provides architectural completeness and exact rank equivalence with NumPy dot-product search rather than an ANN speed advantage (which emerges at $N \ge 10{,}000$ passages).

---

## 10. Sample Questions to Try

1. *"What is the Okapi BM25 algorithm?"*  
   $\rightarrow$ Returns definition and highlights probabilistic retrieval and TF-IDF improvement.
2. *"What problem does dense semantic retrieval solve that BM25 struggles with?"*  
   $\rightarrow$ Returns vocabulary mismatch limitation.
3. *"Why is a two-stage retrieval pipeline used with cross-encoders?"*  
   $\rightarrow$ Explains latency constraints when scoring millions of passages.
4. *"What is the role of parameter k1 in BM25?"*  
   $\rightarrow$ Identifies non-linear term frequency saturation curve.
5. *"What is the capital of planet Mars?"* (Out-of-Domain Negative Test)  
   $\rightarrow$ **Triggers Abstention Policy Across All 5 Modes:** Returns *"Insufficient evidence in the knowledge base to answer this question reliably."*

