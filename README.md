# RAG Pipeline From Scratch

A production-style Retrieval-Augmented Generation pipeline built from first principles.
No tutorials. Every design decision explained.

## Pipeline Stages

| File | Stage | What it does |
|------|-------|--------------|
| `01_chunking.py` | Chunking | Fixed vs recursive chunking, context loss problem |
| `02_embeddings_and_index.py` | Embeddings | Sentence transformers + FAISS, naive retrieval failure |
| `03_contextual_enrichment.py` | Enrichment | LLM prepends context to each chunk before embedding |
| `04_hybrid_search.py` | Retrieval | BM25 (sparse) + FAISS (dense) fused with RRF |
| `05_reranking.py` | Reranking | Cross-encoder reranker on hybrid candidates |
| `06_generation.py` | Generation | Grounded generation with hallucination guard |
| `07_evaluation_ragas.py` | Evaluation | RAGAS metrics: faithfulness, relevancy, precision, recall |
| `08_retrieval_failures.py` | Debugging | Intentional failure analysis with quantitative measurement |

## Key Insights

**Why hybrid search:** Dense embeddings fail on exact terms (model names, numbers, rare vocab). BM25 handles lexical precision. RRF fuses both without score normalization.

**Why contextual enrichment:** Chunks lose context when split. Prepending an LLM-generated summary moves correct chunk from rank 3 → rank 1.

**Why cross-encoder reranking:** Bi-encoders encode query and doc separately. Cross-encoders see them together — full attention, much better relevance scoring.

**Why RAGAS:** "Looks correct" is not a metric. Faithfulness catches hallucination. Context recall catches missing chunks. You need numbers to know what's broken.

## Results

Faithfulness:      1.000  
Answer Relevancy:  0.841  
Context Precision: 1.000  
Context Recall:    0.542   (toy corpus — improves on real data)


## Failure Analysis

| Experiment | Faithfulness | Recall | What broke |
|------------|-------------|--------|------------|
| Baseline | 0.964 | 1.000 | — |
| top_k=1 | 0.700 | 0.750 | LLM hallucinated from incomplete context |
| candidate_pool=1 | 0.900 | 1.000 | Survived on small corpus |
| No enrichment | 0.875 | 1.000 | Survived on small corpus |

## Failure Analysis

### Generation Layer — `09_generation_failures.py`

Adversarial stress test against the generation pipeline.

| # | Test | Failure Triggered | System Safe? |
|---|------|-------------------|--------------|
| 1 | Prompt Injection | ✅ Yes | ❌ No |
| 2 | Context Overflow | ✅ Yes | ✅ Yes |
| 3 | Hallucination under Weak Retrieval | No | ✅ Yes |
| 4 | Refusal Failure (Plausible Context) | No | ✅ Yes |
| 5 | Guard Stress Test (Query-Side Injection) | ✅ Yes | ❌ No |

**Findings:**
- Tests 1 and 5 fail on `llama-3.1-8b-instant`. The system-role instruction is applied correctly but the 8b model lacks the robustness to enforce it under adversarial input. Both pass on `llama-3.3-70b-versatile`.
- Context overflow is silently truncated by Groq — information loss is invisible without a pre-send token budget check.
- Refusal behavior is strong — weak and plausible-but-incomplete context both correctly returned "I don't have enough context."

## Stack
Python · PyTorch · HuggingFace · Sentence Transformers
FAISS · rank-bm25 · Groq API · RAGAS · LangChain

## Setup

```bash
git clone https://github.com/shiblimaroof/rag-from-scratch.git
cd rag-from-scratch
pip install -r requirements.txt
```

Create a `.env` file:
GROQ_API_KEY=your_key_here

Run any stage:
```bash
python3 06_generation.py
```

## Author

Shibli | [@BuildwithShibli](https://x.com/BuildwithShibli) |
