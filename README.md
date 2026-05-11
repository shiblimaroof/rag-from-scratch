# RAG Pipeline From Scratch
 
A production-style Retrieval-Augmented Generation pipeline built from first principles.
No LangChain. No tutorials. Every design decision explained inline.
 
---
 
## Pipeline
 
```
Query
  → Chunking              [01]  fixed vs recursive, context loss problem
  → Embeddings + Index    [02]  sentence-transformers + FAISS
  → Contextual Enrichment [03]  LLM prepends context before embedding
  → Hybrid Search         [04]  BM25 (sparse) + FAISS (dense) + RRF fusion
  → Reranking             [05]  cross-encoder on hybrid candidates
  → Generation            [06]  grounded generation + hallucination guard
  → Evaluation            [07]  RAGAS metrics
  → Retrieval Failures    [08]  intentional failure analysis
  → Generation Failures   [09]  adversarial stress testing
  → Live Diagnostic       [10]  health scores + latency per stage
```
 
---
 
## Files
 
| File | Stage | What it does |
|------|-------|--------------|
| `01_chunking.py` | Chunking | Fixed vs recursive chunking, context loss problem |
| `02_embeddings_and_index.py` | Embeddings | Sentence transformers + FAISS, naive retrieval failure |
| `03_contextual_enrichment.py` | Enrichment | LLM prepends context to each chunk before embedding |
| `04_hybrid_search.py` | Retrieval | BM25 + FAISS fused with Reciprocal Rank Fusion |
| `05_reranking.py` | Reranking | Cross-encoder reranker on hybrid candidates |
| `06_generation.py` | Generation | Grounded generation + hallucination guard + sanitize_chunks |
| `07_evaluation_ragas.py` | Evaluation | RAGAS: faithfulness, relevancy, precision, recall |
| `08_retrieval_failures.py` | Debugging | Intentional failure analysis with quantitative measurement |
| `09_generation_failures.py` | Adversarial | 5 adversarial tests against the generation layer |
| `10_debugging_workflow.py` | Diagnostic | Live pipeline health scores + latency breakdown |
 
---
 
## Results
 
```
Faithfulness       1.000
Answer Relevancy   0.841
Context Precision  1.000
Context Recall     0.542   (toy corpus — improves on real data)
```
 
---
 
## Design Decisions
 
**Why hybrid search over pure vector search**
Dense embeddings fail on exact terms — model names, numbers, rare vocabulary. BM25 handles lexical precision. RRF fuses both without requiring score normalization across different scales.
 
**Why contextual enrichment**
Chunks lose surrounding context when split. A chunk that says "it uses this method" tells the embedder nothing. Prepending an LLM-generated summary of the surrounding document moves the correct chunk from rank 3 → rank 1.
 
**Why cross-encoder reranking**
Bi-encoders encode query and document separately — the vectors never interact. Cross-encoders see query and document together with full attention. Much better relevance scoring. Slower, so it runs only on the top candidates from hybrid search.
 
**Why RAGAS**
"Looks correct" is not a metric. Faithfulness catches hallucination. Context recall catches missing chunks. Context precision catches irrelevant chunks being included. You need numbers to know what's broken and where.
 
**Why geometric mean for overall health**
Arithmetic mean of [1.0, 1.0, 1.0, 0.0] = 0.75 — looks acceptable. Geometric mean = 0.0 — correctly broken. One broken stage breaks the pipeline. The score should reflect that.
 
**Why build from scratch instead of LangChain**
Abstractions leak at the worst moments. Rolling your own pipeline takes two weeks and you understand every layer. When retrieval breaks, you know exactly which component to fix.
 
---
 
## Failure Analysis
 
### Retrieval Layer — `08_retrieval_failures.py`
 
| Experiment | Faithfulness | Recall | What broke |
|------------|-------------|--------|------------|
| Baseline | 0.964 | 1.000 | — |
| top_k=1 | 0.700 | 0.750 | LLM hallucinated from incomplete context |
| candidate_pool=1 | 0.900 | 1.000 | Survived on small corpus |
| No enrichment | 0.875 | 1.000 | Survived on small corpus |
 
### Generation Layer — `09_generation_failures.py`
 
Adversarial stress tests against the full generation pipeline.
 
| # | Test | Failure Triggered | System Safe? |
|---|------|-------------------|--------------|
| 1 | Prompt injection via chunk | Yes | No → fixed via `sanitize_chunks()` |
| 2 | Context overflow | Yes | Yes — Groq truncates silently |
| 3 | Hallucination under weak retrieval | No | Yes |
| 4 | Refusal failure (plausible context) | No | Yes |
| 5 | Query-side injection | Yes | No → fix belongs in FastAPI layer |
 
**Findings:**
- Tests 1 and 5 fail on `llama-3.1-8b-instant`. System-role instruction is applied correctly but the 8b model lacks robustness under adversarial input. Both pass on `llama-3.3-70b-versatile`.
- Context overflow is silently truncated by Groq — information loss is invisible without a pre-send token budget check.
- Refusal behavior is strong on tests 3 and 4. Weak and plausible-but-incomplete context both correctly returned "I don't have enough context."
---
 
## Live Diagnostic — `10_debugging_workflow.py`
 
Run a health check on any query:
 
```bash
python 10_debugging_workflow.py "your query here"
```
 
Output: health score (0.0–1.0) + latency per stage + actionable fix suggestions.
Overall health = geometric mean. One broken stage collapses the score to 0.
 
```
STAGE 1 - CHUNKING    — Is the answer in the corpus?
STAGE 2 - RETRIEVAL   — Did the right chunks come back?
STAGE 3 - RERANKING   — Did reranking help or hurt?
STAGE 4 - GENERATION  — Did the model answer from context?
STAGE 5 - GUARD CHECK — Does it refuse on empty context?
```
 
Example output (healthy pipeline):
 
```
  St   Stage         Health                  Latency   Status
  1    Chunking      [███████████████] 1.00    0.1ms   HEALTHY
  2    Retrieval     [███████████████] 1.00  240.4ms   HEALTHY
  3    Reranking     [███████████████] 1.00    0.0ms   HEALTHY
  4    Generation    [███████████████] 1.00  827.6ms   HEALTHY
  5    Guard Check   [███████████████] 1.00  146.0ms   HEALTHY
 
  Overall pipeline health: 1.00  HEALTHY
  All stages healthy — no weak stage detected.
```
 
Use files 07/08/09 in CI. Use file 10 when a user reports a bad answer.
 
---
 
## Stack
 
Python · PyTorch · HuggingFace Transformers · Sentence Transformers  
FAISS · rank-bm25 · Groq API · RAGAS
 
---
 
## Setup
 
```bash
git clone https://github.com/shiblimaroof/rag-from-scratch.git
cd rag-from-scratch
pip install -r requirements.txt
```
 
Create a `.env` file:
 
```
GROQ_API_KEY=your_key_here
```
 
Run any stage:
 
```bash
python 06_generation.py
python 10_debugging_workflow.py "how does attention work?"
```
 
RAGAS notes:
- Use `ragas==0.1.21` — newer versions break Groq compatibility
- Use `llama-3.3-70b-versatile` as the RAGAS judge model
- Daily limit on 70b: 100k tokens, resets at 5:30am IST
---
 
## Author
 
Shibli 
[@BuildwithShibli](https://twitter.com/BuildwithShibli) 
