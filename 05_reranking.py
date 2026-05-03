"""
05_reranking.py
================
Cross-encoder reranking on top of hybrid search candidates.
 
Full pipeline:
  query
    --> hybrid search (BM25 + FAISS + RRF)  --> top-10 candidates  [cheap, high recall]
    --> cross-encoder reranker              --> top-3 final results [expensive, high precision]
 
Why this order matters:
  Cross-encoder on full corpus = O(n) forward passes at inference. Too slow.
  Cross-encoder on 10 candidates = 10 forward passes. Fast enough.
  Hybrid search gives you a good candidate pool. Reranker refines it.
"""
import json
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer, CrossEncoder
import faiss
from rank_bm25 import BM25Okapi

# ── 1. Load enriched chunks ─────────────────────────────────────────────────────

CHUNKS_FILE = Path('enriched_chunks.json')

if not CHUNKS_FILE.exists():
    print("[warn] enriched_chunks.json not found — using toy corpus")
    enriched_chunks = [
        {
            "original": "The transformer architecture uses self-attention to process tokens in parallel.",
            "enriched": "Context: Chapter on transformer internals. The transformer architecture uses self-attention to process tokens in parallel.",
        },
        {
            "original": "BERT is pretrained on masked language modeling and next sentence prediction.",
            "enriched": "Context: Chapter on BERT pretraining. BERT is pretrained on masked language modeling and next sentence prediction.",
        },
        {
            "original": "GPT uses causal language modeling, predicting the next token autoregressively.",
            "enriched": "Context: Chapter on GPT architecture. GPT uses causal language modeling, predicting the next token autoregressively.",
        },
        {
            "original": "Attention is computed as softmax(QK^T / sqrt(d_k)) * V.",
            "enriched": "Context: Chapter on attention math. Attention is computed as softmax(QK^T / sqrt(d_k)) * V.",
        },
        {
            "original": "LoRA fine-tunes large models by injecting low-rank matrices into attention layers.",
            "enriched": "Context: Chapter on parameter-efficient fine-tuning. LoRA fine-tunes large models by injecting low-rank matrices into attention layers.",
        },
        {
            "original": "The vocabulary size of GPT-2 is 50257 tokens using BPE tokenization.",
            "enriched": "Context: Chapter on tokenization. The vocabulary size of GPT-2 is 50257 tokens using BPE tokenization.",
        },


    ]
else:
    with open(CHUNKS_FILE) as f:
        enriched_chunks = json.load(f)
    print(f"[info] Loaded {len(enriched_chunks)} enriched chunks")

texts = [c['enriched'] for c in enriched_chunks]

# ── 2. Build BM25 index ─────────────────────────────────────────────────────────

def tokenize(text: str) -> list[str]:
    return text.lower().split()

bm25 = BM25Okapi([tokenize(t) for t in texts])

# ── 3. Build FAISS index ────────────────────────────────────────────────────────

EMBED_model = "all-MiniLM-L6-v2"
embedder = SentenceTransformer(EMBED_model)

print(f"[info] encoding corpus with {EMBED_model}...")
embeddings = embedder.encode(texts, show_progress_bar= True, normalize_embeddings=True)

dim = embeddings.shape[1]
faiss_index = faiss.IndexFlatIP(dim)
faiss_index.add(embeddings.astype(np.float32))


# ── 4. Load cross-encoder ───────────────────────────────────────────────────────

#3ms-marco-MiniLM-L-6-v2 is the standard lightweight cross-encoder.
# Trained on MS MARCO passage ranking — exactly the retrieve-then-rerank use case.
# It outputs a raw logit (not a probability). Higher = more relevant.
# You don't need to sigmoid it — only relative ordering matters.
#
# Larger option: cross-encoder/ms-marco-electra-base — more accurate, slower.
# For production with latency constraints, stick with MiniLM.

RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
print(f"[info] loading cross-encoder : {RERANK_MODEL}...")
reranker = CrossEncoder(RERANK_MODEL)

# ── 5. Retrieval helpers (same as 04) ──────────────────────────────────────────

def retrieve_bm25(query: str, top_k: int) -> list[tuple[int, float]]:
    scores = bm25.get_scores(tokenize(query))
    ranked = np.argsort((scores)[::-1][:top_k])
    return[(int(i), float(scores[i])) for i in ranked]

def retrieve_dense(query: str, top_k: int) -> list[tuple[int, float]]:
    vec = embedder.encode([query], normalize_embeddings=True).astype(np.float32)
    sims, idxs = faiss_index.search(vec, top_k)
    return [(int(i), float(s)) for i, s in zip(idxs[0], sims[0]) if i != -1]

def reciprocal_rank_fusion(
    ranked_lists: list[list[tuple[int, float]]],
    k: int = 60,
) -> list[tuple[int, float]]:
    scores: dict[int, float] = {}
    for ranked_list in ranked_lists:
        for rank, (doc_idx, _) in enumerate(ranked_list, start=1):
            scores[doc_idx] = scores.get(doc_idx, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)

def hybrid_search(query: str, candidate_pool: int = 10) -> list[tuple[int, float]]:
    candidate_pool = min(candidate_pool, len(texts))
    bm25_results = retrieve_bm25(query, top_k = candidate_pool)
    dense_results = retrieve_dense(query, top_k = candidate_pool)
    return reciprocal_rank_fusion([bm25_results, dense_results])

 
# ── 6. Reranking ────────────────────────────────────────────────────────────────

def rerank(query: str, candidates: list[tuple[int, float]], top_k: int = 3) -> list[dict]:
    """
    Takes hybrid search candidates, runs cross-encoder on each, returns top_k.
 
    The cross-encoder receives pairs: [(query, doc_text), (query, doc_text), ...]
    It scores each pair independently. No comparison between pairs happens inside
    the model — comparison happens when you sort by score afterward.
 
    Args:
        query:      The user query string.
        candidates: Output of hybrid_search — [(doc_idx, rrf_score), ...]
        top_k:      How many results to return after reranking.
    """
    if not candidates:
        return[]
    # Build input pairs for the cross-encoder
    # We pass the original (non-enriched) text to the reranker.
    # Reason: enriched text has the prepended context summary, which was useful
    # for improving embedding quality. But the cross-encoder does full attention
    # across query+doc — it doesn't need the crutch. Original text is cleaner.
    pairs = [(query, enriched_chunks[doc_idx]['original']) for doc_idx, _ in candidates]
    
    # scores is a numpy array of raw logits, one per pair
    # Higher = more relevant. No threshold needed — you just sort.

    scores = reranker.predict(pairs)

    # Attach scores back to doc indices

    scored = [
        {
            "doc_idx" : candidates[i][0],
            "rrf_score": round(candidates[i][1], 5),
            "ce_score" : round(float(scores[i]), 4),
            "text" : texts[candidates[i][0]],
            "original" : enriched_chunks[candidates[i][0]]['original'],
        
        }
        for i in range(len(candidates))
    ]
    # Sort by cross-encoder score, descending
    scored.sort(key = lambda x:x['ce_score'], reverse = True )

    # Add final rank
    for rank, item in enumerate(scored[:top_k], start = 1):
        item['rank'] = rank

    return scored[:top_k]

# ── 7. Full pipeline ────────────────────────────────────────────────────────────
def retrieve_and_rerank(query: str, candidate_pool: int = 10, top_k: int = 3) -> list[dict]:
    candidates = hybrid_search(query, candidate_pool= candidate_pool)
    results = rerank(query, candidates, top_k=top_k)
    return results

# ── 8. Demo ─────────────────────────────────────────────────────────────────────
def print_results(query : str, results : list[dict]) -> None:
    print(f"\n{'='*65}")
    print(f"Query: {query!r}")
    print('='*65)
    for r in results:
        print(f"\nRank {r['rank']}  [ce_score={r['ce_score']}  rrf_score={r['rrf_score']}]")
        print(f"  {r['original']}")


if __name__ == "__main__":

    queries = [
        "how does the attention mechanism work mathematically?",
        "GPT-2 vocabulary size",
        "what is LoRA and how does it work?",

    ]
    for q in queries:
        results = retrieve_and_rerank(q, candidate_pool=10, top_k=3)
        print_results(q,results)

    print("n\[done]")