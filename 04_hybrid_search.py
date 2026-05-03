
"""
04_hybrid_search.py
====================
Hybrid retrieval: BM25 (sparse) + FAISS (dense) fused with Reciprocal Rank Fusion (RRF).
 
Pipeline:
  chunks (from 03) --> BM25 index  \
                                    --> RRF fusion --> ranked docs
                   --> FAISS index /
 
Design decisions explained inline.
"""



import json
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
import faiss
from rank_bm25 import BM25Okapi

# ── 1. Load enriched chunks from 03 ────────────────────────────────────────────
 
# We use the contextually enriched chunks (not raw chunks).
# Reason: BM25 on enriched text benefits from the prepended context summary,
# which contains keywords the original chunk might have lacked

CHUNKS_FILE = Path("enriched_chunks.json")

if not CHUNKS_FILE.exists():
    #Fallback : build toy data so this file runs standalone
    print('[warn]enriched_chunks not found - using toy corpus')
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

else :
    with open(CHUNKS_FILE) as f:
        enriched_chunks = json.fload(f)
        print(f"[info] Loaded {len(enriched_chunks)} enriched chunks")

texts = [c['enriched']for c in enriched_chunks]

# ── 2. Build BM25 index ─────────────────────────────────────────────────────────

# BM25Okapi expects a list of tokenized documents (list of lists of strings).
# Tokenization here is naive whitespace split + lowercase.
# Production note: you'd strip punctuation, apply a stemmer (e.g. SnowballStemmer),
# or use the same tokenizer as your embedding model. For RAG, this is usually good enough.

def tokenize(text:str) -> list[str]:
    return text.lower().split()

tokenized_corpus = [tokenize(t) for t in texts]
bm25 = BM25Okapi(tokenized_corpus)

print(f'[info] BM25 index build over {len(texts)} documents')

# ── 3. Build FAISS index ────────────────────────────────────────────────────────

# Reusing the same model from 02/03 for consistency.
# If you load a different model here, embeddings are incompatible — dimensions differ.

EMBED_MODEL = "all-MiniLM-L6-v2"
embedder = SentenceTransformer(EMBED_MODEL)

print(f"[info] encoding {len(texts)} chunks with {EMBED_MODEL}")
embeddings = embedder.encode(texts, show_progress_bar=True, normalize_embeddings=True)
# normalize_embeddings=True → cosine similarity becomes dot product → use IndexFlatIP


dim = embeddings.shape[1]
faiss_index = faiss.IndexFlatIP(dim)
faiss_index.add(embeddings.astype(np.float32))

# ── 4. Retrieval functions ──────────────────────────────────────────────────────

def retrieve_bm25(query: str, top_k:int = 10) -> list [tuple[int, float]]:
        """
    Returns [(doc_idx, bm25_score), ...] sorted by score descending.
    top_k > final_k intentionally: RRF needs a wider candidate pool.
    Pulling top 10 when you only want top 3 gives fusion more signal.
    """
        tokenized_query = tokenize(query)
        scores = bm25.get_scores(tokenized_query)
        # argsort ascending, flip to get descending
        ranked_indices = np.argsort(scores)[::-1][:top_k]
        return[(int(idx), float(scores[idx])) for idx in ranked_indices]

def retrieve_dense (query: str, top_k:int=10) -> list[tuple[int,float]]:
        """
    Returns [(doc_idx, cosine_sim), ...] sorted by similarity descending.
    """
        query_vec = embedder.encode([query], normalize_embeddings= True). astype(np.float32)
        similarities, indices = faiss_index.search(query_vec, top_k)
        return [
        (int(idx), float(sim))
        for idx, sim in zip(indices[0], similarities[0])
        if idx != -1  # ← add this
    ]

# ── 5. RRF Fusion ───────────────────────────────────────────────────────────────

def reciprocal_rank_fusion(
          ranked_lists : list[list[tuple[int,float]]],
          k: int = 60 ) -> list[tuple[int,float]]:
        """
    Fuse multiple ranked lists using RRF.
 
    Args:
        ranked_lists: Each element is a retriever's output: [(doc_idx, score), ...]
                      sorted best-first. Scores are ignored — only rank matters.
        k:            Damping constant. k=60 is the value from the original RRF paper
                      (Cormack et al., 2009). Higher k → flatter score distribution
                      → less reward for being ranked #1 specifically.
 
    Returns:
        [(doc_idx, rrf_score), ...] sorted by rrf_score descending.
    """
        rrf_scores: dict[int,float] = {}


        for ranked_list in ranked_lists:
              for rank,(doc_idx, _score) in enumerate(ranked_list, start =1):
                    # rank starts at 1, not 0
                    rrf_scores[doc_idx] = rrf_scores.get(doc_idx, 0.0)+1.0/(k+rank)

        sorted_docs = sorted(rrf_scores.items(), key = lambda x:x[1],reverse = True)
        return sorted_docs   # [(doc_idx, rrf_score), ...]

# ── 6. Full hybrid retrieval pipeline ──────────────────────────────────────────

def hybrid_search(query: str, top_k: int = 3, candidate_pool: int = 10) -> list[dict]:
    candidate_pool = min(candidate_pool, len(texts)) # ← add this
    """
    Args:
        query:          User query string.
        top_k:          Final number of results to return.
        candidate_pool: How many candidates each retriever fetches before fusion.
                        Larger pool = better recall at the cost of more computation.
                        Rule of thumb: candidate_pool = 3-5x top_k.
 
    Returns:
        List of result dicts with rank, text, and scores.
    """
    bm25_results = retrieve_bm25(query, top_k=candidate_pool)
    dense_results = retrieve_dense(query, top_k = candidate_pool)

    fused = reciprocal_rank_fusion([bm25_results, dense_results], k=60) 

    results = []
    for rank, (doc_idx, rrf_score) in enumerate(fused[:top_k], start=1):
            # Look up individual retriever scores for transparency
            bm25_rank = next((i+1 for i, (idx, _) in enumerate(bm25_results)if idx == doc_idx), None)
            dense_rank = next((i+1 for i, (idx, _) in enumerate(dense_results)if idx == doc_idx), None)

            results.append({
                  "rank" : rank,
                  "doc_idx" :doc_idx,
                  "rrf_score" : round(rrf_score, 5),
                  "bm25_rank" : bm25_rank,
                  "dense_rank" : dense_rank,
                  "text" : texts[doc_idx]
            })
            return results
      
# ── 7. Demo ─────────────────────────────────────────────────────────────────────
def print_results(query: str, results: list[dict]) -> None:
    print(f"\n{'='*65}")
    print(f"Query: {query!r}")
    print('='*65)
    for r in results:
        print(f"\nRank {r['rank']}  [rrf={r['rrf_score']}  "
              f"bm25_rank={r['bm25_rank']}  dense_rank={r['dense_rank']}]")
        print(f"  {r['text'][:120]}...")

if __name__ == "__main__" :
      queries = [
             # Dense-friendly: semantic, no exact keywords
        "how does the attention mechanism work mathematically?",
 
        # BM25-friendly: exact term that may not embed well
        "GPT-2 vocabulary size",
 
        # Both should agree
        "what is LoRA and how does it work?",
      ]
      for q in queries:
            results = hybrid_search(q, top_k =3, candidate_pool=10)
            print_results(q,results)

            # Show what each retriever found independently for comparison
            bm25_top = retrieve_bm25(q,top_k=3)
            dense_top = retrieve_dense(q, top_k=3)

            print(f"\n  -- BM25 alone  top-3: {[i for i,_ in bm25_top]}")
            print(f"  -- Dense alone top-3: {[i for i,_ in dense_top]}")
            print(f"  -- Fused       top-3: {[r['doc_idx'] for r in results]}")
 
      print("\n[done]")
