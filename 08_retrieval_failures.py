"""
retrieval_failures.py
=========================
Intentionally break retrieval in 4 ways, measure with RAGAS, then fix.
 
Failure modes covered:
  1. Chunk size too small     → answer spans chunk boundaries
  2. Top-k too small          → correct chunk never retrieved
  3. No contextual enrichment → naive embeddings miss semantic context
  4. Candidate pool too small → RRF has insufficient candidates to fuse
 
Why this matters:
  Most RAG tutorials show you a working pipeline.
  Real engineering is knowing exactly how and why it breaks,
  so you can diagnose it from metrics alone — without looking at outputs.
"""
 
import os
import json
import numpy as np
from pathlib import Path
from datasets import Dataset
from sentence_transformers import SentenceTransformer, CrossEncoder
import faiss
from rank_bm25 import BM25Okapi
from dotenv import load_dotenv
from groq import Groq
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
import time

 
 
# ── Config ──────────────────────────────────────────────────────────────────────
 
load_dotenv()
key = os.environ.get("GROQ_API_KEY")
EMBED_MODEL   = "all-MiniLM-L6-v2"
RERANK_MODEL  = "cross-encoder/ms-marco-MiniLM-L-6-v2"
 
 
# ── Corpus ───────────────────────────────────────────────────────────────────────
# Deliberately written so answers SPAN multiple chunks.
# This simulates a real book where a concept is explained across paragraphs.
# Chunk size failures are only visible when the corpus has cross-chunk information.
 
RAW_CORPUS = [
    # Chunk 0 + 1 together answer "how does attention work"
    "The attention mechanism takes three inputs: queries, keys, and values. "
    "These are linear projections of the input embeddings.",
 
    "The dot product of queries and keys is scaled by sqrt(d_k) to prevent "
    "vanishing gradients. Then softmax is applied to get attention weights. "
    "Finally the weights are multiplied with values to get the output: softmax(QK^T / sqrt(d_k)) * V.",
 
    # Chunk 2 + 3 together answer "what is LoRA"
    "LoRA stands for Low-Rank Adaptation. It is a parameter-efficient fine-tuning method. "
    "Instead of updating all model weights, LoRA freezes the original weights.",
 
    "LoRA injects trainable low-rank matrices A and B into attention layers. "
    "The update is represented as W + AB^T where A and B are low-rank. "
    "This reduces trainable parameters by up to 10,000x compared to full fine-tuning.",
 
    # Chunk 4 answers "GPT-2 vocab size" alone
    "GPT-2 uses Byte Pair Encoding tokenization with a vocabulary size of 50257 tokens. "
    "The tokenizer was trained on WebText, a dataset of 40GB of internet text.",
 
    # Chunk 5 answers "BERT pretraining" alone
    "BERT is pretrained using two objectives: masked language modeling where 15% of tokens "
    "are masked and predicted, and next sentence prediction where the model learns "
    "sentence relationships.",
]
 
# Enriched versions — context prepended (simulating 03_contextual_enrichment.py)
ENRICHED_CORPUS = [
    "Context: Introduction to attention mechanism components. " + RAW_CORPUS[0],
    "Context: Attention computation formula and scaling. "      + RAW_CORPUS[1],
    "Context: Introduction to LoRA fine-tuning method. "       + RAW_CORPUS[2],
    "Context: LoRA low-rank matrix injection details. "        + RAW_CORPUS[3],
    "Context: GPT-2 tokenization and vocabulary. "             + RAW_CORPUS[4],
    "Context: BERT pretraining objectives. "                   + RAW_CORPUS[5],
]
 
EVAL_SAMPLES = [
    {
        "question":     "How is attention computed mathematically?",
        "ground_truth": "Attention takes queries, keys, and values as input. "
                        "The dot product of queries and keys is scaled by sqrt(d_k), "
                        "softmax is applied to get attention weights, then multiplied "
                        "with values: softmax(QK^T / sqrt(d_k)) * V.",
    },
    {
        "question":     "What is LoRA and how does it reduce parameters?",
        "ground_truth": "LoRA is a parameter-efficient fine-tuning method that freezes "
                        "original weights and injects low-rank matrices A and B. "
                        "The update W + AB^T reduces trainable parameters by up to 10,000x.",
    },
    {
        "question":     "What is the vocabulary size of GPT-2?",
        "ground_truth": "GPT-2 has a vocabulary size of 50257 tokens using BPE tokenization.",
    },
    {
        "question":     "How is BERT pretrained?",
        "ground_truth": "BERT is pretrained on masked language modeling (15% of tokens masked) "
                        "and next sentence prediction.",
    },
]
 
 
# ── RAGAS evaluator ─────────────────────────────────────────────────────────────
 
ragas_llm = LangchainLLMWrapper(
    ChatGroq(model="llama-3.3-70b-versatile", api_key=key, temperature=0.0)
)
ragas_embeddings = LangchainEmbeddingsWrapper(
    HuggingFaceEmbeddings(model_name=EMBED_MODEL)
)
 
def run_ragas(questions, answers, contexts, ground_truths) -> dict:
    dataset = Dataset.from_dict({
        "question":     questions,
        "answer":       answers,
        "contexts":     contexts,
        "ground_truth": ground_truths,
    })
    results = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=ragas_llm,
        embeddings=ragas_embeddings,
    )
    df = results.to_pandas()
    return {
        "faithfulness":      round(df["faithfulness"].mean(),      3),
        "answer_relevancy":  round(df["answer_relevancy"].mean(),  3),
        "context_precision": round(df["context_precision"].mean(), 3),
        "context_recall":    round(df["context_recall"].mean(),    3),
    }
 
 
# ── Pipeline builder ────────────────────────────────────────────────────────────
# Parameterized so we can swap corpus, top_k, candidate_pool to simulate failures.
 
def build_pipeline(corpus: list[str], raw_corpus: list[str]):
    """
    Build BM25 + FAISS + CrossEncoder pipeline over the given corpus.
    Returns a rag() function closed over the built indexes.
    """
    # BM25
    tokenized = [t.lower().split() for t in corpus]
    bm25      = BM25Okapi(tokenized)
 
    # FAISS
    embedder   = SentenceTransformer(EMBED_MODEL)
    embeddings = embedder.encode(corpus, show_progress_bar=False, normalize_embeddings=True)
    dim        = embeddings.shape[1]
    index      = faiss.IndexFlatIP(dim)
    index.add(embeddings.astype(np.float32))
 
    # CrossEncoder
    reranker = CrossEncoder(RERANK_MODEL)
 
    # Groq
    groq_client = Groq(api_key=key)
 
    def retrieve(query: str, top_k: int, candidate_pool: int) -> list[str]:
        candidate_pool = min(candidate_pool, len(corpus))
 
        # BM25
        scores  = bm25.get_scores(query.lower().split())
        ranked  = np.argsort(scores)[::-1][:candidate_pool]
        bm25_results = [(int(i), float(scores[i])) for i in ranked]
 
        # Dense
        vec  = embedder.encode([query], normalize_embeddings=True).astype(np.float32)
        sims, idxs = index.search(vec, candidate_pool)
        dense_results = [(int(i), float(s)) for i, s in zip(idxs[0], sims[0]) if i != -1]
 
        # RRF
        rrf_scores: dict[int, float] = {}
        for ranked_list in [bm25_results, dense_results]:
            for rank, (doc_idx, _) in enumerate(ranked_list, start=1):
                rrf_scores[doc_idx] = rrf_scores.get(doc_idx, 0.0) + 1.0 / (60 + rank)
        candidates = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
 
        # Rerank
        pairs  = [(query, raw_corpus[doc_idx]) for doc_idx, _ in candidates]
        ce_scores = reranker.predict(pairs)
        scored = sorted(
            [(candidates[i][0], float(ce_scores[i])) for i in range(len(candidates))],
            key=lambda x: x[1], reverse=True
        )
        return [raw_corpus[idx] for idx, _ in scored[:top_k]]
 
    def rag(query: str, top_k: int, candidate_pool: int) -> tuple[str, list[str]]:
        chunks = retrieve(query, top_k=top_k, candidate_pool=candidate_pool)
        context_block = "\n\n".join(f"[{i+1}] {c}" for i, c in enumerate(chunks))
        prompt = f"""Answer using ONLY the context below.
If insufficient, say "I don't have enough context to answer this."
 
Context:
{context_block}
 
Question: {query}
Answer:"""
        response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=256,
        )
        answer = response.choices[0].message.content.strip()
        return answer, chunks
 
    return rag

def run_experiment(name, corpus, raw_corpus, top_k:int, candidate_pool:int):
    """Run a full experiment and return RAGAS scores."""
    print(f"\n{'='*60}")
    print(f"EXPERIMENT: {name}")
    print(f"  top_k={top_k}  candidate_pool={candidate_pool}  corpus='{corpus[0][:40]}...'")
    print('='*60)

    rag = build_pipeline(corpus, raw_corpus)

    questions, answers, contexts, ground_truths = [],[],[],[]
    for sample in EVAL_SAMPLES :
        q= sample['question']
        print(f"->{q}")
        answer, retrieved = rag(q, top_k=top_k, candidate_pool= candidate_pool)
        questions.append(q)
        answers.append(answer)
        contexts.append(retrieved)
        ground_truths.append(sample["ground_truth"])

    scores = run_ragas(questions,answers,contexts,ground_truths)
    print(f"\n Results : {scores}")
    return scores

# ── Run experiments ─────────────────────────────────────────────────────────────

if __name__ == "__main__":

    all_results = {}

    # ── Baseline: correct setup ─────────────────────────────────────────────────
    # Enriched corpus, top_k=3, candidate_pool=6
    # This is your working pipeline from 06/07.
    # all_results['baseline'] = run_experiment(
    #     name = "Baseline (correct setup)",
    #     corpus= ENRICHED_CORPUS,
    #     raw_corpus= RAW_CORPUS,
    #     top_k= 3,
    #     candidate_pool= 6,
    # )
    # time.sleep(30)
# ── Failure 1: top_k too small ──────────────────────────────────────────────
    # top_k=1 means only 1 chunk retrieved.
    # Attention and LoRA answers SPAN 2 chunks — one chunk is never enough.
    # Expected: context_recall drops sharply. faithfulness stays high
    # (whatever is retrieved is still used correctly, just incomplete).

    # all_results["failure_top_k"] = run_experiment(
    #     name = "Failure 1: top_k=1 (too small)",
    #     corpus= ENRICHED_CORPUS,
    #     raw_corpus= RAW_CORPUS,
    #     top_k = 1,
    #     candidate_pool= 6,
    # ) 
    # time.sleep(30)
# ── Failure 2: candidate pool too small ─────────────────────────────────────
    # candidate_pool=1 means BM25 and dense each only fetch 1 doc.
    # RRF has almost nothing to fuse — correct chunk may never be a candidate.
    # Expected: context_recall and context_precision both drop.

    # all_results["failure_pool"] = run_experiment(
    #     name = "Failure 2: candidate_pool=1 (too small)",
    #     corpus= ENRICHED_CORPUS,
    #     raw_corpus= RAW_CORPUS,
    #     top_k = 3,
    #     candidate_pool= 1,
    # ) 
    # time.sleep(30)
# ── Failure 3: no contextual enrichment ─────────────────────────────────────
    # Use raw corpus (no context prepended) for embedding.
    # Naive embeddings have no document-level context — chunks look similar
    # to each other because they lack the disambiguating context summary.
    # Expected: context_recall drops. context_precision may also drop
    # as the reranker struggles with decontextualized chunks.
    all_results["failure_no_enrichment"] = run_experiment(
        name           = "Failure 3: no contextual enrichment",
        corpus         = RAW_CORPUS,   # ← raw, not enriched
        raw_corpus     = RAW_CORPUS,
        top_k          = 3,
        candidate_pool = 6,
    )
    time.sleep(30)  # add this between each run_experiment() call
    # ── Fix: all corrections applied ────────────────────────────────────────────
    # This should match or exceed baseline.
    # Included to confirm fixes work, not just break things.
    all_results["fixed"] = run_experiment(
        name           = "Fixed (enriched + top_k=3 + pool=6)",
        corpus         = ENRICHED_CORPUS,
        raw_corpus     = RAW_CORPUS,
        top_k          = 3,
        candidate_pool = 6,
    )
    time.sleep(30)
# ── Summary table ───────────────────────────────────────────────────────────
    print(f"\n\n{'='*60}")
    print("SUMMARY — ALL EXPERIMENTS")
    print(f"{'='*60}")
    print(f"{'Experiment':<30} {'Faith':>6} {'Relev':>6} {'Prec':>6} {'Rec':>6}")
    print("-"*60)
    for name, scores in all_results.items():
        print(
            f"{name:<30} "
            f"{scores['faithfulness']:>6.3f} "
            f"{scores['answer_relevancy']:>6.3f} "
            f"{scores['context_precision']:>6.3f} "
            f"{scores['context_recall']:>6.3f}"
        )
 
    print(f"\n{'='*60}")
    print("DIAGNOSIS")
    print('='*60)
    print("""
  If context_recall drops with top_k=1:
    → Your answers require multiple chunks. Increase top_k.
 
  If context_recall drops with candidate_pool=1:
    → RRF needs more candidates. Increase candidate_pool to 3-5x top_k.
 
  If context_recall drops without enrichment:
    → Your embedding model needs document context to distinguish chunks.
      Keep contextual enrichment in your pipeline.
 
  If faithfulness drops in any experiment:
    → The LLM is hallucinating when context is incomplete.
      Tighten your prompt or add a confidence threshold.
    """)
 
    print("[done]")
 
