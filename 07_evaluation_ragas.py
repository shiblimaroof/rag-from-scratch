"""
07_evaluation_ragas.py
=======================
Evaluate the full RAG pipeline using RAGAS metrics.
 
RAGAS measures 4 things:
  Faithfulness      — is the answer grounded in retrieved context?
  Answer Relevancy  — does the answer address the question?
  Context Precision — are retrieved chunks relevant?
  Context Recall    — did we retrieve everything needed?
 
Each metric targets a different component:
  Faithfulness      → generation quality  (LLM hallucinating?)
  Answer Relevancy  → generation quality  (LLM answering the right thing?)
  Context Precision → retrieval quality   (retrieving noise?)
  Context Recall    → retrieval quality   (missing key information?)
 """

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datasets import Dataset
from sentence_transformers import SentenceTransformer , CrossEncoder
import faiss
from rank_bm25 import BM25Okapi
from groq import Groq
from ragas import evaluate
from dotenv import load_dotenv
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

# ── 1. Load enriched chunks ─────────────────────────────────────────────────────

CHUNKS_FILE = Path("enriched_chunks.json")
 
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
 
texts = [c["enriched"] for c in enriched_chunks]


# ── 2. Build retrieval stack (same as 04-06) ────────────────────────────────────

def tokenize(text):
    return text.lower().split()

bm25 = BM25Okapi([tokenize(t) for t in texts])

EMBED_MODEL = "all-MiniLM-L6-v2"
embedder = SentenceTransformer(EMBED_MODEL)

print(f"[info] Encoding corpus...")

embeddings = embedder.encode(texts, show_progress_bar= True, normalize_embeddings=True)
dim = embeddings.shape[1]
faiss_index = faiss.IndexFlatIP(dim)
faiss_index.add(embeddings.astype(np.float32))

RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
print(f"[info] Loading cross-encoder...")
reranker = CrossEncoder(RERANK_MODEL)

load_dotenv()
key = os.environ.get("GROQ_API_KEY")
groq_client = Groq(api_key=key)

# ── 3. RAG pipeline (same as 06) ───────────────────────────────────────────────

def retrieve_bm25(query, top_k):
    scores = bm25.get_scores(tokenize(query))
    ranked = np.argsort(scores)[::-1][:top_k]
    return[(int(i), float(scores[i])) for i in ranked]

def retrieve_dense(query, top_k):
    vec = embedder.encode([query], normalize_embeddings= True).astype(np.float32)
    sims, idxs = faiss_index.search(vec, top_k)
    return[(int(i), float(s)) for i,s in zip (idxs[0],sims[0]) if i != -1]

def reciprocal_rank_fusiom(ranked_lists, k=60):
    scores = {}
    for ranked_list in ranked_lists:
        for rank ,(doc_idx, _) in enumerate(ranked_list, start =1):
            scores[doc_idx] = scores.get(doc_idx, 0.0) +1.0/(k+rank)
    return sorted(scores.items(), key = lambda x:x[1], reverse = True)

def hybrid_search(query, candidate_pool = 10):
    candidate_pool = min(candidate_pool, len(texts))
    return reciprocal_rank_fusiom([
        retrieve_bm25(query, top_k= candidate_pool),
        retrieve_dense(query, top_k = candidate_pool)])

def rerank(query, candidates, top_k =3):
    if not candidates:
        return []
    pairs = [(query, enriched_chunks[doc_idx]["original"]) for doc_idx,_ in candidates]
    scores  = reranker.predict(pairs)
    scored = [
        {
            "doc_idx" :candidates[i][0], "ce_score": float(scores[i]),
            "text": enriched_chunks[candidates[i][0]]['original']}
            for i in range (len(candidates))
    ]
    scored.sort(key = lambda x: x["ce_score"], reverse = True)
    return scored[:top_k]

def build_prompt(query, chunks):
    context_block = "\n\n".join(f"[{i+1}] {c['text']}" for i, c in enumerate(chunks))
    return f"""you are a helpful assistant. Answer the question using Only the context provided below.
    If the context does not contain enough information, say "i don't have enough context to answer this."
    Do not use any oudsite knowledge

Context : {context_block}

Question : {query}

Answer :"""

def generate(query, chunks):
    prompt = build_prompt(query, chunks)
    response = groq_client.chat.completions.create(
        model = "llama-3.1-8b-instant",
        messages= [{"role" : "user", "content" : prompt}],
        temperature= 0.0,
        max_tokens = 512,
    )
    return response.choices[0].message.content.strip()

def rag(query, top_k =3):
    candidates = hybrid_search(query)
    chunks = rerank(query, candidates, top_k = top_k)
    answer = generate(query,chunks)
    contexts = [c["text"] for c in chunks]
    return answer, contexts

# ── 4. Evaluation dataset ───────────────────────────────────────────────────────


# RAGAS needs 4 things per sample:
#   question        — the query
#   answer          — what your RAG pipeline returned
#   contexts        — the chunks your pipeline retrieved (list of strings)
#   ground_truth    — the ideal answer (you write this manually)
#
# ground_truth is required for context_recall.
# It lets RAGAS ask: "given the ideal answer, did the retrieved chunks
# contain the information needed to produce it?"
# Without ground_truth, you can still measure faithfulness + answer_relevancy.


eval_samples = [
       {
        "question":     "How is attention computed mathematically?",
        "ground_truth": "Attention is computed as softmax(QK^T / sqrt(d_k)) * V, where Q is the query matrix, K is the key matrix, V is the value matrix, and d_k is the key dimension.",
    },
    {
        "question":     "What is LoRA?",
        "ground_truth": "LoRA fine-tunes large models by injecting low-rank matrices into attention layers, enabling parameter-efficient fine-tuning.",
    },
    {
        "question":     "What is the vocabulary size of GPT-2?",
        "ground_truth": "GPT-2 has a vocabulary size of 50257 tokens using BPE tokenization.",
    },
    {
        "question":     "How is BERT pretrained?",
        "ground_truth": "BERT is pretrained on masked language modeling and next sentence prediction.",
    },
]

# ── 5. Run pipeline on eval set ─────────────────────────────────────────────────
 
print("\[info] Running RAG pipeline on eval set..")

questions = []
answers = []
contexts = []
ground_truths = []

for sample in eval_samples:
    q = sample["question"]
    print(f"f -> {q}")
    answer, retrieved_context = rag(q)

    questions.append(q)
    answers.append(answer)
    contexts.append(retrieved_context)
    ground_truths.append(sample["ground_truth"])

# RAGAS expects a HuggingFace Dataset with these exact column names

eval_dataset = Dataset.from_dict({
    "question" : questions,
    "answer" :answers,
    "contexts" : contexts,
    "ground_truth" : ground_truths,
})

print(f"\n[info] eval dataset built : {len(eval_dataset)} samples")

# ── 6. Configure RAGAS to use Groq + HuggingFace embeddings ────────────────────
 
# RAGAS uses an LLM internally to judge faithfulness and answer relevancy.
# We point it at Groq (free) instead of OpenAI (paid).
# We also give it a local embedding model for answer_relevancy scoring.

ragas_llm = LangchainLLMWrapper(
    ChatGroq(model="llama-3.1-8b-instant", api_key=key, temperature=0.0)
)
 
ragas_embeddings = LangchainEmbeddingsWrapper(
    HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
)

# ── 7. Run RAGAS evaluation ─────────────────────────────────────────────────────

print("\n[info] Running RAGAS evaluation...")
print("This makes LLM calls for each metric — takes ~30-60 seconds.\n")

results = evaluate(
    dataset=eval_dataset,
    metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    llm=ragas_llm,
    embeddings=ragas_embeddings,
)
 
# ── 8. Print results ────────────────────────────────────────────────────────────
 
print("\n" + "="*50)
print("RAGAS EVALUATION RESULTS")
print("="*50)

df = results.to_pandas()
print(df.columns.tolist())

# Per-sample scores
print("\nPer-sample scores:")
for i, row in df.iterrows():
    print(f"\n  Q: {row['question']}") 
    print(f"    Faithfulness:     {row['faithfulness']:.3f}")
    print(f"    Answer Relevancy: {row['answer_relevancy']:.3f}")
    print(f"    Context Precision:{row['context_precision']:.3f}")
    print(f"    Context Recall:   {row['context_recall']:.3f}")

# Aggregate scores
print("\n" + "-" * 50)
print('Aggregate (mean across all samples):')
print(f" Faithfulness: {df['faithfulness'].mean():.3f}")
print(f"Answer relevancy : {df['answer_relevancy'].mean():.3f}")
print(f"Context Precision: {df['context_precision'].mean():.3f} ")
print(f"Context Recall: {df['context_recall'].mean():.3f}")
print("_"*50)

# Diagnosis
print("\nDiagnosis:")
faith     = df['faithfulness'].mean()
relevancy = df['answer_relevancy'].mean()
precision = df['context_precision'].mean()
recall    = df['context_recall'].mean()

if faith < 0.7:
    print("  ⚠ Faithfulness low  → LLM is hallucinating. Fix: tighten prompt, lower temperature.")
if relevancy < 0.7:
    print("  ⚠ Answer Relevancy low → LLM answering wrong question. Fix: rewrite prompt instruction.")
if precision < 0.7:
    print("  ⚠ Context Precision low → retrieving noisy chunks. Fix: reduce top_k, improve reranker.")
if recall < 0.7:
    print("  ⚠ Context Recall low → missing key chunks. Fix: increase candidate_pool, fix chunking.")
if all(s >= 0.7 for s in [faith, relevancy, precision, recall]):
    print("  ✅ All metrics above 0.7 — pipeline is working well.")
 
print("\n[done]")
