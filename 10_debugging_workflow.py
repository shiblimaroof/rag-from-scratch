"""
10_debugging_workflow.py
=========================
Full RAG pipeline diagnostic — health scores + latency breakdown.

What this file does:
  Runs a query through every stage of the pipeline.
  Each stage gets a health score (0.0 → 1.0).
  Overall pipeline health = geometric mean of all stage scores.
  Geometric mean is the right formula here: one broken stage (0.0)
  collapses the entire score to 0 regardless of other stages being perfect.
  That's the correct behavior — a broken stage breaks the pipeline.

What makes this different from 07, 08, 09:
  07 — measures quality metrics on a test set   → HOW BAD is your system?
  08 — retrieval failure analysis               → WHERE does retrieval fail?
  09 — generation failure harness               → WHICH failure modes exist?
  10 — live diagnostic on a real query          → WHERE did THIS query break?

  Use 07/08/09 in CI. Use 10 when a user reports a bad answer.

Stages:
  1. Chunking    — is the answer even in your corpus?
  2. Retrieval   — did the right chunks come back?
  3. Reranking   — did reranking help or hurt?
  4. Generation  — did the model answer from context?
  5. Guard check — does the model refuse on empty context?

Latency:
  Every stage is timed. This tells you where your pipeline is slow.
  Healthy targets:
    Retrieval  : < 200ms
    Reranking  : < 500ms
    Generation : < 3000ms (network + model)

CLI mode:
  python 10_debugging_workflow.py "your query here"
  Pass the actual query that produced the bad answer.
  That's the whole point of this file.

Prerequisites:
  - 06_generation.py in same directory
  - GROQ_API_KEY in .env
  - enriched_chunks.json optional (toy corpus used as fallback)
"""

import os
import sys
import time
import textwrap
import importlib
import importlib.util
import numpy as np
from dataclasses import dataclass, field

# ── Terminal colours ─────────────────────────────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
RESET  = "\033[0m"
BOLD   = "\033[1m"

def ok(msg):
    print(f"{GREEN}[PASS]{RESET} {msg}")

def fail(msg):
    print(f"{RED}[FAIL]{msg}")

def warn(msg):
    print(f"{YELLOW}[WARN]{msg}")

def info(msg):
    print(f"[INFO] {msg}")

# ── Timer ────────────────────────────────────────────────────────────────────────

class Timer:
    def __enter__(self):
        self.start = time.perf_counter()
        return self
    def __exit__(self, *args):
        self.elapsed_ms = (time.perf_counter() - self.start) *1000

# ── Dataclasses ──────────────────────────────────────────────────────────────────

@dataclass
class StageResult:
    stage_id    :       int
    name        :       str
    health      :       float
    passed      :       bool
    finding     :       str
    fix         :       str     =   ""
    latency_ms  :       float   =   0.0
    details     :       dict    =   field(default_factory=dict)


@dataclass
class DiagnosticReport:
    query           :     str
    stages          :     list   =  field(default_factory=list)
    final_answer    :     str    = ""
    weakest_stage   :     object = None

    def add(self, result : StageResult):
        self.stages.append(result)
        if self.weakest_stage is None or result.health  < self.weakest_stage.health:
            self.weakest_stage = result

# ── Visual helpers ───────────────────────────────────────────────────────────────

def health_bar (score:float, width: int =20) -> str:
     """
    Visual health bar: [████████░░░░░░░░░░░░] 0.40
    At a glance you can see which stage is the weakest.
    """
     filled = int(score * width)
     bar = "█" * filled + "░" * (width - filled)
     return f"[{bar}] {score:.2f}"

def health_label(score:float) -> str:
    if score >= 0.85 : return f"{GREEN}HEALTHY{RESET}"
    if score >= 0.60 : return f"{YELLOW}DEGRADED{RESET}"
    return f"{RED}BROKEN{RESET}"

def print_divider(char = '-', width = 65):
    print(char*width)

def print_stage_header(stage_id:int, name:str):
    print_divider('=')
    print(f"{BOLD}STAGE {stage_id} - {name}{RESET}")
    print_divider('=')

def print_result_compact(r:StageResult):
    print(f"\n Health : {health_bar(r.health)} {health_label(r.health)}")
    print(f" Latency : {r.latency_ms:.1f}ms")
    print(f" Finding : {r.finding}")
    if r.fix:
        print(f"\n  {YELLOW}Fix needed:{RESET}")
        print(textwrap.fill(r.fix, width=60,
                            initial_indent="    ",
                            subsequent_indent="    "))
    print()

# ── Utilities ────────────────────────────────────────────────────────────────────
def keyword_overlap(query: str, text : str) -> float:
     """
    Fraction of query keywords found in text.
    Stopwords removed so meaningful terms are matched.
    Used as a lightweight relevance proxy — no model call needed.
    Catches obvious failures where retrieved chunks are off-topic.
    """
     stopwords = {
         "the", "a", "an", "is", "in", "of", "to", "and", "for",
        "what", "how", "why", "does", "did", "was", "are", "be", "it"
     }
     tokens = {
         t.lower().strip("?.,") for t in query.split()
         if t.lower() not in stopwords and len(t) >2
     }
     if not tokens:
         return 0.0
     text_lower = text.lower()
     hits = sum(1 for t in tokens if t in text_lower)    
     return hits/ len(tokens)

def refusal_detected(answer : str) -> bool:
    """Same refusal phrase list as 09 for consistency."""
    lower = answer.lower()
    phrases = [
        "don't have enough context", "do not have enough context",
        "cannot answer", "can't answer", "not enough information",
        "i don't know", "the context does not", "context does not contain",
        "not provided in the context", "not mentioned in the context",
    ]
    return any(p in lower for p in phrases)

# ════════════════════════════════════════════════════════════════════════════════
# LOAD 06_generation.py
# ════════════════════════════════════════════════════════════════════════════════
#
# Design decision — import only from 06, not from 01/02/04/05:
#   06_generation.py initializes the full pipeline at module level:
#   BM25 index, FAISS index, embedder, cross-encoder, Groq client.
#   Importing it gives us everything.
#   Importing earlier files separately would re-initialize all those models
#   a second time, doubling load time, and risks function signature mismatches.
#   Always import from the most downstream module that has what you need.

print_divider("═")
print(f"{BOLD}RAG PIPELINE DIAGNOSTIC{RESET}")
print_divider("═")
print()

try:
    gen06_path = os.path.join(os.getcwd(), "06_generation.py")
    spec = importlib.util.spec_from_file_location("gen06", gen06_path)
    gen06 = importlib.util.module_from_spec(spec)
    with Timer() as t:
        spec.loader.exec_module(gen06)
    ok(f"06_generation.py loaded in {t.elapsed_ms:.0f}ms")
except Exception as e:
    print(f"{RED}[error]{RESET} Could not load 06_generation.py:{e}")
    sys.exit(1)

generate = gen06.generate
build_prompt = gen06.build_prompt
rag = gen06.rag
hybrid_search = gen06.hybrid_search
rerank = gen06.rerank
enriched_chunks = gen06.enriched_chunks
GENERATION_MODEL = getattr(gen06, "GENERATION_MODEL", "llama-3.1-8b-instant")

print(f" -> Model        : {GENERATION_MODEL}")
print(f" -> Corpus size  : {len(enriched_chunks)} chunks")

# ════════════════════════════════════════════════════════════════════════════════
# DIAGNOSTIC RUNNER
# ════════════════════════════════════════════════════════════════════════════════

def run_diagnostic(query: str, top_k =3) -> DiagnosticReport:
     """
    Run a full pipeline diagnostic on a single query.

    Parameters
    ----------
    query  : the user question to trace through the pipeline
    top_k  : chunks to retrieve/rerank (default 3, matching 06)
    """
     report = DiagnosticReport(query=query)

    # ── STAGE 1: CHUNKING ────────────────────────────────────────────────────
    #
    # Is the answer in the corpus at all?
    # If no chunks contain query keywords, no retriever can find the answer.
    # The content simply isn't there. Tuning retrieval won't fix this.
    #
    # Health formula:
    #   content_score (0.6) — max keyword overlap across all chunks
    #   size_score    (0.4) — chunk sizes are in a reasonable range

     print_stage_header(1, "CHUNKING - Is the content in the corpus?")
     
     with Timer() as t:
         chunk_texts = [c["original"] for c in enriched_chunks]
         lengths = [len(t.split()) for t in chunk_texts]
         avg_len = sum(lengths) / len(lengths) if lengths else 0
         too_short = sum(1 for l in lengths if l<10)
         too_long = sum(1 for l in lengths if l>500)
         overlaps = [keyword_overlap(query, t) for t in chunk_texts]
         max_overlap = max(overlaps) if overlaps else 0
         best_idx = overlaps.index(max_overlap) if overlaps else 0

         size_score = 1.0
         if len(lengths) > 0:
             if too_short / len(lengths) > 0.3 : size_score -= 0.3
             if too_long / len(lengths) > 0.2 : size_score -= 0.3
         content_score = min(1.0, max_overlap /0.4)
         health_1 = 0.4 * size_score + 0.6 * content_score

         finding = (
             f"Corpus : {len(enriched_chunks)} chunks, avg {avg_len:.0f}"
             f"Best keyword overlap: {max_overlap:.2f} (chunk #{best_idx})"
         )
         fix = ""
         if max_overlap <0.2:
             fix =(
                 
              "No chunks match query keywords. The answer may not be in your corpus. "
                 "Check: (1) Did you ingest the right document? "
                 "(2) Is your chunker splitting relevant sentences across chunks? "
                 "(3) Try query expansion — corpus may use different vocabulary."
             )
         elif too_short / max(len(lengths), 1) > 0.3:
             fix = (
                 f"{too_short} chunks are under 10 words. Too-small chunks lose context. "
                "Increase min_chunk_size in 01_chunking.py."
             )
     info(f"Avg chunk length : {avg_len:.0f} words")
     info(f"Too short (<10w) : {too_short}")
     info(f"Too long (>500w) : {too_long}")
     info(f"Best overlap     : {max_overlap:.2f}")

     r1= StageResult(1, "Chunking", health_1, health_1 >=0.6,
                     finding, fix, t.elapsed_ms,
                     details= {"n_chunks : len(enriched_chunks),"
                     "max_overlap" : max_overlap})
    
     report.add(r1)
     print_result_compact(r1)


    # ── STAGE 2: RETRIEVAL ───────────────────────────────────────────────────
     # Did the right chunks come back?
    # Keyword overlap = lightweight relevance proxy without ground-truth labels.
    # For precision/recall with labels: use 07_evaluation_ragas.py.
    #
    # Health formula:
    #   avg keyword overlap across top-k, scaled 0-1
    #   penalty if best chunk is ranked outside top 3

     print_stage_header(2, "RETRIEVAL - Did the right chunks come back.?" )

     chunks_ret = []
     with Timer() as t:
        try:
            candidates = hybrid_search(query, candidate_pool = 10)
            chunks_ret = rerank(query, candidates, top_k = top_k)

            if not chunks_ret :
                r2 = StageResult(2, "Retrievel, 0.0, False,"
                                "hybrid_search() returned no results,"
                                "Check BM25 and FAISS indexes are built.",
                                t.elapsed_ms)
                report.add(2)
                print_result_compact(r2)
                return report
            
            ret_overlaps = [keyword_overlap(query, c["text"]) for c in chunks_ret]
            avg_overlap = sum(ret_overlaps) / len(ret_overlaps)
            best_overlap = max(ret_overlaps)
            best_rank = ret_overlaps.index(best_overlap) +1
            rank_ok = best_rank <= 3
            health_2 = min(1.0, avg_overlap / 0.4) * (1.0 if rank_ok else 0.75)

            finding = (
                f"Retrieved {len(chunks_ret)} chunks "
                f" Avg overlap : {avg_overlap:.2f}"
                f" Best overlap : {best_overlap:.2f} at rank #{best_rank}"
            )

            fix = ""
            if avg_overlap <0.2:
                fix = (
                    "Retrieved chunks have low keyword overlap. "
                    "Possible causes: (1) Query vocabulary differs from corpus. "
                    "(2) FAISS index is stale — re-run 02_embeddings_and_index.py. "
                    "(3) Check Stage 1 — if corpus overlap was also low, "
                    "the content isn't in the corpus."
                )

            elif not rank_ok:
                fix = (
                    f"Best chunk is at rank #{best_rank}."
                    "Increase candidate_pool in hybrid_search() so reranking "
                    "has more candidates to work with."
                )

            info(f"AVG OVERLAP : {avg_overlap :.2f}")
            info(f"Best overlap : {best_overlap:.2f} at rank #{best_rank}")
            info(f"Top chunk : {chunks_ret[0]['text'][:80]}")

        except Exception as e:
            r2 = StageResult(2, "Retrieval", 0.0, False, f"hybrid_search() raised: {e}","check 04_hybrid_search.py", t.elapsed_ms)
            report.add(r2)
            print_result_compact(r2)
            return report
        
     r2 = StageResult(2, "Retrieval", health_2, health_2>0.6, finding, fix, t.elapsed_ms,
                        details={"avg_overlap" : avg_overlap, "best_rank" : best_rank})
     report.add(r2)
     print_result_compact(r2)

# ── STAGE 3: RERANKING ───────────────────────────────────────────────────
    #
    # Did reranking help or hurt?
    # A cross-encoder trained on a different domain will actively MIS-rank chunks.
    # If Stage 3 health < Stage 2 health → reranker is hurting you.
    #
    # Health formula:
    #   starts at 1.0
    #   -0.3 if best chunk moved DOWN after reranking
    #   -0.3 if cross-encoder scores are all zero or None
    #   -0.2 if best chunk is still outside top 3

     print_stage_header(3, "RERANKING - Did reeranking help or hurt")

     with Timer() as t:
         try:
             rerank_overlaps = [keyword_overlap(query, c["text"]) for c in chunks_ret]
             post_best_rank = rerank_overlaps.index(max(rerank_overlaps)) +1
             pre_best_rank = r2.details.get("best_rank", top_k)
             rank_improved = post_best_rank <= pre_best_rank
             top_ce_score = chunks_ret[0].get("ce_score", None) if chunks_ret else None
             ce_ok = top_ce_score is not None and top_ce_score > -10

             health_3 = 1.0
             if not rank_improved : health_3 -= 0.3
             if not ce_ok : health_3 -= 0.3
             if post_best_rank >3 : health_3 -= 0.2
             health_3 = max(0.0, health_3)

             ce_str = f"{top_ce_score: .4f}" if top_ce_score is not None else "N/A"
             finding = (
                  f"Best chunk: rank #{pre_best_rank} → #{post_best_rank} after reranking. "
                f"Top ce_score: {ce_str}."
             )
             fix = ""
             if not rank_improved:
                 fix = (
                     "Reranking moved the best chunk DOWN. "
                    "Cross-encoder may be domain-mismatched. "
                    "Fix: (1) Remove reranking and compare RAGAS scores with/without it. "
                    "(2) Fine-tune cross-encoder on your corpus."
                 )
             elif not ce_ok:
                 fix = (
                     "Cross-encoder score missing or invalid. "
                    "Check 05_reranking.py returns 'ce_score' in each chunk dict."
                 )
                 info(f"Pre-rerank best rank  : #{pre_best_rank}")
                 info(f"Post-rerank best rank : #{post_best_rank}")
                 info(f"Top ce_score          : {ce_str}")

         except Exception as e:
             r3 = StageResult(3, "reranking", 0.0, False, 
                              f"Reranking check raised: {e}",
                             "Check reranked chunks have 'ce_score' field.",
                             t.elapsed_ms)
            
             report.add(r3)
             print_result_compact(r3)
             health_3 = 0.0
     
     r3 = StageResult(3, "Reranking", health_3, health_3 >= 0.6, finding, fix, t.elapsed_ms,
                      details = {"pre_best_rank": pre_best_rank,
                              "post_best_rank": post_best_rank,
                              "rank_improved": rank_improved})
     report.add(r3)
     print_result_compact(r3)


# ── STAGE 4: GENERATION ──────────────────────────────────────────────────
    #
    # Did the model answer from context?
    # Grounding check: a grounded answer should contain query keywords.
    # If the answer has zero keyword overlap with the query, it likely
    # hallucinated off-topic or answered a different question entirely.
    #
    # Special case: if Stage 2 retrieval was bad (health < 0.5),
    # a refusal is the CORRECT behavior — upgrade health for correct refusal.

     print_stage_header(4, "GENERATION — Did the model answer from context?")

     with Timer() as t:
         try:
             answer = generate(query, chunks_ret[:top_k])
             report.final_answer = answer

             word_count = len(answer.split())
             ans_overlap = keyword_overlap(query, answer)
             is_refusal = refusal_detected(answer)
             grounded = ans_overlap >= 0.3 and not is_refusal

             health_4 = 1.0
             if word_count <3 :                  health_4 = 0.0
             elif is_refusal  :                  health_4 = 0.5
             elif not grounded :                 health_4 -= 0.4
             if word_count <15 and not is_refusal: health_4 -= 0.2
             if word_count >400:                 health_4 -= 0.1
             health_4 = max(0.0, min(1.0, health_4))

              # Correct refusal on bad retrieval = healthy behavior
             if is_refusal and r2.health < 0.5:
                 health_4 = 0.85
            
             finding = (
                 f"Answer : {word_count} words."
                 f"Keyword overlap: {ans_overlap:.2f}."+
                 ("RESUFAL." if is_refusal else "Grounded." if grounded else "Possible hallucination.")
             )
             fix = ""
             if word_count < 3:
                 fix = "generate() returned empty string. Check GROQ_API_KEY and model name"

             elif not grounded and not is_refusal:
                 fix = (
                     f"Answer has low keyword overlap ({ans_overlap:.2f}). "
                    "Model may have used parametric memory instead of context. "
                    "Move grounding instruction to system role in 06_generation.py."
                 )

             elif word_count <15 and not is_refusal:
                 fix = "Answer too short. Increase max_tokens in generate()"

            
             info(f"Word count : {word_count}")
             info(f"Overlap  : {ans_overlap:.2f}")
             info(f"Refusal  : {is_refusal}")
             info(f"Grounded  : {grounded}")

         except Exception as e:
             report.final_answer = ""
             r4 = StageResult(4, "Generation", 0.0, False, f"generate() raised : {e}", "Check GROQ_API_KEY and model name in 06_generation.py.",t.latency_ms)
             
             
             report.add(r4)
             print_result_compact(r4)
             return report
         
     r4 = StageResult(4, "Generation", health_4, health_4 > 0.6, finding, fix, t.elapsed_ms,
                          details= {"word_count": word_count,
                              "ans_overlap": ans_overlap,
                              "is_refusal": is_refusal,
                              "grounded": grounded})
         
     report.add(r4)
     print_result_compact(r4)

    # ── STAGE 5: GUARD CHECK ─────────────────────────────────────────────────
    #
    # Does the model refuse when context is empty?
    # Call generate() with NO chunks.
    # If it answers confidently → guard is broken.
    #
    # Why separate from Stage 4:
    #   Stage 4 = generation on GOOD context
    #   Stage 5 = guard on BAD context
    #   A system that works on good context but hallucinates on empty context
    #   will fail silently in production whenever retrieval returns nothing.


     print_stage_header(5, "GUARD CHECK — Does the model refuse on empty context?")

     with Timer() as t:
         try:
             empty_answer = generate(query, [])
             guard_fired = refusal_detected(empty_answer)
             health_5 = 1.0 if guard_fired else 0.2

             finding = (
                 "Guard fired corrextly on empty context."
                 if guard_fired else
                 f"Guard FAILED -model answered with {len(empty_answer.split())} words"
                              "despite empty context"          
                     )
                              
             fix = (
                     ""if guard_fired else
                     "Move the 'say i dont know' instruction to system role in 06_generation.py"

                     "messages=[{'role': 'system', 'content': guard_instruction}, ...]. "
                     "System-role instructions are harder for the model to override. "
                     "See 09_generation_failures.py Test 5 for the full guard stress test."

                 )
             info(f"Guard fired : {guard_fired}")
             info(f"Response : {empty_answer[:80]}")

         except Exception as e:
             r5 = StageResult(5, "Guard Check", 0.0, False,
                              f"generate() raised on empty context : {e}",
                              "Check generate() handles empty chunk list.", t.elapsed_ms)
             
             report.add(r5)
             print_result_compact(r5)
             return report
         

     r5 = StageResult(5, "Guard Check", health_5, guard_fired, finding, fix,t.elapsed_ms,
                          details = {"guard_fired" : guard_fired})
         
     report.add(r5)
     print_result_compact(r5)

     return report

# ════════════════════════════════════════════════════════════════════════════════
# FINAL REPORT
# ════════════════════════════════════════════════════════════════════════════════

def print_final_report(report: DiagnosticReport):
    print_divider("═")
    print(f"{BOLD}DIAGNOSTIC SUMMARY{RESET}")
    print_divider("═")
    print(f"\n  Query: {report.query}\n")
 
    print_divider()
    print(f"  {'St':<4} {'Stage':<18} {'Health':<30} {'Latency':>9}  Status")
    print_divider()
    for r in report.stages:
        bar = health_bar(r.health, width=15)
        label = health_label(r.health)
        lat = f"{r.latency_ms:0f}ms"
        print(f"  {r.stage_id:<4} {r.name:<18} {bar:<30} {lat:>9}  {label}")
    print_divider()

    # Overall pipeline health — geometric mean
    # Why geometric mean not arithmetic mean:
    #   Arithmetic mean of [1.0, 1.0, 1.0, 1.0, 0.0] = 0.8  → looks okay
    #   Geometric mean    of [1.0, 1.0, 1.0, 1.0, 0.0] = 0.0 → correctly broken
    # One broken stage breaks the pipeline. The score should reflect that.

    scores = [r.health for r in report.stages]
    geo_mean = float(np.prod(scores) ** (1.0/len(scores))) if scores else 0.0

    print(f"\n  Overall pipeline health : {health_bar(geo_mean)}  {health_label(geo_mean)}")
    print(f"  (geometric mean — one broken stage collapses overall score to 0)\n")

    # Weakest stage + fix

    if report.weakest_stage:
        ws = report.weakest_stage
        all_healthy = all(r.health >= 0.85 for r in report.stages)
    if all_healthy:
        print(f"  {GREEN}All stages healthy — no weak stage detected.{RESET}")
    else:
        print(f"  {YELLOW}Weakest stage:{RESET} Stage {ws.stage_id} — {ws.name} "
              f"(health: {ws.health:.2f})")
        
    if ws.fix:
        print(f"\n  Recommended fix:")
        print(textwrap.fill(ws.fix, width=60,
                                initial_indent="    ",
                                subsequent_indent="    "))

        # Final answer
    if report.final_answer:
        print(f"\n{'─'*65}")
        print(f"  {BOLD}FINAL ANSWER:{RESET}")
        print(f"{'─'*65}")
        print(textwrap.fill(
            report.final_answer[:500] + ("…" if len(report.final_answer) > 500 else ""),
            width=62, initial_indent="  ", subsequent_indent="  "
        ))
 
    print(f"\n{'═'*65}")
    print("Next: FastAPI + Docker + Gradio")
    print(f"{'═'*65}\n")


# ════════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ════════════════════════════════════════════════════════════════════════════════
 
if __name__ == "__main__":
    if len(sys.argv) > 1:
        # CLI mode: pass the actual bad query
        # python 10_debugging_workflow.py "your broken query here"
        query = " ".join(sys.argv[1:])
    else:
        # Default demo query — in domain for the toy corpus
        query = "How does scaled dot-product attention work in transformers?"
 
    print(f"\n  Query  : {query}")
    print(f"  Stages : Chunking → Retrieval → Reranking → Generation → Guard")
    print(f"  Tip    : python 10_debugging_workflow.py \"your query\"\n")
 
    report = run_diagnostic(query, top_k=3)
    print_final_report(report)
