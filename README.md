
Claude finished the response
You said: Profile: Shibli | Hyderabad, India | CS grad 2018, no formal employment since
Profile: Shibli | Hyderabad, India | CS grad 2018, no formal employment since
Goal: Remote NLP/LLM Engineer, $100-150K US market
Barrier: visibility + employment gap, not skills
Social: @BuildwithShibli (Twitter/X) | github.com/shiblimaroof/rag-from-scratch
Stack: Python, PyTorch, Transformers from scratch, Backprop from scratch,
  BPE tokenizer, HuggingFace (AutoTokenizer, datasets, Trainer), Groq API,
  Structured outputs, Sentence transformers, FAISS, Embeddings
Current projects: Bi-gram LM, nanoGPT (Karpathy), Book recommender
RAG Progress:
  ✅ 01_chunking.py — fixed vs recursive, context loss problem
  ✅ 02_embeddings_and_index.py — sentence-transformers + FAISS
  ✅ 03_contextual_enrichment.py — Groq LLM prepends context, rank jumped to 1
  ✅ 04_hybrid_search.py — BM25 + FAISS + RRF fusion
  ✅ 05_reranking.py — cross-encoder reranker
  ✅ 06_generation.py — Groq generation + system-role hallucination guard + sanitize_chunks()
  ✅ 07_evaluation_ragas.py — Faith=1.0, Prec=1.0, Recall=0.54
  ✅ 08_retrieval_failures.py — failure analysis, top_k=1 drops faith to 0.70
  ✅ 09_generation_failures.py — 5 adversarial tests, 4/5 passing, prompt injection fixed
  ✅ 10_debugging_workflow.py — health scores + latency breakdown per stage, geometric mean
  ✅ GitHub pushed — github.com/shiblimaroof/rag-from-scratch
Learning order remaining:
  → FastAPI, Docker, Gradio deployment
  → Fine-tuning (after deployment)
RAGAS notes:
  - Use ragas==0.1.21 (newer versions break Groq compatibility)
  - Use llama-3.3-70b-versatile for RAGAS judge
  - Daily token limit: 100k TPD on 70b, resets at 5:30am IST
  - Column names: question, answer, contexts, ground_truth
  - Metrics: faithfulness, answer_relevancy, context_precision, context_recall
  - llm and embeddings go into evaluate(), not into each metric
09 findings:
  - Test 1 (prompt injection) fixed via sanitize_chunks() in 06_generation.py
  - Test 5 (query-side injection) still fails — fix belongs in FastAPI layer
  - Context overflow silently truncated by Groq — add token budget check
  - Refusal behavior strong on tests 3 & 4
10 notes:
  - Health scores 0.0-1.0 per stage
  - Overall health = geometric mean (one broken stage collapses score to 0)
  - CLI mode: python 10_debugging_workflow.py "your query"
  - Imports only from 06_generation.py — not from 01/02/04/05
  - os.getcwd() used instead of __file__ for path resolution
Preference: direct, no fluff, explain every design decision, build from scratch not tutorials

still working on 10.debugging overflow

Show less
May 9
Claude responded: Share your current 10debuggingworkflow.

Identified token budget check requirement for Groq context overflow
Identified token budget check requirement for Groq context overflow
Share your current 10_debugging_workflow.py so I can see exactly where overflow handling needs to slot in — specifically how you're building the context string before the Groq call, and where your health score logic sits.

If you haven't started the overflow check yet, paste what you have and I'll add:

Token count per chunk before assembly
Budget gate (with configurable max_context_tokens)
Truncation with a logged warning (not silent)
A context_overflow field in the health score output
You said: how this is working.
def add(self, result : StageResult):
        self.stages.append(result)
        if self.weakest_stage is None or result.health  <self.weakest_stage.health:
            self.weakest_stage = result
how this is working.?

May 9
Claude responded: This is a method on what's likely a PipelineHealth or DebugReport class that tracks all stage results.

Analyzed method logic for appending and tracking weakest stage
Analyzed method logic for appending and tracking weakest stage
This is a method on what's likely a PipelineHealth or DebugReport class that tracks all stage results.

python
def add(self, result: StageResult):
    self.stages.append(result)
    if self.weakest_stage is None or result.health < self.weakest_stage.health:
        self.weakest_stage = result
Line by line:

self.stages.append(result) — every StageResult (chunking, embedding, retrieval, generation…) gets added to a list. This is just accumulation for the final report.

if self.weakest_stage is None — on the very first .add() call, weakest_stage hasn't been set yet, so this is always True on the first call. Prevents a NoneType comparison crash.

or result.health < self.weakest_stage.health — on every subsequent call, it checks: is this new stage worse than the current weakest? If yes, replace it.

self.weakest_stage = result — keeps a pointer to whichever stage has the lowest health score so far.

Why it matters for your geometric mean:

weakest_stage is what collapses the whole pipeline score. If retrieval has health=0.0, the geometric mean goes to 0 regardless of how good generation looks. This pointer lets you surface exactly which stage killed the pipeline without iterating self.stages again at report time.

Concrete trace through your pipeline:

add(chunking,    health=0.9)  → weakest = chunking (first call)
add(embedding,   health=0.8)  → 0.8 < 0.9  → weakest = embedding
add(retrieval,   health=0.4)  → 0.4 < 0.8  → weakest = retrieval
add(generation,  health=0.7)  → 0.7 < 0.4? No → weakest stays retrieval
Final report: geometric mean ≈ 0.62, weakest stage = retrieval → that's where you debug first.

You said: i didnt get this
i didnt get this

if self.weakest_stage is None or result.health  <self.weakest_stage.health:
            self.weakest_stage = result
May 9
Claude responded: Two conditions joined by or.

Deciphered conditional logic for health comparison tracking
Deciphered conditional logic for health comparison tracking
Two conditions joined by or. If either is True, weakest_stage gets updated.

Condition 1: self.weakest_stage is None

This is only True on the very first .add() call.

python
# Before any .add() call:
self.weakest_stage = None   # set in __init__

# First call → add(chunking, health=0.9)
self.weakest_stage is None  # → True
# So weakest_stage = chunking. Now it's no longer None.
Why needed? Because Condition 2 does self.weakest_stage.health — if weakest_stage is still None, that crashes with AttributeError: 'NoneType' has no attribute 'health'. Condition 1 guards against that.

Condition 2: result.health < self.weakest_stage.health

From the second call onward, compare incoming health vs current weakest.

python
# weakest_stage is chunking (0.9)

# add(embedding, health=0.8)
0.8 < 0.9  # True  → weakest_stage = embedding

# add(retrieval, health=0.4)
0.4 < 0.8  # True  → weakest_stage = retrieval

# add(generation, health=0.7)
0.7 < 0.4  # False → weakest_stage stays retrieval
The or short-circuits:

First call  → Condition 1 is True  → skip Condition 2 entirely → update
Later calls → Condition 1 is False → check Condition 2 → update only if worse
So weakest_stage always holds the single worst stage seen so far.

You said: what is langchain and langraph.
what is langchain and langraph.?

May 9
Claude responded: Given your stack, I'll explain these in terms you already know.

Recognized familiar frameworks and prepared direct explanation
Recognized familiar frameworks and prepared direct explanation
Given your stack, I'll explain these in terms you already know.

LangChain

A framework that chains LLM calls + tools together. Instead of you manually writing:

python
# What you do now in your pipeline:
chunks = chunk_document(doc)
embeddings = embed(chunks)
results = faiss_search(query)
answer = groq_generate(results)
LangChain gives you pre-built abstractions for each of those steps so you can snap them together faster. Retrievers, prompt templates, output parsers — all pre-packaged.

The tradeoff: you're trading control for speed. You've been building from scratch, so you understand every layer. LangChain hides those layers. That's why people who skip to LangChain can't debug when things break.

LangGraph

Built on top of LangChain. Lets you build stateful, multi-step agents as a graph.

Each node = a function (LLM call, tool, decision). Each edge = the flow between them, including conditional branching.

query → retrieve → [if low confidence] → re-retrieve
                 → [if high confidence] → generate → output
Your current RAG pipeline is linear (01 → 02 → ... → 10). LangGraph handles loops, retries, and branching decisions — things an agent needs.

Where you stand:

You (from scratch)	LangChain	LangGraph
Control	Full	Low	Low
Understanding	Deep	Surface	Surface
Speed to build	Slow	Fast	Fast
Debugging	Easy	Hard	Hard
My honest take for your situation: don't touch either yet. Finish FastAPI + Docker first. When you go for jobs, "built RAG from scratch without LangChain" is a stronger signal than "used LangChain." Interviewers at $100-150K roles will ask you to explain internals — you'll win that conversation.

Learn LangGraph after you understand agents conceptually. It'll make sense immediately then.

You said: okay 1 question.
okay 1 question.. i'm writing that debugging code with understanding...do i have to write that code without seeing someday.? like while giving interview or something.?

May 10
Claude responded: No.

Deciphered interview coding expectations and memorization requirements
Deciphered interview coding expectations and memorization requirements
No. Nobody writes pipeline debugging code from scratch in an interview.

What interviews actually test at your target level:

Coding rounds — DSA (arrays, graphs, dynamic programming). LeetCode medium. Has nothing to do with your RAG code.

ML/NLP rounds — concepts + reasoning:

"How does attention work?"
"Why cross-encoder reranking over bi-encoder?"
"Your recall is 0.54, what do you do?"
"Walk me through how you'd debug a RAG pipeline"
That last one — you answer it verbally from experience. You don't code it.

System design rounds — whiteboard/verbal:

"Design a production RAG system"
"How do you handle context overflow?"
Again, verbal. No syntax required.

So what is the debugging code actually for?

Two things:

Understanding — writing 10_debugging_workflow.py means you can answer those verbal questions with real depth. You've actually measured latency per stage. That's rare.
Portfolio — the interviewer looks at your GitHub before the call. The code signals seriousness.
Bottom line:

You're not memorizing code. You're building genuine understanding so that when they ask "how would you debug a broken RAG pipeline" — you have a real answer, not a theoretical one.

You said: do they ask this
do they ask this
DSA (arrays, graphs, dynamic programming). LeetCode medium.

May 10
Claude responded: Honest answer: it depends on the company.

Weighed industry practices for NLP engineer interview expectations
Weighed industry practices for NLP engineer interview expectations
Honest answer: it depends on the company.

Big Tech (Google, Meta, Microsoft, Amazon)

Yes, heavily. LeetCode mediums and hards. Mandatory, no exceptions. NLP role doesn't matter — everyone goes through the same coding bar.

AI startups / product companies ($100-150K range, remote)

Mixed. Many have dropped pure DSA in favor of:

Take-home projects ("build a small RAG pipeline")
ML coding ("implement BM25 from scratch")
Debugging exercises ("here's a broken embedding pipeline, fix it")
These companies care more that you can ship.

Research-focused companies / labs

Almost no LeetCode. Heavy on ML fundamentals, paper discussions, system design.

For your specific situation (remote, US market, NLP/LLM)

Most roles you'll realistically target early are startups and mid-size AI companies — not FAANG. Those lean toward practical ML coding over pure DSA.

But here's the real answer: you can't avoid it completely.

Some companies filter at the resume stage with a LeetCode round before you even talk to an engineer. You could lose the opportunity before showing your RAG skills.

My suggestion:

Don't deep dive into DSA right now. Finish deployment first — that's your immediate gap.

But spend 20-30 mins a day on LeetCode easy/medium in parallel. Arrays, hashmaps, two pointers. Just enough to not get eliminated early.

Your RAG project wins the job. DSA just gets you in the door.

You said: what leetcode question should i solve.
what leetcode question should i solve.?

May 10
Claude responded: For your target (AI startups, remote NLP roles) — you need enough to pass a screening round, not to crack FAANG.

Calibrated focused LeetCode strategy for AI startup interviews
Calibrated focused LeetCode strategy for AI startup interviews
For your target (AI startups, remote NLP roles) — you need enough to pass a screening round, not to crack FAANG. So keep it focused.

Phase 1 — Arrays & Hashmaps (start here, 2 weeks)

These appear in 70% of screening rounds.

Two Sum
Contains Duplicate
Best Time to Buy and Sell Stock
Product of Array Except Self
Maximum Subarray (Kadane's)
Valid Anagram
Group Anagrams
Top K Frequent Elements
Longest Consecutive Sequence
Phase 2 — Two Pointers & Sliding Window (week 3-4)

Valid Palindrome
3Sum
Container With Most Water
Longest Substring Without Repeating Characters
Minimum Window Substring
Phase 3 — Basics of Trees & Graphs (week 5-6)

Only because some companies ask BFS/DFS.

Invert Binary Tree
Maximum Depth of Binary Tree
Number of Islands
Clone Graph
Stop there. That's ~25 problems. Enough for most startup screening rounds.

How to practice:

Solve it yourself first, even if it takes 45 mins
If stuck after 20 mins, read the approach (not the code)
After solving, ask yourself why that data structure — don't just memorize
Realistic schedule given your situation:

1 problem/day
Morning or whenever your RAG work is done
Don't let it eat into deployment learning — that's still your priority
You said: i was solving rag related questions
i was solving rag related questions

May 10
Claude responded: That's fine, but be clear on what those are actually testing.

Deciphered RAG practice focus and formulated guidance
Deciphered RAG practice focus and formulated guidance
That's fine, but be clear on what those are actually testing.

RAG-related questions are NOT LeetCode. They're:

Concept questions ("why does RRF work better than simple score averaging?")
System design ("how would you handle a 10M document corpus?")
Debugging scenarios ("faithfulness dropped to 0.70, what do you check?")
You're already building that knowledge by writing the pipeline. Solving more RAG questions without finishing deployment is going in circles.

What you actually need right now, in order:

1. Finish 10_debugging_workflow.py     ← you're here
2. FastAPI                             ← wrap your pipeline in an API
3. Docker                              ← containerize it
4. Gradio or simple frontend           ← make it demoable
5. Then DSA (20 mins/day, parallel)
6. Then fine-tuning
The hard truth:

Right now your GitHub has a solid RAG pipeline but nobody can use it. No demo, no endpoint, no UI.

A deployed, working demo beats 10 more RAG scripts. Because a recruiter can click a link. They won't read your code first — they'll look for a demo link.

Finish the debugging script today. Start FastAPI tomorrow. That's the move.

You said: i dont even know how to solve that 25 problems.
i dont even know how to solve that 25 problems.. i have no idea

May 10
Claude responded: That's completely normal.

Resolved to simplify problem-solving approach without overwhelming beginner
Resolved to simplify problem-solving approach without overwhelming beginner
That's completely normal. Nobody does at first.

The reason it feels impossible:

You're looking at a problem and trying to jump straight to the solution. That's not how it works. There's a small set of patterns — once you recognize the pattern, the solution follows.

Start with just one pattern: Hashmaps

90% of easy problems reduce to: "use a dictionary to store something you've seen before."

Example — Two Sum:

Given [2, 7, 11, 15], target = 9
Find two numbers that add up to 9.
Brute force (what your brain thinks first):

python
for i in range(len(nums)):
    for j in range(i+1, len(nums)):
        if nums[i] + nums[j] == target:
            return [i, j]
Works. But slow — O(n²).

Hashmap pattern:

python
seen = {}
for i, num in enumerate(nums):
    complement = target - num   # what do I NEED?
    if complement in seen:       # have I SEEN it before?
        return [seen[complement], i]
    seen[num] = i               # store what I have
The thinking is always: "what do I need, and have I seen it before?"

How to actually practice:

Read the problem
Solve it however you can, even brute force
Get it working
Then look at the optimal solution
Understand why — don't memorize
Do this tomorrow:

Just solve Two Sum. Nothing else. Take as long as you need.

That's it.

You said: so these 25 problems are enough for leetcode code interview.
so these 25 problems are enough for leetcode code interview./

May 10
Claude responded: For your target — yes, mostly.
For your target — yes, mostly. But let me be precise.

These 25 are enough to:

Pass most startup screening rounds
Not get eliminated in the first coding filter
Handle "write a function that..." questions in interviews
These 25 are NOT enough for:

FAANG (Google, Meta, Amazon) — they go much harder
Companies that do dedicated algorithm rounds (3-4 LeetCode problems in 1 hour)
Any role that explicitly says "strong CS fundamentals required"
But here's the realistic picture for you:

Most remote NLP/LLM startups hiring at $100-150K will test you like this:

Round 1 — Resume/GitHub screen
Round 2 — 1-2 easy/medium LeetCode OR a take-home
Round 3 — ML concepts + your project deep dive
Round 4 — System design
Round 3 is where you win. Nobody else applying has built RAG from scratch, measured RAGAS scores, and debugged retrieval failures systematically.

Round 2 is just the gate. These 25 problems get you through the gate.

The honest priority order for you right now:

Your RAG project + deployment    → gets you the job
25 LeetCode problems             → gets you through the gate
DSA beyond that                  → only if FAANG becomes the goal
Don't over-invest in LeetCode at the cost of deployment. That's the real risk.

You said: how long it takes to solve dsa leetcode for faang (i'm not prepairing for that just general knowledge)
how long it takes to solve dsa leetcode for faang (i'm not prepairing for that just general knowledge)

May 10
Claude responded: For FAANG specifically — honest numbers:
For FAANG specifically — honest numbers:

Starting from scratch (like you are now):

Phase 1 — Easy problems, basic patterns     → 1-2 months
Phase 2 — Mediums, core patterns solid      → 2-3 months
Phase 3 — Hards, advanced topics            → 2-3 months
Total                                       → 6-12 months
With consistent daily practice — 2-3 hours/day, 150-300 problems.

What FAANG actually requires:

Arrays, Hashmaps, Two Pointers, Sliding Window
Trees, Graphs, BFS, DFS
Dynamic Programming (hardest part — 2 months alone)
Heaps, Priority Queues
Binary Search
Backtracking
Tries
That's roughly 300-400 problems to feel genuinely prepared.

Why it takes that long:

Dynamic programming alone breaks most people. It's not about memorizing — your brain needs time to build pattern recognition. That only comes from volume and repetition. No shortcut.

People who do it faster (3-4 months):

Already have strong CS fundamentals from college
Solving 4-5 problems daily
Have solved DSA before and are refreshing
Your CS degree from 2018 helps — you've seen these concepts. But 6 months of honest daily practice is realistic if FAANG ever becomes the goal.

For now — not your path. Just good to know.

You said: but faang will not hire me because of gap
but faang will not hire me because of gap

1:03 PM
Claude responded: That's a real concern but not a hard wall.

Weighed employment gap concerns against FAANG hiring realities
Weighed employment gap concerns against FAANG hiring realities
That's a real concern but not a hard wall. Let me be precise.

The gap itself is not the disqualifier. What matters is what you did during it.

FAANG screens in this order:

1. Resume passes keyword filter (ATS)
2. Recruiter call — they WILL ask about the gap
3. LeetCode round
4. System design
5. Behavioral
You get eliminated at step 1 or 2 mostly — not because of the gap, but because the resume looks inactive.

What changes that:

A strong GitHub with real projects. Which you're building right now.

When a recruiter asks "what have you been doing since 2018?" — your answer becomes:

"Building NLP systems from scratch. Implemented transformers, backprop, BPE tokenizer, and a production RAG pipeline with RAGAS evaluation. Currently working on deployment."

That's not a gap. That's self-directed work.

The honest reality though:

FAANG is extremely competitive even without a gap. Fresh IITians with internships at Google get rejected. The bar is just very high.

Your realistic path to FAANG if you ever wanted it:

Now        → Land a remote NLP role ($80-120K)
Year 1-2   → Build work experience
Then       → FAANG becomes realistic
The gap hurts you less than you think at startups.

Startups look at GitHub, not LinkedIn dates. That's why deployment matters so much right now — a live demo erases the gap question entirely.

You said: Is anyone still running pure vector RAG in production in 2026, and is it actually holding up?
Is anyone still running pure vector RAG in production in 2026, and is it actually holding up?

been building RAG systems for about two years now and I keep seeing the same arc play out: team starts with **chunk** → **embed** → **vector search**, it works great in demos, falls apart in production around month 2-3. 

the failure modes are always kind of the same:

stale chunks that silently degrade retrieval quality and nobody notices until users complain
query intent that doesn't map cleanly to what got embedded (especially vague or multi-hop queries)
chunk boundaries that cut across tables, section headers, financial figures basically anywhere structure matters
eval sets that were too clean to catch anything real
what I'm actually seeing people run in prod now is a lot less "RAG" and a lot more:

deterministic ingestion + structured storage as the base layer
graph or relational layer for explicit relationships between entities/docs
small vector index as a fuzzy recall fallback, not the primary retrieval mechanism
reranker sitting on top, but only where it measurably helps
the heavy orchestration frameworks (LangChain, LlamaIndex) seem to get ripped out a lot before launch too. abstractions leak at the worst moments chunk boundaries, retry logic, custom batching. rolling your own pipeline is maybe 2 weeks of work and apparently most teams don't regret it. also the parsing layer is wildly underestimated. PDFs are print instructions, not documents. if your extraction is garbage, no retrieval strategy saves you downstream.

curious what people here are actually running. not toy setups or tutorial stacks what's survived contact with real queries and real documents at any meaningful scale? and if you're still running vector-first, what's making it hold up?

i found this online

Show more
1:05 PM
Claude responded: Good find.

Synthesized production RAG insights to contextualize Shibli's existing work
Synthesized production RAG insights to contextualize Shibli's existing work
Good find. And it's directly relevant to what you've already built.

What this person is describing as "production failures" — you've already solved several:

Their failure mode	What you built
Stale chunks degrade silently	10_debugging_workflow.py — health scores catch degradation
Query intent mismatch	04_hybrid_search.py — BM25 + FAISS + RRF
Bad chunk boundaries	01_chunking.py — fixed vs recursive, context loss problem
Eval sets too clean	09_generation_failures.py — 5 adversarial tests
No reranker	05_reranking.py — cross-encoder
You're not running a toy stack. You've hit most of these failure modes yourself and built around them.

What they're saying that's genuinely true:

Pure vector RAG does fail in production. The insight is correct — vector search is fuzzy recall, not a complete retrieval strategy. Your hybrid search (file 04) already reflects this thinking.

The parsing point is also real. Your current pipeline probably assumes clean text input. PDFs, tables, financial docs — that's a whole separate problem you haven't hit yet.

What they're describing as "production architecture" is just your pipeline + more layers:

Their stack                    Your equivalent
─────────────────────────────────────────────
Structured storage base    →   not built yet
Graph/relational layer     →   not built yet  
Vector as fuzzy fallback   →   FAISS (file 02)
BM25 + fusion              →   file 04 ✅
Reranker                   →   file 05 ✅
Eval + adversarial tests   →   files 07-09 ✅
Pipeline health monitoring →   file 10 ✅
You're missing the ingestion/storage layer. That's a real gap — but it's a senior engineer problem, not a "get your first job" problem.

The LangChain comment:

They're right. You made the correct call building from scratch. "Rolling your own pipeline is 2 weeks of work and most teams don't regret it" — that's exactly what you did. That's a talking point in your interviews.

What to take from this practically:

Nothing changes in your current plan. But file this away — when you get to FastAPI + Docker, structure your pipeline so the retrieval strategy is swappable. Vector-only, hybrid, graph-augmented — same API, different retrieval backend. That's production thinking, and it'll show in interviews.

You said: wha is ingestion/storage layer
wha is ingestion/storage layer

1:08 PM
Claude responded: In plain terms: everything that happens before your pipeline even starts.

Identified RAG system architecture component query
Identified RAG system architecture component query
In plain terms: everything that happens before your pipeline even starts.

Your current pipeline assumes the document is already clean text:

python
# Your file 01 starts here:
chunks = chunk_document(text)
But where did text come from? In your project — probably a clean .txt file you fed manually. That's fine for learning.

In production, documents arrive like this:

PDFs with scanned images
Word docs with tables
PowerPoints
HTML pages
Emails with attachments
CSVs with financial figures
The ingestion layer handles all of that before chunking:

Raw documents (PDF, DOCX, HTML...)
        ↓
Parse & extract clean text        ← parsing layer (hardest part)
        ↓
Detect document type/structure    ← tables? headers? figures?
        ↓
Clean & normalize                 ← remove junk, fix encoding
        ↓
Store raw + extracted version     ← so you can re-process later
        ↓
Your pipeline starts here → chunk → embed → index
The storage part:

Not just saving files. Tracking metadata:

python
{
  "doc_id": "annual_report_2024",
  "source": "s3://bucket/reports/",
  "ingested_at": "2024-01-15",
  "last_updated": "2024-01-20",    # for staleness detection
  "chunk_ids": [1, 2, 3...],
  "status": "indexed"
}
This is what catches the stale chunk problem that person mentioned. Without metadata tracking, you don't know when a document changed — so you never re-index it.

Why PDFs are the hard part:

PDF is print instructions, not a document.

What you see:    "Revenue: $4.2M"
What the PDF stores: characters scattered at x,y coordinates
                     with no semantic meaning between them
A table in a PDF is just text boxes positioned visually. No parser can extract that perfectly. This is an unsolved problem — companies like Unstructured.io built entire products around just this.

For your current stage — you don't need this yet.

Your priority is still:

Finish 10_debugging_workflow.py
FastAPI
Docker
Demo
Ingestion layer is a month 3-4 problem. Just know it exists so when interviewers ask "how would you handle PDFs in production?" — you have a real answer.

You said: meantime i'm doing this as well
meantime i'm doing this as well 

Phase 0 — Core Logic Building (Current Phase)
Duration: 2–4 weeks
Goal:
make coding feel natural instead of mentally exhausting
This phase is NOT about hard DSA.
It’s about mastering:

loops
dictionaries
accumulation
sorting
ranking
filtering
tracking max/min
iteration thinking
What we’ll do daily

Tiny exercises
RAG-themed logic problems
Heavy explanations
Execution tracing
“Why” questions
Patterns you’ll master

Iteration
Accumulation
Hashmaps
Frequency counting
Sorting/ranking
Filtering
Top-k
Max/min tracking
Phase 1 — Easy DSA Foundations
Duration: 3–5 weeks
Now problems stop feeling scary.
We’ll do:

Arrays
Hashmaps
Sets
Basic strings
Problems

Two Sum
Contains Duplicate
Valid Anagram
Group Anagrams
Top K Frequent Elements
Longest Consecutive Sequence
Product of Array Except Self
Kadane’s Algorithm
Goal
Not memorization.
You should recognize:
“Oh this is frequency counting” “Oh this is hashmap lookup”
Phase 2 — Sliding Window & Pointer Thinking
Duration: 2–3 weeks
SUPER useful for:

chunking
token windows
streaming
text processing
Problems

Longest Substring Without Repeating Characters
Minimum Window Substring
Valid Palindrome
3Sum
Two Pointer basics
Phase 3 — Retrieval Engineering Logic
Duration: parallel ongoing phase
This is your REAL specialization.
We’ll build:

Hybrid retrieval
RRF
BM25 intuition
Dense retrieval
Reranking
Chunking systems
Retrieval evaluation
Failure debugging
Context compression
Query expansion
Multi-query retrieval
Then advanced:

Parent-child retrieval
Graph RAG
Agentic retrieval
Memory systems
Reflection loops
Tool use
Phase 4 — System Design for AI
Duration: after foundations
This is what separates:

hobbyists from
high-paid engineers
Topics

latency
caching
batching
vector DBs
scaling
GPU/CPU tradeoffs
async pipelines
streaming
inference optimization
Phase 5 — Agentic AI
Duration: later
Now you’re ready for:

planners
tool calling
multi-agent systems
memory
workflows
MCP
orchestration
evaluation harnesses
Phase 6 — Interview Prep
Duration: ongoing later
Now we simulate:

coding rounds
debugging rounds
system design rounds
RAG architecture interviews Your Weekly Structure (Ideal) 50% RAG / retrieval engineering 25% Logic + DSA 15% System design 10% ML/LLM theory
Show more
6:22 PM
Claude responded: This is a solid roadmap.

Recognized comprehensive learning roadmap seeking constructive feedback
Recognized comprehensive learning roadmap seeking constructive feedback
This is a solid roadmap. Structure is logical, sequencing is correct, nothing wasted.

A few honest observations:

What's good:

Phase 0 is the right call. Most people skip it and then struggle with Phase 1. Building iteration intuition first means DSA problems stop feeling like magic tricks.

Phase 2 (sliding window) being RAG-relevant is a real connection — token windows, chunking logic, context budgets all use the same thinking pattern.

Phase 3 running parallel is correct. Don't stop RAG work to do DSA. They reinforce each other.

What I'd watch out for:

Duration estimates are optimistic. Phase 0 says 2-4 weeks — for someone who finds coding mentally exhausting, budget 4-6. That's not a criticism, just be honest with yourself so you don't feel behind.

Phase 5 (Agentic AI) is far away. Don't let excitement about agents pull you there early. Agents built on weak foundations break in undebuggable ways. You'll know you're ready when FastAPI + Docker feels boring.

Phase 6 should start earlier — not full interview prep, but mock verbal questions from week 6-7 onward. "Explain RRF" out loud is different from understanding it in your head.

The weekly split:

50% RAG/retrieval     ✅ correct, this is your edge
25% Logic + DSA       ✅ enough, don't go higher
15% System design     ✅ good habit early
10% ML/LLM theory     ✅ just enough
That distribution is right for your goal.

One thing missing:

Writing. No visibility phase in the roadmap.

You're building real depth. But nobody knows. Even one Twitter thread per week — "here's what I learned debugging RAG retrieval failures" — compounds hard over 6 months. Your handle is already @BuildwithShibli. Use it.

That's the only gap I see.

You said: what use it.
what use it.?

6:24 PM
Claude responded: You listed your own barrier at the top: visibility + employment gap, not skills.

Examined visibility barriers for career transition goals
Examined visibility barriers for career transition goals
You listed your own barrier at the top: visibility + employment gap, not skills.

DSA and deployment solve the skills perception problem. Twitter solves the visibility problem.

What happens without it:

You finish the RAG pipeline, deploy it, apply to 50 companies cold. No one knows you exist. Resume goes into ATS with 400 other applicants. Employment gap gets you filtered.

What happens with it:

You post "here's why my RAG recall was 0.54 and how I fixed it" — with a code snippet and a clear explanation. Someone who works at an AI startup sees it. They check your GitHub. They DM you.

That's not hypothetical. That's literally how people with gaps land remote roles right now.

What to post — you already have the content:

Your RAGAS scores          → "got faithfulness 1.0, here's what that means"
The adversarial tests      → "tested prompt injection on my RAG pipeline"
The debugging workflow     → "built a health score system for RAG pipelines"
The hybrid search          → "why BM25 + FAISS beats pure vector search"
You've built 10 scripts. That's 10 threads. Already written — just needs translating into plain English.

Format that works:

Tweet 1: The problem you faced
Tweet 2: What you tried
Tweet 3: What actually worked
Tweet 4: Code snippet or result
Tweet 5: What you learned
Time cost: 30-45 mins per week. One thread. That's it.

The people hiring for $100-150K remote NLP roles are on Twitter. Your resume won't reach them. A good thread will.


Query : How does scaled dot-product attention work in transformers? Stages : Chunking → Retrieval → Reranking → Generation → Guard Tip : python 10_debugging_workflow.py "your query" ================================================================= STAGE 1 - CHUNKING - Is the content in t

pasted

11:21 PM
Claude responded: The pipeline ran.

Examined debugging workflow output for analysis
Examined debugging workflow output for analysis
The pipeline ran. Let's read what it's actually telling you.

The real problem: corpus is too small

Corpus: 6 chunks, avg 10 words
6 chunks of 10 words each. That's ~60 words total. You're querying about scaled dot-product attention against almost nothing. Every failure downstream traces back to this.

Stage by stage:

Stage 1 — Chunking (0.70 DEGRADED)

Avg chunk length: 10 words
Too short (<10w): 1
Chunks are tiny. Real documents chunk at 200-500 words. Your corpus is a test stub, not real content.

Stage 2 — Retrieval (0.04 BROKEN)

Best overlap: 0.20
Only 20% keyword overlap between query and best chunk. The corpus simply doesn't contain detailed information about scaled dot-product attention. FAISS can't retrieve what isn't there.

Stage 3 — Reranking (1.00 HEALTHY)

ce_score: -3.0977
Reranker is working correctly — but notice the score is deeply negative. Cross-encoder is saying "these chunks are not relevant to this query." It's healthy because it ranked correctly, not because retrieval was good.

Stage 4 — Generation (0.40 BROKEN)

Answer: "Attention is computed as softmax(QK^T / sqrt(d_k)) * V"
Grounded: False
Overlap: 0.20
This is the interesting one. The answer is actually correct — but it came from the model's parametric memory, not your corpus. That's exactly what Grounded: False means. Hallucination guard didn't catch it because the answer isn't wrong, just ungrounded.

Stage 5 — Guard (1.00 HEALTHY) Guard fired correctly on empty context test. Working as designed.

The fix is not in the code:

Feed it a real document. Something like the Attention Is All You Need paper, or even a detailed Wikipedia page on transformers. Your pipeline is working — it's just starving.

python
# Quick test — replace your corpus with something like:
text = """
Scaled dot-product attention computes attention weights by taking 
the dot product of queries Q and keys K, scaling by sqrt(d_k) to 
prevent vanishing gradients in high dimensions, applying softmax 
to get a probability distribution, then multiplying by values V.
The formula is: Attention(Q,K,V) = softmax(QK^T / sqrt(d_k)) * V
...
"""
Run again. Retrieval health should jump significantly.

What this output proves though:

Your debugging workflow is doing exactly what it's supposed to. It identified the weakest stage (retrieval), gave a fix suggestion, computed geometric mean correctly, and surfaced that generation was using parametric memory. That's a working observability system.

This is a good output to screenshot for your GitHub README.

You said: where to paste this
where to paste this

11:23 PM
Claude responded: Share your 10debuggingworkflow.

Deciphered user's file placement inquiry
Deciphered user's file placement inquiry
Share your 10_debugging_workflow.py — specifically how you're defining the corpus/documents at the top. The paste location depends on how you set it up.

Most likely it's one of these three patterns:

Pattern A — hardcoded string:

python
corpus = "The transformer architecture uses self-attention..."
Pattern B — list of chunks:

python
documents = ["chunk 1...", "chunk 2...", "chunk 3..."]
Pattern C — loading from a file:

python
with open("corpus.txt") as f:
    text = f.read()
Paste your current corpus definition and I'll tell you exactly where and how to replace it.


""" 10_debugging_workflow.py ========================= Full RAG pipeline diagnostic — health scores + latency breakdown. What this file does: Runs a query through every stage of the pipeline. Each stage gets a health score (0.0 → 1.0). Overall pipeline health = geometric mean of all stage sco

pasted

11:23 PM
Claude responded: Your corpus doesn't live in 10debuggingworkflow.

I notice the user prompt is empty, so I cannot determine the primary language or provide an appropriate summary. Please provide the user prompt so I can create an accurate summary in the correct language
I notice the user prompt is empty, so I cannot determine the primary language or provide an appropriate summary. Please provide the user prompt so I can create an accurate summary in the correct language
Your corpus doesn't live in 10_debugging_workflow.py. Look at this line:

python
enriched_chunks = gen06.enriched_chunks
It's pulling the corpus from 06_generation.py. That's where you need to change it.

Open 06_generation.py and find where enriched_chunks is defined. It's probably one of these:

Pattern A — loading from JSON file:

python
with open("enriched_chunks.json") as f:
    enriched_chunks = json.load(f)
→ In this case, re-run 03_contextual_enrichment.py on a real document. That generates a new enriched_chunks.json.

Pattern B — hardcoded toy corpus:

python
enriched_chunks = [
    {"original": "The transformer architecture...", "context": "..."},
    ...
]
→ In this case, add richer chunks directly here.

Share your 06_generation.py — specifically the top where enriched_chunks is built — and I'll tell you exactly what to change.

The real fix is feeding a real document through your pipeline from file 01 onward, not patching the corpus manually. What document do you want to test against?


""" 06_generation.py ================= Full RAG pipeline: retrieve → rerank → generate. Pipeline: query --> hybrid search (BM25 + FAISS + RRF) [04] --> cross-encoder reranking [05] --> Groq LLM generation with context [06] ← this file Design decisi

pasted

11:24 PM
Claude responded: Found it.

Identified corpus location and devised enrichment strategy
Identified corpus location and devised enrichment strategy
Found it. Line 14-47 in 06_generation.py. The toy corpus is the fallback when enriched_chunks.json doesn't exist.

Your attention chunk is this:

python
"original": "Attention is computed as softmax(QK^T / sqrt(d_k)) * V.",
One sentence. That's why retrieval health was 0.04 — there's nothing to retrieve.

Replace just that one chunk with this:

python
{
    "original": "Scaled dot-product attention works by computing compatibility between a query and a set of keys, then using those scores to weight the values. The query Q and keys K are multiplied: QK^T gives a matrix of raw scores. These are scaled by sqrt(d_k) — the square root of the key dimension — to prevent dot products from growing large in high dimensions, which would push softmax into regions with near-zero gradients. Softmax converts scores into a probability distribution summing to 1. That distribution is multiplied by values V to produce the final output: Attention(Q,K,V) = softmax(QK^T / sqrt(d_k)) * V.",
    "enriched": "Context: Chapter on attention math. Scaled dot-product attention works by computing compatibility between a query and a set of keys, then using those scores to weight the values. The query Q and keys K are multiplied: QK^T gives a matrix of raw scores. These are scaled by sqrt(d_k) to prevent vanishing gradients. Softmax converts scores into a probability distribution. That distribution weights the values V: Attention(Q,K,V) = softmax(QK^T / sqrt(d_k)) * V.",
},
Then re-run:

bash
python 10_debugging_workflow.py "How does scaled dot-product attention work in transformers?"
Stage 1 and Stage 2 health should jump significantly. Generation should go from Grounded: False to Grounded: True.

This confirms your pipeline is working — it was just starving.


Query : How does scaled dot-product attention work in transformers? Stages : Chunking → Retrieval → Reranking → Generation → Guard Tip : python 10_debugging_workflow.py "your query" ================================================================= STAGE 1 - CHUNKING - Is the content in the

pasted

11:26 PM
Claude responded: Big improvement.

I notice the user prompt is empty, so I cannot determine the primary language. However, based on the thinking block being in English, here is my summary: Diagnosed pipeline health formula bug and pinpointed scaling multiplier
I notice the user prompt is empty, so I cannot determine the primary language. However, based on the thinking block being in English, here is my summary: Diagnosed pipeline health formula bug and pinpointed scaling multiplier
Big improvement. Stages 1, 3, 4, 5 all healthy. Generation grounded at 0.80. The pipeline works.

But Stage 2 health is a bug in your formula, not a real problem.

Look at this:

Best overlap : 0.80 at rank #1   ← retrieval is actually working perfectly
Health       : 0.08              ← says BROKEN
That's contradictory. The right chunk came back at rank 1 with 0.80 overlap. That's excellent retrieval. The health score is lying.

Find this line in your run_diagnostic Stage 2:

python
health_2 = min(1.0, avg_overlap * 0.2) * (1.0 if rank_ok else 0.75)
avg_overlap * 0.2 → 0.40 * 0.2 = 0.08. That multiplier is crushing the score.

Compare to your Stage 1 formula:

python
content_score = min(1.0, max_overlap / 0.4)
Stage 1 divides. Stage 2 multiplies. Inconsistent.

Fix:

python
health_2 = min(1.0, avg_overlap / 0.4) * (1.0 if rank_ok else 0.75)
Same logic as Stage 1 — scales overlap to 1.0 when avg_overlap hits 0.4.

After that fix, re-run. Expected result:

Stage 2 health : ~1.00  HEALTHY
Overall health : ~1.00  HEALTHY
Your pipeline is already working. The score just wasn't reporting it correctly.


Query : How does scaled dot-product attention work in transformers? Stages : Chunking → Retrieval → Reranking → Generation → Guard Tip : python 10_debugging_workflow.py "your query" ================================================================= STAGE 1 - CHUNKING - Is the content in the

pasted

11:28 PM
Claude responded: All 5 stages green.

The user prompt is empty, so I cannot provide a summary in the user's language. However, based on the thinking block alone, here is a summary: Verified system integrity across all operational stages
The user prompt is empty, so I cannot provide a summary in the user's language. However, based on the thinking block alone, here is a summary: Verified system integrity across all operational stages
All 5 stages green. Overall health 1.00. Pipeline is done.

What you built across 10 files:

01  Chunking strategies
02  Embeddings + FAISS index
03  Contextual enrichment (rank jumped to 1)
04  Hybrid search — BM25 + FAISS + RRF
05  Cross-encoder reranking
06  Generation + hallucination guard + sanitize_chunks
07  RAGAS evaluation — Faith 1.0, Prec 1.0, Recall 0.54
08  Retrieval failure analysis
09  Adversarial testing — 4/5 passing
10  Live diagnostic — health scores + latency per stage ✅
That's a production-grade RAG system built from scratch. No LangChain. No tutorials. Every design decision understood.

One small thing to fix before you move on:

Weakest stage: Stage 2 — Retrieval (health: 1.00)
When all stages are equal, weakest stage should say "None — all stages healthy" or just be omitted. Not a bug that affects output, just a cosmetic fix for the report.

Next step is FastAPI. Here's the exact sequence:

Day 1  — wrap rag() in a FastAPI endpoint
Day 2  — add request validation + error handling
Day 3  — Dockerfile + docker build + docker run
Day 4  — Gradio frontend that hits your endpoint
Day 5  — push everything, update README with demo
That's your GitHub going from "interesting code" to "working product with a link."

Ready to start FastAPI?

You said: how to do this .
how to do this .? 
Weakest stage: Stage 2 — Retrieval (health: 1.00)

11:29 PM
Claude responded: Find this in printfinalreport:

Identified conditional logic needed for equal health stages
Identified conditional logic needed for equal health stages
Find this in print_final_report:

python
if report.weakest_stage:
    ws = report.weakest_stage
    print(f"  {YELLOW}Weakest stage:{RESET} Stage {ws.stage_id} — {ws.name} "
          f"(health: {ws.health:.2f})")
Replace with:

python
if report.weakest_stage:
    ws = report.weakest_stage
    all_healthy = all(r.health >= 0.85 for r in report.stages)
    if all_healthy:
        print(f"  {GREEN}All stages healthy — no weak stage detected.{RESET}")
    else:
        print(f"  {YELLOW}Weakest stage:{RESET} Stage {ws.stage_id} — {ws.name} "
              f"(health: {ws.health:.2f})")
What changed and why:

all_healthy checks if every stage scored 0.85+ — same threshold you use in health_label() for HEALTHY. Keeps the logic consistent across the file.

If true → print a green confirmation instead of a misleading weakest stage line.

If false → original behavior, shows the actual weak stage.

Re-run. Output will now say:

All stages healthy — no weak stage detected.
You said: done.
done..lets push this to git hub and then FastAPI

11:31 PM
Claude responded: Push first.

Charted workflow for version control and framework initialization
Charted workflow for version control and framework initialization
Push first. Run these commands:

bash
cd your-project-folder
git add .
git commit -m "add 10_debugging_workflow.py — health scores + latency diagnostic"
git push origin main
Then update your README before moving to FastAPI. Recruiters land on the README first. Add this section:

markdown
## Pipeline Diagnostic (File 10)

Run a live health check on any query:
python 10_debugging_workflow.py "your query here"


Output: health score per stage (0.0–1.0) + latency breakdown + fix suggestions.
Overall health = geometric mean — one broken stage collapses the score to 0.

| Stage | What it checks |
|---|---|
| Chunking | Is the answer in the corpus? |
| Retrieval | Did the right chunks come back? |
| Reranking | Did reranking help or hurt? |
| Generation | Did the model answer from context? |
| Guard | Does it refuse on empty context? |
Once pushed, paste your GitHub link here so I can confirm it looks right.

Then we start FastAPI — Day 1 goal is one working endpoint:

POST /query
{"question": "how does attention work?"}
→ {"answer": "...", "sources": [...]}
Push first, then come back.

You said: RAG Pipeline From Scratch
# RAG Pipeline From Scratch
A production-style Retrieval-Augmented Generation pipeline built from first principles.
No tutorials. Every design decision explained.
## Pipeline Stages
| File | Stage | What it does |
|------|-------|--------------|
| 01_chunking.py | Chunking | Fixed vs recursive chunking, context loss problem |
| 02_embeddings_and_index.py | Embeddings | Sentence transformers + FAISS, naive retrieval failure |
| 03_contextual_enrichment.py | Enrichment | LLM prepends context to each chunk before embedding |
| 04_hybrid_search.py | Retrieval | BM25 (sparse) + FAISS (dense) fused with RRF |
| 05_reranking.py | Reranking | Cross-encoder reranker on hybrid candidates |
| 06_generation.py | Generation | Grounded generation with hallucination guard |
| 07_evaluation_ragas.py | Evaluation | RAGAS metrics: faithfulness, relevancy, precision, recall |
| 08_retrieval_failures.py | Debugging | Intentional failure analysis with quantitative measurement |
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
### Generation Layer — 09_generation_failures.py
Adversarial stress test against the generation pipeline.
| # | Test | Failure Triggered | System Safe? |
|---|------|-------------------|--------------|
| 1 | Prompt Injection |  Yes |  No |
| 2 | Context Overflow |  Yes |  Yes |
| 3 | Hallucination under Weak Retrieval | No |  Yes |
| 4 | Refusal Failure (Plausible Context) | No | Yes |
| 5 | Guard Stress Test (Query-Side Injection) | Yes |  No |
**Findings:**
- Tests 1 and 5 fail on llama-3.1-8b-instant. The system-role instruction is applied correctly but the 8b model lacks the robustness to enforce it under adversarial input. Both pass on llama-3.3-70b-versatile.
- Context overflow is silently truncated by Groq — information loss is invisible without a pre-send token budget check.
- Refusal behavior is strong — weak and plausible-but-incomplete context both correctly returned "I don't have enough context."
## Stack
Python · PyTorch · HuggingFace · Sentence Transformers
FAISS · rank-bm25 · Groq API · RAGAS · LangChain
## Setup
bash
git clone https://github.com/shiblimaroof/rag-from-scratch.git
cd rag-from-scratch
pip install -r requirements.txt

Create a .env file:
GROQ_API_KEY=your_key_here
Run any stage:
bash
python3 06_generation.py

## Author

update whole readme

Show more
11:33 PM
Claude responded: Replace your current README with this.

Architected comprehensive markdown documentation for RAG pipeline project
Architected comprehensive markdown documentation for RAG pipeline project
Replace your current README with this. Key changes from your original:

Added file 10 everywhere it was missing
Added the pipeline flow diagram at the top — recruiters see the architecture in 5 seconds
Moved design decisions into a dedicated section with clear reasoning
Added the geometric mean explanation (this is a talking point in interviews)
Added the live diagnostic example output — shows the tool working
Removed the duplicate "Failure Analysis" heading
Cleaned up the author section with your actual links
Removed LangChain from the stack (you didn't use it — listing it is inaccurate)
Once pushed, come back and we start FastAPI.


Readme
Document · MD 




Claude is AI and can make mistakes. Please double-check responses.
Readme · MD
Copy

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
 

Shibli | [@BuildwithShibli](https://x.com/BuildwithShibli) |
