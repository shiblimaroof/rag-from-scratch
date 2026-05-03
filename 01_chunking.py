#!/usr/bin/env python
# coding: utf-8

# In[1]:


from langchain_text_splitters import RecursiveCharacterTextSplitter
import json


# In[12]:


text = """
Paul Graham's essay: What I Worked On

What I worked on before college was writing and programming. I didn't write essays. I wrote what beginning writers were supposed to write then, fiction. But I was terrible at fiction. My stories had hardly any plot, just characters with strong feelings, which I imagined made them deep.

The first programs I tried writing were on the IBM 1401 that our school district used for what was then called "data processing." This was in 9th grade. The school district's 1401 had 4k of memory, in the form of what were called core memory.

I wrote simple games, a program to predict how high model rockets would fly, and a word processor that my father used to write at least one book. It was terrifying using the word processor, because the program took up most of the memory, leaving only a small region to store the document. 

When I was in high school I spent a lot of time working on a novel. It was terrible. I wrote it using a word processor I built myself on an Apple II. I thought of programming as a tool for writing, not an end in itself.

College changed all that. I went to Cornell intending to study philosophy, which seemed to be the most interesting thing one could study. But I found that philosophy courses were boring. The answers to the questions they raised were always a letdown.

So I switched to AI. My logic was that if intelligence can be programmed, then that's the most important thing to work on. And at the time the main approach seemed promising: rule-based systems. I spent a lot of time on this, but ultimately concluded it was a dead end.

I applied to art school after college. I got into RISD and the Accademia di Belle Arti in Florence. I went to RISD. It was a great experience and I made good friends there. The foundation year was hard but illuminating.

After RISD I moved to New York. I had no idea what to do with my life. I started writing essays and realized I was much better at that than fiction. I also wrote software for a startup, but the startup didn't work out.

Then I started Viaweb with Robert Morris. We were building a web application for making web stores. We got acquired by Yahoo in 1998. That was a big relief, because we were running out of money.

After Yahoo I worked on painting for a while. Then I wrote some essays. Then I started Y Combinator with Jessica Livingston, Robert Morris, and Trevor Blackwell. Y Combinator has now funded over 3000 startups including Airbnb, Stripe, Dropbox, and Reddit.

The most important thing I've learned is that you have to work on things that matter to you. Not things that matter to other people, or things that look impressive on a resume, but things that you genuinely care about. Otherwise you'll never do your best work.

The second most important thing is that it's better to make something people want than to make something impressive. A lot of startup founders confuse these two. They make something impressive that nobody wants.

The third most important thing is that you should try to be around smart people. Not just to learn from them, but because the conversations you have with them will shape the problems you think about.

Working on hard problems is more fun than working on easy ones. Not because they're harder, but because the solutions are more surprising and interesting.

I've also learned that most of what people believe about business is wrong. They believe you need a business plan, a management team, and lots of capital. What you actually need is good people, a product people want, and as little money as possible.

Another thing I've learned is that it's important to have the right relationship with money. Money is a means to an end, not an end in itself. If you start treating money as an end in itself, you'll make bad decisions.

One of the most important things I've learned about writing is that you should write the way you talk. Not literally, but the same level of formality. Most people write too formally, and the result is stiff and boring.

The same principle applies to code. The best code is the simplest code that does the job. Not the cleverest code, not the most sophisticated code, but the simplest. Simple code is easier to read, easier to debug, and easier to extend.

I think the most valuable thing I can offer anyone is this: work on things that matter, be around smart people, and don't worry too much about money. If you do those three things, the rest will take care of itself.
"""

print(f"Document Length:{len(text)} characters \n")

# --- Strategy 1: Fixed-size chunking ---

def fixed_chunks(text, size = 500, overlap = 50):
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        start += size - overlap
    return chunks

fixed = fixed_chunks(text)
print(f"Fixed chunking → {len(fixed)} chunks")
print(f"Sample chunk:\n{fixed[5]}\n")
print("-" * 60)

# --- Strategy 2: Recursive character splitting ---

splitter = RecursiveCharacterTextSplitter(
    chunk_size = 500,
    chunk_overlap = 50,
    separators= ["\n\n", "\n", ".", " ", ""]
)

recursive = splitter.split_text(text)
print(f"Recursive chunking -> {len(recursive)} chunks")
print(f"Sample chunk :\{recursive[5]}\n")
print('-' * 60)

print("PROBLEM DEMO:")
print("This chunk makes no sense without surrounding context:")
print(recursive[10])


# In[11]:


# --- Strategy 2: Recursive character splitting ---

splitter = RecursiveCharacterTextSplitter(
    chunk_size = 500,
    chunk_overlap = 50,
    separators= ["\n\n", "\n", ".", " ", ""]
)

recursive = splitter.split_text(text)
print(f"Recursive chunking -> {len(recursive)} chunks")
print(f"Sample chunk :\{recursive[5]}\n")
print('-' * 60)

print("PROBLEM DEMO:")
print("This chunk makes no sense without surrounding context:")
print(recursive[10])


# In[ ]:




