import gradio as gr
import os
from generation import rag

def ask(question):
    if not question.strip():
        return "Please enter a question.", ""
    result = rag(question)
    answer = result["answer"]
    sources = "\n\n".join(
        f"[{i+1}] {src}" for i, src in enumerate(result["sources"])
    )
    return answer, sources

demo = gr.Interface(
    fn=ask,
    inputs=gr.Textbox(label="Question", placeholder="Ask something..."),
    outputs=[
        gr.Textbox(label="Answer"),
        gr.Textbox(label="Sources")
    ],
    title="RAG Pipeline",
    description="Built from scratch. No LangChain.",
)

demo.launch()