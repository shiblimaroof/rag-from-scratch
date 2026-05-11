import gradio as gr
import requests

API_URL = "http://localhost:8000/query"

def ask(question):
    if not question.strip():
        return "Please enter a question.", ""
    
    response = requests.post(API_URL, json={"question": question})
    
    if response.status_code != 200:
        return f"API error: {response.status_code}", ""
    
    data = response.json()
    answer = data["answer"]
    sources = "\n\n".join(
        f"[{i+1}] {src}" for i, src in enumerate(data["sources"])
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