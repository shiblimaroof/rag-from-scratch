from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from generation import rag

app = FastAPI(title="RAG Pipeline API")

class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str
    sources: list[str]

@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    result = rag(request.question)
    return QueryResponse(
        answer=result["answer"],
        sources=result["sources"]
    )

@app.get("/health")
def health():
    return {"status": "ok"}