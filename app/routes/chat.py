from fastapi import APIRouter
from pydantic import BaseModel
from typing import List, Optional
from app.routes.pdf import PDF_STORE
from app.core.embeddings import search_embedding
from app.core.llm import answer_chat_query

router = APIRouter()

class ChatRequest(BaseModel):
    user_id: str
    message: str
    pdf_name: Optional[str] = None  # Optional, if context from a PDF is needed

@router.post("/")
def chat_endpoint(request: ChatRequest):
    if request.pdf_name and request.pdf_name in PDF_STORE:
        pdf_data = PDF_STORE[request.pdf_name]
        chunks = pdf_data["chunks"]
        embeddings = pdf_data["embeddings"]
        context_snippets = search_embedding(request.message, embeddings, chunks)
    else:
        context_snippets = []

    reply = answer_chat_query(request.message, context_snippets)
    return {"reply": reply, "context": context_snippets}
