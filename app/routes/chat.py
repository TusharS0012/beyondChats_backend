from fastapi import APIRouter
from pydantic import BaseModel
from app.services.llm_service import generate_answer

router = APIRouter()

# Use the in-memory DOCUMENTS from upload.py or replace with DB
from app.routes.upload import DOCUMENTS

class ChatRequest(BaseModel):
    question: str

class ChatResponse(BaseModel):
    answer: str
    context: list

@router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    if not DOCUMENTS:
        return {"answer": "No documents found", "context": []}

    # Use top 3 chunks as context (placeholder logic)
    context_chunks = DOCUMENTS[:3]
    context_text = "\n\n".join([c["text"] for c in context_chunks])

    answer = generate_answer(context_text, req.question)
    return {"answer": answer, "context": context_chunks}
