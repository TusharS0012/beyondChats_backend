from fastapi import APIRouter
from pydantic import BaseModel
from typing import Dict, Optional
from app.routes.pdf import PDF_STORE
from app.core.llm import generate_quiz_from_text
from app.core.supabase import save_quiz_attempt

router = APIRouter()

class QuizSubmitRequest(BaseModel):
    user_id: str
    quiz_id: str
    answers: Dict[str, str]
    score: Optional[float] = None

@router.post("/generate/{pdf_name}")
def generate_quiz(pdf_name: str):
    if pdf_name not in PDF_STORE:
        return {"error": "PDF not found"}

    text = PDF_STORE[pdf_name]["text"]
    quiz = generate_quiz_from_text(text)
    return {"quiz": quiz}

@router.post("/submit")
def submit_quiz(data: QuizSubmitRequest):
    score = data.score if data.score is not None else 0.0
    saved = save_quiz_attempt(data.user_id, data.quiz_id, score, data.answers)

    return {
        "message": f"Quiz {data.quiz_id} submitted by user {data.user_id}",
        "score": score,
        "saved_attempt": saved
    }
