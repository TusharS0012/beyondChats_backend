from fastapi import APIRouter
from pydantic import BaseModel
from typing import Dict, Any
from app.core.supabase import save_quiz_attempt, get_user_progress
import statistics

router = APIRouter()

class ProgressSubmitRequest(BaseModel):
    user_id: str
    quiz_id: str
    score: float
    answers: Dict[str, str]

class ProgressRequest(BaseModel):
    user_id: str

@router.post("/submit")
def submit_progress(data: ProgressSubmitRequest):
    """
    Save a user's quiz attempt to Supabase
    """
    result = save_quiz_attempt(data.user_id, data.quiz_id, data.score, data.answers)
    return {"message": "Quiz attempt saved", "data": result}

@router.post("/get")
def get_progress(data: ProgressRequest) -> Dict[str, Any]:
    """
    Retrieve all quiz attempts and basic stats for dashboard
    """
    attempts = get_user_progress(data.user_id)
    
    if not attempts:
        return {"progress": [], "stats": {}}

    scores = [a.get("score", 0.0) for a in attempts]
    stats = {
        "total_quizzes": len(attempts),
        "average_score": round(statistics.mean(scores), 2),
        "max_score": max(scores),
        "min_score": min(scores)
    }

    # Optionally, you can compute per-topic strengths/weaknesses if your quiz has topic metadata
    return {"progress": attempts, "stats": stats}
