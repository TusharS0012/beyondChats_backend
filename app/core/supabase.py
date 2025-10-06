import os
from dotenv import load_dotenv
from supabase import create_client, Client

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)  # type: ignore

def save_quiz_attempt(user_id: str, quiz_id: str, score: float, answers: dict):
    """
    Save a user's quiz attempt to Supabase
    """
    data = {
        "user_id": user_id,
        "quiz_id": quiz_id,
        "score": score,
        "answers": answers
    }
    response = supabase.from_("quiz_attempts").insert(data).execute()  # type: ignore
    return response.data

def save_progress(user_id: str, quiz_id: str, score: float):
    """
    Save user progress (can be used for dashboard)
    """
    data = {
        "user_id": user_id,
        "quiz_id": quiz_id,
        "score": score
    }
    response = supabase.from_("progress").insert(data).execute()  # type: ignore
    return response.data

def get_user_progress(user_id: str):
    """
    Retrieve all progress entries for a user
    """
    response = supabase.from_("progress").select("*").eq("user_id", user_id).execute()  # type: ignore
    return response.data
