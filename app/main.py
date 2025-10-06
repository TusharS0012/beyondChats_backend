from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routes import pdf, quiz, chat, progress, videos

app = FastAPI(
    title="BeyondChats - Student Revision Backend",
    description="Backend for PDF-based quizzes, progress tracking, LLM chat, and video recommendations",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(pdf.router, prefix="/pdf", tags=["PDF"])
app.include_router(quiz.router, prefix="/quiz", tags=["Quiz"])
app.include_router(chat.router, prefix="/chat", tags=["Chat"])
app.include_router(progress.router, prefix="/progress", tags=["Progress"])
app.include_router(videos.router, prefix="/videos", tags=["Videos"])

@app.get("/")
def root():
    return {"message": "BeyondChats backend is running!"}
