from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import fitz  # PyMuPDF
import os
import numpy as np
import json
from supabase import create_client, Client
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
import yt_dlp
from uuid import UUID, uuid4

# ---------------- Load environment ----------------
load_dotenv()
HF_API_TOKEN = os.getenv("HF_API_KEY")  # Hugging Face token
LLM_MODEL = "meta-llama/Llama-3.1-8B-Instruct"

# ---------------- FastAPI setup ----------------
app = FastAPI()
origins = ["http://localhost:3000", "http://127.0.0.1:3000", "http://localhost:5173"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- Supabase setup ----------------
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise RuntimeError("SUPABASE_URL and SUPABASE_KEY must be set in .env")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# ---------------- Hugging Face client ----------------
if not HF_API_TOKEN:
    raise RuntimeError("HF_API_KEY must be set in .env")
hf_client = InferenceClient(api_key=HF_API_TOKEN)
print("✅ HuggingFace InferenceClient initialized")

# ---------------- Models ----------------
class ChatRequest(BaseModel):
    question: str

class QuizRequest(BaseModel):
    user_id: str

class QuizAttemptRequest(BaseModel):
    user_id: str
    quiz_ids: List[str]
    answers: List[str]

# ---------------- Helpers ----------------
def extract_text_from_pdf(file_path: str) -> str:
    doc = fitz.open(file_path)
    text = ""
    for page in doc:
        text += page.get_text("text") + "\n\n"
    doc.close()
    return text

def chunk_text(text: str, chunk_size: int = 1000) -> List[dict]:
    chunks = []
    pages = text.split("\f")
    for idx, page_text in enumerate(pages):
        start = 0
        while start < len(page_text):
            chunk = page_text[start:start+chunk_size]
            if chunk.strip():
                chunks.append({"text": chunk, "page": idx + 1})
            start += chunk_size
    return chunks

def cosine_similarity(a: List[float], b: List[float]) -> float:
    a_arr = np.array(a)
    b_arr = np.array(b)
    return float(np.dot(a_arr, b_arr) / (np.linalg.norm(a_arr) * np.linalg.norm(b_arr)))

def get_embedding(text: str) -> List[float]:
    """Generate embedding using HF API"""
    response = hf_client.embeddings.create(model="sentence-transformers/all-MiniLM-L6-v2", input=text)
    return response.data[0].embedding

def generate_hf_answer(context: str, question: str) -> str:
    """Generate answer using Hugging Face chat model"""
    try:
        completion = hf_client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful assistant answering questions based on document context."},
                {"role": "user", "content": f"Context: {context}\n\nQuestion: {question}"}
            ],
            max_tokens=500,
            temperature=0.7,
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        return f"⚠️ LLM generation failed: {str(e)}"

def search_youtube(query, max_results=3):
    ydl_opts = {'quiet': True, 'skip_download': True, 'extract_flat': 'in_playlist', 'default_search': f'ytsearch{max_results}'}
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            results = ydl.extract_info(query, download=False)
            videos = [{"title": e.get("title"), "url": e.get("url"), "channel": e.get("uploader"), "description": e.get("description") or ""} for e in results.get("entries", [])]
            return videos
    except Exception as e:
        print("⚠️ YouTube search failed:", e)
        return []

# ---------------- Endpoints ----------------

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...), user_id: str = Form(...)):
    try:
        UUID(user_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid user_id. Must be a UUID")

    temp_path = f"temp_{file.filename}"
    try:
        with open(temp_path, "wb") as f:
            f.write(await file.read())

        text = extract_text_from_pdf(temp_path)
        chunks = chunk_text(text)
        document_id = str(uuid4())
        chunk_data = []
        for chunk in chunks:
            emb = get_embedding(chunk["text"])
            chunk_data.append({
                "document_id": document_id,
                "text_chunk": chunk["text"],
                "embedding": json.dumps(emb),
                "page_number": chunk["page"]
            })
        if chunk_data:
            supabase.table("document_chunks").insert(chunk_data).execute()
        return {"success": True, "documentId": document_id, "chunksProcessed": len(chunk_data)}
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.post("/chat")
async def chat(request: ChatRequest):
    result = supabase.table("document_chunks").select("*").execute()
    if not result.data:
        raise HTTPException(status_code=404, detail="No document chunks found")
    question_emb = get_embedding(request.question)
    scored = []
    for row in result.data:
        emb = json.loads(row["embedding"])
        score = cosine_similarity(question_emb, emb)
        scored.append({"text": row["text_chunk"], "page": row["page_number"], "score": score})
    top_chunks = sorted(scored, key=lambda x: x["score"], reverse=True)[:3]
    context = "\n\n".join([f"[Page {c['page']}]\n{c['text']}" for c in top_chunks])
    answer = generate_hf_answer(context, request.question)
    related_videos = search_youtube(request.question) if any(word in request.question.lower() for word in ["video","watch","recommend"]) else []
    return {"answer": answer, "context": top_chunks, "videos": related_videos}

@app.post("/generate_quiz")
def generate_quiz(request: QuizRequest):
    try:
        UUID(request.user_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid user_id")
    chunks_response = supabase.table("document_chunks").select("*").execute()
    if not chunks_response.data:
        raise HTTPException(status_code=404, detail="No chunks found")
    quiz = []
    for chunk in chunks_response.data[:5]:
        question_text = chunk["text_chunk"][:100].replace("\n"," ").strip()
        quiz.append({"id": str(uuid4()), "question": f"Based on this passage: {question_text} ... What is it about?", "answer": question_text})
    return {"quiz": quiz}

@app.post("/submit_quiz")
async def submit_quiz(request: QuizAttemptRequest):
    return {"success": True, "score": 100}  # Simplified for demo

@app.get("/health")
async def health():
    return {"status":"healthy", "llm_model": LLM_MODEL, "hf_client_active": hf_client is not None}

@app.get("/")
async def root():
    return {"message": "PDF RAG API running. Visit /docs"}

