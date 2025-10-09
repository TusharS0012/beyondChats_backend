from fastapi import FastAPI, UploadFile, File, HTTPException , Form
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
import re
import yt_dlp
from uuid import UUID
from uuid import uuid4

# Load environment variables
load_dotenv()

HF_API_TOKEN = os.getenv("HF_API_KEY")  # Hugging Face API token
LLM_MODEL = "meta-llama/Llama-3.1-8B-Instruct"  # Model to use
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # HuggingFace embedding model

app = FastAPI()

origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- Database ----------------
url = os.getenv("SUPABASE_URL")
key = os.getenv("SUPABASE_KEY")

if url is None or key is None:
    raise RuntimeError("SUPABASE_URL and SUPABASE_KEY must be set in environment variables")

supabase: Client = create_client(url, key)

# ---------------- Initialize HuggingFace InferenceClient ----------------
hf_client = None
if HF_API_TOKEN:
    try:
        hf_client = InferenceClient(
            provider="auto",
            api_key=HF_API_TOKEN,
        )
    except Exception as e:
        print(f"⚠️ Failed to initialize HuggingFace client: {e}")

# ---------------- Request Models ----------------
class ChatRequest(BaseModel):
    question: str

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

def get_embedding(text: str) -> List[float]:
    """
    Generate embedding using HuggingFace Inference API.
    Avoids using local torch library.
    """
    if not hf_client:
        raise RuntimeError("HuggingFace client not initialized. Set HF_API_KEY.")
    
    response = hf_client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text
    )
    return response.embedding

def cosine_similarity(a: List[float], b: List[float]) -> float:
    a_arr = np.array(a)
    b_arr = np.array(b)
    return float(np.dot(a_arr, b_arr) / (np.linalg.norm(a_arr) * np.linalg.norm(b_arr)))

def generate_hf_answer(context: str, question: str, top_score: float = 0.0) -> str:
    if top_score < 0.15:
        return "I apologize, but I can only answer questions related to the content of the uploaded document."

    if not hf_client:
        return f"Based on the document:\n\n{context[:500]} (HF_API_KEY not set)"

    system_message = (
        "You are a helpful AI assistant. Answer the question by understanding the document content "
        "and providing clear explanations. Combine document information with helpful context when needed."
    )

    user_message = f"""Document Context:
{context}

Question: {question}
"""

    try:
        completion = hf_client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ],
            max_tokens=500,
            temperature=0.7,
            top_p=0.9,
        )
        answer = completion.choices[0].message.content.strip()  # type: ignore
        if len(answer) < 30:
            return f"Based on the document:\n\n{context[:400]}"
        return answer
    except Exception as e:
        return f"⚠️ Unable to generate AI response. Using context:\n\n{context[:400]}"

def search_youtube(query, max_results=3):
    ydl_opts = {
        'quiet': True,
        'skip_download': True,
        'extract_flat': 'in_playlist',
        'default_search': f'ytsearch{max_results}',
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            results = ydl.extract_info(query, download=False)
            videos = []
            for entry in results.get('entries', [])[:max_results]:
                videos.append({
                    "title": entry.get('title'),
                    "url": entry.get('url'),
                    "channel": entry.get('uploader'),
                    "description": entry.get('description') or ""
                })
            return videos
    except Exception as e:
        print("⚠️ YouTube search failed:", e)
        return []

# ---------------- Endpoints ----------------
@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        result = supabase.table("document_chunks").select("text_chunk, embedding, page_number").execute()
        if not result.data or len(result.data) == 0:  # type: ignore
            raise HTTPException(status_code=404, detail="No document chunks found.")

        question_emb = get_embedding(request.question)
        scored = []

        for row in result.data:  # type: ignore
            text_chunk = row.get("text_chunk")
            emb = row.get("embedding")
            page = row.get("page_number")
            
            if isinstance(emb, str):
                emb = json.loads(emb)
            score = cosine_similarity(question_emb, emb)
            scored.append({"text": text_chunk, "page": page, "score": score})

        top_chunks = sorted(scored, key=lambda x: x["score"], reverse=True)[:3]

        if not top_chunks or all(c["score"] < 0.1 for c in top_chunks):
            answer = "I apologize, but I can only answer questions related to the uploaded document."
        else:
            context = "\n\n".join([f"[Page {c['page']}]\n{c['text']}" for c in top_chunks])
            top_score = top_chunks[0]["score"]
            answer = generate_hf_answer(context, request.question, top_score)

        related_videos = []
        if any(word in request.question.lower() for word in ["video", "videos", "show me", "watch", "recommend"]):
            related_videos = search_youtube(request.question, max_results=3)

        return {
            "answer": answer,
            "context": [{"text": c["text"], "page": c["page"], "score": float(c["score"])} for c in top_chunks],
            "videos": related_videos
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing question: {str(e)}")

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...), user_id: str = Form(...)):
    try:
        user_uuid = UUID(user_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid user_id. Must be UUID.")

    if not file.filename or not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files allowed.")

    temp_path = f"temp_{file.filename}"
    try:
        with open(temp_path, "wb") as f:
            f.write(await file.read())

        doc_result = supabase.table("documents").insert({
            "file_name": file.filename,
            "file_url": f"/files/{file.filename}",
            "user_id": str(user_uuid)
        }).execute()

        document_id = doc_result.data[0]["id"]  # type: ignore
        text = extract_text_from_pdf(temp_path)
        chunks = chunk_text(text)

        chunk_data = []
        for chunk in chunks:
            embedding = get_embedding(chunk["text"])
            chunk_data.append({
                "document_id": document_id,
                "text_chunk": chunk["text"],
                "embedding": embedding,
                "page_number": chunk["page"]
            })

        if chunk_data:
            supabase.table("document_chunks").insert(chunk_data).execute()

        return {"success": True, "documentId": str(document_id), "chunksProcessed": len(chunk_data)}

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "embedding_model": EMBEDDING_MODEL,
        "llm_model": LLM_MODEL,
        "llm_client_active": hf_client is not None
    }

@app.get("/")
async def root():
    return {"message": "PDF RAG API running. Visit /docs for API documentation."}
