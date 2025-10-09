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
from sentence_transformers import SentenceTransformer
from huggingface_hub import InferenceClient

# Load environment variables
load_dotenv()

HF_API_TOKEN = os.getenv("HF_API_KEY")  # Hugging Face API token
LLM_MODEL = "meta-llama/Llama-3.1-8B-Instruct"  # Model to use

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- Database ----------------
url = "https://gsgiwpwyvfascngdjlfm.supabase.co"
key = "sb_secret_YhmFOf5rWfNgtRkoFU3aXg_ad9gUsc1"

if url is None or key is None:
    print("❌ SUPABASE_URL and SUPABASE_KEY environment variables must be set")
    raise RuntimeError("SUPABASE_URL and SUPABASE_KEY environment variables must be set")

try:
    supabase: Client = create_client(url, key)
    print("✅ Supabase client initialized successfully!")
except Exception as e:
    print(f"❌ Supabase client initialization failed: {e}")
    raise

# ---------------- Load Embedding Model ----------------
print("📥 Loading embedding model...")
try:
    embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    print("✅ Embedding model loaded successfully!")
except Exception as e:
    print(f"❌ Failed to load embedding model: {e}")
    raise

# ---------------- Initialize HuggingFace InferenceClient ----------------
hf_client = None
if HF_API_TOKEN:
    try:
        hf_client = InferenceClient(
            provider="auto",
            api_key=HF_API_TOKEN,
        )
        print("✅ HuggingFace InferenceClient initialized successfully!")
    except Exception as e:
        print(f"⚠️ Failed to initialize HuggingFace client: {e}")
else:
    print("⚠️ No HF_API_KEY found. LLM features will use fallback responses.")

# ---------------- Request Models ----------------
class ChatRequest(BaseModel):
    question: str

# ---------------- Helpers ----------------
def extract_text_from_pdf(file_path: str) -> str:
    doc = fitz.open(file_path)
    text = ""
    for page in doc:  # type: ignore
        text += page.get_text("text") + "\n\n"  # type: ignore
    doc.close()
    return text

def chunk_text(text: str, chunk_size: int = 1000) -> List[dict]:
    chunks = []
    pages = text.split("\f")
    for idx, page_text in enumerate(pages):
        start = 0
        while start < len(page_text):
            chunk = page_text[start:start+chunk_size]
            if chunk.strip():  # Only add non-empty chunks
                chunks.append({"text": chunk, "page": idx + 1})
            start += chunk_size
    return chunks

def ensure_user_in_profiles(user_id: str):
    existing = supabase.table("profiles").select("id").eq("id", user_id).execute()
    if not existing.data or len(existing.data) == 0: # type: ignore
        supabase.table("profiles").insert({"id": user_id}).execute()
        
def get_embedding(text: str) -> List[float]:
    """Generate embedding using local sentence-transformers model"""
    embedding = embedding_model.encode(text)
    return embedding.tolist()

def generate_hf_answer(context: str, question: str, top_score: float = 0.0) -> str:
    """Generate answer using Hugging Face InferenceClient with context-awareness"""
    
    # Determine if question is relevant based on similarity score
    if top_score < 0.15:
        return "I apologize, but I can only answer questions related to the content of the uploaded document. Your question appears to be outside the scope of this document."
    
    # If no HF client, provide context-based answer
    if not hf_client:
        if top_score > 0.3:
            return f"Based on the document:\n\n{context[:500]}\n\nNote: Using fallback mode. Configure HF_API_KEY for better responses."
        else:
            return f"The document mentions:\n\n{context[:500]}"
    
    # Build system message based on relevance score
    if top_score > 0.3:
        system_message = """You are a helpful AI assistant analyzing a document. Your task is to:
1. Answer questions naturally based on the document context
2. Explain concepts clearly in your own words
3. Be conversational and informative
4. Don't just quote - synthesize and explain the information
5. If the context doesn't contain enough information, say so politely"""
    else:
        system_message = """You are a helpful AI assistant. Answer the question by understanding the document content and providing clear explanations. Combine document information with helpful context when needed."""
    
    # Build user message with context
    user_message = f"""Based on the following document excerpt, please answer my question:

Document Context:
{context}

My Question: {question}

Please provide a clear, conversational answer based on the document content."""
    
    try:
        print(f"🤖 Generating answer with {LLM_MODEL}...")
        
        # Use the new InferenceClient API
        completion = hf_client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": system_message
                },
                {
                    "role": "user",
                    "content": user_message
                }
            ],
            max_tokens=500,
            temperature=0.7,
            top_p=0.9,
        )
        
        # Extract the answer from the completion
        answer = completion.choices[0].message.content.strip() # type: ignore
        
        # If answer is too short or empty, use context
        if len(answer) < 30:
            return f"Based on the document:\n\n{context[:400]}"
        
        print("✅ Answer generated successfully!")
        return answer
        
    except Exception as e:
        error_msg = str(e)
        print(f"⚠️ LLM generation failed: {error_msg}")
        
        # Provide informative fallback based on error type
        if "404" in error_msg or "not found" in error_msg.lower():
            fallback = f"⚠️ Model '{LLM_MODEL}' not available. "
        elif "401" in error_msg or "unauthorized" in error_msg.lower():
            fallback = "⚠️ Invalid API token. "
        elif "503" in error_msg or "loading" in error_msg.lower():
            fallback = "⚠️ Model is loading. "
        else:
            fallback = "⚠️ Unable to generate AI response. "
        
        # Return context with error info
        if top_score > 0.3:
            return f"{fallback}Here's what the document says:\n\n{context[:400]}"
        else:
            return f"{fallback}Related information from document:\n\n{context[:400]}"

def cosine_similarity(a: List[float], b: List[float]) -> float:
    a_arr = np.array(a)
    b_arr = np.array(b)
    return float(np.dot(a_arr, b_arr) / (np.linalg.norm(a_arr) * np.linalg.norm(b_arr)))

# ---------------- Endpoints ----------------         
@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        # Fetch all document chunks using Supabase
        result = supabase.table("document_chunks").select("text_chunk, embedding, page_number").execute()
        
        if not hasattr(result, 'data') or not result.data or len(result.data) == 0:  # type: ignore
            raise HTTPException(status_code=404, detail="No document chunks found. Please upload a PDF first.")

        print(f"🔍 Searching through {len(result.data)} chunks...")  # type: ignore
        question_emb = get_embedding(request.question)
        scored = []
        
        for row in result.data:  # type: ignore
            text_chunk = row.get("text_chunk")  # type: ignore
            emb = row.get("embedding")  # type: ignore
            page = row.get("page_number")  # type: ignore
            
            # Convert embedding from database format to list if needed
            if isinstance(emb, str):
                emb = json.loads(emb)
            elif not isinstance(emb, list):
                emb = list(emb)  # type: ignore
            
            score = cosine_similarity(question_emb, emb)  # type: ignore
            scored.append({"text": text_chunk, "page": page, "score": score})

        # Get top 3 most similar chunks
        top_chunks = sorted(scored, key=lambda x: x["score"], reverse=True)[:3]
        
        # Log the top scores for debugging
        print(f"🎯 Top 3 scores: {[round(c['score'], 3) for c in top_chunks]}")
        
        if not top_chunks or all(c["score"] < 0.1 for c in top_chunks):
            return {
                "answer": "I apologize, but I can only answer questions related to the content of the uploaded document. Your question appears to be outside the scope of this document.",
                "context": []
            }
        
        context = "\n\n".join([f"[Page {c['page']}]\n{c['text']}" for c in top_chunks])
        
        # Pass the top score to the answer generator
        top_score = top_chunks[0]["score"]
        print(f"📊 Generating answer with top similarity score: {round(top_score, 3)}")
        answer = generate_hf_answer(context, request.question, top_score)
        
        return {
            "answer": answer, 
            "context": [{"text": c["text"], "page": c["page"], "score": float(c["score"])} for c in top_chunks]
        }
    
    except Exception as e:
        print(f"❌ Error processing question: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing question: {str(e)}")
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from pydantic import BaseModel
from uuid import UUID, uuid4
import os

app = FastAPI()

# Assuming you already have supabase client initialized:
# from supabase import create_client, Client
# supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# ---------------- PDF Upload ----------------
@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...), user_id: str = Form(...)):
    """
    Upload a PDF and save chunks with embeddings.
    Ensure the user_id exists in profiles.
    """
    # Validate UUID
    try:
        user_uuid = UUID(user_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid user_id. Must be a valid UUID.")

    # Check user exists in profiles
    existing_user = supabase.table("profiles").select("id").eq("id", str(user_uuid)).execute()
    if not existing_user.data or len(existing_user.data) == 0: # type: ignore
        # Insert user into profiles
        supabase.table("profiles").insert({"id": str(user_uuid)}).execute()
        print(f"✅ Created new user in profiles: {user_uuid}")

    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files allowed")

    temp_path = f"temp_{file.filename}"
    try:
        with open(temp_path, "wb") as f:
            f.write(await file.read())

        # Insert document record
        doc_result = supabase.table("documents").insert({
            "file_name": file.filename,
            "file_url": f"/files/{file.filename}",
            "user_id": str(user_uuid)
        }).execute()

        if not doc_result.data or len(doc_result.data) == 0: # type: ignore
            raise HTTPException(status_code=500, detail="Failed to insert document")

        document_id = doc_result.data[0]["id"] # type: ignore
        print(f"✅ Document inserted: {document_id} for user {user_uuid}")

        # Extract text, chunk, generate embeddings
        text = extract_text_from_pdf(temp_path)
        chunks = chunk_text(text)

        chunk_data = []
        for idx, chunk in enumerate(chunks):
            embedding = get_embedding(chunk["text"])
            chunk_data.append({
                "document_id": document_id,
                "text_chunk": chunk["text"],
                "embedding": embedding,
                "page_number": chunk["page"]
            })

        if chunk_data:
            supabase.table("document_chunks").insert(chunk_data).execute()
            print(f"✅ Inserted {len(chunk_data)} chunks")

        return {"success": True, "documentId": str(document_id), "chunksProcessed": len(chunk_data)}

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

# ---------------- Generate Quiz ----------------
class QuizRequest(BaseModel):
    user_id: str

@app.post("/generate_quiz")
def generate_quiz(request: QuizRequest):
    """
    Example: Fetch the latest document for this user and generate quiz
    """
    # Validate UUID
    try:
        user_uuid = UUID(request.user_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid user_id. Must be a valid UUID.")

    # Fetch latest document for this user
    try:
        doc_response = supabase.table("documents") \
            .select("id") \
            .eq("user_id", str(user_uuid)) \
            .order("created_at", desc=True) \
            .limit(1) \
            .execute()
        
        if not doc_response.data or len(doc_response.data) == 0: # type: ignore
            raise HTTPException(status_code=404, detail="No documents found for this user.")

        document_id = doc_response.data[0].get("id") # type: ignore
        print(f"✅ Latest document for user {user_uuid}: {document_id}")

        # Fetch chunks for this document
        chunks_response = supabase.table("document_chunks") \
            .select("*") \
            .eq("document_id", document_id) \
            .execute()
        
        chunks = chunks_response.data # type: ignore
        print(f"📄 Found {len(chunks)} chunks for document {document_id}")

        # Example: generate a simple quiz from chunks
        quiz = []
        for idx, chunk in enumerate(chunks[:5]):  # limit to first 5 chunks
            quiz.append({
                "question": f"Question from chunk {idx+1}",
                "answer": chunk["text_chunk"][:100] + "..."  # type: ignore
            })

        return {"documentId": str(document_id), "quiz": quiz}

    except Exception as e:
        print(f"❌ Error generating quiz: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating quiz: {str(e)}")

class QuizAttemptRequest(BaseModel):
    user_id: str
    quiz_ids: List[str]
    answers: List[str]
@app.post("/submit_quiz")
async def submit_quiz(request: QuizAttemptRequest): # type: ignore
    try:
        # Fetch correct answers
        quizzes = supabase.table("quizzes").select("id, correct_answer").in_("id", request.quiz_ids).execute()
        if not quizzes.data: # type: ignore
            raise HTTPException(status_code=404, detail="No quizzes found.")

        # Score calculation
        correct_count = 0
        for i, q in enumerate(quizzes.data): # type: ignore
            if i < len(request.answers) and request.answers[i].upper() == q["correct_answer"].upper(): # type: ignore
                correct_count += 1

        score = (correct_count / len(request.quiz_ids)) * 100

        # Save attempt
        attempt = supabase.table("quiz_attempts").insert({
            "user_id": request.user_id,
            "quiz_id": str(request.quiz_ids[0]),  # could link to first quiz of that set
            "score": score,
            "answers": json.dumps(request.answers),
        }).execute()

        # Update progress
        supabase.table("progress").insert({
            "user_id": request.user_id,
            "quiz_id": str(request.quiz_ids[0]),
            "score": score,
        }).execute()

        return {"success": True, "score": score, "correct": correct_count, "total": len(request.quiz_ids)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Quiz submission failed: {e}")
    
@app.get("/quiz/{document_id}")
async def get_quiz(document_id: str):
    result = supabase.table("quizzes").select("*").eq("document_id", document_id).execute()
    if not result.data: # type: ignore
        raise HTTPException(status_code=404, detail="No quiz found for this document.")
    return result.data  # type: ignore
        
@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        "llm_model": LLM_MODEL,
        "llm_client_active": hf_client is not None
    }

@app.get("/")
async def root():
    return {"message": "PDF RAG API is running. Visit /docs for API documentation."}