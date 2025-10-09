from fastapi import APIRouter, UploadFile, File
from app.services.pdf_service import extract_text_from_pdf, chunk_text
from app.services.llm_service import generate_embedding

router = APIRouter()

# Example: store chunks in-memory (replace with Supabase / DB)
DOCUMENTS = []

@router.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    contents = await file.read()
    file_path = f"temp_{file.filename}"
    
    with open(file_path, "wb") as f:
        f.write(contents)

    text = extract_text_from_pdf(file_path)
    chunks = chunk_text(text)

    # Optionally generate embeddings here if needed
    for chunk in chunks:
        chunk["embedding"] = generate_embedding(chunk["text"])
        DOCUMENTS.append(chunk)

    return {"success": True, "chunks": len(chunks), "documentId": f"doc-{file.filename}"}
