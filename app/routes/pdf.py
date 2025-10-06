from fastapi import APIRouter, UploadFile, File
from typing import List
from app.core.pdf_parser import extract_pdf_with_pages, split_text_into_chunks
from app.core.embeddings import embed_texts
import os

router = APIRouter()

PDF_STORE = {}

@router.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    file_path = f"/tmp/{file.filename}"
    with open(file_path, "wb") as f:
        f.write(await file.read())

    pages = extract_pdf_with_pages(file_path)

    chunks = []
    for p in pages:
        page_chunks = split_text_into_chunks(p["text"], max_tokens=300)
        for chunk in page_chunks:
            chunks.append({"page": p["page"], "text": chunk})

    embeddings = embed_texts([c["text"] for c in chunks])

    PDF_STORE[file.filename] = {
        "text": " ".join([p["text"] for p in pages]),
        "chunks": chunks,
        "embeddings": embeddings
    }

    return {"filename": file.filename, "chunks": len(chunks)}

@router.get("/list")
def list_pdfs() -> List[str]:
    return list(PDF_STORE.keys())

@router.get("/download/{pdf_name}")
def download_pdf(pdf_name: str):
    file_path = f"/tmp/{pdf_name}"
    if not os.path.exists(file_path):
        return {"error": "File not found"}
    return {"file_path": file_path}

@router.delete("/delete/{pdf_name}")
def delete_pdf(pdf_name: str):
    if pdf_name in PDF_STORE:
        del PDF_STORE[pdf_name]
    file_path = f"/tmp/{pdf_name}"
    if os.path.exists(file_path):
        os.remove(file_path)
    return {"message": f"{pdf_name} deleted"}
