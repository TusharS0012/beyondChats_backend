from typing import List
import fitz  # PyMuPDF

def extract_text_from_pdf(file_path: str) -> str:
    doc = fitz.open(file_path)
    text = ""
    for i in range(len(doc)):
        page = doc[i]
        text += page.get_text("text") + "\n\n"  # type: ignore
    return text


def chunk_text(text: str, chunk_size: int = 1000) -> List[dict]:
    chunks = []
    pages = text.split("\f")  # simple page split
    for idx, page_text in enumerate(pages):
        start = 0
        while start < len(page_text):
            chunk = page_text[start:start+chunk_size]
            chunks.append({"text": chunk, "page": idx+1})
            start += chunk_size
    return chunks
