import fitz  # type: ignore

from typing import List, Dict

def extract_text_from_pdf(file_path: str) -> str:
    """Extract full text from a PDF file."""
    text = ""
    doc = fitz.open(file_path)
    for page in doc:
        page_text = page.get_text()
        text += f"\n\n{page_text}"
    return text

def split_text_into_chunks(text: str, max_tokens: int = 300) -> List[str]:
    """
    Split text into chunks of approx max_tokens words.
    """
    words = text.split()
    chunks = []
    current_chunk = []

    for word in words:
        current_chunk.append(word)
        if len(current_chunk) >= max_tokens:
            chunks.append(" ".join(current_chunk))
            current_chunk = []

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

def extract_pdf_with_pages(file_path: str) -> List[Dict]:
    """
    Extract text per page for RAG citation.
    Returns list of dicts: {"page": page_number, "text": page_text}
    """
    doc = fitz.open(file_path)
    pdf_pages = []
    for i, page in enumerate(doc, start=1):
        page_text = page.get_text()
        pdf_pages.append({"page": i, "text": page_text})
    return pdf_pages
