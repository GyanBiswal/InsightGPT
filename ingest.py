# ingest.py
import pdfplumber
from typing import List, Dict

def chunk_text(text: str, chunk_size: int = 500) -> List[str]:
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunks.append(" ".join(words[i:i+chunk_size]))
    return chunks

def ingest_documents(pdf_paths: List[str], urls: List[str] = []) -> List[Dict]:
    docs = []
    for idx, pdf_file in enumerate(pdf_paths):
        with pdfplumber.open(pdf_file) as pdf:
            text = " ".join([p.extract_text() for p in pdf.pages if p.extract_text()])
        chunks = chunk_text(text)
        docs.append({
            "id": str(idx),
            "source": pdf_file.split("/")[-1],
            "chunks": [{"chunk_id": f"{idx}_{i}", "text": c} for i, c in enumerate(chunks)]
        })
    return docs
