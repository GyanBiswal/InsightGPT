# utils.py
import re
from typing import List

def clean_text(text: str) -> str:
    text = text.replace('\n', ' ').replace('\r', ' ')
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def chunk_text(text: str, max_chars=2000, overlap=400) -> List[str]:
    """Simple character-based chunking with overlap (fast and effective)."""
    text = text.strip()
    if len(text) <= max_chars:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = start + max_chars
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap
        if start < 0:
            start = 0
        if start >= len(text):
            break
    return chunks
