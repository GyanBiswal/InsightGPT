# summarizer.py
from transformers import pipeline
from typing import List

_summarizer = None

def get_summarizer():
    global _summarizer
    if _summarizer is None:
        # CPU-friendly summarization model
        _summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
    return _summarizer

def summarize_text(text: str) -> str:
    summarizer = get_summarizer()
    summary = summarizer(text, max_length=150, min_length=40, do_sample=False)
    return summary[0]['summary_text']

def summarize_chunks(chunks: List[str]) -> str:
    return " ".join([summarize_text(c) for c in chunks])
