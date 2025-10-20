# compare.py
from transformers import pipeline
from typing import Dict

_gen = None

def get_generation_pipeline():
    global _gen
    if _gen is None:
        # Small CPU-friendly model
        _gen = pipeline("text-generation", model="distilgpt2", max_new_tokens=300)
    return _gen

def generate_comparison(per_source_summaries: Dict[str, str]) -> str:
    prompt_text = (
        "You are an expert research analyst. "
        "Do NOT repeat the source summaries verbatim. "
        "Focus on synthesizing information into:\n"
        "1) Key themes (3-6 bullets)\n"
        "2) Agreements and disagreements\n"
        "3) Open questions / recommended next steps\n\n"
        "SOURCE SUMMARIES:\n"
    )
    for source, summary in per_source_summaries.items():
        prompt_text += f"--- Source: {source}\n{summary}\n"

    gen = get_generation_pipeline()
    out = gen(prompt_text, max_new_tokens=400, do_sample=False)
    text = out[0]['generated_text']
    return text
