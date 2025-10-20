# app.py
import streamlit as st
from ingest import ingest_documents
from embeddings import EmbeddingIndex
from summarizer import summarize_chunks, summarize_text
from compare import generate_comparison

st.set_page_config(page_title="InsightGPT", layout="wide")

st.title("InsightGPT — AI Research Analyst")

with st.sidebar:
    st.header("Input")
    uploaded_files = st.file_uploader("Upload PDFs (multiple)", type=['pdf'], accept_multiple_files=True)
    urls_text = st.text_area("Or paste URLs (one per line)")
    max_results = st.slider("Top chunks to show per source", 1, 10, 3)
    run_btn = st.button("Run analysis")

if run_btn:
    # Collect inputs
    pdf_paths = []
    for f in uploaded_files:
        path = f"/tmp/{f.name}"
        with open(path, "wb") as out:
            out.write(f.read())
        pdf_paths.append(path)

    urls = [u.strip() for u in urls_text.splitlines() if u.strip()]

    if not pdf_paths and not urls:
        st.error("Please upload PDFs or paste at least one URL.")
    else:
        st.info("Ingesting documents...")
        docs = ingest_documents(pdf_paths, urls)

        st.info("Building embeddings (Chroma)...")
        index = EmbeddingIndex()
        index.add_documents(docs)

        st.success("Generating per-source summaries...")
        per_source_summaries = {}
        for d in docs:
            # Summarize using chunk-level summarizer
            chunks = [c['text'] for c in d['chunks']]
            summary = summarize_chunks(chunks)
            per_source_summaries[d['source']] = summary
            st.subheader(f"Summary — {d['source']}")
            st.write(summary)

        st.info("Generating comparison and final brief...")
        comparison = generate_comparison(per_source_summaries)
        st.header("Cross-source Comparison & Research Brief")
        st.write(comparison)
