# embeddings.py
import chromadb
from chromadb.utils import embedding_functions
from typing import List, Dict

class EmbeddingIndex:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.client = chromadb.Client()
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=model_name
        )
        self.collection = self.client.create_collection(
            name="insightgpt",
            embedding_function=self.embedding_function
        )

    def add_documents(self, docs: List[Dict]):
        for d in docs:
            for ch in d["chunks"]:
                self.collection.add(
                    documents=[ch["text"]],
                    metadatas=[{
                        "source": d["source"],
                        "doc_id": d["id"],
                        "chunk_id": ch["chunk_id"]
                    }],
                    ids=[ch["chunk_id"]]
                )

    def query(self, query_text: str, n_results: int = 5):
        return self.collection.query(
            query_texts=[query_text],
            n_results=n_results
        )
