import json
import os
from typing import List, Tuple, Dict
from sentence_transformers import SentenceTransformer
import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter


class RAGPipeline:
    def __init__(self, docs_dir: str, strategy: str = "fixed"):
        self.docs_dir = docs_dir
        self.strategy = strategy
        self.chunks = []
        self.chunk_ids = []
        self.embeddings = None
        self.model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

    def load_documents(self) -> Dict[str, str]:
        docs = {}
        for filename in os.listdir(self.docs_dir):
            if filename.endswith('.txt'):
                doc_id = filename.replace('.txt', '')
                with open(os.path.join(self.docs_dir, filename), 'r', encoding='utf-8') as f:
                    docs[doc_id] = f.read()
        return docs

    def chunk_fixed(self, text: str, doc_id: str, chunk_size: int = 2000) -> List[Tuple[str, str]]:
        chunks = []
        for i, start in enumerate(range(0, len(text), chunk_size)):
            chunk = text[start:start + chunk_size]
            if chunk.strip():
                chunk_id = f"{doc_id}__{i}"
                chunks.append((chunk, chunk_id))
        return chunks

    def chunk_recursive(self, text: str, doc_id: str) -> List[Tuple[str, str]]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=400,
            chunk_overlap=80,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
            length_function=len
        )
        chunk_texts = splitter.split_text(text)
        chunks = []
        for i, chunk in enumerate(chunk_texts):
            if chunk.strip():
                chunk_id = f"{doc_id}__{i}"
                chunks.append((chunk, chunk_id))
        return chunks

    def build_index(self):
        docs = self.load_documents()
        self.chunks = []
        self.chunk_ids = []

        for doc_id, content in docs.items():
            if self.strategy == "fixed":
                chunks = self.chunk_fixed(content, doc_id)
            else:
                chunks = self.chunk_recursive(content, doc_id)

            for chunk_text, chunk_id in chunks:
                self.chunks.append(chunk_text)
                self.chunk_ids.append(chunk_id)

        self.embeddings = self.model.encode(self.chunks, show_progress_bar=False)
        print(f"  {self.strategy}: {len(self.chunks)} chunks")
        return self.chunks

    def retrieve(self, query: str, k: int = 5) -> List[Tuple[str, float, str]]:
        query_embedding = self.model.encode([query])[0]
        similarities = np.dot(self.embeddings, query_embedding) / (
            np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_embedding) + 1e-8
        )
        top_indices = np.argsort(similarities)[-k:][::-1]
        results = []
        for idx in top_indices:
            chunk_text = self.chunks[idx]
            chunk_id = self.chunk_ids[idx]
            preview = chunk_text[:150].replace('\n', ' ') + "..."
            results.append((preview, float(similarities[idx]), chunk_id))
        return results

    def corpus_stats(self) -> dict:
        docs = self.load_documents()
        total_chars = sum(len(content) for content in docs.values())
        return {
            'num_documents': len(docs),
            'total_chars': total_chars,
            'num_chunks': len(self.chunks),
            'avg_chunk_size': total_chars / len(self.chunks) if self.chunks else 0
        }