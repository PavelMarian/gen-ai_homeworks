import json
import re
import time
from pathlib import Path
import chromadb
from chromadb.utils import embedding_functions
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi
import pandas as pd

DATA_DIR = Path(__file__).parent / "data"
BM25_CACHE = Path(__file__).parent / "bm25_cache.json"
CHROMA_PATH = "./chroma_db"

embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="paraphrase-multilingual-MiniLM-L12-v2"
)
client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = client.get_or_create_collection(
    name="amazon_reviews",
    embedding_function=embed_fn,
    metadata={"hnsw:space": "cosine"}
)

splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=80,
    separators=["\n\n", "\n", ". ", "? ", "! ", " "]
)

def tokenize(text: str):
    return re.findall(r"[a-z0-9]{2,}", text.lower())

def chunk_text(text: str):
    return [c.strip() for c in splitter.split_text(text) if c.strip()]

def ingest_from_csv(csv_path: str):
    """Читает CSV, для каждой строки создаёт чанки и индексирует."""
    df = pd.read_csv(csv_path)
    all_chunks, all_ids, all_meta = [], [], []

    for idx, row in df.iterrows():
        text = f"ProductId: {row['ProductId']}\nSummary: {row['Summary']}\nText: {row['Text']}"
        chunks = chunk_text(text)
        for i, c in enumerate(chunks):
            cid = f"review_{idx}__{i}"
            all_chunks.append(c)
            all_ids.append(cid)
            all_meta.append({"product_id": row['ProductId'], "chunk_id": i})

    existing = collection.get()
    if existing["ids"]:
        collection.delete(ids=existing["ids"])

    collection.add(documents=all_chunks, ids=all_ids, metadatas=all_meta)

    bm25_data = {
        "ids": all_ids,
        "tokens": [tokenize(c) for c in all_chunks],
        "texts": all_chunks,
    }
    BM25_CACHE.write_text(json.dumps(bm25_data, ensure_ascii=False))

    print(f"Индексировано {len(all_chunks)} чанков из {len(df)} отзывов")

def _load_bm25():
    data = json.loads(BM25_CACHE.read_text())
    bm25 = BM25Okapi(data["tokens"])
    return bm25, data["ids"], data["texts"]

def hybrid_retrieve(query: str, k: int = 5, top: int = 15, c: int = 60):
    """Гибридный поиск: dense (Chroma) + BM25 с RRF."""
    dense = collection.query(query_texts=[query], n_results=top)
    dense_ids = dense["ids"][0]
    dense_docs = dense["documents"][0]

    bm25, bm25_ids, bm25_texts = _load_bm25()
    tokens = tokenize(query)
    scores = bm25.get_scores(tokens)
    bm25_order = sorted(range(len(bm25_ids)), key=lambda i: scores[i], reverse=True)[:top]
    sparse_ids = [bm25_ids[i] for i in bm25_order]
    sparse_texts = [bm25_texts[i] for i in bm25_order]

    rrf = {}
    for rank, cid in enumerate(dense_ids):
        rrf[cid] = rrf.get(cid, 0.0) + 1.0 / (c + rank)
    for rank, cid in enumerate(sparse_ids):
        rrf[cid] = rrf.get(cid, 0.0) + 1.0 / (c + rank)

    ordered = sorted(rrf.items(), key=lambda kv: kv[1], reverse=True)[:k]
    top_ids = [cid for cid, _ in ordered]

    text_by_id = dict(zip(bm25_ids, bm25_texts))
    for i, did in enumerate(dense_ids):
        text_by_id[did] = dense_docs[i]
    documents = [text_by_id[i] for i in top_ids]

    return {"ids": [top_ids], "documents": [documents]}