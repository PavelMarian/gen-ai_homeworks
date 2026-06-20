import pandas as pd
from llm_client import make_client, get_model
from schema import ProductStats, ProductAspects

_df = None

def set_dataframe(df):
    global _df
    _df = df

def search_reviews(query: str, k: int = 5):
    """Инструмент: поиск отзывов по запросу."""
    from rag import hybrid_retrieve
    hits = hybrid_retrieve(query, k=k)
    return {
        "ids": hits["ids"][0],
        "texts": hits["documents"][0],
    }

def get_product_stats(product_id: str):
    """Инструмент: средний рейтинг и количество отзывов."""
    if _df is None:
        return {"error": "DataFrame not loaded"}
    subset = _df[_df['ProductId'] == product_id]
    if len(subset) == 0:
        return {"error": f"Product {product_id} not found"}
    return {
        "product_id": product_id,
        "avg_rating": subset['Score'].mean(),
        "review_count": len(subset),
    }

def extract_aspects(texts: list[str]):
    """Инструмент: извлечение плюсов и минусов из текстов."""
    client = make_client()
    model = get_model()
    prompt = (
        "Извлеки плюсы и минусы из следующих отзывов. "
        "Верни JSON с полями pros (список строк) и cons (список строк).\n\n"
        + "\n---\n".join(texts)
    )
    return client.chat.completions.create(
        model=model,
        response_model=ProductAspects,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_retries=2,
    )