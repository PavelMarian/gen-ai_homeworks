from llm_client import make_client, get_model
from pydantic import BaseModel, Field
from typing import List, Dict, Any
from rag import hybrid_retrieve
from schema import AssertionVerdict, JudgeReport

def judge_report(query: str, report: Dict[str, Any]) -> JudgeReport:
    """
    Проверяет утверждения из отчёта (pros/cons) через RAG.
    Для каждого утверждения делает поиск в индексе и просит LLM подтвердить или опровергнуть.
    """
    client = make_client()
    model = get_model()

    assertions = []
    for p in report.get("pros", []):
        assertions.append({"text": p, "type": "pro"})
    for c in report.get("cons", []):
        assertions.append({"text": c, "type": "con"})

    verdicts = []
    for a in assertions:
        hits = hybrid_retrieve(a["text"], k=5)
        retrieved_texts = hits["documents"][0] if hits["documents"] else []
        context = "\n---\n".join(retrieved_texts[:5]) if retrieved_texts else "(нет контекста)"

        prompt = (
            f"Ты — судья, проверяющий, подтверждается ли утверждение из отчёта найденными отзывами.\n"
            f"Утверждение: {a['text']}\n"
            f"Тип: {a['type']} (плюс или минус)\n"
            f"Контекст из отзывов:\n{context}\n\n"
            "Оцени:\n"
            "1. Подтверждается ли это утверждение контекстом? (supported: true/false)\n"
            "2. Какие цитаты из контекста подтверждают или опровергают? (evidence: список цитат)\n"
            "3. Уверенность в решении (0-1)\n"
            "4. Короткий комментарий (comment)\n"
            "Верни JSON."
        )
        verdict = client.chat.completions.create(
            model=model,
            response_model=AssertionVerdict,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_retries=2,
        )
        verdict.assertion = a["text"]
        verdicts.append(verdict)

    hallucinated_assertions = [v for v in verdicts if not v.supported]
    overall_hallucinated = len(hallucinated_assertions) > 0
    overall_score = 5 - min(len(hallucinated_assertions), 4)
    summary = f"Из {len(verdicts)} утверждений {len(hallucinated_assertions)} не подтверждены контекстом."

    return JudgeReport(
        assertions=verdicts,
        overall_hallucinated=overall_hallucinated,
        overall_score=overall_score,
        summary=summary
    )