# analyst.py
from llm_client import make_client, get_model
from schema import ResearchData, ProductReport

SYSTEM_PROMPT = """Ты — Analyst. Твоя задача — на основе исследовательских данных (ResearchData) сформировать финальный структурированный отчёт (ProductReport) для пользователя.

Входные данные содержат:
- product_id
- список текстов отзывов (texts)
- средний рейтинг и количество отзывов
- плюсы и минусы (извлечены)
- уверенность исследователя

Ты должен:
1. Проверить согласованность данных (если плюсы/минусы пустые — можно указать, что информации недостаточно).
2. Сформулировать краткую сводку (1-2 предложения), которая обобщает мнение пользователей.
3. Выставить свою уверенность в отчёте (может быть ниже, чем у исследователя, если данные противоречивы или неполны).
4. Вернуть структуру ProductReport.

Будь краток, но содержателен.
"""

class Analyst:
    def __init__(self):
        self.client = make_client()
        self.model = get_model()
        self.trace = []

    def analyze(self, research_data: ResearchData) -> ProductReport:
        prompt = (
            f"Исследовательские данные:\n"
            f"- Product ID: {research_data.product_id}\n"
            f"- Средний рейтинг: {research_data.avg_rating} (на основе {research_data.review_count} отзывов)\n"
            f"- Плюсы: {research_data.pros}\n"
            f"- Минусы: {research_data.cons}\n"
            f"- Уверенность исследователя: {research_data.confidence}\n"
            f"- Тексты отзывов (первые 3): {research_data.texts[:3]}\n\n"
            "Сформируй финальный отчёт (ProductReport)."
        )
        step_info = {
            "step": "analyst",
            "input": {
                "product_id": research_data.product_id,
                "avg_rating": research_data.avg_rating,
                "review_count": research_data.review_count,
                "pros_count": len(research_data.pros),
                "cons_count": len(research_data.cons),
                "confidence_researcher": research_data.confidence
            }
        }
        report = self.client.chat.completions.create(
            model=self.model,
            response_model=ProductReport,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_retries=2
        )
        step_info["output"] = report.model_dump()
        self.trace.append(step_info)

        return report