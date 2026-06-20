from llm_client import make_client, get_model
from schema import ResearchData, ProductReport

SYSTEM_PROMPT = """Ты — критик, проверяющий отчёт на основе исследовательских данных.
Твоя задача:
1. Проверить, что все утверждения в отчёте (плюсы, минусы, сводка) обоснованы данными из ResearchData.
2. Проверить согласованность: нет ли противоречий между плюсами и минусами.
3. Если данных недостаточно (например, мало отзывов), указать это в сводке.
4. Если есть недостатки, исправь их: дополни сводку, уточни плюсы/минусы, скорректируй уверенность.
5. Верни исправленный ProductReport.

Будь критичен, но конструктивен.
"""

class Critic:
    def __init__(self):
        self.client = make_client()
        self.model = get_model()
        self.trace = []

    def critique(self, research_data: ResearchData, report: ProductReport) -> ProductReport:
        prompt = (
            f"Исследовательские данные:\n"
            f"- Product ID: {research_data.product_id}\n"
            f"- Средний рейтинг: {research_data.avg_rating} (из {research_data.review_count} отзывов)\n"
            f"- Плюсы: {research_data.pros}\n"
            f"- Минусы: {research_data.cons}\n"
            f"- Уверенность исследователя: {research_data.confidence}\n\n"
            f"Сгенерированный отчёт:\n"
            f"- Средний рейтинг: {report.avg_rating}\n"
            f"- Плюсы: {report.pros}\n"
            f"- Минусы: {report.cons}\n"
            f"- Сводка: {report.summary}\n"
            f"- Уверенность: {report.confidence}\n\n"
            "Проверь и при необходимости исправь отчёт. Верни исправленный ProductReport."
        )
        improved_report = self.client.chat.completions.create(
            model=self.model,
            response_model=ProductReport,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_retries=2
        )
        self.trace.append({
            "step": "critique",
            "original_report": report.model_dump(),
            "improved_report": improved_report.model_dump()
        })
        return improved_report