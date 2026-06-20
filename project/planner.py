from llm_client import make_client, get_model
from schema import Plan, SubTask

PLANNER_SYSTEM = """Ты — планировщик мультиагентной системы анализа отзывов.

Пользователь задаёт вопрос о продукте. Твоя задача — разбить его на подзадачи и распределить между агентами.

Доступные агенты:
- researcher: собирает информацию: ищет отзывы, получает статистику, извлекает плюсы и минусы.
- analyst: на основе данных researcher формирует финальный отчёт.
- critic: проверяет отчёт аналитика на полноту и согласованность, исправляет при необходимости.

Правила:
1. Всегда соблюдай порядок: researcher → analyst → critic.
2. Укажи зависимости (depends_on) между задачами.
3. Верни план в JSON с полями: reasoning (строка) и subtasks (список).
4. Для каждой подзадачи укажи id, description, agent, depends_on (список id).

Пример ответа:
{
  "reasoning": "Сначала Researcher соберёт данные, затем Analyst сделает отчёт, потом Critic проверит и улучшит его.",
  "subtasks": [
    {"id": 1, "description": "Собрать информацию о продукте", "agent": "researcher", "depends_on": []},
    {"id": 2, "description": "Сформировать финальный отчёт", "agent": "analyst", "depends_on": [1]},
    {"id": 3, "description": "Проверить и улучшить отчёт", "agent": "critic", "depends_on": [2]}
  ]
}
"""

def planner(query: str) -> Plan:
    client = make_client()
    model = get_model()
    return client.chat.completions.create(
        model=model,
        response_model=Plan,
        messages=[
            {"role": "system", "content": PLANNER_SYSTEM},
            {"role": "user", "content": f"Запрос пользователя: {query}"}
        ],
        temperature=0.0,
        max_retries=2,
    )