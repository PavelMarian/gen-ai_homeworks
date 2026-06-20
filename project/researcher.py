import json
import re
from typing import Any, Dict, List, Optional
from llm_client import make_raw_client, get_model
from schema import ResearchData
from tools import search_reviews, get_product_stats, extract_aspects

RESEARCHER_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_reviews",
            "description": "Находит релевантные отзывы по текстовому запросу. Возвращает список текстов.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Поисковый запрос"},
                    "k": {"type": "integer", "description": "Количество результатов (по умолчанию 5)"}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_product_stats",
            "description": "Возвращает средний рейтинг и количество отзывов для продукта по его ID.",
            "parameters": {
                "type": "object",
                "properties": {
                    "product_id": {"type": "string", "description": "ID продукта (например, B001E4KFG0)"}
                },
                "required": ["product_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "extract_aspects",
            "description": "Извлекает плюсы и минусы из списка текстов отзывов.",
            "parameters": {
                "type": "object",
                "properties": {
                    "texts": {"type": "array", "items": {"type": "string"}, "description": "Список текстов отзывов"}
                },
                "required": ["texts"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "finish_research",
            "description": "Вызывается, когда собрано достаточно информации. Передаёт собранные данные для передачи Analyst.",
            "parameters": {
                "type": "object",
                "properties": {
                    "product_id": {"type": "string"},
                    "texts": {"type": "array", "items": {"type": "string"}},
                    "avg_rating": {"type": "number", "minimum": 1, "maximum": 5},
                    "review_count": {"type": "integer"},
                    "pros": {"type": "array", "items": {"type": "string"}},
                    "cons": {"type": "array", "items": {"type": "string"}},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1}
                },
                "required": ["product_id", "avg_rating", "review_count", "confidence"]
            }
        }
    }
]

def _to_serializable(obj):
    """Преобразует Pydantic-модели в словари для JSON-сериализации."""
    if hasattr(obj, 'model_dump'):
        return obj.model_dump()
    return obj

TOOLS_IMPL = {
    "search_reviews": search_reviews,
    "get_product_stats": get_product_stats,
    "extract_aspects": extract_aspects,
}

SYSTEM_PROMPT = """Ты — Researcher, агент, собирающий информацию о продукте по запросу пользователя.

У тебя есть инструменты:
- search_reviews(query, k): найти отзывы по текстовому запросу.
- get_product_stats(product_id): получить средний рейтинг и количество отзывов.
- extract_aspects(texts): выделить плюсы и минусы из списка отзывов.
- finish_research(...): завершить сбор и передать данные дальше.

Правила:
1. Сначала определи продукт (обычно ID вида BXXXXXXXXX). Если ID не указан, используй search_reviews, чтобы найти его.
2. Собери достаточно информации: рейтинг, количество отзывов, плюсы и минусы.
3. Ты можешь вызывать инструменты в любом порядке и несколько раз, если нужно уточнить данные.
4. Когда соберёшь всё необходимое, вызови finish_research и передай итоговые данные.
5. Если данных недостаточно, можно сделать дополнительные поиски или уточнить запрос.
6. Не выдумывай факты — только из результатов инструментов.
7. В поле confidence укажи уверенность в собранных данных (0–1).
"""

class Researcher:
    def __init__(self, max_iter: int = 10):
        self.max_iter = max_iter
        self.client = make_raw_client()
        self.model = get_model()
        self.trace = []

    def research(self, query: str) -> ResearchData:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": query}
        ]
        trace = []

        for step in range(1, self.max_iter + 1):
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=RESEARCHER_TOOLS,
                tool_choice="auto",
                temperature=0.0
            )
            msg = resp.choices[0].message
            messages.append(msg.model_dump(exclude_none=True))

            if not msg.tool_calls:
                # Нет вызовов — напоминаем
                messages.append({
                    "role": "user",
                    "content": "Ты не вызвал ни одного инструмента. Используй доступные инструменты, чтобы собрать данные, и затем вызови finish_research."
                })
                continue

            for tc in msg.tool_calls:
                name = tc.function.name
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    obs = {"error": "Некорректный JSON аргументов"}
                else:
                    if name == "finish_research":
                        # Завершаем сбор данных
                        try:
                            data = ResearchData(**args)
                            trace.append({"step": step, "action": "finish_research", "data": data.model_dump()})
                            self.trace = trace
                            return data
                        except Exception as e:
                            obs = {"error": f"Ошибка валидации данных: {e}"}
                            messages.append({
                                "role": "tool",
                                "tool_call_id": tc.id,
                                "content": json.dumps(obs, ensure_ascii=False)
                            })
                            continue
                    else:
                        fn = TOOLS_IMPL.get(name)
                        if fn is None:
                            obs = {"error": f"Неизвестный инструмент: {name}"}
                        else:
                            try:
                                obs = fn(**args)
                                obs_serializable = _to_serializable(obs)
                            except Exception as e:
                                obs = {"error": f"{type(e).__name__}: {e}"}
                                obs_serializable = obs
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": json.dumps(obs_serializable, ensure_ascii=False)
                        })
                        trace.append({"step": step, "tool": name, "args": args, "result": obs_serializable})

        raise RuntimeError(f"Researcher не вызвал finish_research за {self.max_iter} шагов")