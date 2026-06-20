## Запуск

Индексация отзывов для RAG

```

python pipeline.py ingest input/reviews_sample.csv

```



Прогон на одном товаре (тест)

```

python pipeline.py ask "Что говорят о продукте B00014IVPQ?"

```



Прогон на всех 15-ти товарах

```

python eval.py

```



## Где что лежит

```

project/
├── .env.example                  # пример переменных окружения
├── requirements.txt              # зависимости
├── README.md
├── schema.py                     # Pydantic-модели с валидаторами
├── rag.py                        # RAG
├── tools.py                      # инструменты агента
├── planner.py                    # планировщик – генерирует план задач
├── researcher.py                 # агент-исследователь с тулами
├── analyst.py                    # агент-аналитик (генерирует отчёт)
├── critic.py                     # агент-критик (исправляет отчёт)
├── judge.py                      # LLM-as-judge – проверяет галлюцинации через RAG
├── orchestrator.py               # оркестратор – запускает план, собирает трейс
├── pipeline.py                   # CLI: ingest и ask
├── eval.py                       # прогон по 15 продуктам, подсчёт метрик
├── llm\_client.py                 │
├── input/                        # исходные данные
│   └── reviews\_sample.csv        # сэмпл отзывов (750 отзывов по 15 продуктам)
│
├── output/                       # результаты
&#x20;   ├── eval\_results.csv          # таблица оценок по 15 продуктам
&#x20;   ├── eval\_summary.json         # сводка метрик (pass-rate, галлюцинации)
&#x20;   ├── product\_\*/                # артефакты по каждому продукту
&#x20;   │   ├── report.json           # финальный отчёт
&#x20;   │   ├── trace.json            # трассировка
&#x20;   │   ├── plan.json             # сгенерированный план
&#x20;   │   └── judge\_report.json     # оценка судьи
&#x20;   └── report.json / plan.json   # (при одиночном запросе)

```

