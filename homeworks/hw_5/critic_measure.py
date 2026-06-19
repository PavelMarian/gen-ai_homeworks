import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from critic import critic
from schemas_pwc import Plan, SubQuestion, WorkerAnswer, Verdict

FAKE_BROKEN = [
    {
        "name": "Арифметика без calculate",
        "question": "Во сколько раз USD подорожал?",
        "plan": Plan(
            reasoning="Нужны курсы USD на две даты",
            subquestions=[
                SubQuestion(id=1, question="Курс USD на 2022-01-01", expected_tools=["get_fx_rate"]),
                SubQuestion(id=2, question="Курс USD сегодня", expected_tools=["get_fx_rate"]),
            ]
        ),
        "answers": {
            1: WorkerAnswer(
                subquestion_id=1,
                question_snippet="Курс USD на 2022-01-01",
                answer="USD=74.29",
                used_tools=["get_fx_rate"],
                raw_trace=[]
            ),
            2: WorkerAnswer(
                subquestion_id=2,
                question_snippet="Курс USD сегодня",
                answer="USD=71.91, разница=-2.38",
                used_tools=["get_fx_rate"],
                raw_trace=[]
            )
        }
    },
    {
        "name": "Выдуманное число",
        "question": "Какая инфляция в марте 2026?",
        "plan": Plan(
            reasoning="Получить инфляцию за март 2026",
            subquestions=[
                SubQuestion(id=1, question="Инфляция март 2026", expected_tools=["get_inflation"]),
            ]
        ),
        "answers": {
            1: WorkerAnswer(
                subquestion_id=1,
                question_snippet="Инфляция март 2026",
                answer="ИПЦ=15.2%",
                used_tools=["get_inflation"],  # но инструмент вернёт ошибку, если данных нет
                raw_trace=[]
            )
        }
    },
    {
        "name": "Несогласованные данные",
        "question": "Сравни курс USD и EUR",
        "plan": Plan(
            reasoning="Получить курсы",
            subquestions=[
                SubQuestion(id=1, question="Курс USD", expected_tools=["get_fx_rate"]),
                SubQuestion(id=2, question="Курс EUR", expected_tools=["get_fx_rate"], depends_on=[1]),
            ]
        ),
        "answers": {
            1: WorkerAnswer(
                subquestion_id=1,
                question_snippet="Курс USD",
                answer="USD=71.91",
                used_tools=["get_fx_rate"],
                raw_trace=[]
            ),
            2: WorkerAnswer(
                subquestion_id=2,
                question_snippet="Курс EUR",
                answer="EUR=85.3",
                used_tools=["get_fx_rate"],
                raw_trace=[]
            )
        }
    },
    {
        "name": "Неполный план",
        "question": "Какая реальная ставка?",
        "plan": Plan(
            reasoning="Нужна ключевая ставка и инфляция",
            subquestions=[
                SubQuestion(id=1, question="Ключевая ставка", expected_tools=["get_key_rate"]),
                # нет подвопроса про инфляцию!
            ]
        ),
        "answers": {
            1: WorkerAnswer(
                subquestion_id=1,
                question_snippet="Ключевая ставка",
                answer="16%",
                used_tools=["get_key_rate"],
                raw_trace=[]
            )
        }
    },
    {
        "name": "Галлюцинация инструмента",
        "question": "Накопленная инфляция",
        "plan": Plan(
            reasoning="Нужна накопленная инфляция",
            subquestions=[
                SubQuestion(id=1, question="Накопленная инфляция с 2022", expected_tools=["get_cumulative_inflation"]),
            ]
        ),
        "answers": {
            1: WorkerAnswer(
                subquestion_id=1,
                question_snippet="Накопленная инфляция",
                answer="35%",
                used_tools=["get_cumulative_inflation"],
                raw_trace=[]
            )
        }
    }
]

def measure():
    results = []
    for case in FAKE_BROKEN:
        row = {"name": case["name"], "temp_0_0": 0, "temp_0_7": 0}
        for _ in range(10):
            verdict = critic(
                question=case["question"],
                plan=case["plan"],
                answers=case["answers"],
                temperature=0.0
            )
            if verdict.ok:
                row["temp_0_0"] += 1
        for _ in range(10):
            verdict = critic(
                question=case["question"],
                plan=case["plan"],
                answers=case["answers"],
                temperature=0.7
            )
            if verdict.ok:
                row["temp_0_7"] += 1
        results.append(row)
    print("Битый кейс\t\tT=0.0, ложных принятий\tT=0.7, ложных принятий")
    for r in results:
        print(f"{r['name']:30}\t{r['temp_0_0']}/10\t\t\t{r['temp_0_7']}/10")

if __name__ == "__main__":
    measure()