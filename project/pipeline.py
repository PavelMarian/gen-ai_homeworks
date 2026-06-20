import sys
import pandas as pd
from pathlib import Path
from rag import ingest_from_csv, hybrid_retrieve
from orchestrator import run_mas
from judge import judge_report
import json

def ask(query: str, df_path: str = "input/reviews_sample.csv"):
    df = pd.read_csv(df_path)
    result = run_mas(query, df)

    if "error" in result:
        print(f"Ошибка: {result['error']}")
        return

    report = result["answer"]
    print("\n=== ОТЧЁТ ===")
    print(f"Продукт: {report.product_id}")
    print(f"Средний рейтинг: {report.avg_rating:.2f} ({report.total_reviews_analyzed} отзывов)")
    print("Плюсы:", ", ".join(report.pros))
    print("Минусы:", ", ".join(report.cons))
    print(f"Сводка: {report.summary}")
    print(f"Уверенность: {report.confidence:.2f}")

    report_dict = report.model_dump()
    judge_result = judge_report(query, report_dict)
    print("\n=== ОЦЕНКА СУДЬИ ===")
    print(f"Галлюцинации: {'ЕСТЬ' if judge_result.overall_hallucinated else 'НЕТ'}")
    print(f"Оценка качества: {judge_result.overall_score}/5")
    print(f"Сводка: {judge_result.summary}")
    for v in judge_result.assertions:
        if not v.supported:
            print(f"  - Не подтверждено: {v.assertion} | {v.comment}")

    out_dir = Path("output")
    out_dir.mkdir(exist_ok=True)
    with open(out_dir / "report.json", "w", encoding="utf-8") as f:
        json.dump(report.model_dump(), f, indent=2, ensure_ascii=False)
    with open(out_dir / "trace.json", "w", encoding="utf-8") as f:
        json.dump(result["trace"], f, indent=2, ensure_ascii=False)
    with open(out_dir / "plan.json", "w", encoding="utf-8") as f:
        json.dump(result["plan"].model_dump(), f, indent=2, ensure_ascii=False)
    with open(out_dir / "judge_report.json", "w", encoding="utf-8") as f:
        json.dump(judge_result.model_dump(), f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python pipeline.py ingest <csv_path>")
        print("   or: python pipeline.py ask <query>")
        sys.exit(1)
    cmd = sys.argv[1]
    if cmd == "ingest":
        ingest_from_csv(sys.argv[2])
    elif cmd == "ask":
        ask(" ".join(sys.argv[2:]))
    else:
        print("Unknown command")