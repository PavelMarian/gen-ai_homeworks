import pandas as pd
import random
from pathlib import Path
from orchestrator import run_mas
from judge import judge_report
from rag import hybrid_retrieve
import json

def extract_steps_and_tools(trace):
    """Извлекает общее количество шагов и использованные инструменты из трассировки."""
    steps = 0
    tools_set = set()
    for key, value in trace.items():
        if isinstance(key, str) and key.endswith("_steps"):
            if isinstance(value, list):
                steps += len(value)
                for step in value:
                    if isinstance(step, dict) and "tool" in step:
                        tools_set.add(step["tool"])
    return steps, list(tools_set)

def to_serializable(obj):
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    elif hasattr(obj, "dict"):
        return obj.dict()
    return obj

def save_product_artifacts(product_id: str, report, trace, plan, judge_result):
    """Сохраняет артефакты для одного продукта в папку output/product_{product_id}/"""
    out_dir = Path("output") / f"product_{product_id}"
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "report.json", "w", encoding="utf-8") as f:
        json.dump(report.model_dump(), f, indent=2, ensure_ascii=False)

    with open(out_dir / "trace.json", "w", encoding="utf-8") as f:
        json.dump(trace, f, indent=2, ensure_ascii=False)

    with open(out_dir / "plan.json", "w", encoding="utf-8") as f:
        json.dump(plan.model_dump(), f, indent=2, ensure_ascii=False)

    with open(out_dir / "judge_report.json", "w", encoding="utf-8") as f:
        json.dump(judge_result.model_dump(), f, indent=2, ensure_ascii=False)

def run_eval(sample_csv: str, n_products: int = 15):
    """Прогоняет MAS на n случайных продуктах, считает метрики с использованием судьи."""
    df = pd.read_csv(sample_csv)
    products = df['ProductId'].dropna().unique()
    selected = random.sample(list(products), min(n_products, len(products)))

    results = []
    hallucination_count = 0

    for pid in selected:
        query = f"Что говорят о продукте {pid}? Выдели плюсы и минусы."

        mas_result = run_mas(query, df)
        if "error" in mas_result:
            print(f"Ошибка для {pid}: {mas_result['error']}")
            continue

        report = mas_result["answer"]
        trace = mas_result["trace"]

        steps, tools_used = extract_steps_and_tools(trace)

        real_avg = df[df['ProductId'] == pid]['Score'].mean()
        agent_avg = report.avg_rating

        report_dict = report.model_dump()
        verdict = judge_report(query, report_dict)

        if verdict.overall_hallucinated:
            hallucination_count += 1

        numeric_ok = abs(agent_avg - real_avg) < 0.5
        pass_ = numeric_ok and not verdict.overall_hallucinated

        hallucinated_facts = [v.assertion for v in verdict.assertions if not v.supported]

        pros_str = json.dumps(report.pros, ensure_ascii=False)
        cons_str = json.dumps(report.cons, ensure_ascii=False)

        results.append({
            "product_id": pid,
            "pros": pros_str,
            "cons": cons_str,
            "real_avg": real_avg,
            "agent_avg": agent_avg,
            "numeric_ok": numeric_ok,
            "hallucinated": verdict.overall_hallucinated,
            "hallucinated_facts": hallucinated_facts,
            "judge_score": verdict.overall_score,
            "judge_comment": verdict.summary,
            "steps": steps,
            "tools_used": json.dumps(tools_used, ensure_ascii=False),
            "pass": pass_,
        })

        save_product_artifacts(pid, report, mas_result["trace"], mas_result["plan"], verdict)

    out_dir = Path("output")
    out_dir.mkdir(exist_ok=True)
    df_res = pd.DataFrame(results)
    df_res.to_csv(out_dir / "eval_results.csv", index=False)

    pass_rate = df_res['pass'].mean() * 100
    print(f"Pass-rate: {pass_rate:.1f}% ({df_res['pass'].sum()}/{len(results)})")
    print(f"Галлюцинаций: {hallucination_count} из {len(results)}")

    summary = {
        "pass_rate": pass_rate,
        "total": len(results),
        "hallucinations": hallucination_count,
        "hallucination_rate": hallucination_count / len(results) if results else 0,
    }
    with open(out_dir / "eval_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    return df_res

if __name__ == "__main__":
    run_eval("input/reviews_sample.csv", n_products=15)