from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from critic import critic
from llm_client import get_model, make_raw_client
from planner import planner
from schemas_pwc import Plan, SubQuestion, WorkerAnswer
from worker import worker
from concurrent.futures import ThreadPoolExecutor, as_completed


VALID_TOOLS = {"get_fx_rate", "get_key_rate", "get_inflation", "calculate"}

def validate_plan(plan: Plan) -> list[str]:
    errors = []
    for sq in plan.subquestions:
        for tool in sq.expected_tools:
            if tool not in VALID_TOOLS:
                errors.append(f"Подвопрос {sq.id}: недопустимый инструмент '{tool}'")
    return errors


def _topological_sort(subqs: list[SubQuestion]) -> list[SubQuestion]:
    by_id = {s.id: s for s in subqs}
    ordered: list[SubQuestion] = []
    visited: set[int] = set()

    def visit(node_id: int, path: list[int]):
        if node_id in visited:
            return None
        if node_id in path:
            raise ValueError(f"Цикл в depends_on : {path + node[ids]}")
        if node_id not in by_id:
            return None
        for dep in by_id[node_id].depends_on:
            visit(dep, path + [node_id])
        visited.add(node_id)
        ordered.append(by_id[node_id])

    for sq in subqs:
        visit(sq.id, [])
    return ordered

def _topological_levels(subqs: list[SubQuestion]) -> list[list[SubQuestion]]:
    """Вернуть список уровней (список списков подвопросов). Внутри уровня нет зависимостей."""
    by_id = {sq.id: sq for sq in subqs}
    indegree = {sq.id: len(sq.depends_on) for sq in subqs}
    adj = {sq.id: [] for sq in subqs}
    for sq in subqs:
        for dep in sq.depends_on:
            if dep in by_id:
                adj[dep].append(sq.id)
    levels = []
    queue = [sid for sid, deg in indegree.items() if deg == 0]
    while queue:
        level = [by_id[sid] for sid in queue]
        levels.append(level)
        next_queue = []
        for sid in queue:
            for dep_id in adj.get(sid, []):
                indegree[dep_id] -= 1
                if indegree[dep_id] == 0:
                    next_queue.append(dep_id)
        queue = next_queue
    if any(deg > 0 for deg in indegree.values()):
        raise ValueError("Цикл в depends_on")
    return levels


def _synthesize(
    question: str,
    plan: Plan,
    answers: dict[int, WorkerAnswer],
) -> str:
    parts = [f"{i}. {answers[i].answer}" for i in sorted(answers)]
    combined = "\n".join(parts)

    prompt = f"""На основе следующих фактов дай краткий ответ (1-2 предложения) на вопрос пользователя.

    Исходный вопрос: {question}

    Факты:
    {combined}

    Ответ:"""

    client = make_raw_client()
    try:
        resp = client.chat.completions.create(
            model=get_model(),
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        return resp.choices[0].message.content.strip()
    except Exception:
        return " · ".join(answers[i].answer for i in sorted(answers))

def run_pwc(
    question: str, *, max_iter: int = 3, verbose: bool = True, validate: bool = False
) -> dict[str, Any]:
    """Запустить цикл Планировщик-Исполнитель-Критик."""
    trace: list[dict[str, Any]] = []
    errors = []
    plan = None

    for attempt in range(2):
        feedback = None if plan is None else f"Ошибки в плане: {errors}"
        plan = planner(question, feedback=feedback)
        if validate:
            errors = validate_plan(plan)
            if not errors:
                break
            continue
        else:
            break

    else:
        if validate:
            print("Валидация не удалась после двух попыток, продолжаем с невалидным планом", file=sys.stderr)

    if verbose:
        print(f"\n[plan] {plan.reasoning}")
        for sq in plan.subquestions:
            print(f"  {sq.id}. [{','.join(sq.expected_tools)}] {sq.question}")

    answers: dict[int, WorkerAnswer] = {}

    for iter_num in range(1, max_iter + 1):
        levels = _topological_levels(plan.subquestions)
        for level in levels:
            with ThreadPoolExecutor(max_workers=len(level)) as executor:
                futures = {executor.submit(worker, sq, answers): sq.id for sq in level}
                for future in as_completed(futures):
                    sq_id = futures[future]
                    answers[sq_id] = future.result()
                    if verbose:
                        ans = answers[sq_id]
                        print(f"  [{sq_id}] → {ans.answer}   tools={ans.used_tools}")

        verdict = critic(question, plan, answers)
        trace.append({
            "iter": iter_num,
            "kind": "verdict",
            "ok": verdict.ok,
            "action": verdict.action,
            "reason": verdict.reason,
            "rework_ids": verdict.rework_ids,
        })

        if verbose:
            mark = "✅" if verdict.ok else "❌"
            print(f"  [critic {mark}] {verdict.action}: {verdict.reason}")

        # ---- Обработка вердикта ----
        if verdict.ok:
            final = _synthesize(question, plan, answers)
            return {
                "answer": final,
                "plan": plan,
                "answers": answers,
                "trace": trace,
                "iterations": iter_num,
            }

        if verdict.action == "replan":
            plan = planner(question, feedback=verdict.reason)
            continue
        elif verdict.action == "rework":
            feedback = f"Переделай подвопросы с id {verdict.rework_ids}. Причина: {verdict.reason}"
            plan = planner(question, feedback=feedback)
            continue
        else:
            break

    return {
        "answer": None,
        "error": f"не удалось получить вердикт 'accept' за {max_iter} итераций",
        "plan": plan,
        "answers": answers,
        "trace": trace,
        "iterations": max_iter,
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("query", nargs="+", help="Вопрос к агенту")
    ap.add_argument("--max-iter", type=int, default=3)
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument(
        "--trace", type=Path, default=None, help="Куда сохранить JSON-лог (если задан)"
    )
    args = ap.parse_args()

    q = " ".join(args.query)
    res = run_pwc(q, max_iter=args.max_iter, verbose=not args.quiet)

    print("\n=== ВОПРОС ===")
    print(q)
    print("\n=== ОТВЕТ ===")
    print(res.get("answer") or res.get("error"))
    print(f"\n(итераций: {res.get('iterations', '?')})")

    if args.trace:
        args.trace.write_text(
            json.dumps(
                {"query": q, **_serialize(res)},
                ensure_ascii=False,
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )
        print(f"Трейс сохранён: {args.trace}")


def _serialize(res: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in res.items():
        if k == "plan" and v is not None:
            out[k] = v.model_dump()
        elif k == "answers":
            out[k] = {i: a.model_dump() for i, a in v.items()}
        else:
            out[k] = v
    return out


if __name__ == "__main__":
    main()
