import pandas as pd
from planner import planner
from researcher import Researcher
from analyst import Analyst
from critic import Critic
import tools

class Orchestrator:
    """
    Оркестратор с планировщиком:
    1. Генерирует план задач.
    2. Выполняет задачи в порядке зависимостей.
    3. Собирает результаты и передаёт их между агентами.
    """
    def __init__(self, df: pd.DataFrame):
        self.df = df
        tools._df = df
        self.results = {}
        self.context = {}

    def run(self, query: str) -> dict:
        plan = planner(query)
        trace = {"plan": plan.model_dump()}

        for task in sorted(plan.subtasks, key=lambda t: t.id):
            if task.depends_on:
                for dep_id in task.depends_on:
                    if dep_id not in self.results:
                        raise Exception(f"Зависимость {dep_id} не выполнена для задачи {task.id}")

            if task.agent == "researcher":
                researcher = Researcher()
                try:
                    research_data = researcher.research(query)
                    self.results[task.id] = research_data
                    self.context["research_data"] = research_data
                    trace[f"task_{task.id}"] = {
                        "agent": "researcher",
                        "result": research_data.model_dump()
                    }
                    trace[f"task_{task.id}_steps"] = researcher.trace
                except Exception as e:
                    return {"error": f"Researcher failed in task {task.id}: {e}", "trace": trace}

            elif task.agent == "analyst":
                if "research_data" not in self.context:
                    raise Exception("Нет данных от Researcher для Analyst")
                research_data = self.context["research_data"]
                analyst = Analyst()
                try:
                    report = analyst.analyze(research_data)
                    self.results[task.id] = report
                    self.context["report"] = report
                    trace[f"task_{task.id}"] = {
                        "agent": "analyst",
                        "result": report.model_dump()
                    }
                    trace[f"task_{task.id}"] = {
                        "agent": "analyst",
                        "result": report.model_dump()
                    }
                    trace[f"task_{task.id}_steps"] = analyst.trace
                except Exception as e:
                    return {"error": f"Analyst failed in task {task.id}: {e}", "trace": trace}

            elif task.agent == "critic":
                if "research_data" not in self.context or "report" not in self.context:
                    raise Exception("Нет данных для критика (нужны ResearchData и Report)")
                research_data = self.context["research_data"]
                report = self.context["report"]
                critic = Critic()
                try:
                    improved_report = critic.critique(research_data, report)
                    self.results[task.id] = improved_report
                    self.context["report"] = improved_report
                    trace[f"task_{task.id}"] = {
                        "agent": "critic",
                        "result": improved_report.model_dump()
                    }
                    trace[f"task_{task.id}"] = {
                        "agent": "critic",
                        "result": improved_report.model_dump()
                    }
                    trace[f"task_{task.id}_steps"] = critic.trace
                except Exception as e:
                    return {"error": f"Critic failed in task {task.id}: {e}", "trace": trace}

            else:
                raise Exception(f"Неизвестный агент: {task.agent}")


        if "report" in self.context:
            return {
                "answer": self.context["report"],
                "trace": trace,
                "plan": plan
            }
        else:
            return {"error": "Не удалось получить финальный отчёт", "trace": trace}

def run_mas(query: str, df: pd.DataFrame) -> dict:
    orchestrator = Orchestrator(df)
    return orchestrator.run(query)