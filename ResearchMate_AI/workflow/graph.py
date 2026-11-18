import operator
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END
from core import agents as A, db

class State(TypedDict, total=False):
    topic: str; description: str; depth: str; rtype: str; use_web: bool; use_academic: bool
    questions: list; queries: list; plan: list; searched: list; sources: list; selected: list; model_name: str
    findings: list; analysis: dict; factchecks: list; iteration: int; report: str; status: str; errors: list; project_id: int
    logs: Annotated[list, operator.add]

def save_node(st):
    return dict(project_id=db.save_project(st), logs=[dict(agent="Database", status="completed", message="Research saved")])

def build():
    g = StateGraph(State)
    for n, f in [("planner", A.planner_node), ("researcher", A.researcher_node), ("evaluator", A.evaluator_node), ("ml", A.ml_node),
                 ("analysis", A.analysis_node), ("factcheck", A.factcheck_node), ("report", A.report_node), ("save", save_node)]:
        g.add_node(n, f)
    g.add_edge(START, "planner")
    g.add_conditional_edges("planner", lambda s: END if s.get("status") == "failed" else "researcher")
    for a, b in [("researcher", "evaluator"), ("evaluator", "ml"), ("ml", "analysis"), ("analysis", "factcheck"), ("report", "save"), ("save", END)]:
        g.add_edge(a, b)
    g.add_conditional_edges("factcheck", A.decide, {"research": "researcher", "report": "report"})
    return g.compile()
