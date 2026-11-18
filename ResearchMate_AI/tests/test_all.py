import os, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
os.environ["RM_DB"] = tempfile.mktemp(suffix=".db")
import pytest
from core import tools, agents, db, export
from ml import train_model
from ml.relevance_model import predict
from workflow.graph import build

FAKE = lambda q, n=4: [dict(title=f"AI healthcare diagnosis {i}", url=f"https://example.org/{i}", publisher="example.org", published_date="2024",
    snippet="AI improves diagnosis in healthcare.", content="Artificial intelligence improves diagnosis accuracy in healthcare systems. Privacy risks and bias remain challenges for AI in healthcare. AI in healthcare reduces costs and improves efficiency for hospitals.",
    source_type="web", search_query=q) for i in range(3)]

def test_ml_metrics():
    m = train_model.train(); assert 0 <= m["accuracy"] <= 1 and m["dataset_size"] == 24
    p, _ = predict("AI in healthcare", [("AI diagnosis in medicine", "deep learning for hospitals"), ("Pasta recipes", "cooking")]); assert p[0] > p[1]

def test_workflow_with_mock(monkeypatch):
    monkeypatch.setattr(tools, "web_search", FAKE); monkeypatch.setattr(tools, "academic_search", lambda q, n=3: [])
    r = build().invoke(dict(topic="AI in healthcare", depth="Quick", use_web=True, use_academic=True))
    assert r["status"] == "completed" and r["project_id"] and len(r["sources"]) == 3  # duplicates removed
    assert db.get_project(r["project_id"])["report"].startswith("# Research Report"); export.to_pdf(r["report"]); export.to_docx(r["report"])

def test_empty_topic():
    assert build().invoke(dict(topic=" ", depth="Quick"))["status"] == "failed"

def test_no_results_and_loop_limit(monkeypatch):
    monkeypatch.setattr(tools, "web_search", lambda q, n=4: []); monkeypatch.setattr(tools, "academic_search", lambda q, n=3: [])
    r = build().invoke(dict(topic="zzz unknown", depth="Standard")); assert r["iteration"] <= 2 and r["errors"]

def test_chat_and_delete(monkeypatch):
    monkeypatch.setattr(tools, "web_search", FAKE); monkeypatch.setattr(tools, "academic_search", lambda q, n=3: [])
    r = build().invoke(dict(topic="AI in healthcare", depth="Quick")); p = db.get_project(r["project_id"])
    assert "[" in agents.chat("What are the privacy risks?", p); db.delete_project(r["project_id"]); assert db.get_project(r["project_id"]) is None
