# ResearchMate AI — Multi-Agent Research Assistant
Streamlit + LangGraph app: Planner → Researcher → Source Evaluator → ML Relevance Classifier → Analysis → Fact-Checker → (loop if evidence weak) → Report Writer → SQLite.

## Run locally
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env        # add TAVILY_API_KEY and OPENAI_API_KEY or GEMINI_API_KEY
python -m ml.train_model    # optional; app trains on first start
streamlit run app.py
pytest -q tests
```
## Go live (free): Streamlit Community Cloud
Push to GitHub (`.env` is git-ignored) → share.streamlit.io → New app → `app.py` → Settings → Secrets: paste the keys as `TAVILY_API_KEY = "..."` etc. Note: SQLite on Streamlit Cloud resets on redeploy; use PostgreSQL for persistence (SQL in `core/db.py` is plain).

## Modes & honesty notes
- No Tavily key → web search returns nothing (no fake results); academic search (Semantic Scholar/Crossref) needs no key.
- No LLM key → heuristic planner, extractive analysis, template report/chat (labelled).
- ML model: TF-IDF + Logistic Regression on `ml/dataset.csv` (24 sample rows). Metrics (see ML Analytics) come from a held-out split of this *small* dataset; expand the CSV for meaningful numbers.
- Fact-checking is TF-IDF similarity against collected sources, an estimate, not proof. Source scores = 40% relevance, 25% credibility (domain heuristic), 20% content quality, 15% recency.
- Only API-provided content/abstracts are used; pages are not scraped.

## Structure
`app.py` UI · `core/` llm, tools, agents, db, export · `workflow/graph.py` LangGraph · `ml/` dataset, training, model · `tests/`
