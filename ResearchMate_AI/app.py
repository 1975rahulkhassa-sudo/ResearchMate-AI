"""ResearchMate AI - Streamlit UI. Run: streamlit run app.py"""
import json, os
from pathlib import Path
import pandas as pd, plotly.express as px, streamlit as st
from core import db, agents, export, llm
from ml import train_model
from ml.relevance_model import MODEL

st.set_page_config("ResearchMate AI", "🔬", layout="wide")
st.markdown("""<style>.block-container{padding-top:1.5rem}div[data-testid=stMetric]{background:#f4f5fb;border-radius:12px;padding:12px 16px;border:1px solid #e3e5f3}
.badge{padding:2px 10px;border-radius:12px;font-size:12px;color:#fff}h1,h2,h3{letter-spacing:-.02em}</style>""", unsafe_allow_html=True)
if not MODEL.exists(): train_model.train()
metrics = json.loads((MODEL.parent / "metrics.json").read_text())
BADGE = {"Supported": "#16a34a", "Partially Supported": "#d97706", "Conflicting Evidence": "#dc2626", "Not Verified": "#6b7280"}

st.sidebar.title("🔬 ResearchMate AI")
page = st.sidebar.radio("Navigate", ["Dashboard", "New Research", "Research History", "Chat with Research", "ML Analytics", "Agent Activity", "Settings"])
st.sidebar.caption(f"LLM: {'configured' if llm.available() else 'not set (heuristic mode)'} · Search: {'configured' if os.getenv('TAVILY_API_KEY') else 'not set'}")

def show(p):
    t = st.tabs(["Overview", "Questions", "Findings", "Sources", "Academic", "Fact Checking", "Final Report"])
    srcs = p["sources"]; sel = [s for s in srcs if s.get("ml_relevant")]
    with t[0]:
        c = st.columns(3); c[0].metric("Sources collected", len(srcs)); c[1].metric("ML-relevant", len(sel)); c[2].metric("Findings", len(p["findings"]))
        st.write(p.get("description") or p["topic"])
    with t[1]:
        for i, q in enumerate(p.get("questions") or [], 1): st.write(f"{i}. {q}")
        if not p.get("questions"): st.info("Questions are stored in the report's Research Questions section.")
    with t[2]:
        for f in p["findings"]:
            ids = f.get("source_ids"); ids = json.loads(ids) if isinstance(ids, str) else ids
            st.markdown(f"**{f.get('category') or f.get('explanation')}** · {f['finding']}  \n<small>Sources: {ids}</small>", unsafe_allow_html=True)
    def src_table(rows):
        if not rows: return st.info("No sources of this type.")
        df = pd.DataFrame(rows)[["id", "title", "publisher", "published_date", "source_type", "ml_prob", "score", "label", "url"]]
        sort = st.selectbox("Sort by", ["score", "ml_prob", "id"], key=f"s{id(rows)}"); st.dataframe(df.sort_values(sort, ascending=sort == "id"), use_container_width=True, hide_index=True)
        for s in rows:
            with st.expander(f"[{s['id']}] {s['title']} — {s['label']} ({s['score']})"): st.write(s["snippet"]); st.caption(s["why"]); st.link_button("Open source", s["url"])
    with t[3]: src_table(srcs)
    with t[4]: src_table([s for s in srcs if s["source_type"] == "academic"])
    with t[5]:
        for c in p["factchecks"]:
            st.markdown(f"<span class='badge' style='background:{BADGE[c['status']]}'>{c['status']}</span> {c['claim']}", unsafe_allow_html=True); st.caption(c["explanation"])
    with t[6]:
        st.markdown(p["report"]); a, b, c, d, e = st.columns(5); n = p["topic"][:30].replace(" ", "_")
        a.download_button("PDF", export.to_pdf(p["report"]), f"{n}.pdf"); b.download_button("DOCX", export.to_docx(p["report"]), f"{n}.docx")
        c.download_button("Markdown", p["report"], f"{n}.md"); d.download_button("JSON", export.to_json(p), f"{n}.json"); e.download_button("Sources CSV", export.to_csv(p), f"{n}.csv")

if page == "Dashboard":
    st.title("Dashboard"); ps = db.list_projects()
    if not ps: st.info("No research yet. Start one from **New Research**.")
    else:
        full = [db.get_project(p["id"]) for p in ps]; all_s = [s for f in full for s in f["sources"]]
        c = st.columns(4); c[0].metric("Projects", len(ps)); c[1].metric("Sources", len(all_s)); c[2].metric("Reports", sum(bool(f["report"]) for f in full))
        c[3].metric("Avg ML relevance", f"{sum(s['ml_prob'] for s in all_s) / max(1, len(all_s)):.0%}")
        df = pd.DataFrame(ps); df["day"] = df["created"].str[:10]; st.plotly_chart(px.bar(df.groupby("day").size().reset_index(name="projects"), x="day", y="projects", title="Research activity"), use_container_width=True)
        st.dataframe(df[["id", "topic", "depth", "status", "n_sources", "created"]].head(10), hide_index=True, use_container_width=True)
    st.caption(f"ML model accuracy on held-out sample test set: {metrics['accuracy']:.0%} (n={metrics['test_size']})")

elif page == "New Research":
    st.title("New Research")
    with st.form("f"):
        topic = st.text_input("Topic or research question"); desc = st.text_area("Description (optional)")
        c = st.columns(3); depth = c[0].selectbox("Depth", ["Quick", "Standard", "Deep"], 1); rtype = c[1].selectbox("Type", ["General", "Academic", "Market", "Technical"])
        web = c[2].toggle("Web search", True); acad = c[2].toggle("Academic research", True); go = st.form_submit_button("Start research", type="primary")
    if go:
        if len(topic.strip()) < 3: st.error("Please enter a topic (at least 3 characters).")
        elif not (web or acad): st.error("Enable at least one search source.")
        else:
            if not os.getenv("TAVILY_API_KEY") and web: st.warning("TAVILY_API_KEY missing: web search returns nothing. Academic search still works.")
            from workflow.graph import build
            state = dict(topic=topic.strip(), description=desc, depth=depth, rtype=rtype, use_web=web, use_academic=acad, logs=[]); box = st.status("Running agents…", expanded=True)
            try:
                for upd in build().stream(state, stream_mode="updates"):
                    for node, out in upd.items():
                        state.update({k: v for k, v in out.items() if k != "logs"}); state["logs"] += out.get("logs", [])
                        box.write(f"**{node}** — " + (out["logs"][-1]["message"] if out.get("logs") else "done"))
                box.update(label="Finished", state="complete")
                if state.get("status") == "completed": st.session_state["last"] = state["project_id"]; st.success("Saved.")
                for e in state.get("errors", []): st.error(e)
            except Exception as e: box.update(label="Failed", state="error"); st.error(f"Workflow error: {type(e).__name__}: {e}")
    if "last" in st.session_state:
        p = db.get_project(st.session_state["last"])
        if p: show(p)

elif page == "Research History":
    st.title("Research History"); ps = db.list_projects()
    if not ps: st.info("Nothing here yet.")
    for p in ps:
        with st.expander(f"#{p['id']} {p['topic']} · {p['depth']} · {p['n_sources']} sources"):
            c = st.columns(2)
            if c[0].button("Open", key=f"o{p['id']}"): st.session_state["open"] = p["id"]
            if c[1].button("Delete", key=f"d{p['id']}"): db.delete_project(p["id"]); st.rerun()
    if st.session_state.get("open"):
        pr = db.get_project(st.session_state["open"])
        if pr: show(pr)

elif page == "Chat with Research":
    st.title("Chat with Research"); ps = db.list_projects()
    if not ps: st.info("Run a research first.")
    else:
        pid = st.selectbox("Project", [p["id"] for p in ps], format_func=lambda i: next(p["topic"] for p in ps if p["id"] == i)); proj = db.get_project(pid)
        h = st.session_state.setdefault(f"chat{pid}", [])
        for m in h: st.chat_message(m[0]).write(m[1])
        if q := st.chat_input("Ask about this research…"):
            h += [("user", q), ("assistant", agents.chat(q, proj))]; st.rerun()

elif page == "ML Analytics":
    st.title("ML Analytics"); c = st.columns(4)
    for col, k in zip(c, ["accuracy", "precision", "recall", "f1"]): col.metric(k.title(), f"{metrics[k]:.2f}")
    st.write(f"**{metrics['model']}** · dataset {metrics['dataset_size']} rows (train {metrics['train_size']} / test {metrics['test_size']}) · {metrics['n_features']} TF-IDF features")
    st.warning(metrics["note"]); cm = metrics["confusion_matrix"]
    st.plotly_chart(px.imshow(cm, text_auto=True, x=["Pred irrelevant", "Pred relevant"], y=["Actual irrelevant", "Actual relevant"], title="Confusion matrix"))
    if st.button("Retrain model"): train_model.train(); st.rerun()

elif page == "Agent Activity":
    st.title("Agent Activity"); ps = db.list_projects()
    if not ps: st.info("No runs yet.")
    else:
        pid = st.selectbox("Run", [p["id"] for p in ps], format_func=lambda i: next(p["topic"] for p in ps if p["id"] == i))
        st.dataframe(pd.DataFrame(db.get_project(pid)["logs"])[["agent", "status", "message", "ts"]], hide_index=True, use_container_width=True)

else:
    st.title("Settings"); st.write("Configure via `.env` (see `.env.example`). Keys are never displayed.")
    for k in ["OPENAI_API_KEY", "GEMINI_API_KEY", "TAVILY_API_KEY"]: st.write(f"{k}: {'✅ set' if os.getenv(k) else '❌ not set'}")
    st.write("Provider:", os.getenv("LLM_PROVIDER", "openai"))
