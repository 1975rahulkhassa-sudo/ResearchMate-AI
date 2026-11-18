"""All agents as LangGraph node functions: take state, return partial state update."""
import re, datetime
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from . import llm as L, tools as T
from ml.relevance_model import predict

DEPTH = {"Quick": (4, 5, 1), "Standard": (6, 12, 2), "Deep": (10, 20, 3)}  # questions, sources, max iterations
TEMPL = ["What is {t} and how does it work?", "How is {t} currently used?", "What are the benefits of {t}?",
         "What challenges exist for {t}?", "What are the risks of {t}?", "What are the limitations of {t}?",
         "What are recent trends in {t}?", "What is the future of {t}?", "What ethical or regulatory issues surround {t}?",
         "What evidence or studies exist on {t}?", "How does {t} compare with alternatives?", "What research gaps exist in {t}?"]
QSUF = ["applications", "benefits", "challenges risks", "limitations", "future trends", "case study", "statistics", "review"]
CATS = {"benefit": r"benefit|improv|advantage|efficien|enable|reduc", "challenge": r"challenge|difficult|barrier|cost|complex",
        "risk": r"risk|threat|privacy|bias|safety|harm", "limitation": r"limit|lack|insufficient|constraint",
        "trend": r"trend|future|growth|emerging|increasing", "conflict": r"however|contrast|controvers|mixed|debate|contradict"}
logs = lambda a, s, m: [dict(agent=a, status=s, message=m)]

def sents(t): return [s.strip() for s in re.split(r"(?<=[.!?])\s+", t) if 40 < len(s.strip()) < 400]

def planner_node(st):
    topic = (st.get("topic") or "").strip()
    if len(topic) < 3: return dict(status="failed", errors=["Topic too short"], logs=logs("Planner", "failed", "Empty/invalid topic"))
    nq, _, _ = DEPTH[st["depth"]]
    data = L.llm_json(f"Topic: {topic}\nContext: {st.get('description','')}\nReturn {{\"research_questions\":[{nq} strings],\"search_queries\":[{nq} strings],\"research_plan\":[strings]}}")
    if not data or not data.get("search_queries"):
        qs = [t.format(t=topic) for t in TEMPL[:nq]]
        data = dict(research_questions=qs, search_queries=[f"{topic} {s}" for s in QSUF[:nq]] if nq <= 8 else [f"{topic} {s}" for s in QSUF] + [q[:-1] for q in qs[8:]],
                    research_plan=["Search web/academic sources", "Score & classify sources", "Extract findings", "Fact-check", "Write report"])
    return dict(questions=data["research_questions"], queries=data["search_queries"], plan=data.get("research_plan", []), iteration=0, sources=[], searched=[], errors=[],
                logs=logs("Planner", "completed", f"{len(data['research_questions'])} questions, {len(data['search_queries'])} queries (LLM={'on' if L.available() else 'off; heuristic'})"))

def researcher_node(st):
    it = st.get("iteration", 0); _, target, _ = DEPTH[st["depth"]]
    qs = [q for q in st["queries"] if q not in st["searched"]] if it == 0 else [f"{st['topic']} {s}" for s in QSUF[3 * it: 3 * it + 3]]
    srcs, seen, new_logs = list(st["sources"]), {T.norm_url(s["url"]) for s in st["sources"]}, []
    for q in qs:
        if len(srcs) >= target * 2: break
        res = (T.web_search(q, 4) if st.get("use_web", True) else []) + (T.academic_search(q, 3) if st.get("use_academic", True) else [])
        added = 0
        for r in res:
            u = T.norm_url(r["url"])
            if u in seen: new_logs += logs("Researcher", "rejected", f"Duplicate: {r['url'][:60]}"); continue
            seen.add(u); r["id"] = len(srcs) + 1; srcs.append(r); added += 1
        new_logs += logs("Researcher", "search", f"'{q}' -> {added} new sources")
    errs = [] if srcs else ["No sources retrieved (missing/failed API keys or no results). Set TAVILY_API_KEY."]
    return dict(sources=srcs, searched=st["searched"] + qs, iteration=it + 1, errors=errs, logs=new_logs + logs("Researcher", "completed", f"{len(srcs)} total sources"))

GOOD = r"\.edu|\.gov|\.ac\.|nature\.com|science|who\.int|nih|ieee|acm\.org|springer|elsevier|arxiv|doi\.org|semanticscholar|mit\.edu|oecd|worldbank|un\.org"
def evaluator_node(st):
    out = []; yr = datetime.date.today().year
    for s in st["sources"]:
        cred = 90 if s["source_type"] == "academic" else (85 if re.search(GOOD, s["url"].lower()) else 55)
        qual = min(100, len(s["content"]) / 8); m = re.search(r"(19|20)\d\d", s["published_date"] or "")
        rec = max(0, 100 - 12 * (yr - int(m.group(0)))) if m else 40
        out.append(dict(s, cred=cred, qual=qual, rec=rec))
    return dict(sources=out, logs=logs("Source Evaluator", "completed", f"Scored {len(out)} sources (relevance 40/credibility 25/quality 20/recency 15)"))

def ml_node(st):
    items = [(s["title"], s["snippet"]) for s in st["sources"]]
    probs, name = predict(st["topic"], items) if items else ([], "none")
    ss = []
    for s, p in zip(st["sources"], probs):
        sc = round(40 * p + 0.25 * s["cred"] + 0.2 * s["qual"] + 0.15 * s["rec"], 1)
        lab = "Excellent" if sc >= 80 else "Good" if sc >= 60 else "Average" if sc >= 40 else "Poor"
        ss.append(dict(s, ml_prob=round(p, 3), ml_relevant=p >= 0.5, score=sc, label=lab,
                       why=f"Estimate: relevance {p:.0%}, credibility {s['cred']:.0f}, quality {s['qual']:.0f}, recency {s['rec']:.0f}"))
    _, target, _ = DEPTH[st["depth"]]
    sel = sorted([s for s in ss if s["ml_relevant"]], key=lambda s: -s["score"])[:target]
    return dict(sources=ss, selected=[s["id"] for s in sel], model_name=name, logs=logs("ML Classifier", "completed", f"{name}: {len(sel)}/{len(ss)} relevant"))

def analysis_node(st):
    sel = [s for s in st["sources"] if s["id"] in st.get("selected", [])]
    pool = [(x, s["id"]) for s in sel for x in sents(s["content"])]
    if not pool: return dict(findings=[], analysis={}, logs=logs("Analysis", "completed", "No evidence to analyse"))
    v = TfidfVectorizer(stop_words="english").fit([x for x, _ in pool] + st["questions"])
    sim = cosine_similarity(v.transform(st["questions"] + [st["topic"]]), v.transform([x for x, _ in pool])).max(axis=0)
    finds, used = [], set()
    for i in np.argsort(-sim)[:40]:
        x, sid = pool[i]; cat = next((c for c, p in CATS.items() if re.search(p, x.lower())), "finding")
        if x in used or sim[i] < 0.05: continue
        used.add(x); finds.append(dict(finding=x, category=cat, source_ids=[sid]))
        if len(finds) >= (6 if st["depth"] == "Quick" else 14): break
    grp = {}
    for f in finds: grp.setdefault(f["category"], []).append(f)
    terms = v.get_feature_names_out()[np.argsort(-np.asarray(v.transform([x for x, _ in pool]).sum(axis=0)).ravel())[:8]].tolist()
    return dict(findings=finds, analysis=dict(by_category=grp, themes=terms), logs=logs("Analysis", "completed", f"{len(finds)} findings, themes: {', '.join(terms[:4])}"))

def factcheck_node(st):
    sel = [s for s in st["sources"] if s["id"] in st.get("selected", [])]
    pool = [(x, s["id"]) for s in sel for x in sents(s["content"])]; checks = []
    if pool and st["findings"]:
        v = TfidfVectorizer(stop_words="english").fit([x for x, _ in pool] + [f["finding"] for f in st["findings"]])
        S = cosine_similarity(v.transform([f["finding"] for f in st["findings"]]), v.transform([x for x, _ in pool]))
        for f, row in zip(st["findings"], S):
            sup = sorted({pool[j][1] for j in np.where(row >= 0.3)[0]})
            stt = "Conflicting Evidence" if f["category"] == "conflict" else "Supported" if len(sup) >= 2 else "Partially Supported" if sup else "Not Verified"
            checks.append(dict(claim=f["finding"], status=stt, source_ids=sup or f["source_ids"],
                               explanation=f"{len(sup)} source(s) contain closely matching text (TF-IDF cosine >= 0.3)." if sup else "No matching evidence found in collected sources."))
    ok = sum(c["status"] in ("Supported", "Partially Supported") for c in checks)
    return dict(factchecks=checks, logs=logs("Fact Checker", "completed", f"{ok}/{len(checks)} claims at least partially supported"))

def decide(st):
    _, target, mx = DEPTH[st["depth"]]
    weak = len(st.get("selected", [])) < max(2, target // 2)
    return "research" if weak and st["iteration"] < mx and st.get("status") != "failed" else "report"

def report_node(st):
    sel = [s for s in st["sources"] if s["id"] in st.get("selected", [])]; f = st["findings"]
    ref = lambda ids: ", ".join(f"[{i}]" for i in ids)
    sec = lambda c: "\n".join(f"- {x['finding']} {ref(x['source_ids'])}" for x in f if x["category"] == c) or "_No evidence collected for this section._"
    summ = L.llm("Write a 120-word executive summary using ONLY these findings:\n" + "\n".join(x["finding"] for x in f[:10])) if f else None
    summ = summ or (f"This report is based on {len(sel)} selected sources. Key findings: " + " ".join(x["finding"] for x in f[:3]) if f else "Insufficient evidence was collected to summarise this topic.")
    cnt = {k: sum(c["status"] == k for c in st["factchecks"]) for k in ("Supported", "Partially Supported", "Conflicting Evidence", "Not Verified")}
    md = f"""# Research Report: {st['topic']}
_Generated {datetime.date.today()} by ResearchMate AI. Source scores are system estimates, not guarantees of truth._

## Executive Summary
{summ}

## Introduction & Objectives
{st.get('description') or 'Automated research on: ' + st['topic']}

## Research Questions
""" + "\n".join(f"{i+1}. {q}" for i, q in enumerate(st["questions"])) + f"""

## Methodology
Queries were run against {'Tavily web search' if st.get('use_web', True) else ''}{' and ' if st.get('use_web', True) and st.get('use_academic', True) else ''}{'Semantic Scholar/Crossref' if st.get('use_academic', True) else ''}. Sources were deduplicated, scored (relevance 40%, credibility 25%, quality 20%, recency 15%), classified by a {st.get('model_name')} relevance model, analysed, and fact-checked by text similarity. Iterations: {st['iteration']}.

## Main Findings
{sec('finding')}

## Benefits
{sec('benefit')}

## Challenges
{sec('challenge')}

## Risks
{sec('risk')}

## Limitations
{sec('limitation')}

## Trends & Future Scope
{sec('trend')}

## Conflicting Evidence
{sec('conflict')}

## Fact-Checking Summary
""" + ", ".join(f"{k}: {v}" for k, v in cnt.items()) + "\n\n## Conclusion\n" + ("Findings above are limited to the collected sources; see fact-check statuses before relying on any claim." if f else "No conclusion can be drawn without evidence.") + "\n\n## References\n" + "\n".join(f"[{s['id']}] {s['title']}. {s['publisher']}. {s['published_date']}. {s['url']}" for s in sel)
    return dict(report=md, status="completed", logs=logs("Report Writer", "completed", "Report generated"))

def chat(question, proj):
    sel = proj["sources"]; pool = [(x, s["id"], s["title"]) for s in sel for x in sents(s["content"])]
    if not pool: return "No source material is stored for this research."
    v = TfidfVectorizer(stop_words="english").fit([x for x, _, _ in pool] + [question])
    sc = cosine_similarity(v.transform([question]), v.transform([x for x, _, _ in pool]))[0]; top = np.argsort(-sc)[:5]
    if sc[top[0]] < 0.05: return "That is not covered by the collected sources."
    ctx = "\n".join(f"[{pool[i][1]}] {pool[i][0]}" for i in top)
    a = L.llm(f"Answer using ONLY this context; say so if the answer is not in it. Cite [id].\nContext:\n{ctx}\n\nQuestion: {question}")
    return a or "**(LLM not configured - showing most relevant passages)**\n\n" + ctx
