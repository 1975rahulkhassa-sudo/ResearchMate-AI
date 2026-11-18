"""SQLite layer (plain SQL, easy to port to PostgreSQL)."""
import sqlite3, json, os
from pathlib import Path
DB = Path(os.getenv("RM_DB", Path(__file__).parent.parent / "data" / "researchmate.db"))
S = """CREATE TABLE IF NOT EXISTS projects(id INTEGER PRIMARY KEY, topic TEXT, description TEXT, depth TEXT, rtype TEXT, status TEXT, created TEXT DEFAULT CURRENT_TIMESTAMP, completed TEXT);
CREATE TABLE IF NOT EXISTS sources(id INTEGER PRIMARY KEY, project_id INT, sid INT, title TEXT, url TEXT, publisher TEXT, published_date TEXT, source_type TEXT, snippet TEXT, content TEXT, ml_score REAL, quality_score REAL, raw TEXT);
CREATE TABLE IF NOT EXISTS findings(id INTEGER PRIMARY KEY, project_id INT, finding TEXT, explanation TEXT, source_ids TEXT);
CREATE TABLE IF NOT EXISTS factchecks(id INTEGER PRIMARY KEY, project_id INT, claim TEXT, status TEXT, explanation TEXT, sources TEXT);
CREATE TABLE IF NOT EXISTS reports(id INTEGER PRIMARY KEY, project_id INT, content TEXT, created TEXT DEFAULT CURRENT_TIMESTAMP);
CREATE TABLE IF NOT EXISTS agent_logs(id INTEGER PRIMARY KEY, project_id INT, agent TEXT, status TEXT, message TEXT, ts TEXT DEFAULT CURRENT_TIMESTAMP);"""

def conn():
    DB.parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(DB); c.row_factory = sqlite3.Row; c.executescript(S); return c

def save_project(st):
    c = conn(); cur = c.cursor()
    cur.execute("INSERT INTO projects(topic,description,depth,rtype,status,completed) VALUES(?,?,?,?,?,datetime('now'))",
                (st["topic"], st.get("description", ""), st.get("depth", ""), st.get("rtype", ""), st.get("status", "completed")))
    pid = cur.lastrowid
    for s in st.get("sources", []):
        cur.execute("INSERT INTO sources(project_id,sid,title,url,publisher,published_date,source_type,snippet,content,ml_score,quality_score,raw) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                    (pid, s["id"], s["title"], s["url"], s["publisher"], s["published_date"], s["source_type"], s["snippet"], s["content"], s.get("ml_prob"), s.get("score"), json.dumps(s)))
    for f in st.get("findings", []):
        cur.execute("INSERT INTO findings(project_id,finding,explanation,source_ids) VALUES(?,?,?,?)", (pid, f["finding"], f["category"], json.dumps(f["source_ids"])))
    for k in st.get("factchecks", []):
        cur.execute("INSERT INTO factchecks(project_id,claim,status,explanation,sources) VALUES(?,?,?,?,?)", (pid, k["claim"], k["status"], k["explanation"], json.dumps(k["source_ids"])))
    cur.execute("INSERT INTO reports(project_id,content) VALUES(?,?)", (pid, st.get("report", "")))
    for l in st.get("logs", []):
        cur.execute("INSERT INTO agent_logs(project_id,agent,status,message) VALUES(?,?,?,?)", (pid, l["agent"], l["status"], l["message"]))
    c.commit(); c.close(); return pid

def list_projects():
    c = conn(); r = [dict(x) for x in c.execute("SELECT p.*, (SELECT COUNT(*) FROM sources WHERE project_id=p.id) n_sources FROM projects p ORDER BY id DESC")]; c.close(); return r

def get_project(pid):
    c = conn(); p = c.execute("SELECT * FROM projects WHERE id=?", (pid,)).fetchone()
    if not p: c.close(); return None
    q = lambda t: [dict(x) for x in c.execute(f"SELECT * FROM {t} WHERE project_id=?", (pid,))]
    out = dict(p); out["sources"] = [json.loads(x["raw"]) for x in q("sources")]
    out["findings"] = q("findings"); out["factchecks"] = q("factchecks"); out["logs"] = q("agent_logs")
    r = q("reports"); out["report"] = r[0]["content"] if r else ""; c.close(); return out

def delete_project(pid):
    c = conn()
    for t in ("sources", "findings", "factchecks", "reports", "agent_logs"): c.execute(f"DELETE FROM {t} WHERE project_id=?", (pid,))
    c.execute("DELETE FROM projects WHERE id=?", (pid,)); c.commit(); c.close()
