"""Web + academic search tools. Real APIs only; failures return [] (never fabricated results)."""
import os, time, requests, logging
from urllib.parse import urlparse
log = logging.getLogger("tools")

def _get(url, **kw):
    for i in range(2):  # simple retry with backoff
        try:
            r = requests.get(url, timeout=20, **kw)
            if r.status_code == 429: time.sleep(2 * (i + 1)); continue
            r.raise_for_status(); return r.json()
        except Exception as e:
            log.warning("GET failed (%s)", type(e).__name__)
    return None

def web_search(query, n=5):
    key = os.getenv("TAVILY_API_KEY")
    if not key: return []
    try:
        r = requests.post("https://api.tavily.com/search", headers={"Authorization": "Bearer " + key},
                          json={"query": query, "max_results": n, "search_depth": "advanced"}, timeout=30)
        r.raise_for_status()
        return [dict(title=x.get("title", ""), url=x.get("url", ""), publisher=urlparse(x.get("url", "")).netloc,
                     published_date=x.get("published_date", "") or "", snippet=(x.get("content") or "")[:400],
                     content=x.get("content") or "", source_type="web", search_query=query)
                for x in r.json().get("results", []) if x.get("url")]
    except Exception as e:
        log.warning("Tavily failed (%s)", type(e).__name__); return []

def academic_search(query, n=5):
    out = []
    d = _get("https://api.semanticscholar.org/graph/v1/paper/search",
             params={"query": query, "limit": n, "fields": "title,url,abstract,year,venue"})
    for p in (d or {}).get("data", []):
        if p.get("abstract"):
            out.append(dict(title=p["title"], url=p.get("url", ""), publisher=p.get("venue") or "Semantic Scholar",
                            published_date=str(p.get("year") or ""), snippet=p["abstract"][:400], content=p["abstract"],
                            source_type="academic", search_query=query))
    if not out:  # fallback: Crossref
        d = _get("https://api.crossref.org/works", params={"query": query, "rows": n, "select": "title,URL,abstract,issued,container-title"})
        for p in (d or {}).get("message", {}).get("items", []):
            if p.get("abstract") and p.get("title"):
                ab = __import__("re").sub("<[^>]+>", "", p["abstract"])
                out.append(dict(title=p["title"][0], url=p["URL"], publisher=(p.get("container-title") or ["Crossref"])[0],
                                published_date=str((p.get("issued", {}).get("date-parts") or [[""]])[0][0]),
                                snippet=ab[:400], content=ab, source_type="academic", search_query=query))
    return [o for o in out if o["url"]]

def norm_url(u): return u.lower().split("#")[0].rstrip("/").replace("http://", "https://").replace("www.", "")
