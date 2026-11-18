import io, json, re, csv
from html import escape

def to_json(p): return json.dumps({k: p.get(k) for k in ("topic", "depth", "sources", "findings", "factchecks", "report")}, indent=2, default=str)

def to_csv(p):
    b = io.StringIO(); cols = ["id", "title", "url", "publisher", "published_date", "source_type", "ml_prob", "score", "label"]
    w = csv.DictWriter(b, cols, extrasaction="ignore"); w.writeheader(); w.writerows(p["sources"]); return b.getvalue()

def to_docx(md):
    from docx import Document
    d = Document()
    for l in md.splitlines():
        if l.startswith("# "): d.add_heading(l[2:], 0)
        elif l.startswith("## "): d.add_heading(l[3:], 1)
        elif l.startswith("- "): d.add_paragraph(re.sub(r"[_*]", "", l[2:]), style="List Bullet")
        elif l.strip(): d.add_paragraph(re.sub(r"[_*]", "", l))
    b = io.BytesIO(); d.save(b); return b.getvalue()

def to_pdf(md):
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
    ss = getSampleStyleSheet(); b = io.BytesIO(); el = []
    for l in md.splitlines():
        if not l.strip(): el.append(Spacer(1, 6)); continue
        t = re.sub(r"(https?://\S+)", r'<a href="\1" color="blue">\1</a>', escape(re.sub(r"[_*]", "", l.lstrip("#- "))))
        el.append(Paragraph(t, ss["Title"] if l.startswith("# ") else ss["Heading2"] if l.startswith("## ") else ss["BodyText"]))
    SimpleDocTemplate(b, title="Research Report").build(el); return b.getvalue()
