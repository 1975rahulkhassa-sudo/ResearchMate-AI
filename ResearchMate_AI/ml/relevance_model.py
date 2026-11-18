"""TF-IDF + Logistic Regression relevance model (inference + shared feature builder)."""
import re, joblib
from pathlib import Path
MODEL = Path(__file__).parent / "saved_models" / "model.joblib"
_t = lambda s: set(re.findall(r"[a-z0-9]+", s.lower()))

def overlap(q, title, desc):
    qt = _t(q); return len(qt & _t(title + " " + desc)) / max(1, len(qt))

def feat(q, title, desc):  # text + query-overlap bucket token so the model sees query/source match
    return f"{title} {desc} ovl{min(4, int(overlap(q, title, desc) * 5))}"

def predict(query, items):
    """items: [(title, desc)] -> (probabilities, model_name). Falls back to word overlap if no trained model."""
    try: pipe = joblib.load(MODEL)
    except Exception: pipe = None
    if pipe is None:
        return [overlap(query, t, d) for t, d in items], "overlap-fallback"
    return [float(pipe.predict_proba([feat(query, t, d)])[0][1]) for t, d in items], "tfidf-logreg"
