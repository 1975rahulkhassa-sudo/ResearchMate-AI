"""Train + evaluate. Metrics are measured on a held-out stratified test split of the (small) sample dataset."""
import json, joblib, pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from ml.relevance_model import feat, MODEL
D = Path(__file__).parent

def train():
    df = pd.read_csv(D / "dataset.csv").fillna("")
    X = [feat(r.query, r.title, r.description) for r in df.itertuples()]
    Xtr, Xte, ytr, yte = train_test_split(X, df.label, test_size=0.3, stratify=df.label, random_state=42)
    pipe = Pipeline([("tfidf", TfidfVectorizer(ngram_range=(1, 2), stop_words="english")),
                     ("clf", LogisticRegression(max_iter=1000))]).fit(Xtr, ytr)  # vectorizer fit on train only: no leakage
    pred = pipe.predict(Xte)
    p, r, f, _ = precision_recall_fscore_support(yte, pred, average="binary", zero_division=0)
    m = dict(model="TF-IDF + Logistic Regression", dataset_size=len(df), train_size=len(Xtr), test_size=len(Xte),
             accuracy=accuracy_score(yte, pred), precision=p, recall=r, f1=f,
             confusion_matrix=confusion_matrix(yte, pred, labels=[0, 1]).tolist(),
             n_features=len(pipe["tfidf"].vocabulary_),
             note="Trained on a small sample dataset; not a large real-world benchmark.")
    MODEL.parent.mkdir(exist_ok=True)
    joblib.dump(pipe, MODEL); (MODEL.parent / "metrics.json").write_text(json.dumps(m, indent=2))
    return m

if __name__ == "__main__": print(json.dumps(train(), indent=2))
