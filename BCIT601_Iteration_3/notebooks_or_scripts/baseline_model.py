from pathlib import Path
import json
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "processed_service_requests.csv"
MODEL_DIR = ROOT / "model"
OUT_DIR = ROOT / "outputs"
MODEL_DIR.mkdir(exist_ok=True)
OUT_DIR.mkdir(exist_ok=True)

df = pd.read_csv(DATA)
X_train, X_test, y_train, y_test = train_test_split(
    df["request_text_clean"], df["category"],
    test_size=0.20, random_state=42, stratify=df["category"]
)

vectoriser = TfidfVectorizer()
X_train_vec = vectoriser.fit_transform(X_train)
X_test_vec = vectoriser.transform(X_test)

model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)
pred = model.predict(X_test_vec)

report = classification_report(y_test, pred, digits=4)
print("Baseline accuracy:", accuracy_score(y_test, pred))
print(report)

joblib.dump(model, MODEL_DIR / "model.joblib")
joblib.dump(vectoriser, MODEL_DIR / "vectoriser.joblib")

# Five test examples as required by the brief.
examples = pd.DataFrame({
    "input_text": X_test.tolist()[:5],
    "true_label": y_test.tolist()[:5],
    "predicted_label": pred[:5]
})
examples.to_csv(OUT_DIR / "five_example_predictions.csv", index=False)

metrics = {
    "model": "LogisticRegression",
    "vectoriser": "TfidfVectorizer",
    "test_size": 0.20,
    "random_state": 42,
    "train_rows": int(len(X_train)),
    "test_rows": int(len(X_test)),
    "accuracy": float(accuracy_score(y_test, pred)),
    "macro_f1": float(__import__('sklearn.metrics').metrics.f1_score(y_test, pred, average='macro')),
    "labels": sorted(y_test.unique().tolist()),
    "classification_report": __import__('sklearn.metrics').metrics.classification_report(y_test, pred, output_dict=True),
    "confusion_matrix": confusion_matrix(y_test, pred).tolist()
}
(OUT_DIR / "baseline_metrics.json").write_text(json.dumps(metrics, indent=2), encoding='utf-8')
(OUT_DIR / "classification_report.txt").write_text(report, encoding='utf-8')

pd.DataFrame({"input_text": X_test, "true_label": y_test, "predicted_label": pred}).to_csv(OUT_DIR / "test_predictions.csv", index=False)
