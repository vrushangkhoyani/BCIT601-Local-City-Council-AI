"""Leakage-aware Iteration 3 model comparison and final model training.

Models are compared on the same 80/20 GroupShuffleSplit. Exact cleaned request
texts are assigned wholly to one partition. After evaluation, the selected
configuration is refitted on all available records for API deployment.
"""
from pathlib import Path
import json
import pandas as pd
import joblib
from sklearn.model_selection import GroupShuffleSplit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "processed_service_requests.csv"
OUT = ROOT / "outputs"
MODEL_DIR = ROOT / "model"
OUT.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

df = pd.read_csv(DATA).dropna(subset=["request_text_clean", "category"]).copy()
X = df["request_text_clean"].astype(str)
y = df["category"].astype(str)
groups = X

splitter = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
train_idx, test_idx = next(splitter.split(X, y, groups=groups))
X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
if set(y_train) != set(y) or set(y_test) != set(y):
    raise RuntimeError("Grouped split does not contain all categories in both partitions; collect more distinct examples or revise the validation design.")
if set(X_train) & set(X_test):
    raise RuntimeError("Leakage check failed: identical cleaned texts occur in both partitions.")

experiments = [
    ("Logistic Regression + word TF-IDF (1,1)", TfidfVectorizer(ngram_range=(1, 1)), LogisticRegression(max_iter=2000, C=1.0)),
    ("Logistic Regression + word TF-IDF (1,2), sublinear", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True), LogisticRegression(max_iter=2000, C=1.0)),
    ("Logistic Regression + character TF-IDF (3,5), char_wb", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True), LogisticRegression(max_iter=2000, C=1.0)),
    ("LinearSVC + word TF-IDF (1,1)", TfidfVectorizer(ngram_range=(1, 1)), LinearSVC()),
    ("LinearSVC + word TF-IDF (1,2)", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True), LinearSVC()),
    ("MultinomialNB + word TF-IDF (1,1)", TfidfVectorizer(ngram_range=(1, 1)), MultinomialNB()),
    ("ComplementNB + word TF-IDF (1,1)", TfidfVectorizer(ngram_range=(1, 1)), ComplementNB()),
]

rows = []
for name, vectoriser, model in experiments:
    train_features = vectoriser.fit_transform(X_train)
    test_features = vectoriser.transform(X_test)
    model.fit(train_features, y_train)
    predictions = model.predict(test_features)
    rows.append({
        "experiment": name,
        "accuracy": float(accuracy_score(y_test, predictions)),
        "macro_f1": float(f1_score(y_test, predictions, average="macro")),
        "features": int(train_features.shape[1]),
    })

results = pd.DataFrame(rows).sort_values(["macro_f1", "accuracy"], ascending=False).reset_index(drop=True)
results.to_csv(OUT / "iteration3_model_comparison.csv", index=False)

# The winner is selected by macro F1 first so each category contributes equally.
best_name = str(results.iloc[0]["experiment"])
if best_name != "Logistic Regression + character TF-IDF (3,5), char_wb":
    raise RuntimeError(f"Unexpected selected model: {best_name}; update the explicit final-model configuration before deployment.")

# Evaluate and save the selected model on the held-out grouped test set.
selected_vectoriser = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True)
Xtr = selected_vectoriser.fit_transform(X_train)
Xte = selected_vectoriser.transform(X_test)
selected_model = LogisticRegression(max_iter=2000, C=1.0)
selected_model.fit(Xtr, y_train)
pred = selected_model.predict(Xte)
report = classification_report(y_test, pred, digits=4, zero_division=0)
(OUT / "iteration3_selected_classification_report.txt").write_text(report, encoding="utf-8")

metrics = {
    "evaluation_method": "80/20 GroupShuffleSplit grouped by exact cleaned request_text",
    "random_state": 42,
    "train_rows": int(len(train_idx)),
    "test_rows": int(len(test_idx)),
    "train_unique_texts": int(X_train.nunique()),
    "test_unique_texts": int(X_test.nunique()),
    "overlapping_text_groups": int(len(set(X_train) & set(X_test))),
    "categories": sorted(y.unique().tolist()),
    "selected_model": best_name,
    "selection_metric": "macro F1, with accuracy as tie-breaker",
    "accuracy": float(accuracy_score(y_test, pred)),
    "macro_f1": float(f1_score(y_test, pred, average="macro")),
    "classification_report": classification_report(y_test, pred, output_dict=True, zero_division=0),
    "confusion_matrix": confusion_matrix(y_test, pred, labels=sorted(y.unique())).tolist(),
    "confusion_matrix_labels": sorted(y.unique().tolist()),
    "limitations": "One grouped holdout on a small synthetic 80-row dataset; metrics are unstable and do not establish real-world performance."
}
(OUT / "iteration3_selected_model_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

# Refit the selected pipeline on all supplied records for API use.
final_vectoriser = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True)
all_features = final_vectoriser.fit_transform(X)
final_model = LogisticRegression(max_iter=2000, C=1.0)
final_model.fit(all_features, y)
joblib.dump(final_model, MODEL_DIR / "model.joblib")
joblib.dump(final_vectoriser, MODEL_DIR / "vectoriser.joblib")

summary = {
    "selected_model": best_name,
    "selection_reason": "Highest macro F1 on the same leakage-aware grouped holdout; accuracy used as a tie-breaker.",
    "grouped_holdout_accuracy": metrics["accuracy"],
    "grouped_holdout_macro_f1": metrics["macro_f1"],
    "all_records_used_for_final_deployment_fit": int(len(df)),
    "deployment_fit_note": "The saved API model is refitted on all available records after holdout evaluation; reported holdout metrics come only from the held-out split."
}
(OUT / "model_selection_notes.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(results.to_string(index=False))
print("\nSelected model metrics:")
print(json.dumps(metrics, indent=2))
print("\nSaved final model and vectoriser for API deployment.")
