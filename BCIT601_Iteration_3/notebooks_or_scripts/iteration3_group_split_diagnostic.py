"""Iteration 3 diagnostic: compare row split with duplicate-group split.

This script does not replace the required Iteration 2 80/20 stratified result.
It reports how repeated request text can affect evaluation. Identical cleaned
texts are kept in the same partition to reduce train/test leakage.
"""
from pathlib import Path
import json
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "processed_service_requests.csv"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

df = pd.read_csv(DATA).dropna(subset=["request_text_clean", "category"]).copy()
groups = df["request_text_clean"].astype(str)
splitter = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
train_idx, test_idx = next(splitter.split(df, df["category"], groups=groups))
train, test = df.iloc[train_idx], df.iloc[test_idx]

# Check whether the group split retains every class in both partitions.
train_labels = set(train["category"])
test_labels = set(test["category"])
missing_train = sorted(set(df["category"]) - train_labels)
missing_test = sorted(set(df["category"]) - test_labels)

result = {
    "method": "GroupShuffleSplit by exact cleaned request text",
    "random_state": 42,
    "train_rows": int(len(train)),
    "test_rows": int(len(test)),
    "train_unique_texts": int(train["request_text_clean"].nunique()),
    "test_unique_texts": int(test["request_text_clean"].nunique()),
    "overlapping_text_groups": int(
        len(set(train["request_text_clean"]) & set(test["request_text_clean"]))
    ),
    "missing_train_categories": missing_train,
    "missing_test_categories": missing_test,
}

if not missing_train and not missing_test:
    vec = TfidfVectorizer()
    X_train = vec.fit_transform(train["request_text_clean"].astype(str))
    X_test = vec.transform(test["request_text_clean"].astype(str))
    clf = LogisticRegression(max_iter=2000)
    clf.fit(X_train, train["category"])
    pred = clf.predict(X_test)
    result["accuracy"] = float(accuracy_score(test["category"], pred))
    result["macro_f1"] = float(f1_score(test["category"], pred, average="macro"))
    result["classification_report"] = classification_report(
        test["category"], pred, zero_division=0, output_dict=True
    )
else:
    result["evaluation_note"] = (
        "This single grouped split did not contain every category in both sets. "
        "Do not compare its metrics directly; use repeated stratified group "
        "cross-validation or collect more distinct examples."
    )

(OUT / "iteration3_group_split_diagnostic.json").write_text(
    json.dumps(result, indent=2), encoding="utf-8"
)
print(json.dumps(result, indent=2))
