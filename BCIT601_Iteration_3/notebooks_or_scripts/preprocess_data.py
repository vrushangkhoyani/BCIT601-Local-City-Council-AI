from pathlib import Path
import pandas as pd
import re
import json

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data" / "raw" / "BCIT601_starter_service_requests_dataset.csv"
OUTPUT = ROOT / "data" / "processed" / "processed_service_requests.csv"

REQUIRED_COLUMNS = ["request_id", "request_text", "category", "suburb", "priority", "date_reported"]

def clean_text(value):
    if pd.isna(value):
        return ""
    text = str(value).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

df = pd.read_csv(INPUT)
missing_columns = [c for c in REQUIRED_COLUMNS if c not in df.columns]
if missing_columns:
    raise ValueError(f"Missing required columns: {missing_columns}")

missing_before = df[REQUIRED_COLUMNS].isna().sum().astype(int).to_dict()
exact_duplicates = int(df.duplicated().sum())
repeated_text_rows = int(df["request_text"].duplicated(keep=False).sum())
print("Rows before cleaning:", len(df))
print("Missing values before cleaning:")
print(df[REQUIRED_COLUMNS].isna().sum())
print("Exact duplicate records:", exact_duplicates)
print("Duplicate request_text rows:", repeated_text_rows)

df["request_text_clean"] = df["request_text"].apply(clean_text)
empty_text = df["request_text_clean"].eq("")
missing_target = df["category"].isna() | df["category"].astype(str).str.strip().eq("")

# Missing request text cannot be classified, so those rows are removed.
# The supplied dataset has no missing request_text or category rows.
df = df.loc[~empty_text & ~missing_target].copy()
df["category"] = df["category"].astype(str).str.strip().str.lower()

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUTPUT, index=False)
quality_summary = {
    "source_file": "data/raw/BCIT601_starter_service_requests_dataset.csv",
    "processed_file": "data/processed/processed_service_requests.csv",
    "rows_before_cleaning": int(len(pd.read_csv(INPUT))),
    "rows_after_cleaning": int(len(df)),
    "missing_values_by_column_before_cleaning": missing_before,
    "exact_duplicate_records": exact_duplicates,
    "rows_with_repeated_request_text": repeated_text_rows,
    "category_counts_after_cleaning": {str(k): int(v) for k, v in df["category"].value_counts().sort_index().items()},
    "text_cleaning": ["lowercase", "remove punctuation and non-alphanumeric noise", "normalise whitespace", "convert missing text to empty string and exclude unclassifiable empty text"],
    "limitation": "Repeated request text is retained because the records have distinct request IDs; a row-based train/test split can place identical text in both sets, so holdout performance may be optimistic."
}
(ROOT / "outputs" / "data_quality_summary.json").write_text(json.dumps(quality_summary, indent=2), encoding="utf-8")
print("Rows after cleaning:", len(df))
print("Saved processed data: data/processed/processed_service_requests.csv")
print("Saved quality summary: outputs/data_quality_summary.json")
