"""FastAPI service for the BCIT601 council request classifier.

The saved model and vectoriser are loaded once when this module is imported.
No model training takes place inside an API request.
"""
from pathlib import Path
import re
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field, field_validator

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "model" / "model.joblib"
VECTORIZER_PATH = ROOT / "model" / "vectoriser.joblib"

if not MODEL_PATH.is_file() or not VECTORIZER_PATH.is_file():
    raise RuntimeError(
        "Saved model files are missing. Run the training script before starting the API."
    )

model = joblib.load(MODEL_PATH)
vectoriser = joblib.load(VECTORIZER_PATH)

app = FastAPI(
    title="BCIT601 Local City Council Service Request Classifier",
    description=(
        "A teaching prototype that classifies service-request text. "
        "Predictions support staff triage and must not replace human review."
    ),
    version="1.2.0",
)
CATEGORIES = sorted(str(label) for label in model.classes_)


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    request_text: str = Field(..., min_length=1, max_length=2000)

    @field_validator("request_text")
    @classmethod
    def text_must_not_be_whitespace(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("request_text must contain non-whitespace text.")
        return value


def clean_text(value: str) -> str:
    """Apply the same basic normalisation used by the training pipeline."""
    text = value.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": True}


@app.get("/categories")
def categories():
    return {"categories": CATEGORIES}


@app.post("/predict")
def predict(payload: PredictionRequest):
    cleaned = clean_text(payload.request_text)
    if not cleaned:
        raise HTTPException(
            status_code=422,
            detail="request_text contains no usable text after cleaning.",
        )

    features = vectoriser.transform([cleaned])
    probabilities = model.predict_proba(features)[0]
    best_index = int(probabilities.argmax())
    predicted = str(model.classes_[best_index])
    confidence = float(probabilities[best_index])

    return {
        "request_text": payload.request_text,
        "predicted_category": predicted,
        "confidence": round(confidence, 4),
        "review_recommended": confidence < 0.60,
        "notice": (
            "Confidence is a model score, not proof of correctness. "
            "Staff should review urgent, safety-related or uncertain requests."
        ),
    }
