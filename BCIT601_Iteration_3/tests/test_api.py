from fastapi.testclient import TestClient
from api.app import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["model_loaded"] is True


def test_categories_endpoint():
    response = client.get("/categories")
    assert response.status_code == 200
    assert set(response.json()["categories"]) == {"road", "waste", "water", "safety"}


def test_five_valid_prediction_examples():
    examples = [
        "There is a large pothole on the road",
        "The rubbish collection was missed",
        "Water is leaking from a pipe",
        "There is broken glass creating a public safety risk",
        "A street light is not working near the crossing",
    ]
    for text in examples:
        response = client.post("/predict", json={"request_text": text})
        assert response.status_code == 200
        body = response.json()
        assert body["predicted_category"] in {"road", "waste", "water", "safety"}
        assert 0.0 <= body["confidence"] <= 1.0
        assert body["request_text"] == text
        assert isinstance(body["review_recommended"], bool)


def test_missing_request_text_is_rejected():
    response = client.post("/predict", json={})
    assert response.status_code == 422


def test_empty_request_text_is_rejected():
    response = client.post("/predict", json={"request_text": ""})
    assert response.status_code == 422


def test_whitespace_only_request_text_is_rejected():
    response = client.post("/predict", json={"request_text": "   "})
    assert response.status_code == 422


def test_non_string_request_text_is_rejected():
    response = client.post("/predict", json={"request_text": 123})
    assert response.status_code == 422


def test_extra_json_field_is_rejected():
    response = client.post(
        "/predict",
        json={"request_text": "Pothole outside school", "unexpected": "value"},
    )
    assert response.status_code == 422


def test_oversized_request_text_is_rejected():
    response = client.post("/predict", json={"request_text": "a" * 2001})
    assert response.status_code == 422


def test_invalid_json_content_type_is_rejected():
    response = client.post(
        "/predict",
        content="not-json",
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 422
