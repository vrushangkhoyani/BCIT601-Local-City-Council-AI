"""Run API endpoint checks with TestClient and save console-style evidence."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fastapi.testclient import TestClient
from api.app import app

OUT = ROOT / "outputs" / "api_test_results_iteration3.txt"
client = TestClient(app)
lines = []

def record(label, response):
    lines.append(f"{label}: HTTP {response.status_code}")
    lines.append(response.text)

record("GET /health", client.get("/health"))
record("GET /categories", client.get("/categories"))
for i, text in enumerate([
    "There is a pothole on the road",
    "Missed rubbish collection",
    "Water is leaking from a pipe",
    "Broken glass is creating a safety risk",
    "Street light is not working",
], start=1):
    record(f"POST /predict valid example {i}", client.post("/predict", json={"request_text": text}))
record("POST /predict missing field", client.post("/predict", json={}))
record("POST /predict empty text", client.post("/predict", json={"request_text": ""}))
record("POST /predict wrong type", client.post("/predict", json={"request_text": 123}))

OUT.write_text("\\n".join(lines) + "\\n", encoding="utf-8")
print("\\n".join(lines))
print(f"Evidence saved to: {OUT}")
