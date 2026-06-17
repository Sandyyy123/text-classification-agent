"""FastAPI service exposing the text-classification agent.

Run:  uvicorn main:app --reload
Then: curl -X POST localhost:8000/classify -H 'content-type: application/json' \
            -d '{"text": "where is my package"}'

If FastAPI/uvicorn are not installed, run this file directly for a CLI demo:
      python main.py
"""
from __future__ import annotations

from classifier import HybridClassifier
from data import TRAIN

# Train once at startup. In production, persist centroids and load them.
CLASSIFIER = HybridClassifier(TRAIN)

try:
    from fastapi import FastAPI
    from pydantic import BaseModel

    app = FastAPI(title="Text Classification Agent", version="1.0.0")

    class ClassifyIn(BaseModel):
        text: str

    class BatchIn(BaseModel):
        texts: list[str]

    @app.get("/categories")
    def categories() -> dict:
        return {"categories": CLASSIFIER.labels}

    @app.post("/classify")
    def classify(body: ClassifyIn) -> dict:
        return CLASSIFIER.predict(body.text).as_dict()

    @app.post("/classify_batch")
    def classify_batch(body: BatchIn) -> dict:
        return {"results": [p.as_dict() for p in CLASSIFIER.predict_batch(body.texts)]}

except ImportError:  # FastAPI not installed - CLI demo fallback
    app = None


def _cli_demo() -> None:
    samples = [
        "where is my package, it never showed up",
        "i was billed twice this month",
        "the app keeps crashing on login",
        "do you offer a discount for nonprofits",
        "asdf qwer zxcv",  # ambiguous - should escalate / flag for review
    ]
    print("\nText Classification Agent - CLI demo")
    print(f"categories: {CLASSIFIER.labels}\n")
    for s in samples:
        p = CLASSIFIER.predict(s)
        flag = "  <-- needs_review" if p.needs_review else ""
        print(f"[{p.label:<10}] conf={p.confidence:.2f} via {p.backend:<9} | {s}{flag}")
    print()


if __name__ == "__main__":
    _cli_demo()
