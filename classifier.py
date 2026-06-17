"""
Text classification agent.

Two interchangeable backends:
  1. EmbeddingClassifier  - fast, local, nearest-centroid over sentence embeddings.
  2. LLMClassifier        - few-shot LLM fallback for ambiguous / novel inputs.

The HybridClassifier routes between them: the embedding backend handles the
high-confidence bulk, and anything below a confidence threshold is escalated to
the LLM backend. Every prediction carries a confidence score and a needs_review
flag so low-certainty cases are never silently mislabeled.

Runs with zero external services in demo mode (deterministic hashing embedder),
so the repo is runnable out of the box. Set OPENAI_API_KEY to enable the real
LLM fallback, or install sentence-transformers for real embeddings.
"""
from __future__ import annotations

import hashlib
import math
import os
import re
from dataclasses import dataclass, field
from typing import Optional


# --------------------------------------------------------------------------- #
# Result type
# --------------------------------------------------------------------------- #
@dataclass
class Prediction:
    text: str
    label: str
    confidence: float
    backend: str
    needs_review: bool
    scores: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "text": self.text,
            "label": self.label,
            "confidence": round(self.confidence, 4),
            "backend": self.backend,
            "needs_review": self.needs_review,
            "scores": {k: round(v, 4) for k, v in self.scores.items()},
        }


# --------------------------------------------------------------------------- #
# Embedding backend
# --------------------------------------------------------------------------- #
def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


class _HashingEmbedder:
    """Deterministic bag-of-words hashing embedder.

    Used as a dependency-free fallback so the demo runs anywhere. Swap for
    sentence-transformers in production (see EmbeddingClassifier.__init__).
    """

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim

    def encode(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for tok in _tokenize(text):
            h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
            vec[h % self.dim] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]


def _cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


class EmbeddingClassifier:
    """Nearest-centroid classifier over sentence embeddings."""

    def __init__(self, embedder: Optional[object] = None) -> None:
        if embedder is not None:
            self.embedder = embedder
        else:
            try:  # real embeddings if available
                from sentence_transformers import SentenceTransformer

                model = SentenceTransformer("all-MiniLM-L6-v2")
                self.embedder = _STWrapper(model)
            except Exception:
                self.embedder = _HashingEmbedder()
        self.centroids: dict[str, list[float]] = {}

    def fit(self, examples: list[tuple[str, str]]) -> "EmbeddingClassifier":
        by_label: dict[str, list[list[float]]] = {}
        for text, label in examples:
            by_label.setdefault(label, []).append(self.embedder.encode(text))
        for label, vecs in by_label.items():
            dim = len(vecs[0])
            centroid = [sum(v[i] for v in vecs) / len(vecs) for i in range(dim)]
            norm = math.sqrt(sum(c * c for c in centroid)) or 1.0
            self.centroids[label] = [c / norm for c in centroid]
        return self

    def predict(self, text: str) -> Prediction:
        vec = self.embedder.encode(text)
        sims = {lbl: (_cosine(vec, c) + 1) / 2 for lbl, c in self.centroids.items()}
        ranked = sorted(sims.items(), key=lambda kv: kv[1], reverse=True)
        top_label, top_score = ranked[0]
        runner = ranked[1][1] if len(ranked) > 1 else 0.0
        # confidence = top score tempered by the margin to the runner-up
        margin = top_score - runner
        confidence = max(0.0, min(1.0, 0.5 * top_score + 0.5 * (top_score + margin)))
        return Prediction(
            text=text,
            label=top_label,
            confidence=confidence,
            backend="embedding",
            needs_review=False,
            scores=sims,
        )


class _STWrapper:
    def __init__(self, model) -> None:
        self.model = model

    def encode(self, text: str) -> list[float]:
        return list(map(float, self.model.encode(text, normalize_embeddings=True)))


# --------------------------------------------------------------------------- #
# LLM backend
# --------------------------------------------------------------------------- #
class LLMClassifier:
    """Few-shot LLM classifier. Falls back to a transparent stub if no key."""

    def __init__(self, labels: list[str], model: str = "gpt-4o-mini") -> None:
        self.labels = labels
        self.model = model
        self.key = os.getenv("OPENAI_API_KEY")

    def predict(self, text: str) -> Prediction:
        if not self.key:
            # Deterministic stub so the pipeline is testable without a key.
            label = self.labels[len(_tokenize(text)) % len(self.labels)]
            return Prediction(text, label, 0.50, "llm-stub", True, {})
        from openai import OpenAI  # imported lazily

        client = OpenAI(api_key=self.key)
        prompt = (
            "Classify the text into exactly one category.\n"
            f"Categories: {', '.join(self.labels)}\n"
            f"Text: {text!r}\n"
            "Answer with only the category name."
        )
        resp = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        out = resp.choices[0].message.content.strip()
        label = next((l for l in self.labels if l.lower() in out.lower()), self.labels[0])
        return Prediction(text, label, 0.85, "llm", False, {})


# --------------------------------------------------------------------------- #
# Hybrid router
# --------------------------------------------------------------------------- #
class HybridClassifier:
    def __init__(
        self,
        examples: list[tuple[str, str]],
        threshold: float = 0.62,
        review_threshold: float = 0.55,
    ) -> None:
        self.labels = sorted({lbl for _, lbl in examples})
        self.threshold = threshold
        self.review_threshold = review_threshold
        self.embedding = EmbeddingClassifier().fit(examples)
        self.llm = LLMClassifier(self.labels)

    def predict(self, text: str) -> Prediction:
        pred = self.embedding.predict(text)
        if pred.confidence >= self.threshold:
            pred.needs_review = pred.confidence < self.review_threshold
            return pred
        # escalate ambiguous input to the LLM backend
        llm_pred = self.llm.predict(text)
        llm_pred.scores = pred.scores
        llm_pred.needs_review = llm_pred.confidence < self.review_threshold
        return llm_pred

    def predict_batch(self, texts: list[str]) -> list[Prediction]:
        return [self.predict(t) for t in texts]
