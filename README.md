# Text Classification Agent

A small, production-shaped agent that takes **short text** and classifies it into
**predefined categories**, with a **confidence score** and a **needs-review flag**
on every prediction so low-certainty inputs are never silently mislabeled.

Built to be runnable out of the box (zero external services), with drop-in
upgrades to real embeddings and an LLM fallback.

## Architecture

```
                 ┌─────────────────────────┐
   short text ─▶ │  EmbeddingClassifier     │  fast, local, handles the bulk
                 │  (nearest-centroid)      │
                 └───────────┬─────────────┘
                             │ confidence ≥ threshold ? ─▶ return label
                             │ else escalate
                             ▼
                 ┌─────────────────────────┐
                 │  LLMClassifier (few-shot)│  ambiguous / novel inputs
                 └───────────┬─────────────┘
                             ▼
              { label, confidence, backend, needs_review, scores }
```

- **EmbeddingClassifier** - encodes text and assigns the nearest category
  centroid. Uses `sentence-transformers` if installed, otherwise a deterministic
  hashing embedder so the demo runs anywhere.
- **LLMClassifier** - few-shot prompt to an LLM for inputs the embedding backend
  is unsure about. Uses OpenAI if `OPENAI_API_KEY` is set, otherwise a
  transparent deterministic stub.
- **HybridClassifier** - routes by confidence and sets `needs_review` below a
  configurable threshold.

## Run the demo (no dependencies)

```bash
python main.py        # CLI demo over sample inputs
python evaluate.py    # per-category precision / recall / F1 on a held-out set
```

## Run the API

```bash
pip install -r requirements.txt
uvicorn main:app --reload
curl -X POST localhost:8000/classify \
     -H 'content-type: application/json' \
     -d '{"text": "where is my package"}'
```

```json
{ "text": "where is my package", "label": "shipping",
  "confidence": 0.74, "backend": "embedding", "needs_review": false }
```

Endpoints: `GET /categories`, `POST /classify`, `POST /classify_batch`.

## Adapting to your categories

Edit `data.py` with your own `(text, label)` examples - the pipeline learns
centroids from whatever schema you provide. No code changes needed to add or
rename categories.

## Production upgrades

| Demo default | Production swap |
|---|---|
| Hashing embedder | `sentence-transformers` (uncomment in `requirements.txt`) |
| LLM stub | OpenAI / Anthropic (set `OPENAI_API_KEY`) |
| In-memory centroids | Persist fitted centroids; load at startup |
| Thresholds 0.62 / 0.55 | Tune on your labeled validation set |

---

Demo by Dr. Sandeep Grover - PhD Data Science. Illustrative categories and data only.
