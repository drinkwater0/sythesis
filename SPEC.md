# LaminoRAG — Personal Laminopathy Research Assistant

## Overview

A local RAG pipeline built around the user's growing personal corpus of laminopathy literature, with adjacent corpora in LNP delivery and bioinformatics that the system pulls from for cross-domain reasoning. Articles are chunked into raw text passages by an LLM, tagged with corpus, embedded with a biomedical model, and stored in a local vector DB.

Retrieval is **forced cross-corpus**: a query always pulls a large share from the laminopathy corpus plus a smaller share from each side corpus, so the assistant grounds primarily in the user's research domain while still surfacing relevant connections from delivery and computational literature.

The LLM client talks directly to the user's **university AI gateway** (OpenAI-compatible REST endpoint) via the `openai` SDK with a custom `base_url`. Any model the gateway exposes can be selected by changing a single config string.

Example queries:

*"Can bioinformatics approaches improve LNP targeting for laminopathy?"*

*"Read the connections between changed gene expressions in the laminopathic hearts and propose an experiment to find commonality between them."*

---

## Phase 1 Scope

- Ingest articles the user has read (PDF / pasted text / URL)
- LLM-driven chunking that emits raw text passages tagged with corpus (`laminopathy` / `lnp` / `bioinformatics`) in one pass
- Embed via `embeddings.py` (default: `neuml/pubmedbert-base-embeddings`; swappable to MedCPT asymmetric encoders by a one-line config flip), store in Chroma with corpus metadata
- Forced cross-corpus retrieval: K chunks from laminopathy + N chunks from each side corpus
- Synthesis via university AI gateway (OpenAI-compatible), with chunks grouped by corpus in the prompt
- Jupyter notebook interface

**Out of Phase 1:** PubMed fetcher, full-text retrieval, Streamlit UI, evaluation framework, vanilla-vs-cross-corpus comparison mode.

---

## Tech Stack

| Component | Choice |
|---|---|
| Language | Python 3.11+ |
| Chunker | LLM call (one per article, returns chunks + corpus tags) |
| Embeddings | `embeddings.py` — `sentence-transformers` (pubmedbert) or `transformers` (MedCPT) |
| Vector DB | `chromadb` (local, persistent) |
| LLM | `openai` SDK pointed at the university AI gateway (custom `base_url`) |
| Interface | Jupyter notebook |

CPU only.

---

## Project Structure

```
laminorag/
├── README.md
├── requirements.txt
├── .env                    # LLM_API_BASE, LLM_API_KEY
├── config.py               # Corpora, model names, retrieval params
├── llm.py                  # Thin OpenAI-SDK wrapper around the uni gateway
├── embeddings.py           # Backend-agnostic embed_documents / embed_query
├── ingest.py               # Load article → LLM chunk+tag → embed → store
├── query.py                # Forced cross-corpus retrieve → synthesize
├── data/
│   ├── articles/           # Raw input articles
│   └── chroma_db/          # Persistent Chroma storage
└── notebooks/
    └── demo.ipynb
```

Three small modules. Split further only when there's reason to.

---

## Configuration (`config.py`)

```python
PRIMARY_CORPUS = "laminopathy"
SIDE_CORPORA = ["lnp", "bioinformatics"]
ALL_CORPORA = [PRIMARY_CORPUS, *SIDE_CORPORA]

# Embedding backend: "pubmedbert" (symmetric) or "medcpt" (asymmetric article/query)
EMBEDDING_BACKEND = "pubmedbert"

CHROMA_PATH = "./data/chroma_db"
CHROMA_COLLECTION = "laminorag"

PRIMARY_CHUNKS = 6           # From laminopathy per query
SIDE_CHUNKS = 2              # From each side corpus per query

# LLM via uni AI gateway (OpenAI-compatible)
# Use whatever model identifier the gateway exposes (check uni docs).
LLM_MODEL = "claude-sonnet-4-20250514"
# LLM_API_BASE and LLM_API_KEY come from .env
```

Corpus names are the contract — the chunker is told the corpus list and must pick from it (or return `null` for unclassifiable).

---

## Module Responsibilities

### `llm.py`
Thin wrapper around the `openai` SDK pointed at the uni gateway. Single function:

```python
import os
from openai import OpenAI
from config import LLM_MODEL

_client = OpenAI(
    base_url=os.environ["LLM_API_BASE"],
    api_key=os.environ["LLM_API_KEY"],
)

def complete(prompt: str, system: str | None = None) -> str:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    response = _client.chat.completions.create(
        model=LLM_MODEL,
        messages=messages,
    )
    return response.choices[0].message.content
```

Module-level client reuses the underlying HTTP connection across calls. No streaming, no tool use, no fancy features in Phase 1 — text in, text out. Swapping models is a one-line config change.

### `embeddings.py`
Wraps the embedding model behind a backend-agnostic API so swapping families is a one-line `config.py` change. Exposes two functions:

```python
def embed_documents(texts: list[str]) -> list[list[float]]: ...
def embed_query(text: str) -> list[float]: ...
```

- `EMBEDDING_BACKEND = "pubmedbert"` → both functions delegate to one `SentenceTransformer("neuml/pubmedbert-base-embeddings")` (symmetric retrieval).
- `EMBEDDING_BACKEND = "medcpt"` → loads `ncbi/MedCPT-Article-Encoder` and `ncbi/MedCPT-Query-Encoder` as HF `AutoModel`s. `embed_documents` uses the article encoder (max 512 tokens); `embed_query` uses the query encoder (max 64 tokens). Both pool the CLS token and L2-normalize.

`ingest.py` always calls `embed_documents`; `query.py` always calls `embed_query`. The asymmetric/symmetric distinction never leaks out of this module.

### `ingest.py`
- Input: path to an article (text or PDF)
- Single LLM call: pass article text + corpus list, get back JSON `{chunks: [{text, corpus, source_title}]}` where `text` is a verbatim passage and `corpus` is one of the configured corpora (or `null`)
- Embed each chunk's `text` via `embeddings.embed_documents`
- Upsert into Chroma with metadata `{corpus, source_title, source_path}`
- Idempotent on re-runs of the same article (key by content hash or source path)

### `query.py`
- Input: user question
- Embed the question via `embeddings.embed_query`
- Query Chroma once per corpus with that vector, filtered by `where={"corpus": <name>}`:
  - `PRIMARY_CHUNKS` from laminopathy
  - `SIDE_CHUNKS` from each side corpus
- Build prompt grouped by corpus:
  ```
  You are a biomedical research assistant specialized in laminopathies.
  Answer the user's question using the literature below, grouped by domain.

  [LAMINOPATHY — primary]
  {laminopathy_chunks with source titles}

  [LNP DELIVERY]
  {lnp_chunks}

  [BIOINFORMATICS]
  {bioinformatics_chunks}

  Question: {question}

  Ground answers in the cited passages. Where the side corpora suggest
  non-obvious connections to laminopathy, surface them as hypotheses
  rather than claims. Cite sources by title.
  ```
- Call `llm.complete`, return response + per-corpus source titles

---

## Demo Notebook Flow

1. Load config, init Chroma, load embedding model, verify `LLM_API_BASE` / `LLM_API_KEY` are set
2. Run `ingest.py` over a handful of articles spanning all three corpora (one-time, persisted)
3. Ask a laminopathy-centered question and a cross-corpus question; show grouped retrieval + final answers

---

## Environment

```bash
python -m venv venv
venv\Scripts\activate
pip install openai chromadb sentence-transformers jupyter python-dotenv pypdf
```

`.env`:
```
LLM_API_BASE=https://<uni-gateway-host>/v1
LLM_API_KEY=...
```

---

## Implementation Notes

- The corpus is **growing** — design for incremental ingestion, not one-shot batches.
- Embedding model first load is slow (pubmedbert ~400MB; MedCPT loads two encoders, ~440MB each). `embeddings.py` loads the model(s) at import time so they cache across queries.
- Switching `EMBEDDING_BACKEND` requires re-ingesting the corpus — vectors from different model families aren't comparable in the same Chroma collection. Delete `data/chroma_db/` (or use a different `CHROMA_COLLECTION`) and re-run ingest.
- Chroma metadata filter syntax: `where={"corpus": "laminopathy"}`. Verify against installed Chroma version.
- The LLM chunker is the quality bottleneck. Chunks should be semantically coherent passages, not arbitrary token windows. Iterate on the chunker prompt before tuning retrieval ratios.
- `PRIMARY_CHUNKS` vs `SIDE_CHUNKS` ratio is a knob to tune. Start at 6:2:2; if side corpora dominate answers, lower side; if laminopathy answers feel narrow, lower primary.
- The uni gateway is OpenAI-compatible, so the `openai` SDK works as-is — just override `base_url`. `LLM_MODEL` is whatever string the gateway accepts (check uni docs for the exact identifier).
- If the gateway adds an unusual auth header or path prefix beyond standard Bearer + `/v1`, the `openai` SDK supports `default_headers=` and the `base_url` can include path segments.

---

## Open Items (Phase 1 follow-ups)

1. **Section-aware embeddings.** Store the Docling section heading in chunk metadata; prepend `{source_title} — {section}` to the *embedded* text only (not the stored document) so retrieval gets structural signal. In the synthesis prompt, print each source title once and group its chunks under it instead of repeating `<<source>>` per chunk — lowers token cost when several chunks share a PDF.
2. **Notebook verification.** `notebooks/demo.ipynb` holds stale empty-collection outputs; run it clean end-to-end once against the populated Chroma.
3. **Chunk-quality spot-check.** Per paper, dump each chunk's section heading + char length + extracted entities; scan for garbage (running-header fragments, reference/author-list bleed, pervasively empty entities, tiny chunks).
4. **Populate side corpora.** `lnp` and `bioinformatics` folders are empty; forced cross-corpus retrieval is unexercised until papers are added there.
