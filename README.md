# LaminoRAG

LaminoRAG is a local retrieval-augmented assistant for laminopathy research. It
indexes a personal library of papers — primarily laminopathy, with smaller
collections on lipid-nanoparticle (LNP) delivery and bioinformatics — and
answers questions grounded in that literature.

## The idea

Ordinary RAG ranks passages purely by similarity to the question, so a
laminopathy query comes back with laminopathy passages and nothing else.
LaminoRAG instead retrieves a fixed quota from every corpus on each query — six
passages from laminopathy plus two from each side collection — so an answer
stays anchored in the core domain while still pulling in the delivery and
computational work that pure similarity would never rank highly enough to
surface. Those cross-domain connections are the point; forcing the quota is
what makes them appear.

## How it works

Chunking is deterministic. A PDF is parsed to markdown with Docling, split on
its section headings, and packed into ~2,000-character passages; reference,
acknowledgment, and supplementary sections are dropped by heading match. Each
passage then gets one lightweight LLM pass (qwen3.6-35b) that pulls out the
genes, methods, and concepts it mentions as searchable metadata and flags
non-content fragments — stray reference lists, vendor boilerplate — to skip.
The survivors are embedded with a biomedical model
(`neuml/pubmedbert-base-embeddings`) and stored in a local Chroma collection,
tagged with the corpus they came from — simply the folder they were filed
under, so there is no classification step. Ingestion is incremental and
idempotent: re-running skips papers already stored.

A query embeds the question, retrieves the per-corpus quota from Chroma, groups
the passages by corpus into a single prompt, and sends it to qwen3.5:122b for
synthesis. The answer comes back with its sources grouped by domain.

| Module | Responsibility |
|---|---|
| `ingest.py` | PDF → markdown → deterministic chunking → embed → store |
| `embeddings.py` | Backend-agnostic embedding (pubmedbert; MedCPT swap is one config flip) |
| `query.py` | Forced cross-corpus retrieval and grouped synthesis |
| `llm.py` | Thin OpenAI-SDK wrapper: qwen3.6-35b for extraction, qwen3.5:122b for synthesis |
| `config.py` | Corpora, retrieval quotas, model identifiers |

## Setup

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env      # set LLM_API_BASE and LLM_API_KEY
```

Drop articles into `data/articles/<corpus>/`, where `<corpus>` is `laminopathy`,
`lnp`, or `bioinformatics`. The folder name becomes the corpus tag.

## Usage

Open `notebooks/demo.ipynb` and run it top to bottom: it ingests anything new,
then answers a laminopathy question and a cross-corpus one, showing the
retrieved sources grouped by corpus. The same thing in code:

```python
import query
r = query.ask("Can bioinformatics approaches improve LNP targeting for laminopathy?")
print(r["answer"])
print(r["sources"])     # source titles, grouped by corpus
```

## Design notes

Three choices shaped the system. Chunk boundaries are deterministic rather than
LLM-driven: an earlier per-article LLM chunker timed out on large PDFs, and
splitting on section headings turned out to be faster, reproducible, and good
enough. The embeddings are biomedical by necessity — raw passages from this
literature need domain embeddings, not a general-purpose model, which is why
pubmedbert is the default and the only sanctioned swap is to another biomedical
encoder. And the two LLM roles are split deliberately: the smaller qwen3.6-35b
does the high-volume per-chunk extraction, while the larger qwen3.5:122b is
reserved for the one quality-critical step — answer synthesis, which can fail
over to Claude (the local Claude Code CLI on a Pro subscription, no API key)
with a single `SYNTH_BACKEND` flip if the endpoint is unavailable.

Phase 1 covers the three corpora, deterministic ingest, forced retrieval, and
the notebook demo. A PubMed fetcher, a web UI, and automated evaluation are
deliberately out of scope.
