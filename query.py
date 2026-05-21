from collections import Counter

import chromadb

import embeddings
import llm
from config import (
    CHROMA_COLLECTION,
    CHROMA_PATH,
    MAX_PER_SOURCE,
    PRIMARY_CHUNKS,
    PRIMARY_CORPUS,
    SIDE_CHUNKS,
    SIDE_CORPORA,
)

_client = chromadb.PersistentClient(path=CHROMA_PATH)
_collection = _client.get_or_create_collection(name=CHROMA_COLLECTION)


SYNTHESIS_SYSTEM = """You are a biomedical research assistant specialized in laminopathies.
Answer the user's question using only the literature passages provided, grouped by domain.
Ground answers in the cited passages. Where the side corpora (LNP delivery, bioinformatics) suggest non-obvious connections to laminopathy, surface them as hypotheses rather than claims. Cite sources by title."""


def _retrieve(query_vec: list[float], corpus: str, k: int,
              max_per_source: int = MAX_PER_SOURCE) -> list[dict]:
    # Over-fetch, then take chunks in similarity order while capping per source so one
    # on-topic paper can't crowd out the rest. Backfill past the cap (still in
    # similarity order) only if too few papers exist to reach k.
    result = _collection.query(
        query_embeddings=[query_vec],
        n_results=k * 4,
        where={"corpus": corpus},
    )
    candidates = [
        {
            "text": d,
            "source_title": m["source_title"],
            "genes_proteins": m.get("genes_proteins") or [],
            "methods": m.get("methods") or [],
            "concepts": m.get("concepts") or [],
        }
        for d, m in zip(result["documents"][0], result["metadatas"][0])
    ]

    selected: list[dict] = []
    overflow: list[dict] = []
    per_source: Counter = Counter()
    for c in candidates:
        if per_source[c["source_title"]] < max_per_source:
            selected.append(c)
            per_source[c["source_title"]] += 1
        else:
            overflow.append(c)
    selected = selected[:k]
    if len(selected) < k:
        selected += overflow[: k - len(selected)]
    return selected


def _entity_tags(chunk: dict) -> str:
    parts: list[str] = []
    for key, label in (("genes_proteins", "Genes"), ("methods", "Methods"), ("concepts", "Concepts")):
        items = chunk.get(key) or []
        if items:
            parts.append(f"[{label}: {', '.join(items)}]")
    return " ".join(parts)


def _format_group(label: str, chunks: list[dict]) -> str:
    if not chunks:
        return f"[{label}]\n(no relevant passages)\n"
    blocks = []
    for c in chunks:
        header = f"<<source: {c['source_title']}>>"
        tags = _entity_tags(c)
        if tags:
            header += " " + tags
        blocks.append(f"{header}\n{c['text']}")
    return f"[{label}]\n" + "\n\n".join(blocks) + "\n"


def ask(question: str) -> dict:
    qvec = embeddings.embed_query(question)

    primary_hits = _retrieve(qvec, PRIMARY_CORPUS, PRIMARY_CHUNKS)
    side_hits = {c: _retrieve(qvec, c, SIDE_CHUNKS) for c in SIDE_CORPORA}

    sections = [_format_group(f"{PRIMARY_CORPUS.upper()} — primary", primary_hits)]
    sections += [_format_group(c.upper(), side_hits[c]) for c in SIDE_CORPORA]

    prompt = "\n".join(sections) + f"\nQuestion: {question}"
    answer = llm.complete(prompt=prompt, system=SYNTHESIS_SYSTEM)

    sources = {PRIMARY_CORPUS: sorted({h["source_title"] for h in primary_hits})}
    for c, hits in side_hits.items():
        sources[c] = sorted({h["source_title"] for h in hits})

    return {"answer": answer, "sources": sources}
