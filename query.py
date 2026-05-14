import chromadb

import embeddings
import llm
from config import (
    CHROMA_COLLECTION,
    CHROMA_PATH,
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


def _retrieve(query_vec: list[float], corpus: str, k: int) -> list[dict]:
    result = _collection.query(
        query_embeddings=[query_vec],
        n_results=k,
        where={"corpus": corpus},
    )
    docs = result["documents"][0]
    metas = result["metadatas"][0]
    return [
        {
            "text": d,
            "source_title": m["source_title"],
            "genes_proteins": m.get("genes_proteins") or [],
            "methods": m.get("methods") or [],
            "concepts": m.get("concepts") or [],
        }
        for d, m in zip(docs, metas)
    ]


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
