import hashlib
import json
from pathlib import Path

import chromadb
from pypdf import PdfReader

import embeddings
import llm
from config import ALL_CORPORA, CHROMA_COLLECTION, CHROMA_PATH

_client = chromadb.PersistentClient(path=CHROMA_PATH)
_collection = _client.get_or_create_collection(name=CHROMA_COLLECTION)


CHUNKER_SYSTEM = """You are a chunking assistant for a biomedical research corpus.

Your task: split the article into semantically coherent passages of verbatim text, and tag each with the corpus it belongs to.

Each passage should:
- Be a contiguous, verbatim span from the original text (no paraphrasing, no summarization).
- Cover one self-contained finding, method, or argument.
- Be roughly 150-300 words. Coherence matters more than exact length.
- Skip headers, page footers, references, and figure captions unless they carry substantive content.

Corpus tagging:
- Available corpora: {corpora}.
- Use the corpus name that best fits the passage's topic.
- If a passage is generic introductory material or doesn't fit any corpus, tag corpus as null.
- A single article may produce chunks tagged with different corpora if it spans topics.

Return ONLY valid JSON (no markdown fence, no commentary) with this shape:
{{
  "source_title": "<the article's title>",
  "chunks": [
    {{"text": "<verbatim passage>", "corpus": "<corpus name or null>"}}
  ]
}}
"""


def _load_text(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        reader = PdfReader(str(path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    return path.read_text(encoding="utf-8")


def _parse_json(text: str) -> dict:
    # LLMs sometimes wrap JSON in ```json ... ``` despite instructions.
    text = text.strip()
    if text.startswith("```"):
        parts = text.split("```")
        if len(parts) >= 3:
            text = parts[1].lstrip()
            if text.startswith("json"):
                text = text[4:].lstrip()
    return json.loads(text)


def _chunk_id(source_path: str, text: str) -> str:
    return hashlib.sha1(f"{source_path}::{text}".encode("utf-8")).hexdigest()[:16]


def ingest(path: str | Path) -> dict:
    path = Path(path)
    article_text = _load_text(path)

    system = CHUNKER_SYSTEM.format(corpora=ALL_CORPORA)
    response = llm.complete(prompt=article_text, system=system)
    parsed = _parse_json(response)

    source_title = parsed.get("source_title") or path.stem
    chunks = [c for c in parsed["chunks"] if c.get("corpus") in ALL_CORPORA]

    if not chunks:
        return {"source_title": source_title, "ingested": 0, "skipped": len(parsed["chunks"])}

    # Idempotency: drop any prior chunks from this source before re-adding.
    _collection.delete(where={"source_path": str(path)})

    texts = [c["text"] for c in chunks]
    ids = [_chunk_id(str(path), t) for t in texts]
    metadatas = [
        {"corpus": c["corpus"], "source_title": source_title, "source_path": str(path)}
        for c in chunks
    ]
    vectors = embeddings.embed_documents(texts)

    _collection.add(ids=ids, documents=texts, embeddings=vectors, metadatas=metadatas)

    return {
        "source_title": source_title,
        "ingested": len(chunks),
        "skipped": len(parsed["chunks"]) - len(chunks),
    }
