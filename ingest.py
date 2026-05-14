import hashlib
import re
from pathlib import Path

import chromadb
from chromadb.api.types import Metadata
from docling.document_converter import DocumentConverter

import embeddings
from config import ALL_CORPORA, CHROMA_COLLECTION, CHROMA_PATH

_client = chromadb.PersistentClient(path=CHROMA_PATH)
_collection = _client.get_or_create_collection(name=CHROMA_COLLECTION)

_converter: DocumentConverter | None = None


def _get_converter() -> DocumentConverter:
    global _converter
    if _converter is None:
        _converter = DocumentConverter()
    return _converter


# Section headings that aren't substantive content — skip these and everything under them.
_SKIP_HEADINGS = (
    "references",
    "bibliography",
    "acknowledg",
    "supplementary",
    "funding",
    "author contributions",
    "competing interests",
    "conflict of interest",
    "appendix",
    "data availability",
)


_FIGURE_CAPTION_RE = re.compile(r"^\s*(figure|fig\.?)\s*\d", re.IGNORECASE)


def _extract_figure_captions(doc) -> list[str]:
    # Docling extracts captions inconsistently across publishers — keep only entries
    # that actually start like a figure caption to avoid injecting body-text bleed.
    captions: list[str] = []
    for pic in doc.pictures:
        parts: list[str] = []
        for ref in (pic.captions or []):
            try:
                target = ref.resolve(doc)
            except Exception:
                target = None
            if target is not None and getattr(target, "text", None):
                parts.append(target.text)
        text = " ".join(parts).strip()
        if text and _FIGURE_CAPTION_RE.match(text):
            captions.append(text)
    return captions


def _load_document(path: Path) -> tuple[str, list[str]]:
    if path.suffix.lower() == ".pdf":
        result = _get_converter().convert(str(path))
        doc = result.document
        body = doc.export_to_markdown().replace("<!-- image -->", "")
        return body, _extract_figure_captions(doc)
    return path.read_text(encoding="utf-8"), []


def _corpus_from_path(path: Path) -> str:
    corpus = path.parent.name
    if corpus not in ALL_CORPORA:
        raise ValueError(
            f"{path} must live under one of {ALL_CORPORA}; found '{corpus}'"
        )
    return corpus


def _extract_title(markdown: str, fallback: str) -> str:
    for line in markdown.split("\n"):
        if line.startswith("# "):
            return line[2:].strip()
    return fallback


def _split_sections(markdown: str) -> list[tuple[str, str]]:
    # Split on level-2 headers. Content before the first ## is a "preamble" section.
    sections: list[tuple[str, list[str]]] = [("", [])]
    for line in markdown.split("\n"):
        if line.startswith("## "):
            sections.append((line[3:].strip(), []))
        else:
            sections[-1][1].append(line)
    return [(h, "\n".join(body).strip()) for h, body in sections if "\n".join(body).strip()]


def _should_skip(heading: str) -> bool:
    h = heading.lower()
    return any(p in h for p in _SKIP_HEADINGS)


def _pack_paragraphs(text: str, target_chars: int) -> list[str]:
    # Greedily pack paragraphs into chunks <= target_chars; never split mid-paragraph.
    chunks: list[str] = []
    buf: list[str] = []
    buf_len = 0
    for para in text.split("\n\n"):
        para = para.strip()
        if not para:
            continue
        para_len = len(para) + 2
        if buf and buf_len + para_len > target_chars:
            chunks.append("\n\n".join(buf))
            buf, buf_len = [], 0
        buf.append(para)
        buf_len += para_len
    if buf:
        chunks.append("\n\n".join(buf))
    return chunks


def _chunk_id(source_path: str, text: str) -> str:
    return hashlib.sha1(f"{source_path}::{text}".encode("utf-8")).hexdigest()[:16]


def ingest(path: str | Path, target_chars: int = 1500) -> dict:
    path = Path(path)
    corpus = _corpus_from_path(path)
    article_text, figure_captions = _load_document(path)
    source_title = _extract_title(article_text, path.stem)

    chunks: list[tuple[str, str]] = []
    for heading, body in _split_sections(article_text):
        if _should_skip(heading):
            continue
        for t in _pack_paragraphs(body, target_chars=target_chars):
            chunks.append(("text", t))
    chunks.extend(("figure_caption", c) for c in figure_captions)

    if not chunks:
        return {"source_title": source_title, "corpus": corpus, "ingested": 0}

    # Idempotency: drop any prior chunks from this source before re-adding.
    _collection.delete(where={"source_path": str(path)})

    # Dedupe by id: a caption may also live inline in the body markdown.
    seen: set[str] = set()
    ids: list[str] = []
    texts: list[str] = []
    metadatas: list[Metadata] = []
    by_type: dict[str, int] = {}
    for ctype, text in chunks:
        cid = _chunk_id(str(path), text)
        if cid in seen:
            continue
        seen.add(cid)
        ids.append(cid)
        texts.append(text)
        metadatas.append({
            "corpus": corpus,
            "source_title": source_title,
            "source_path": str(path),
            "type": ctype,
        })
        by_type[ctype] = by_type.get(ctype, 0) + 1

    vectors = embeddings.embed_documents(texts)
    _collection.add(ids=ids, documents=texts, embeddings=vectors, metadatas=metadatas)

    return {
        "source_title": source_title,
        "corpus": corpus,
        "ingested": len(texts),
        "by_type": by_type,
    }
