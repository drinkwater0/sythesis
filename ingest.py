import hashlib
import re
from pathlib import Path

import chromadb
from chromadb.api.types import Metadata
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend

import embeddings
import llm
from config import ALL_CORPORA, CHROMA_COLLECTION, CHROMA_PATH

_PARSED_DIR = Path(__file__).parent / "data" / "parsed"

_client = chromadb.PersistentClient(path=CHROMA_PATH)
_collection = _client.get_or_create_collection(name=CHROMA_COLLECTION)

_converter: DocumentConverter | None = None


def _get_converter() -> DocumentConverter:
    global _converter
    if _converter is None:
        # The default native parsing backend exhausts memory (std::bad_alloc)
        # past ~page 16 on long PDFs, silently truncating them. Inputs are
        # born-digital (no scans), so the lightweight PyPdfium backend parses
        # the full document with identical section structure, and OCR is
        # unnecessary. The layout model still detects section headings.
        opts = PdfPipelineOptions(do_ocr=False)
        _converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=opts, backend=PyPdfiumDocumentBackend
                )
            }
        )
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


def _cache_path(pdf: Path) -> Path:
    # Don't use with_suffix — pdf.stem may contain dots (e.g. "Genes Dev.-2010-...")
    # and with_suffix would strip everything after the first dot.
    return _PARSED_DIR / pdf.parent.name / f"{pdf.stem}.md"


def _load_document(path: Path) -> tuple[str, bool]:
    """Returns (body_markdown, cached). Figure captions, if any, are appended
    to the markdown under a trailing '## Figure captions' section."""
    if path.suffix.lower() != ".pdf":
        return path.read_text(encoding="utf-8"), False

    md_path = _cache_path(path)
    if md_path.exists() and md_path.stat().st_mtime >= path.stat().st_mtime:
        return md_path.read_text(encoding="utf-8"), True

    result = _get_converter().convert(str(path))
    doc = result.document
    body = doc.export_to_markdown().replace("<!-- image -->", "")
    captions = _extract_figure_captions(doc)
    if captions:
        body = body.rstrip() + "\n\n## Figure captions\n\n" + "\n\n".join(captions) + "\n"

    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(body, encoding="utf-8")
    return body, False


def _corpus_from_path(path: Path) -> str:
    corpus = path.parent.name
    if corpus not in ALL_CORPORA:
        raise ValueError(
            f"{path} must live under one of {ALL_CORPORA}; found '{corpus}'"
        )
    return corpus


def _extract_title(markdown: str, fallback: str) -> str:
    # A real H1 title lives in the preamble (before the first ## section).
    # Stop at the first ## so we don't pick up code-block/shebang lines that
    # Docling mis-renders as '# ' headings deeper in the document.
    for line in markdown.split("\n"):
        if line.startswith("## "):
            break
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


def ingest(path: str | Path, target_chars: int = 2000) -> dict:
    path = Path(path)
    corpus = _corpus_from_path(path)
    article_text, cached = _load_document(path)
    source_title = _extract_title(article_text, path.stem)

    texts: list[str] = []
    for heading, body in _split_sections(article_text):
        if _should_skip(heading):
            continue
        texts.extend(_pack_paragraphs(body, target_chars=target_chars))

    if not texts:
        return {"source_title": source_title, "corpus": corpus, "ingested": 0, "cached": cached}

    # Idempotency: drop any prior chunks from this source before re-adding.
    _collection.delete(where={"source_path": str(path)})

    # Dedupe by id: a caption may appear both inline in the body and in the
    # trailing "## Figure captions" section.
    seen: set[str] = set()
    ids: list[str] = []
    unique_texts: list[str] = []
    for t in texts:
        cid = _chunk_id(str(path), t)
        if cid in seen:
            continue
        seen.add(cid)
        ids.append(cid)
        unique_texts.append(t)

    print(f"[ingest] {path.name}: extracting entities for {len(unique_texts)} chunks...")
    kept_ids: list[str] = []
    kept_texts: list[str] = []
    extractions: list[dict] = []
    skipped = 0
    for i, (cid, t) in enumerate(zip(ids, unique_texts), 1):
        try:
            e = llm.extract_entities(t)
        except Exception as ex:
            print(f"  [{i}/{len(unique_texts)}] extraction failed: {type(ex).__name__}: {ex}")
            e = {"genes_proteins": [], "methods": [], "concepts": [], "skip": False}
        if e["skip"]:
            skipped += 1
            print(f"  [skip] {t[:60]!a}")
            continue
        kept_ids.append(cid)
        kept_texts.append(t)
        extractions.append(e)

    if not kept_texts:
        return {"source_title": source_title, "corpus": corpus, "ingested": 0,
                "skipped": skipped, "cached": cached}

    # Chroma rejects empty list metadata values, so only attach non-empty buckets.
    metadatas: list[Metadata] = []
    for e in extractions:
        m: Metadata = {
            "corpus": corpus,
            "source_title": source_title,
            "source_path": str(path),
        }
        for key in ("genes_proteins", "methods", "concepts"):
            if e[key]:
                m[key] = e[key]
        metadatas.append(m)
    vectors = embeddings.embed_documents(kept_texts)
    _collection.add(ids=kept_ids, documents=kept_texts, embeddings=vectors, metadatas=metadatas)

    return {
        "source_title": source_title,
        "corpus": corpus,
        "ingested": len(kept_texts),
        "skipped": skipped,
        "cached": cached,
    }
