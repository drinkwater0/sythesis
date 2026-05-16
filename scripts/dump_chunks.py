"""Chunk-quality spot-check: one compact line per chunk for every PDF.
Same chunking + extraction as ingest(), but prints instead of storing.
Scan output for tiny chunks, junk section names, reference bleed,
pervasively empty entities."""
import glob
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ingest import _load_document, _pack_paragraphs, _should_skip, _split_sections  # type: ignore
from llm import extract_entities


def chunks_with_sections(text: str) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for heading, body in _split_sections(text):
        if _should_skip(heading):
            continue
        for c in _pack_paragraphs(body, target_chars=2000):
            out.append((heading, c))
    return out


def main() -> None:
    for pdf in sorted(glob.glob("data/articles/**/*.pdf", recursive=True)):
        text, _ = _load_document(Path(pdf))
        chunks = chunks_with_sections(text)
        print(f"\n===== {Path(pdf).name}  ({len(chunks)} chunks) =====")
        for i, (section, body) in enumerate(chunks):
            e = extract_entities(body)
            g = ",".join(e["genes_proteins"]) or "-"
            m = ",".join(e["methods"]) or "-"
            c = ",".join(e["concepts"]) or "-"
            sec = (section[:45] + "…") if len(section) > 46 else section
            flag = " <!>" if len(body) < 150 else ""
            print(f"[{i:2}] {len(body):4}c | {sec:46} | g={g} | m={m} | c={c}{flag}")


if __name__ == "__main__":
    main()
