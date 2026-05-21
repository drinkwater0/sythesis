import json
import os
import shutil
import subprocess

from dotenv import load_dotenv
from openai import OpenAI

from config import INGEST_MODEL, SYNTH_MODEL, SYNTH_BACKEND

load_dotenv()

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    # Lazy so `import llm` works before .env is populated.
    global _client
    if _client is None:
        _client = OpenAI(
            base_url=os.environ["LLM_API_BASE"],
            api_key=os.environ["LLM_API_KEY"],
        )
    return _client


def _complete_endpoint(prompt: str, system: str | None) -> str:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    stream = _get_client().chat.completions.create(
        model=SYNTH_MODEL,
        messages=messages,
        stream=True,
    )
    parts: list[str] = []
    for event in stream:
        delta = event.choices[0].delta.content if event.choices else None
        if delta:
            parts.append(delta)
    return "".join(parts)


def _run_claude(full: str) -> str:
    cmd = ["cmd", "/c", "claude", "-p"] if os.name == "nt" else ["claude", "-p"]
    proc = subprocess.run(
        cmd, input=full, capture_output=True, text=True, encoding="utf-8", timeout=300
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"claude -p failed (exit {proc.returncode}): {proc.stderr.strip()[:300]}"
        )
    return proc.stdout.strip()


def _complete_claude_code(prompt: str, system: str | None) -> str:
    # Synthesis via the local Claude Code CLI (Pro subscription auth, no API key).
    # Prompt goes over stdin to dodge Windows arg-length limits and shell quoting.
    if shutil.which("claude") is None:
        raise RuntimeError("SYNTH_BACKEND='claude_code' but the `claude` CLI is not on PATH")
    full = f"{system}\n\n{prompt}" if system else prompt
    out = _run_claude(full)
    if not out:  # first headless run in a fresh env can return blank; retry once
        out = _run_claude(full)
    if not out:
        raise RuntimeError("claude -p returned empty output twice")
    return out


def complete(prompt: str, system: str | None = None) -> str:
    if SYNTH_BACKEND == "claude_code":
        return _complete_claude_code(prompt, system)
    return _complete_endpoint(prompt, system)


_EXTRACT_SYSTEM = """You extract structured metadata from passages of biomedical research papers.

Return a JSON object with exactly these four keys:

- "genes_proteins": list of gene or protein names mentioned in the passage (e.g. "LMNA", "YAP", "Oct4"). Use the canonical symbol. Only items appearing literally in the passage text. Empty list [] if none.
- "methods": list of experimental techniques, assays, or computational tools mentioned (e.g. "ChIP-seq", "lentiviral knockdown", "quantitative PCR"). Only items appearing literally. Empty list [] if none.
- "concepts": list of 2 to 5 short noun phrases (1-4 words each) capturing the topical focus (e.g. "stem cell pluripotency", "nuclear envelope assembly"). May paraphrase.
- "skip": boolean. Set true ONLY if the passage is clearly not scientific content — an author/affiliation list, a reference/citation list, a journal running-header fragment, garbled or scrambled text, or a boilerplate metadata section (funding, author contributions, data/code availability, additional information). When in any doubt, set false. NEVER set true merely because the passage is short or terse.

Before responding, self-check: for each item in "genes_proteins" and "methods", confirm it appears literally in the passage. Drop any that do not.

Return only the JSON object. No prose, no markdown fences."""


def _parse_extraction(raw: str) -> dict:
    start, end = raw.find("{"), raw.rfind("}")
    if start == -1 or end <= start:
        return {}
    try:
        result = json.loads(raw[start : end + 1])
    except json.JSONDecodeError:
        return {}
    return result if isinstance(result, dict) else {}


def extract_entities(text: str) -> dict:
    """Returns {'genes_proteins': [...], 'methods': [...], 'concepts': [...], 'skip': bool}.
    Falls back to empty lists / skip=False if the model returns unparseable output."""
    response = _get_client().chat.completions.create(
        model=INGEST_MODEL,
        messages=[
            {"role": "system", "content": _EXTRACT_SYSTEM},
            {"role": "user", "content": text},
        ],
        extra_body={"think": False},  # Ollama flag; ~16x faster, extraction needs no reasoning
    )
    raw = response.choices[0].message.content or ""
    parsed = _parse_extraction(raw)
    out: dict = {"skip": bool(parsed.get("skip"))}
    for key in ("genes_proteins", "methods", "concepts"):
        value = parsed.get(key, [])
        out[key] = [str(v) for v in value if isinstance(v, (str, int, float))] if isinstance(value, list) else []
    return out
