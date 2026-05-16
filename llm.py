import json
import os

from dotenv import load_dotenv
from openai import OpenAI

from config import LLM_MODEL

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


def complete(prompt: str, system: str | None = None) -> str:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    stream = _get_client().chat.completions.create(
        model=LLM_MODEL,
        messages=messages,
        stream=True,
    )
    parts: list[str] = []
    for event in stream:
        delta = event.choices[0].delta.content if event.choices else None
        if delta:
            parts.append(delta)
    return "".join(parts)


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
        model=LLM_MODEL,
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
