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

Return a JSON object with exactly these three keys, each a list of strings (empty list [] if nothing applies):

- "genes_proteins": gene or protein names mentioned in the passage (e.g. "LMNA", "YAP", "Oct4"). Use the canonical symbol. Only include items that appear literally in the passage text.
- "methods": experimental techniques, assays, or computational tools mentioned (e.g. "ChIP-seq", "lentiviral knockdown", "quantitative PCR"). Only items appearing in the passage text.
- "concepts": 2 to 5 short noun phrases (1-4 words each) capturing the topical focus of the passage (e.g. "stem cell pluripotency", "nuclear envelope assembly"). These may paraphrase the text.

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


def extract_entities(text: str) -> dict[str, list[str]]:
    """Returns {'genes_proteins': [...], 'methods': [...], 'concepts': [...]}.
    Falls back to empty lists if the model returns unparseable output."""
    response = _get_client().chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": _EXTRACT_SYSTEM},
            {"role": "user", "content": text},
        ],
    )
    raw = response.choices[0].message.content or ""
    parsed = _parse_extraction(raw)
    if not isinstance(parsed, dict):
        return {"genes_proteins": [], "methods": [], "concepts": []}
    out: dict[str, list[str]] = {}
    for key in ("genes_proteins", "methods", "concepts"):
        value = parsed.get(key, [])
        out[key] = [str(v) for v in value if isinstance(v, (str, int, float))] if isinstance(value, list) else []
    return out
