from pathlib import Path

PRIMARY_CORPUS = "laminopathy"
SIDE_CORPORA = ["lnp", "bioinformatics"]
ALL_CORPORA = [PRIMARY_CORPUS, *SIDE_CORPORA]

# Embedding backend: "pubmedbert" (symmetric) or "medcpt" (asymmetric article/query)
EMBEDDING_BACKEND = "pubmedbert"

# Resolved relative to this file so cwd (notebook vs repo root) doesn't matter.
CHROMA_PATH = str(Path(__file__).parent / "data" / "chroma_db")
CHROMA_COLLECTION = "laminorag"

PRIMARY_CHUNKS = 16
SIDE_CHUNKS = 8

# Cap chunks taken from any single paper per corpus, so one on-topic paper can't
# fill the whole quota with near-duplicate passages. Backfilled past the cap only
# when a corpus has too few papers to reach the quota otherwise.
MAX_PER_SOURCE = 4

# Two LLM roles via an OpenAI-compatible endpoint. LLM_API_BASE / LLM_API_KEY from .env.
INGEST_MODEL = "qwen3.6-35b"   # per-chunk entity extraction during ingestion
SYNTH_MODEL = "qwen3.5:122b"   # answer synthesis at query time (endpoint backend)

# Synthesis backend: "endpoint" -> SYNTH_MODEL via the OpenAI-compatible endpoint;
# "claude_code" -> local Claude Code CLI (Pro subscription, no API key). Fallback
# for when the endpoint is unavailable. Ingestion always uses the endpoint.
SYNTH_BACKEND = "endpoint"
